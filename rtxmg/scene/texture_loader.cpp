/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "rtxmg/scene/texture_loader.h"

#include <donut/core/log.h>
#include <donut/engine/DDSFile.h>
#include <donut/engine/KTX2File.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using namespace donut;

namespace rtxmg
{
namespace
{
    std::string LowerExt(const fs::path& p)
    {
        std::string e = p.extension().string();
        std::transform(e.begin(), e.end(), e.begin(),
                       [](unsigned char c) { return char(std::tolower(c)); });
        return e;
    }

    // Read just the leading bytes of a file. Returns empty on failure.
    std::vector<char> ReadHead(const fs::path& path, size_t maxBytes)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f) return {};
        std::vector<char> buf(maxBytes);
        f.read(buf.data(), std::streamsize(maxBytes));
        buf.resize(size_t(f.gcount()));
        return buf;
    }

    bool IsBC5(nvrhi::Format format)
    {
        return format == nvrhi::Format::BC5_UNORM || format == nvrhi::Format::BC5_SNORM;
    }

    // Whether an MR map needs the BC5 component remap supplied from outside the file:
    // true for a two-channel BC5 that does not declare its own mapping.  A .ktx2 written
    // with KTXswizzle answers for itself; .dds has no key/value block, so it always
    // relies on this.  Keyed on the extension because that is what decides which loader
    // donut hands the file to.
    bool NeedsLegacyBC5MRMapping(const fs::path& fp)
    {
        // 4 KB covers a KTX2 header + level index + key/value data, and a DDS header
        // plus its DX10 block.
        const std::vector<char> head = ReadHead(fp, 4096);
        if (head.empty())
            return false;

        const std::string ext = LowerExt(fp);
        const std::string name = fp.generic_string();
#if DONUT_WITH_KTX
        if (ext == ".ktx2")
        {
            engine::KTX2HeaderInfo info;
            if (!engine::ReadKTX2Header(head.data(), head.size(), name.c_str(), info) || !info.supported)
                return false;
            return IsBC5(info.format) && !info.componentMapping;
        }
#endif
        if (ext == ".dds")
        {
            engine::DDSHeaderInfo info;
            if (!engine::ReadDDSHeader(head.data(), head.size(), name.c_str(), info) || !info.supported)
                return false;
            return IsBC5(info.format);
        }
        return false;
    }
} // namespace

fs::path ResolveMediapath(const fs::path& filepath, const fs::path& mediaPath)
{
    if (filepath.empty())
        return {};

    if (fs::is_regular_file(filepath))
        return filepath;

    if (!mediaPath.empty() && fs::is_regular_file(mediaPath / filepath))
        return mediaPath / filepath;

    return {};
}

fs::path TextureLoader::ResolveExisting(const fs::path& absPath) const
{
    if (absPath.empty())
        return {};

    fs::path fp = ResolveMediapath(absPath, m_mediaPath);
    if (!fs::is_regular_file(fp))
    {
        log::warning("Texture %s not found...", absPath.generic_string().c_str());
        return {};
    }
    return fp;
}

std::shared_ptr<engine::LoadedTexture> TextureLoader::LoadResolved(
    const fs::path& resolvedPath,
    engine::ThreadPool* pool,
    engine::SRGBMode sRGBMode,
    std::optional<nvrhi::ComponentMapping> componentMapping) const
{
    engine::TextureLoadOptions options;
    options.sRGBMode = sRGBMode;
    options.overrideComponentMapping = componentMapping;
    // The budget's verdict has to travel with the load: donut folds baseMip into
    // the cache key, so applying it afterwards would miss the texture entirely.
    if (auto it = m_baseMips.find(resolvedPath.generic_string()); it != m_baseMips.end())
        options.baseMip = it->second;

    return pool ? m_textureCache->LoadTextureFromFileAsync(resolvedPath, options, *pool)
                : m_textureCache->LoadTextureFromFileDeferred(resolvedPath, options);
}

std::shared_ptr<engine::LoadedTexture> TextureLoader::Load(
    const fs::path& absPath,
    engine::ThreadPool* pool,
    engine::SRGBMode sRGBMode,
    std::optional<nvrhi::ComponentMapping> componentMapping) const
{
    const fs::path fp = ResolveExisting(absPath);
    return fp.empty() ? nullptr : LoadResolved(fp, pool, sRGBMode, componentMapping);
}

std::shared_ptr<engine::LoadedTexture> TextureLoader::LoadMetallicRoughness(
    const fs::path& absPath, engine::ThreadPool* pool) const
{
    const fs::path fp = ResolveExisting(absPath);
    if (fp.empty())
        return nullptr;

    // A BC5 MR map (from texproc) holds R=roughness, G=metalness, so an SRV
    // component mapping redirects the shader's .g/.b reads onto them.  Files that
    // declare that themselves get an unset override and speak for themselves.
    std::optional<nvrhi::ComponentMapping> mrMapping;
    if (NeedsLegacyBC5MRMapping(fp))
    {
        // The same mapping as the KTXswizzle "0rg1" texproc writes: BC5 carries no
        // third or fourth channel, so those read as the spec's constants.
        mrMapping = nvrhi::ComponentMapping{
            nvrhi::ComponentSwizzle::Zero, // shader .r -> 0 (unused for MR)
            nvrhi::ComponentSwizzle::R,    // shader .g (roughness) -> physical R
            nvrhi::ComponentSwizzle::G,    // shader .b (metalness) -> physical G
            nvrhi::ComponentSwizzle::One };// shader .a -> 1
    }
    return LoadResolved(fp, pool, engine::SRGBMode::ForceLinear, mrMapping);
}

} // namespace rtxmg

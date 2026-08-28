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

#pragma once

#include <donut/engine/TextureCache.h>

#include <nvrhi/nvrhi.h>

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace rtxmg
{
    // Resolve a scene-authored asset path: the path itself if it names a file,
    // else mediaPath/path, else empty.  Models, envmaps and textures all go
    // through this, so a scene file can name assets either way.
    std::filesystem::path ResolveMediapath(const std::filesystem::path& filepath,
                                           const std::filesystem::path& mediaPath);

    // The scene's texture load policy: media-path resolution, the base mip the
    // budget pre-pass chose for each file, and the per-slot sRGB / component
    // swizzle a material needs.  Loads are deferred or pooled; GPU finalization
    // happens on the main thread either way.
    class TextureLoader
    {
    public:
        TextureLoader(std::shared_ptr<donut::engine::TextureCache> textureCache,
                      const std::filesystem::path& mediaPath)
            : m_textureCache(std::move(textureCache))
            , m_mediaPath(mediaPath)
        {
        }

        // Per-texture base mips from the budget pre-pass, keyed by the resolved
        // path's generic_string.  Format-agnostic: donut drops leading mips for
        // DDS as well as KTX2.  Must be set BEFORE any load: donut folds baseMip
        // into the texture cache key, so a later change would not reach an
        // already-loaded texture.  A missing entry keeps every mip.
        void SetBaseMips(std::unordered_map<std::string, uint32_t> baseMips)
        {
            m_baseMips = std::move(baseMips);
        }

        // A non-null `pool` moves the read + decode off the calling thread.  An unset
        // `componentMapping` leaves the file's own KTXswizzle in charge.
        std::shared_ptr<donut::engine::LoadedTexture> Load(
            const std::filesystem::path& absPath,
            donut::engine::ThreadPool* pool,
            donut::engine::SRGBMode sRGBMode,
            std::optional<nvrhi::ComponentMapping> componentMapping = std::nullopt) const;

        // Metallic-roughness may need the BC5 two-channel swizzle supplied from
        // outside the file, so it gets its own entry point instead of leaving that
        // to every call site.
        std::shared_ptr<donut::engine::LoadedTexture> LoadMetallicRoughness(
            const std::filesystem::path& absPath,
            donut::engine::ThreadPool* pool) const;

    private:
        // Resolved path, or empty with a warning when the file is not on disk.
        std::filesystem::path ResolveExisting(const std::filesystem::path& absPath) const;

        std::shared_ptr<donut::engine::LoadedTexture> LoadResolved(
            const std::filesystem::path& resolvedPath,
            donut::engine::ThreadPool* pool,
            donut::engine::SRGBMode sRGBMode,
            std::optional<nvrhi::ComponentMapping> componentMapping) const;

        std::shared_ptr<donut::engine::TextureCache> m_textureCache;
        const std::filesystem::path&                 m_mediaPath;
        std::unordered_map<std::string, uint32_t>    m_baseMips;
    };
}

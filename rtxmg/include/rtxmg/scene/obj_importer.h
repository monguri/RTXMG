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

#include <filesystem>
#include <memory>
#include <optional>

#include <nvrhi/utils.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/DescriptorTableManager.h>

#include "rtxmg/scene/model.h"

namespace donut::vfs
{
    class IFileSystem;
}

namespace donut::engine
{
    struct SceneImportResult;
    class TextureCache;
} // namespace donut::engine

namespace tf
{
    class Executor;
}

namespace fs = std::filesystem;

class TopologyCache;

class ObjImporter
{
protected:
    std::shared_ptr<donut::vfs::IFileSystem> m_fs;
    std::shared_ptr<donut::engine::DescriptorTableManager> m_descriptorTableManager;
    TopologyCache& m_topologyCache;

    fs::path m_modelPath;
    const fs::path& m_mediaPath;

public:
    explicit ObjImporter(
        std::shared_ptr<donut::vfs::IFileSystem> fs,
        const fs::path& mediapath,
        std::shared_ptr<donut::engine::DescriptorTableManager> descriptorTableManager,
        TopologyCache& topologyCache);

    std::optional<Model> Load(const std::filesystem::path& fileName,
        donut::engine::TextureCache& textureCache,
        int2 frameRange, const Instance& parent,
        nvrhi::ICommandList* commandList) const;

    void SetModelPath(fs::path&& path) { m_modelPath = std::move(path); }
};

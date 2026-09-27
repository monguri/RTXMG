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
//

#pragma once

#include <donut/core/math/math.h>
#include <donut/engine/BindingCache.h>
#include <donut/engine/SceneGraph.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/TextureCache.h>
#include <donut/engine/View.h>
#include <nvrhi/nvrhi.h>

#include "rtxmg/hiz/zbuffer.h"

#include "rtxmg/utils/buffer.h"

using namespace donut::engine;
using namespace donut::math;

class Camera;

class ZRenderer
{
public:

    ZRenderer(std::shared_ptr<ShaderFactory> shaderFactory);
    ~ZRenderer();

    void Render(Camera& camera, nvrhi::rt::AccelStructHandle tlas,
        nvrhi::ITexture* zbuffer, nvrhi::ICommandList* commandList);

private:
    void BuildPipeline(nvrhi::IDevice* device);

    std::shared_ptr<ShaderFactory> GetShaderFactory() const
    {
        return m_shaderFactory;
    }

private:
    nvrhi::rt::PipelineHandle m_rayPipeline = nullptr;
    nvrhi::rt::ShaderTableHandle m_shaderTable;
    nvrhi::BindingLayoutHandle m_bindingLayout;
    nvrhi::BindingSetHandle m_bindingSet = nullptr;

    nvrhi::BufferHandle m_params;

    nvrhi::ShaderLibraryHandle m_shaderLibrary;

    std::shared_ptr<ShaderFactory> m_shaderFactory;
};

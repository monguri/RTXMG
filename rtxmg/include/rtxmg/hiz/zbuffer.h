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

#include <memory>

#include <nvrhi/nvrhi.h>
#include <donut/core/math/math.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/ShaderFactory.h>

#include "rtxmg/utils/buffer.h"

#include "rtxmg/hiz/hiz_buffer.h"

using namespace donut::math;

class ZBuffer
{
    nvrhi::TextureHandle m_currentTexture;
    RTXMGBuffer<float> m_minmaxBuffer;

    nvrhi::ShaderHandle m_minmaxShader;
    nvrhi::ShaderHandle m_displayShader;

    nvrhi::BindingLayoutHandle m_minmaxBL;
    nvrhi::ComputePipelineHandle m_minmaxPSO;

    nvrhi::BindingLayoutHandle m_displayBL;
    nvrhi::ComputePipelineHandle m_displayPSO;

    std::unique_ptr<HiZBuffer> m_hierarchy;
    std::shared_ptr<donut::engine::CommonRenderPasses> m_commonPasses;

public:
    static std::unique_ptr<ZBuffer> Create(uint2 size, 
        std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses,
        std::shared_ptr<donut::engine::ShaderFactory> shaderFactory,
        nvrhi::ICommandList* commandList);

    nvrhi::TextureHandle GetCurrent() { return m_currentTexture; }
    const nvrhi::TextureHandle GetCurrent() const { return m_currentTexture; }

    void Display(nvrhi::ITexture *output, nvrhi::ICommandList* commandList);
    void ReduceHierarchy(nvrhi::ICommandList* commandList);
    void Clear(nvrhi::ICommandList* commandList);

    int GetNumHiZLODs() const
    {
        if (m_hierarchy) return m_hierarchy->GetNumLevels();
        return 0;
    }

    float2 GetInvHiZSize() const
    {
        if (m_hierarchy) return m_hierarchy->GetInvSize();
        return float2(0.f, 0.f);
    }
    nvrhi::ITexture* GetHierarchyTexture(uint32_t level) const { return m_hierarchy->GetTextureObject(level); }
    void GetHiZDesc(nvrhi::BindingLayoutDesc* outBindingLayout, nvrhi::BindingSetDesc* outBindingSet) const
    { 
        m_hierarchy->GetDesc(outBindingLayout, outBindingSet, false);
    }
};

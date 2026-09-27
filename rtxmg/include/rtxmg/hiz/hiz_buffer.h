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

#include "hiz_buffer_constants.h"

using namespace donut::math;

class HiZBuffer
{
public:
    ~HiZBuffer() = default;

    static std::unique_ptr<HiZBuffer> Create(uint2 size,
            std::shared_ptr<donut::engine::CommonRenderPasses> commonPasses,
            std::shared_ptr<donut::engine::ShaderFactory> shaderFactory,
            nvrhi::ICommandList* commandList);

    nvrhi::ITexture* GetTextureObject(uint32_t lod) const { return textureObjects[lod]; }

    // resets the depth values across the hi-z mip levels to +inf
    // note: there should be no need to call this on a per-frame basis
    void Clear(nvrhi::ICommandList* commandList);

    // applies max reduction to the input zbuffer data to populate
    // the hi-z mip levels
    void Reduce(nvrhi::ITexture* zbuffer, nvrhi::ICommandList* commandList);

    // composites the hi-z mip levels over an arbitrary rgba texture
    // (starting from a small offset at the bottom left corner)
    void Display(nvrhi::ITexture* output, nvrhi::ICommandList* commandList);
    
    void GetDesc(nvrhi::BindingLayoutDesc* outBindingLayout, nvrhi::BindingSetDesc* outBindingSet, bool writeable = false) const;

    uint32_t GetNumLevels() const { return m_numLODs; }
    float2 GetInvSize() const { return m_invSize; }
private:
    uint2 m_size = { 0, 0 };
    float2 m_invSize = { 0.f, 0.f };
    uint32_t m_numLODs = 0;

    nvrhi::TextureHandle textureObjects[HIZ_MAX_LODS] = { 0 };

    nvrhi::ShaderHandle m_pass1Shader;
    nvrhi::ShaderHandle m_pass2Shader;
    nvrhi::ShaderHandle m_displayShader;

    nvrhi::BindingLayoutHandle m_passBL;
    nvrhi::ComputePipelineHandle m_pass1PSO;
    nvrhi::ComputePipelineHandle m_pass2PSO;

    nvrhi::BindingLayoutHandle m_displayBL;
    nvrhi::ComputePipelineHandle m_displayPSO;

    nvrhi::SamplerHandle m_sampler;

    nvrhi::BufferHandle m_reduceParamsBuffer;
    nvrhi::BufferHandle m_displayParamsBuffer;
};

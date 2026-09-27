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

#include "rtxmg/hiz/zbuffer.h"
#include "rtxmg/hiz/hiz_buffer.h"

#include "rtxmg/utils/buffer.h"
#include "rtxmg/utils/debug.h"

#include <donut/core/log.h>

#include <fstream>

using namespace donut;

std::unique_ptr<ZBuffer> ZBuffer::Create(uint2 size,
    std::shared_ptr<engine::CommonRenderPasses> commonPasses,
    std::shared_ptr<engine::ShaderFactory> shaderFactory,
    nvrhi::ICommandList* commandList)
{
    auto device = commandList->getDevice();
    auto zbuffer = std::make_unique<ZBuffer>();

    nvrhi::TextureDesc desc;
    desc.width = size.x;
    desc.height = size.y;
    desc.isUAV = true;
    desc.keepInitialState = true;
    desc.format = nvrhi::Format::R32_FLOAT;
    desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
    desc.debugName = "ZBuffer";
    zbuffer->m_currentTexture = device->createTexture(desc);
    zbuffer->m_minmaxBuffer.Create(2, "ZBufferMinMax", device);
    zbuffer->m_commonPasses = commonPasses;

    nvrhi::ShaderDesc minmaxDesc(nvrhi::ShaderType::Compute);

    zbuffer->m_minmaxShader = shaderFactory->CreateShader("hiz/zbuffer_minmax.hlsl", "main", nullptr, minmaxDesc);
    if (!zbuffer->m_minmaxShader)
    {
        log::fatal("Failed to create ZBufferMinMax shader");
    }

    zbuffer->m_displayShader = shaderFactory->CreateShader("hiz/zbuffer_display.hlsl", "main", nullptr, nvrhi::ShaderType::Compute);
    if (!zbuffer->m_displayShader)
    {
        log::fatal("Failed to create ZBufferDisplay shader");
    }

    // make level 0 m_size a multiple of 32 to avoid nasty edge cases
    uint2  hizSize = { (((uint32_t(size.x) + 255) & ~255u) / 8),
                      (((uint32_t(size.y) + 255) & ~255u) / 8) };
    zbuffer->m_hierarchy = HiZBuffer::Create(hizSize, commonPasses, shaderFactory, commandList);

    return zbuffer;
}

void ZBuffer::Display(nvrhi::ITexture* output, nvrhi::ICommandList* commandList)
{
    if (!m_currentTexture) return;

    nvrhi::utils::ScopedMarker marker(commandList, "ZBuffer::display");
    uint2 size = { m_currentTexture->getDesc().width, m_currentTexture->getDesc().height };

    uint32_t minmax[2] = { std::numeric_limits<uint32_t>::max(), 0 };
    commandList->writeBuffer(m_minmaxBuffer, minmax, sizeof(minmax));

    auto device = commandList->getDevice();

    auto bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_UAV(0, m_minmaxBuffer))
        .addItem(nvrhi::BindingSetItem::Texture_SRV(0, m_currentTexture));

    nvrhi::BindingSetHandle bindingSet;
    if (!nvrhi::utils::CreateBindingSetAndLayout(device, nvrhi::ShaderType::Compute, 0, bindingSetDesc, m_minmaxBL, bindingSet))
    {
        log::fatal("Failed to create binding set and layout for zbuffer minmax");
    }
    
    if (!m_minmaxPSO)
    {
        auto computePipelineDesc = nvrhi::ComputePipelineDesc()
            .setComputeShader(m_minmaxShader)
            .addBindingLayout(m_minmaxBL);

        m_minmaxPSO = device->createComputePipeline(computePipelineDesc);
    }

    auto state = nvrhi::ComputeState()
        .setPipeline(m_minmaxPSO)
        .addBindingSet(bindingSet);

    commandList->setComputeState(state);
    commandList->dispatch(size.x / 16 + 1, size.y / 16 + 1, 1);

    bindingSetDesc = nvrhi::BindingSetDesc()
        .addItem(nvrhi::BindingSetItem::Texture_SRV(0, m_currentTexture))
        .addItem(nvrhi::BindingSetItem::StructuredBuffer_SRV(1, m_minmaxBuffer))
        .addItem(nvrhi::BindingSetItem::Texture_UAV(0, output));

    bindingSet.Reset();
    if (!nvrhi::utils::CreateBindingSetAndLayout(device, nvrhi::ShaderType::Compute, 0, bindingSetDesc, m_displayBL, bindingSet))
    {
        log::fatal("Failed to create binding set and layout for zbuffer minmax");
    }

    if (!m_displayPSO)
    {
        auto computePipelineDesc = nvrhi::ComputePipelineDesc()
            .setComputeShader(m_displayShader)
            .addBindingLayout(m_displayBL);

        m_displayPSO = device->createComputePipeline(computePipelineDesc);
    }

    state = nvrhi::ComputeState()
        .setPipeline(m_displayPSO)
        .addBindingSet(bindingSet);

    commandList->setComputeState(state);
    commandList->dispatch(size.x / 16 + 1, size.y / 16 + 1, 1);

    if (m_hierarchy)
    {
        m_hierarchy->Display(output, commandList);
    }
}

void ZBuffer::ReduceHierarchy(nvrhi::ICommandList* commandList)
{
    if (m_hierarchy && m_currentTexture)
    {
        nvrhi::utils::ScopedMarker marker(commandList, "ZBuffer::reduceHierarchy");
        m_hierarchy->Reduce(m_currentTexture, commandList);
    }
}

void ZBuffer::Clear(nvrhi::ICommandList* commandList)
{
    if (!m_currentTexture)
        return;
    nvrhi::utils::ScopedMarker marker(commandList, "ZBuffer::clear");
    commandList->clearTextureFloat(m_currentTexture, nvrhi::AllSubresources, nvrhi::Color(std::numeric_limits<float>::max()));
    
    if (m_hierarchy)
    {
        m_hierarchy->Clear(commandList);
    }
}
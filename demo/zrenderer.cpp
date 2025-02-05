//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//


#include "zrender_params.h"
#include "zrenderer.h"


#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/core/math/math.h>
#include <nvrhi/utils.h>

#include <filesystem>
#include <string>
#include <sstream>

using namespace donut;
using namespace donut::math;

#include "rtxmg/utils/buffer.h"
#include "rtxmg/profiler/statistics.h"
#include "rtxmg/scene/camera.h"

ZRenderer::ZRenderer(std::shared_ptr<engine::ShaderFactory> shaderFactory,
    std::shared_ptr<engine::DescriptorTableManager> descriptorTable)
    : m_shaderFactory(shaderFactory), m_descriptorTable(descriptorTable)
{

}

ZRenderer::~ZRenderer() {}

void ZRenderer::BuildPipeline(nvrhi::IDevice* device)
{
    nvrhi::BindingLayoutDesc globalBindingLayoutDesc;
    globalBindingLayoutDesc.visibility = nvrhi::ShaderType::All;
    globalBindingLayoutDesc.bindings = {
        nvrhi::BindingLayoutItem::VolatileConstantBuffer(0), // z render constants
        nvrhi::BindingLayoutItem::RayTracingAccelStruct(0),  // TLAS
        nvrhi::BindingLayoutItem::Texture_UAV(0),          // output
    };
    m_bindingLayout = device->createBindingLayout(globalBindingLayoutDesc);

    m_params =
        device->createBuffer(nvrhi::utils::CreateVolatileConstantBufferDesc(
            sizeof(ZRenderParams), "Z Render Params",
            engine::c_MaxRenderPassConstantBufferVersions));

    m_shaderLibrary =
        m_shaderFactory->CreateShaderLibrary("rtxmg_demo/zrender.hlsl", nullptr);

    if (!m_shaderLibrary)
    {
        log::fatal("Failed to create z render shader library");
    }

    nvrhi::rt::PipelineDesc pipelineDesc;
    pipelineDesc.globalBindingLayouts = { m_bindingLayout };
    pipelineDesc.shaders =
    {
        {"",m_shaderLibrary->getShader("RayGen", nvrhi::ShaderType::RayGeneration),nullptr},
        {"", m_shaderLibrary->getShader("Miss", nvrhi::ShaderType::Miss),nullptr},
    };

    pipelineDesc.hitGroups = { {
        "ZHitGroup",
        m_shaderLibrary->getShader("ClosestHit", nvrhi::ShaderType::ClosestHit),
        nullptr, // anyHit
        nullptr, // intersectionShader
        nullptr, // bindingLayout
        false    // isProceduralPrimitive
    } };

    pipelineDesc.maxPayloadSize = sizeof(ZRayPayload);
    pipelineDesc.maxRecursionDepth = 2;

    m_rayPipeline = device->createRayTracingPipeline(pipelineDesc);

    if (!m_rayPipeline)
    {
        log::fatal("Failed to create Z ray tracing pipeline");
    }

    m_shaderTable = m_rayPipeline->createShaderTable();

    if (!m_shaderTable)
    {
        log::fatal("Failed to create Z shader table");
    }

    m_shaderTable->setRayGenerationShader("RayGen");
    m_shaderTable->addHitGroup("ZHitGroup");
    m_shaderTable->addMissShader("Miss");
}

void ZRenderer::Render(Camera& camera, nvrhi::rt::AccelStructHandle tlas, 
    nvrhi::ITexture* zbuffer, nvrhi::ICommandList* commandList)
{
    if (!m_rayPipeline)
    {
        BuildPipeline(commandList->getDevice());
    }

    nvrhi::utils::ScopedMarker marker(commandList, "Z Render Pass");
    
    nvrhi::BindingSetDesc bindingSetDesc;
    bindingSetDesc.bindings = {
        nvrhi::BindingSetItem::ConstantBuffer(0, m_params),
        nvrhi::BindingSetItem::RayTracingAccelStruct(0, tlas),
        nvrhi::BindingSetItem::Texture_UAV(0, zbuffer),
    };

    m_bindingSet =
        commandList->getDevice()->createBindingSet(bindingSetDesc, m_bindingLayout);

    ZRenderParams params;
    auto const& [u, v, w] = camera.GetBasis();

    params.eye = camera.GetEye();

    params.U = u;
    params.V = v;
    params.W = w;

    commandList->writeBuffer(m_params, &params, sizeof(params));

    nvrhi::rt::State state;
    state.shaderTable = m_shaderTable;
    state.bindings = { m_bindingSet };
    commandList->setRayTracingState(state);

    nvrhi::rt::DispatchRaysArguments args;
    args.width = zbuffer->getDesc().width;
    args.height = zbuffer->getDesc().height;

    stats::frameSamplers.zRenderPassTime.Start(commandList);
    commandList->dispatchRays(args);
    stats::frameSamplers.zRenderPassTime.Stop();
}


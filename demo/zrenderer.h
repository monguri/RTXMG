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

#pragma once

#include <donut/core/math/math.h>
#include <donut/engine/BindingCache.h>
#include <donut/engine/DescriptorTableManager.h>
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

    ZRenderer(std::shared_ptr<ShaderFactory> shaderFactory,
        std::shared_ptr<DescriptorTableManager> descriptorTable);
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
    std::shared_ptr<DescriptorTableManager> m_descriptorTable;
};

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


#include "render_targets.h"

RenderTargets::RenderTargets(nvrhi::IDevice* device, uint32_t width, uint32_t height)
{
    auto CreateCommonTexture = [device, width, height](nvrhi::Format format, const char* debugName, nvrhi::TextureHandle& texture) {
        nvrhi::TextureDesc desc;
        desc.width = width;
        desc.height = height;
        desc.format = format;
        desc.debugName = debugName;
        desc.isVirtual = false;
        desc.initialState = nvrhi::ResourceStates::UnorderedAccess;
        desc.isRenderTarget = false;
        desc.isUAV = true;
        desc.dimension = nvrhi::TextureDimension::Texture2D;
        desc.keepInitialState = true;
        desc.isTypeless = false;

        texture = device->createTexture(desc);
        };

    CreateCommonTexture(nvrhi::Format::R32_FLOAT, "denoiserViewspaceZ", denoiserViewSpaceZ);
    CreateCommonTexture(nvrhi::Format::RGBA16_FLOAT, "denoiserMotionVectors", denoiserMotionVectors);
    CreateCommonTexture(nvrhi::Format::RGBA16_FLOAT, "denoiserNormalRoughness", denoiserNormalRoughness);
    CreateCommonTexture(nvrhi::Format::RGBA16_FLOAT, "denoiserEmissive", denoiserEmissive);
    CreateCommonTexture(nvrhi::Format::RGBA16_FLOAT, "denoiserDiffuseAbedo", denoiserDiffuseAlbedo);
    CreateCommonTexture(nvrhi::Format::RGBA16_FLOAT, "denoiserSpecularAbedo", denoiserSpecularAlbedo);
    CreateCommonTexture(nvrhi::Format::RGBA16_FLOAT, "denoiserInDiffRadianceHitDist", denoiserInDiffRadianceHitDist);
    CreateCommonTexture(nvrhi::Format::RGBA16_FLOAT, "denoiserInSpecRadianceHitDist", denoiserInSpecRadianceHitDist);
    CreateCommonTexture(nvrhi::Format::RGBA16_FLOAT, "denoiserOutDiffRadianceHitDist", denoiserOutDiffRadianceHitDist);
    CreateCommonTexture(nvrhi::Format::RGBA16_FLOAT, "denoiserOutSpecRadianceHitDist", denoiserOutSpecRadianceHitDist);
    CreateCommonTexture(nvrhi::Format::RGBA8_UNORM, "denoiserValidation", denoiserValidation);
}
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
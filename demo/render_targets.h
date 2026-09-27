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

#include <nvrhi/nvrhi.h>

class RenderTargets
{
public:
    RenderTargets(nvrhi::IDevice* device, uint32_t width, uint32_t height);

    nvrhi::TextureHandle denoiserViewSpaceZ;
    nvrhi::TextureHandle denoiserNormalRoughness;
    nvrhi::TextureHandle denoiserMotionVectors;
    nvrhi::TextureHandle denoiserEmissive;
    nvrhi::TextureHandle denoiserDiffuseAlbedo;
    nvrhi::TextureHandle denoiserSpecularAlbedo;

    nvrhi::TextureHandle denoiserInDiffRadianceHitDist;
    nvrhi::TextureHandle denoiserInSpecRadianceHitDist;

    nvrhi::TextureHandle denoiserOutDiffRadianceHitDist;
    nvrhi::TextureHandle denoiserOutSpecRadianceHitDist;

    nvrhi::TextureHandle denoiserValidation;
};
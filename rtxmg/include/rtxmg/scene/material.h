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
#include <string>
#include <donut/core/math/math.h>
#include <donut/engine/TextureCache.h>
#include "rtxmg/scene/material_constants.h"

using namespace donut::math;
using namespace donut::engine;

struct RTXMGMaterial
{
    std::string name;

    float3 baseOrDiffuseColor        = float3(1.f);
    float3 emissiveColor             = float3(0.f);
    float  roughness                 = 0.8f;
    float  metalness                 = 0.f;
    float  opacity                   = 1.f;
    float  alphaCutoff               = 0.5f;
    float  normalOrDisplacementScale = 1.f;
    bool   doubleSided               = false;
    bool   isAlphaMasked             = false;
    bool   isAlphaBlend              = false; // gltf alphaMode BLEND
    bool   isTransmissive            = false; // KHR_materials_transmission
    float  transmissionFactor        = 0.f;
    float  ior                       = 1.5f;

    std::shared_ptr<LoadedTexture> baseOrDiffuseTexture;
    std::shared_ptr<LoadedTexture> metalnessTexture;
    std::shared_ptr<LoadedTexture> roughnessTexture;
    std::shared_ptr<LoadedTexture> specularF0Texture;
    std::shared_ptr<LoadedTexture> emissiveTexture;
    std::shared_ptr<LoadedTexture> normalOrDisplacementTexture;
    bool isDisplacementMap = false;

    void FillMaterialConstants(RTXMGMaterialConstants& mc) const;
};

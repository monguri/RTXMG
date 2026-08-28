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

#ifdef __cplusplus
#include <donut/core/math/math.h>
using namespace donut::math;
#endif

// Flag bits for RTXMGMaterialConstants::flags.  RTXMG-prefixed because donut's
// material_cb.h declares identically-named globals at different values.
static const int RTXMGMaterialFlags_UseBaseOrDiffuseTexture  = (1 << 0);
static const int RTXMGMaterialFlags_UseMetalnessTexture      = (1 << 1);
static const int RTXMGMaterialFlags_UseRoughnessTexture      = (1 << 2);
static const int RTXMGMaterialFlags_UseSpecularF0Texture     = (1 << 3);
static const int RTXMGMaterialFlags_UseEmissiveTexture       = (1 << 4);
static const int RTXMGMaterialFlags_UseNormalTexture         = (1 << 5); // unset: normal maps are not loaded
static const int RTXMGMaterialFlags_UseDisplacementTexture   = (1 << 6); // OBJ subd (map_bump)
static const int RTXMGMaterialFlags_AlphaMask                = (1 << 7);
static const int RTXMGMaterialFlags_AlphaBlend               = (1 << 8); // gltf alphaMode BLEND: emissive alpha-blend "card"
static const int RTXMGMaterialFlags_Transmissive             = (1 << 9); // KHR_materials_transmission: dielectric glass

#ifndef __cplusplus
static const float c_DielectricSpecular = 0.04f;
#endif

struct RTXMGMaterialConstants
{
    // --- 16B ---
    float3 baseOrDiffuseColor;              // GLTF baseColorFactor.rgb / OBJ kd
    int    flags;                           // RTXMGMaterialFlags_* bitmask

    // --- 16B ---
    float  roughness;
    float  metalness;
    float  opacity;
    float  alphaCutoff;

    // --- 16B ---
    float3 emissiveColor;
    float  normalOrDisplacementTextureScale;

    // --- 16B ---
    int    baseOrDiffuseTextureIndex;       // -1 = none
    int    metalnessTextureIndex;           // OBJ map_pm
    int    roughnessTextureIndex;           // OBJ map_pr
    int    specularF0TextureIndex;          // OBJ map_ks

    // --- 16B ---
    int    emissiveTextureIndex;
    int    normalOrDisplacementTextureIndex;
    int    materialID;
    int    _pad;

    // --- 16B ---
    float  transmissionFactor;              // KHR_materials_transmission (0 = opaque)
    float  ior;                             // KHR_materials_ior (default 1.5)
    float  _pad1;
    float  _pad2;
};

#ifdef __cplusplus
static_assert(sizeof(RTXMGMaterialConstants) == 96, "RTXMGMaterialConstants must be 96 bytes");
#endif

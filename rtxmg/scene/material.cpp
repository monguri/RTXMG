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

#include "rtxmg/scene/material.h"

void RTXMGMaterial::FillMaterialConstants(RTXMGMaterialConstants& mc) const
{
    auto getIdx = [](const std::shared_ptr<LoadedTexture>& tex) -> int
    {
        if (!tex || !tex->bindlessDescriptor.IsValid())
            return -1;
        return static_cast<int>(tex->bindlessDescriptor.GetIndexInHeap());
    };

    mc = RTXMGMaterialConstants{};
    mc.baseOrDiffuseColor               = baseOrDiffuseColor;
    mc.emissiveColor                    = emissiveColor;
    mc.roughness                        = roughness;
    mc.metalness                        = metalness;
    mc.opacity                          = opacity;
    mc.alphaCutoff                      = alphaCutoff;
    mc.normalOrDisplacementTextureScale = normalOrDisplacementScale;
    mc.flags = 0;
    if (isAlphaMasked) mc.flags |= RTXMGMaterialFlags_AlphaMask;
    if (isAlphaBlend)  mc.flags |= RTXMGMaterialFlags_AlphaBlend;
    if (isTransmissive) mc.flags |= RTXMGMaterialFlags_Transmissive;
    mc.transmissionFactor = transmissionFactor;
    mc.ior                = ior;

    mc.baseOrDiffuseTextureIndex = getIdx(baseOrDiffuseTexture);
    if (mc.baseOrDiffuseTextureIndex >= 0) mc.flags |= RTXMGMaterialFlags_UseBaseOrDiffuseTexture;

    mc.metalnessTextureIndex = getIdx(metalnessTexture);
    if (mc.metalnessTextureIndex >= 0) mc.flags |= RTXMGMaterialFlags_UseMetalnessTexture;

    mc.roughnessTextureIndex = getIdx(roughnessTexture);
    if (mc.roughnessTextureIndex >= 0) mc.flags |= RTXMGMaterialFlags_UseRoughnessTexture;

    mc.specularF0TextureIndex = getIdx(specularF0Texture);
    if (mc.specularF0TextureIndex >= 0) mc.flags |= RTXMGMaterialFlags_UseSpecularF0Texture;

    mc.emissiveTextureIndex = getIdx(emissiveTexture);
    if (mc.emissiveTextureIndex >= 0) mc.flags |= RTXMGMaterialFlags_UseEmissiveTexture;

    mc.normalOrDisplacementTextureIndex = getIdx(normalOrDisplacementTexture);
    if (mc.normalOrDisplacementTextureIndex >= 0)
    {
        if (isDisplacementMap)
            mc.flags |= RTXMGMaterialFlags_UseDisplacementTexture;
        else
            mc.flags |= RTXMGMaterialFlags_UseNormalTexture;
    }
}

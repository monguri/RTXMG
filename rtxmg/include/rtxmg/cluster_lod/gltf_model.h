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

// gltf_model.h — what ClusterLodGltfImporter::Load() returns, and nothing else:
// the instance list, the material descriptions parsed out of the glTF, and the
// ClusterLodModel that binds them to the baked geometry and to the cache
// mappings that geometry is viewed through.

#pragma once

#include <cstdint>
#include <vector>

#include <donut/core/math/math.h>

#include "rtxmg/cluster_lod/cache.h"  // CacheFileView / ShardCache, held by value in ClusterLodModel

using namespace donut::math;

// ---------------------------------------------------------------------------
// ClusterLodInstance — one instantiation of a cluster LOD geometry in the
// scene, with a full 4x4 world transform in donut's convention: row-major
// storage holding ROW vectors, translation in row 3.  Convert with
// homogeneousToAffine(); ray-tracing instance descs want
// affineToColumnMajor() of that.
// ---------------------------------------------------------------------------

struct ClusterLodInstance
{
    float4x4    transform  = float4x4::identity();
    uint32_t    geometryID = 0;
    uint32_t    materialID = 0;
    std::string name;
};

// ---------------------------------------------------------------------------
// ClusterLodMaterialDesc — lightweight material data parsed from cgltf.
// Mirrors the subset of donut::engine::Material that is populated from a
// glTF PBR metallic-roughness material.
// Texture paths are absolute, resolved at parse time (empty = no texture).
// ---------------------------------------------------------------------------

struct ClusterLodMaterialDesc
{
    std::string name;
    float4      baseColorFactor   = { 1.f, 1.f, 1.f, 1.f };
    float       metalness         = 0.f;
    float       roughness         = 1.f;
    float3      emissiveColor     = { 0.f, 0.f, 0.f };
    float       emissiveIntensity = 1.f;
    float       alphaCutoff       = 0.5f;
    bool        doubleSided       = false;
    // 0 = opaque, 1 = mask, 2 = blend  (matches cgltf_alpha_mode)
    int         alphaModeGltf     = 0;
    // KHR_materials_transmission / KHR_materials_ior (factor 0 = not transmissive)
    float       transmissionFactor = 0.f;
    float       ior                = 1.5f;

    // Texture file paths (absolute; empty = no texture).
    // GLTF metallicRoughness packs metalness (.b) and roughness (.g) in one texture.
    std::string baseColorTexturePath;
    std::string metallicRoughnessTexturePath;
    std::string emissiveTexturePath;
    std::string normalTexturePath;
    float       normalTextureScale = 1.f;
};

// ---------------------------------------------------------------------------
// ClusterLodModel — return value of ClusterLodGltfImporter::Load().
//
// `geometries` is the array consumers read; each entry spans either an owned
// `storages` entry (fresh bake) or a cache mapping held by `cacheView` /
// `shardCache` (cache hit), so both of those must outlive it.
// ---------------------------------------------------------------------------

struct ClusterLodModel
{
    std::vector<GeometryView>          geometries;  // non-owning views
    std::vector<GeometryStorage>       storages;    // owned baked data
    CacheFileView                      cacheView;   // monolith .nvsngeo mmap lifetime
    ShardCache                         shardCache;  // per-geometry shard mmap lifetime
    std::vector<ClusterLodInstance>    instances;
    std::vector<ClusterLodMaterialDesc> materials;  // one entry per GLTF material index
};

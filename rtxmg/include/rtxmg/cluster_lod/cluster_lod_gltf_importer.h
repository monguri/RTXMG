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

#pragma once

#include <filesystem>
#include <optional>

#include "rtxmg/cluster_lod/baker.h"
#include "rtxmg/cluster_lod/gltf_model.h"

namespace fs = std::filesystem;

// cgltf types (full definition included in the .cpp).
struct cgltf_data;
struct cgltf_options;

// Loads a .gltf or .glb file, bakes cluster LOD, reads/writes the bake cache,
// and returns a ClusterLodModel with populated geometries and instances.
//
// Pipeline:
//   1. Parse GLTF → GeometryStorage[] (raw mesh data)
//   2. Cache miss → ClusterLodBaker::Build() each storage, then write it back
//   3. Map the cache → GeometryView[] into model.geometries
//
// Symmetric with ObjImporter; used by RTXMGScene::LoadWithThreadPool().
class ClusterLodGltfImporter
{
public:
    // `mediaPath`  — base directory for resolving relative GLTF buffer URIs.
    // `bakerConfig` — cluster LOD generation parameters (also stored in cache).
    // `log`        — log per-geometry bake/cache stats at info level.
    // `cacheDirOverride` — if non-empty, use this directory for the shard cache
    // instead of deriving a shared "_nvsngeocache" folder next to the gltf. For
    // testing: bake into a scratch dir without clobbering an existing cache.
    // `bakeWorkers` — parallel bake worker count; 0 derives it from the core
    // count capped by physical RAM, since each worker's decode scratch, extracted
    // geometry and baker working set are live at once.
    explicit ClusterLodGltfImporter(const BakerConfig& bakerConfig      = {},
                                    bool               log              = false,
                                    const fs::path&    cacheDirOverride = {},
                                    uint32_t           bakeWorkers      = 0);

    // Load a .gltf/.glb, bake if needed, return model.  nullopt on error.
    std::optional<ClusterLodModel> Load(const fs::path& path) const;

private:
    // Per-geometry content-hash shard cache path.  Populates model.geometries
    // (zero-copy shard mmaps for hits, owned storage for fresh bakes) and
    // model.shardCache.  Returns false on fatal error.
    bool BakeOrLoadShards(const cgltf_data*          gltf,
                          const std::vector<size_t>& geometryToMesh,
                          const fs::path&            gltfPath,
                          cgltf_options&             options,
                          ClusterLodModel&           model) const;

    // Monolithic .nvsngeo path — one cache file for the whole scene.  Populates
    // model.geometries and model.cacheView.  Returns false on fatal error.
    bool BakeOrLoadMonolith(const cgltf_data*          gltf,
                            const std::vector<size_t>& geometryToMesh,
                            const fs::path&            gltfPath,
                            ClusterLodModel&           model) const;

    BakerConfig m_bakerConfig;
    bool        m_log = false;
    // Per-geometry shard cache vs a single monolithic .nvsngeo file.  The shard
    // path bounds bake memory and rebakes only changed geometries.
    bool        m_useShardCache = true;
    // Optional shard-cache directory override (empty = derive next to the gltf).
    fs::path    m_cacheDirOverride;
    // Parallel bake worker count; 0 = derive from cores and physical RAM.
    uint32_t    m_bakeWorkers = 0;
};

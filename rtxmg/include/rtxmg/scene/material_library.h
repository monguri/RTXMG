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

#include "rtxmg/cluster_lod/baked_geometry.h"
#include "rtxmg/cluster_lod/gltf_model.h"
#include "rtxmg/scene/material.h"
#include "rtxmg/scene/texture_loader.h"

#include <nvrhi/nvrhi.h>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

class SubdivisionSurface;

namespace rtxmg
{
    // Every material the scene resolves against, subd first and cluster-LoD after,
    // plus the two GPU buffers the shaders reach them through.  The split point is
    // GetClusterLodMaterialBaseID(), which the cluster-LoD shaders add back.
    class MaterialLibrary
    {
    public:
        explicit MaterialLibrary(nvrhi::IDevice* device)
            : m_device(device)
        {
        }

        // Diagnostic: --nomat, which skips the whole cluster-LoD material set for a
        // single default-gray material.  Must be set BEFORE the materials are built.
        void SetEnableMaterials(bool b) { m_enableMaterials = b; }
        bool GetEnableMaterials() const { return m_enableMaterials; }

        // --normalmaps.  Off, the cluster-LoD normal maps are never read from disk;
        // ApplyKtxTextureBudget must be told the same thing or the budget waterline
        // is solved against a different texture set than the loader reads.
        void SetEnableNormalMaps(bool b) { m_enableNormalMaps = b; }

        // Dedup'd by (resolved mtllib, name, udim) across every subd mesh; also
        // flags the meshes whose material carries a displacement map.
        void BuildSubdMaterials(std::span<const std::unique_ptr<SubdivisionSurface>> subdMeshes,
                                const TextureLoader& textureLoader,
                                donut::engine::ThreadPool* threadPool);

        // Cluster-LoD materials append after the subd ones, so this must run second.
        void BuildClusterLodMaterials(std::span<const ClusterLodMaterialDesc> descs,
                                      const TextureLoader& textureLoader,
                                      donut::engine::ThreadPool* threadPool);

        void BuildMaterialBuffer(nvrhi::ICommandList* commandList);

        // Flatten the per-geometry localMaterialIDs so the cluster-LoD hit shaders
        // can resolve a cluster's local material ID to a t_MaterialConstants slot:
        //   sceneMatID = geom.materialBaseID
        //              + t_ClusterLodLocalMaterialIDs[geom.localMaterialsOffset + localID]
        void BuildClusterLodLocalMaterialIDs(std::span<const GeometryStorage> storages,
                                             nvrhi::ICommandList* commandList);

        const std::vector<std::shared_ptr<RTXMGMaterial>>& GetMaterials() const { return m_materials; }
        nvrhi::IBuffer* GetMaterialBuffer() const { return m_materialBuffer; }

        // Per-subshape index into GetMaterials() for one subd mesh, in the mesh
        // order BuildSubdMaterials was handed.  Feeds BuildSurfaceToMaterialIndex.
        std::span<const uint32_t> GetSubdSubshapeMaterialIndices(size_t meshIndex) const
        {
            return m_subdSubshapeMaterialIndices[meshIndex];
        }

        // The m_materialBuffer slot where the cluster-LoD materials start; the
        // shader adds it to a cluster's local material ID, not the host.
        uint32_t GetClusterLodMaterialBaseID() const { return m_clusterLodMaterialBaseID; }
        nvrhi::IBuffer* GetClusterLodLocalMaterialIDsBuffer() const { return m_clusterLodLocalMaterialIDsBuffer; }
        // Host side of shaderio::Geometry's slice descriptor: one entry per
        // cluster-LoD geometry, in geometry order.  Handed to the resource paths
        // via SetClusterLodLocalMaterialsOffsets so they do not re-derive it.
        const std::vector<uint32_t>& GetClusterLodLocalMaterialsOffsets() const
        {
            return m_clusterLodLocalMaterialsOffsets;
        }

    private:
        nvrhi::IDevice* m_device = nullptr;

        // Scene-level unique materials (subd + cluster-LOD), dedup'd by (mtllib, name).
        std::vector<std::shared_ptr<RTXMGMaterial>> m_materials;
        // Per-mesh, per-subshape index into m_materials.
        std::vector<std::vector<uint32_t>> m_subdSubshapeMaterialIndices;

        nvrhi::BufferHandle m_materialBuffer;  // RTXMGMaterialConstants[]

        uint32_t            m_clusterLodMaterialBaseID = 0;
        nvrhi::BufferHandle m_clusterLodLocalMaterialIDsBuffer;
        // Diagnostic: --nomat — see SetEnableMaterials().
        bool                m_enableMaterials = true;
        bool                m_enableNormalMaps = false;  // --normalmaps
        // Parallel arrays (length = number of cluster-LoD geometries) describing
        // where each geometry's localMaterialIDs slice lives inside
        // m_clusterLodLocalMaterialIDsBuffer.
        std::vector<uint32_t> m_clusterLodLocalMaterialsOffsets;
        std::vector<uint32_t> m_clusterLodLocalMaterialsCounts;
    };
}

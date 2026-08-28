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

#include "rtxmg/scene/instance_grid.h"

#include "rtxmg/subdivision/shape.h"
#include "rtxmg/subdivision/subdivision_surface.h"

#include <donut/core/log.h>

#include <cmath>
#include <random>
#include <string>
#include <utility>

using namespace donut;

namespace rtxmg
{
namespace
{
    // Which axes carry the grid, how many cells per axis, and which axes a copy
    // may be randomly rotated about.
    struct GridLayout
    {
        uint32_t axisBits = 0x7u;
        int      numAxis  = 3;
        uint32_t sq       = 1u;
        float3   rotMask  = float3(0.0f, 0.0f, 0.0f);
        bool     doRot    = false;
    };

    // The rigid transform every instance in one grid cell receives.  The axis and
    // angle travel alongside the matrix because the scene nodes want a quaternion.
    struct GridCellTransform
    {
        affine3 rotation = affine3::identity();  // donut row-vector form
        float3  axis     = float3(0.0f, 1.0f, 0.0f);
        float   angle    = 0.0f;
        float3  shift    = float3(0.0f, 0.0f, 0.0f);
    };

    GridLayout ComputeGridLayout(uint32_t numCopies, uint32_t gridBits)
    {
        GridLayout layout;

        // numAxis = popcount(gridBits & 7). 0 -> fall back to 3D.
        layout.axisBits = gridBits & 0x7u;
        if (layout.axisBits == 0u)
            layout.axisBits = 0x7u;
        layout.numAxis = 0;
        for (int i = 0; i < 3; ++i)
            if (layout.axisBits & (1u << i)) ++layout.numAxis;

        switch (layout.numAxis)
        {
            case 1: layout.sq = numCopies; break;
            case 2: while (layout.sq * layout.sq < numCopies) ++layout.sq; break;
            case 3: while (layout.sq * layout.sq * layout.sq < numCopies) ++layout.sq; break;
        }

        layout.rotMask = float3((gridBits & 0x08u) ? 1.0f : 0.0f,
                                (gridBits & 0x10u) ? 1.0f : 0.0f,
                                (gridBits & 0x20u) ? 1.0f : 0.0f);
        layout.doRot = (layout.rotMask.x + layout.rotMask.y + layout.rotMask.z) > 0.0f;
        return layout;
    }

    float3 GridCellShift(const GridLayout& layout, uint32_t copyIndex, const float3& cellExtent)
    {
        // Counter is unsigned (no centering): copies fan out from the corner.
        float u = 0.0f, v = 0.0f, w = 0.0f;
        switch (layout.numAxis)
        {
            case 1: u = float(copyIndex); break;
            case 2: u = float(copyIndex % layout.sq); v = float(copyIndex / layout.sq); break;
            case 3: u = float(copyIndex % layout.sq);
                    v = float((copyIndex / layout.sq) % layout.sq);
                    w = float(copyIndex / (layout.sq * layout.sq)); break;
        }

        float3 shift = cellExtent;
        float  use   = u;
        if (layout.axisBits & 0x1u) { shift.x *= -use; if (layout.numAxis > 1) use = v; }
        else                        { shift.x = 0.0f; }
        if (layout.axisBits & 0x2u) { shift.y *= use; if (layout.numAxis > 2) use = w;
                                      else if (layout.numAxis > 1) use = v; }
        else                        { shift.y = 0.0f; }
        if (layout.axisBits & 0x4u) { shift.z *= -use; }
        else                        { shift.z = 0.0f; }
        return shift;
    }

    box3 GeometryBounds(const GeometryView& geo)
    {
        return box3(float3(geo.bbox.lo.x, geo.bbox.lo.y, geo.bbox.lo.z),
                    float3(geo.bbox.hi.x, geo.bbox.hi.y, geo.bbox.hi.z));
    }

    // The seeds' WORLD-space bbox, for grid sizing and camera framing.  The full
    // instance transform matters, not just its translation — assets routinely
    // carry a node scale.
    box3 ComputeSeedBounds(const InstanceGridScene& scene, std::span<const Instance> subdSeeds)
    {
        box3 bounds = box3::empty();
        for (const ClusterLodInstance& inst : scene.clusterLodInstances)
        {
            const affine3 instXform = homogeneousToAffine(inst.transform);
            bounds |= GeometryBounds(scene.clusterLodGeometries[inst.geometryID]) * instXform;
        }
        // Union in the subd bounds so grid spacing accounts for the full per-cell
        // content, and so subd-only scenes get a valid extent at all.
        for (Instance seed : subdSeeds)
        {
            seed.UpdateLocalTransform();
            bounds |= scene.subdMeshes[seed.meshID]->GetShape()->aabb * seed.localToWorld;
        }
        return bounds;
    }

    void ReplicateClusterLodInstances(const InstanceGridScene& scene, size_t originalCount,
                                      const GridCellTransform& cell,
                                      const std::shared_ptr<RTXMGSceneNode>& root)
    {
        for (size_t i = 0; i < originalCount; ++i)
        {
            ClusterLodInstance copy = scene.clusterLodInstances[i];

            // Rotate about the origin, then shift into the grid cell.
            affine3 xf = homogeneousToAffine(copy.transform) * cell.rotation;
            xf.m_translation += cell.shift;
            copy.transform = affineToHomogeneous(xf);

            const uint32_t newIdx = uint32_t(scene.clusterLodInstances.size());
            scene.clusterLodInstances.push_back(std::move(copy));

            const ClusterLodInstance& inst = scene.clusterLodInstances.back();
            const float3 worldT            = xf.m_translation;
            const double halfAngle         = double(cell.angle) * 0.5;
            const double sinH              = std::sin(halfAngle);

            auto node = std::make_shared<RTXMGSceneNode>();
            node->name              = "cluster_lod_grid_" + std::to_string(newIdx);
            node->translation       = double3(worldT);
            node->rotation          = dquat::fromWXYZ(std::cos(halfAngle),
                                                       double3(double(cell.axis.x) * sinH,
                                                              double(cell.axis.y) * sinH,
                                                              double(cell.axis.z) * sinH));
            node->scaling           = double3(1.0);
            node->objectSpaceBounds = GeometryBounds(scene.clusterLodGeometries[inst.geometryID]);
            node->clusterLodMeshInstance  = std::make_shared<ClusterLodMeshInstance>(ClusterLodMeshInstance{ inst.geometryID });
            if (root)
                scene.sceneGraph.Attach(root, node);
        }
    }

    // Replicate subd instances into the same grid cell, applying the same shift +
    // rotation as the cluster-LoD copies.
    void ReplicateSubdInstances(const InstanceGridScene& scene, std::span<const Instance> subdSeeds,
                                const GridCellTransform& cell,
                                const std::shared_ptr<RTXMGSceneNode>& root)
    {
        // cell.rotation is donut's row-vector form; R is its column-vector
        // transpose, which the translation product below wants.
        const float3x3 R     = transpose(cell.rotation.m_linear);
        const float    halfF = cell.angle * 0.5f;
        const float    sinHF = sinf(halfF);
        const quat     Rq    = quat::fromWXYZ(cosf(halfF),
                                              float3(cell.axis.x * sinHF,
                                                     cell.axis.y * sinHF,
                                                     cell.axis.z * sinHF));

        for (const Instance& seed : subdSeeds)
        {
            Instance copy = seed;  // preserves meshID and scaling

            const float3 t0 = seed.translation;
            copy.translation = float3(
                R[0][0]*t0.x + R[0][1]*t0.y + R[0][2]*t0.z + cell.shift.x,
                R[1][0]*t0.x + R[1][1]*t0.y + R[1][2]*t0.z + cell.shift.y,
                R[2][0]*t0.x + R[2][1]*t0.y + R[2][2]*t0.z + cell.shift.z);
            copy.rotation     = Rq * seed.rotation;
            copy.localToWorld = affine3::identity();  // recomputed by SceneGraph::Refresh().

            const uint32_t newSubdIdx = uint32_t(scene.subdInstances.size());

            auto node = std::make_shared<RTXMGSceneNode>();
            node->name              = "subd_grid_" + std::to_string(newSubdIdx);
            node->translation       = double3(copy.translation);
            node->rotation          = dquat(copy.rotation);
            node->scaling           = double3(copy.scaling);
            node->objectSpaceBounds = scene.subdMeshes[seed.meshID]->GetShape()->aabb;
            node->subdMeshInstance  =
                std::make_shared<SubdivisionMeshInstance>(SubdivisionMeshInstance{ seed.meshID });
            if (root)
                scene.sceneGraph.Attach(root, node);

            copy.node = node;
            scene.subdInstances.push_back(std::move(copy));
        }
    }
} // namespace

InstanceGridResult ReplicateInstancesOnGrid(const InstanceGridScene& scene,
                                            uint32_t numCopies, float gap, uint32_t gridBits)
{
    InstanceGridResult result;

    const bool hasClusterLod =
        !scene.clusterLodInstances.empty() && !scene.clusterLodGeometries.empty();

    // Copies, not a span: the replication below push_back's into subdInstances,
    // which can reallocate.  Subd copies reuse the cluster-LoD shift + rotation, so
    // a mixed scene's grid cell holds both models co-located.
    const std::vector<Instance> subdSeeds(scene.subdInstances.begin(), scene.subdInstances.end());

    if (!hasClusterLod && subdSeeds.empty())
        return result;

    const size_t originalCount = scene.clusterLodInstances.size();

    result.originalBbox = ComputeSeedBounds(scene, subdSeeds);
    const float3 cellExtent = result.originalBbox.diagonal() * gap;

    if (numCopies <= 1u)
        return result;

    const GridLayout layout = ComputeGridLayout(numCopies, gridBits);

    std::default_random_engine            rng(2342);
    std::uniform_real_distribution<float> randomUnorm(0.0f, 1.0f);

    scene.clusterLodInstances.reserve(originalCount * numCopies);
    auto root = scene.sceneGraph.GetRootNode();

    if (!subdSeeds.empty())
        scene.subdInstances.reserve(scene.subdInstances.size() + subdSeeds.size() * (numCopies - 1));

    for (uint32_t copyIndex = 1; copyIndex < numCopies; ++copyIndex)
    {
        GridCellTransform cell;
        cell.shift = GridCellShift(layout, copyIndex, cellExtent);

        if (layout.doRot)
        {
            float3 dir(randomUnorm(rng) * layout.rotMask.x,
                       randomUnorm(rng) * layout.rotMask.y,
                       randomUnorm(rng) * layout.rotMask.z);
            // Avoid an all-zero vector if the random draws coincided.
            dir = dm::max(dir, layout.rotMask * 1e-5f);
            cell.axis  = normalize(dir);
            cell.angle = randomUnorm(rng) * 2.0f * dm::PI_f;
        }
        cell.rotation = dm::rotation(cell.axis, cell.angle);  // Rodrigues' rotation

        ReplicateClusterLodInstances(scene, originalCount, cell, root);
        ReplicateSubdInstances(scene, subdSeeds, cell, root);
    }

    result.expanded = true;

    // The importer's own load-time count is per the base glTF, so report again
    // after expansion to reflect what is actually instanced.
    log::info("InstanceGrid: %u grid copies -> %zu cluster-LOD instances, %zu subd instances.",
              numCopies, scene.clusterLodInstances.size(), scene.subdInstances.size());
    return result;
}

} // namespace rtxmg

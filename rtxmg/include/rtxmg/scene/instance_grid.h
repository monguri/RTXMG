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
#include "rtxmg/scene/model.h"
#include "rtxmg/scene/scene_graph.h"

#include <donut/core/math/math.h>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

class SubdivisionSurface;

namespace rtxmg
{
    // The scene state the grid replicates over.  The scene owns all of it: the
    // grid only appends to the two instance lists and attaches the new nodes.
    struct InstanceGridScene
    {
        std::span<const GeometryView>                        clusterLodGeometries;
        std::span<const std::unique_ptr<SubdivisionSurface>> subdMeshes;
        std::vector<ClusterLodInstance>&                     clusterLodInstances;
        std::vector<Instance>&                               subdInstances;
        RTXMGSceneGraph&                                     sceneGraph;
    };

    struct InstanceGridResult
    {
        // World-space bbox of the seed instances, before replication.  Empty when
        // there was nothing to replicate.
        box3 originalBbox = box3::empty();
        // False when the call was a no-op (nothing to replicate, or a single copy),
        // which is also what makes a later call with a real copy count still work.
        bool expanded = false;
    };

    // Replicate cluster-LoD and subd instances on a grid with random per-copy
    // rotation.
    //   - bits 0..2 of gridBits select the grid placement axes (XYZ)
    //   - bits 3..5 select the random rotation axes (XYZ); the rotation axis is
    //     a random unit vector inside the masked subspace, the angle is uniform
    //     in [0, 2pi). With 2+ rotation bits set, each copy gets a different axis.
    //   - copies fan out in negative axis directions from the original (which
    //     stays at the corner), so a camera placed at the original looks "down
    //     the grid".
    //   - gap is grid spacing as a multiple of the seed AABB extent.
    InstanceGridResult ReplicateInstancesOnGrid(const InstanceGridScene& scene,
                                                uint32_t numCopies, float gap,
                                                uint32_t gridBits);
}

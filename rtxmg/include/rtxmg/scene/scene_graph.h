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

#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <donut/core/math/math.h>

using namespace donut::math;

struct SubdivisionMeshInstance  { uint32_t meshID;     };
struct ClusterLodMeshInstance   { uint32_t geometryID; };

struct RTXMGSceneNode
{
    std::string name;

    double3 translation = double3(0.0);
    dquat   rotation;
    double3 scaling     = double3(1.0);

    affine3 localToWorld     = affine3::identity();
    affine3 prevLocalToWorld = affine3::identity();

    box3 objectSpaceBounds;

    std::shared_ptr<SubdivisionMeshInstance> subdMeshInstance;
    std::shared_ptr<ClusterLodMeshInstance>  clusterLodMeshInstance;

    std::weak_ptr<RTXMGSceneNode>                parent;
    std::vector<std::shared_ptr<RTXMGSceneNode>> children;
};

class RTXMGSceneGraph
{
public:
    void SetRootNode(std::shared_ptr<RTXMGSceneNode> root);
    std::shared_ptr<RTXMGSceneNode> GetRootNode() const;
    void Attach(std::shared_ptr<RTXMGSceneNode> parent,
                std::shared_ptr<RTXMGSceneNode> child);
    void Refresh();
    box3 GetGlobalBoundingBox() const;

private:
    std::shared_ptr<RTXMGSceneNode> m_root;
    void RefreshNode(RTXMGSceneNode& node, const affine3& parentToWorld);
};

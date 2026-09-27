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

#include "rtxmg/scene/scene_graph.h"

void RTXMGSceneGraph::SetRootNode(std::shared_ptr<RTXMGSceneNode> root)
{
    m_root = std::move(root);
}

std::shared_ptr<RTXMGSceneNode> RTXMGSceneGraph::GetRootNode() const
{
    return m_root;
}

void RTXMGSceneGraph::Attach(std::shared_ptr<RTXMGSceneNode> parent,
                             std::shared_ptr<RTXMGSceneNode> child)
{
    child->parent = parent;
    parent->children.push_back(std::move(child));
}

void RTXMGSceneGraph::RefreshNode(RTXMGSceneNode& node, const affine3& parentToWorld)
{
    node.prevLocalToWorld = node.localToWorld;

    daffine3 local = dm::scaling(node.scaling);
    local *= node.rotation.toAffine();
    local *= dm::translation(node.translation);

    affine3 localF(local);
    node.localToWorld = localF * parentToWorld;

    for (auto& child : node.children)
        RefreshNode(*child, node.localToWorld);
}

void RTXMGSceneGraph::Refresh()
{
    if (m_root)
        RefreshNode(*m_root, affine3::identity());
}

box3 RTXMGSceneGraph::GetGlobalBoundingBox() const
{
    box3 result = box3::empty();

    std::function<void(const RTXMGSceneNode&)> visit = [&](const RTXMGSceneNode& node)
    {
        if (!node.objectSpaceBounds.isempty())
        {
            const box3& b = node.objectSpaceBounds;
            float3 corners[8] = {
                float3(b.m_mins.x, b.m_mins.y, b.m_mins.z),
                float3(b.m_maxs.x, b.m_mins.y, b.m_mins.z),
                float3(b.m_mins.x, b.m_maxs.y, b.m_mins.z),
                float3(b.m_maxs.x, b.m_maxs.y, b.m_mins.z),
                float3(b.m_mins.x, b.m_mins.y, b.m_maxs.z),
                float3(b.m_maxs.x, b.m_mins.y, b.m_maxs.z),
                float3(b.m_mins.x, b.m_maxs.y, b.m_maxs.z),
                float3(b.m_maxs.x, b.m_maxs.y, b.m_maxs.z),
            };
            for (const float3& c : corners)
            {
                float3 world = c * node.localToWorld.m_linear + node.localToWorld.m_translation;
                result |= world;
            }
        }
        for (const auto& child : node.children)
            visit(*child);
    };

    if (m_root)
        visit(*m_root);

    return result;
}

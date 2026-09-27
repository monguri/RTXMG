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

#ifndef OSD_PORTS_TMR_SUBDIVISION_NODE_H
#define OSD_PORTS_TMR_SUBDIVISION_NODE_H

#include "rtxmg/subdivision/osd_ports/tmr/types.h"
#include "rtxmg/subdivision/osd_ports/tmr/nodeDescriptor.h"

struct SubdivisionNode
{
    StructuredBuffer<uint32_t> m_subpatchTrees;
    StructuredBuffer<Index> m_patchPoints;
    int m_nodeOffset;
    int m_treeOffset;
    int m_patchPointsOffset; // global offset m_patchPoints

    static int maxIsolationLevel() { return 10; }

    // patch points
    static int catmarkRegularPatchSize() { return 16; };
    static int catmarkTerminalPatchSize() { return 25; };
    static int loopRegularPatchSize() { return 12; };

    // node sizes (in 'ints', not bytes)
    static int regularNodeSize(bool singleCrease) { return singleCrease ? 3 : 2; }
    static int endCapNodeSize() { return 2; }
    static int terminalNodeSize() { return 3; }
    static int recursiveNodeSize() { return 6; }

    static int getNumChildren(NodeType type)
    {
        switch (type)
        {
        case NodeType::NODE_TERMINAL: return 1;
        case NodeType::NODE_RECURSIVE: return 4;
        default: return 0;
        }
    }

    static int rootNodeOffset() { return 14; }

    // internal node offsets in tree array
    int descriptorOffset() { return m_nodeOffset; }
    int sharpnessOffset() { return m_nodeOffset + 2; }
    int patchPointsOffset() { return m_nodeOffset + 1; }
    int childOffset(int childIndex) { return m_nodeOffset + 2 + childIndex; }

    float GetSharpness()
    {
        return asfloat(m_subpatchTrees[m_treeOffset + sharpnessOffset()]);
    }

    SubdivisionNode GetChild(int childIndex)
    {
        SubdivisionNode child;
        child.m_subpatchTrees = m_subpatchTrees;
        child.m_patchPoints = m_patchPoints;
        child.m_nodeOffset = m_subpatchTrees[m_treeOffset + childOffset(childIndex)];
        child.m_treeOffset = m_treeOffset;
        child.m_patchPointsOffset = m_patchPointsOffset;
        return child;
    }

    NodeDescriptor GetDesc()
    {
        return MakeNodeDescriptor(m_subpatchTrees[m_treeOffset + descriptorOffset()]);
    }

    int GetPatchPointBase()
    {
        return m_subpatchTrees[m_treeOffset + patchPointsOffset()];
    }

    Index GetPatchPoint(
        int pointIndex,
        int quadrant,
        uint16_t maxLevel)
    {
        int offset = GetPatchPointBase();
        if (offset == INDEX_INVALID)
        {
            return INDEX_INVALID;
        }

        NodeDescriptor desc = GetDesc();
        switch (desc.GetType())
        {
        case NODE_REGULAR:
        case NODE_END:
            offset += pointIndex;
            break;
        case NODE_RECURSIVE:
            offset = (desc.GetDepth() >= maxLevel) && desc.HasEndcap() ? offset + pointIndex : INDEX_INVALID;
            break;
        case NODE_TERMINAL:
            // Unsupported, uses quadrant
            break;
        default:
            break;
        }
        return m_patchPoints[m_patchPointsOffset + offset];
    }
};

#endif // OSD_PORTS_TMR_SUBDIVISION_NODE_H
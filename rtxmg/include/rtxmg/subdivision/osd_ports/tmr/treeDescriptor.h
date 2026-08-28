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

#ifndef OSD_PORTS_TMR_TREE_DESCRIPTOR_H
#define OSD_PORTS_TMR_TREE_DESCRIPTOR_H

#include "rtxmg/subdivision/osd_ports/tmr/types.h"

struct TreeDescriptorHLSL
{
    StructuredBuffer<uint32_t> m_subpatchTrees;
    uint32_t m_treeOffset;

    static uint32_t const NumPatchPointsOffset = 2;

    bool IsRegularFace()
    {
        return unpack(m_subpatchTrees[m_treeOffset], 1, 0) != 0;
    }

    uint32_t GetFaceSize()
    {
        return unpack(m_subpatchTrees[m_treeOffset], 16, 16);
    }

    uint32_t GetSubfaceIndex()
    {
        return unpack(m_subpatchTrees[m_treeOffset], 16, 0);
    }

    uint32_t GetNumControlPoints()
    {
        return unpack(m_subpatchTrees[m_treeOffset], 16, 16);
    }

    uint32_t GetNumPatchPoints(uint16_t level)
    {
        return m_subpatchTrees[m_treeOffset + NumPatchPointsOffset + level];
    }
};

#endif // OSD_PORTS_TMR_TREE_DESCRIPTOR_H
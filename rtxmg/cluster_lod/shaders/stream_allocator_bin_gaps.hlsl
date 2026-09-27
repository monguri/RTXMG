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

/*

  Shader Description
  ==================


    Note: The sample showcases two ways to manage CLAS memory on the device.
    One using a persistent allocator system (`stream_allocator...` files),
    and one using a simple compaction scheme (`stream_compaction...` files).
    This file is part of the allocator system.

  This compute shader bins the free gaps based on their size.
  It enables the allocator to provide empty gaps of certain sizes during
  the allocation process within `stream_allocator_alloc_groups.hlsl`.

  We read `streaming.clasAllocator.freeGapsPos` and `streaming.clasAllocator.freeGapsSize`
  and bin into `streaming.clasAllocator.freeGapsPosBinned` using the appropriate
  `streaming.clasAllocator.freeSizeRanges.d[freeGapSize-1].offset`

  One thread operates on one free gap

  rtxmg port notes (D3D12, no GL_EXT_buffer_reference):
  -----------------------------------------------------

  * `freeGapsPos / freeGapsSize / freeGapsPosBinned / freeSizeRanges` all live
    as byte-offset sub-regions inside a single RWByteAddressBuffer
    (`u_AllocatorMem`); the per-region base offsets are in
    `streamingRW[0].clasAllocator.*ByteOffset` (see shaderio::StreamingAllocator).
  * `streaming` (UBO) + `streamingRW` (SSBO) collapse into a single
    RWStructuredBuffer<SceneStreaming>.
  * AllocatorRange is { int32 count; uint32 offset; }, so the byte stride is 8
    and the atomic on `count` targets the slot's +0 offset.
*/

#pragma pack_matrix(row_major)

#include "rtxmg/cluster_lod/shaderio.h"

////////////////////////////////////////////

RWByteAddressBuffer                           u_AllocatorMem : register(u1);
RWStructuredBuffer<shaderio::SceneStreaming>  streamingRW    : register(u7);

////////////////////////////////////////////

[numthreads(shaderio::kStreamAllocatorFreegapsInsertThreads, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint threadID = gid.x;
    bool valid    = threadID < streamingRW[0].clasAllocator.freeGapsCounter;

    if (valid)
    {
        // get the details of the free gap, it was computed in
        // `stream_allocator_scan_gaps.hlsl`.

        uint freeGapPos  = u_AllocatorMem.Load<uint>(streamingRW[0].clasAllocator.freeGapsPosByteOffset + threadID * 4u);
        uint freeGapSize = uint(u_AllocatorMem.Load<uint16_t>(streamingRW[0].clasAllocator.freeGapsSizeByteOffset + threadID * 2u));

        // bin the gap into `streaming.clasAllocator.freeGapsPosBinned` based on size
        uint rangeSlotByteOffset = streamingRW[0].clasAllocator.freeSizeRangesByteOffset
                                 + (freeGapSize - 1u) * uint(sizeof(shaderio::AllocatorRange));

        uint rangeIndex;
        u_AllocatorMem.InterlockedAdd(rangeSlotByteOffset, 1u, rangeIndex);
        uint rangeOffset = u_AllocatorMem.Load<shaderio::AllocatorRange>(rangeSlotByteOffset).offset;

        uint storeOffset = rangeIndex + rangeOffset;
        u_AllocatorMem.Store<uint>(streamingRW[0].clasAllocator.freeGapsPosBinnedByteOffset + storeOffset * 4u, freeGapPos);
    }
}

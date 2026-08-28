/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  Binding notes:
  --------------

  * All allocator sub-arrays are byte-offset regions of the single
    RWByteAddressBuffer u_AllocatorMem (see StreamingAllocator in shaderio.h).
    freeSizeRanges is AllocatorRange (8 B per slot: int32 count at +0,
    uint32 offset at +4).
  * All streaming state lives in one RWStructuredBuffer<SceneStreaming>.
*/

#pragma pack_matrix(row_major)

#include "rtxmg/cluster_lod/shaderio.h"

////////////////////////////////////////////

RWByteAddressBuffer                           u_AllocatorMem  : register(u1);
RWStructuredBuffer<shaderio::SceneStreaming>  streamingRW     : register(u7);

////////////////////////////////////////////

[numthreads(shaderio::kStreamAllocatorSetupInsertionThreads, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint threadID = gid.x;
    bool valid    = threadID < streamingRW[0].clasAllocator.maxAllocationSize;

    if (valid)
    {
        uint rangeBase = streamingRW[0].clasAllocator.freeSizeRangesByteOffset
                       + threadID * uint(sizeof(shaderio::AllocatorRange));

        // from the previous kernel `stream_allocator_scan_gaps.hlsl` we know how
        // many slots the size-binned array will need
        shaderio::AllocatorRange range = u_AllocatorMem.Load<shaderio::AllocatorRange>(rangeBase);

        // get an offset into `streaming.clasAllocator.freeGapsPosBinned` for the list of
        uint rangeOffset;
        InterlockedAdd(streamingRW[0].clasAllocator.freeGapsCounter, uint(range.count), rangeOffset);

        // setup range offset; reset count to zero for insertion done in
        // `stream_allocator_bin_gaps.hlsl`
        shaderio::AllocatorRange newRange;
        newRange.count  = 0;
        newRange.offset = rangeOffset;
        u_AllocatorMem.Store<shaderio::AllocatorRange>(rangeBase, newRange);
    }
}

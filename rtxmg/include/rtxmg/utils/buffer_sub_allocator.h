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

// rtxmg::BufferSubAllocator — port of nvpro_core2/nvvk::BufferSubAllocator onto
// nvrhi + donut's DescriptorTableManager.  The one interface change vs upstream:
// BufferSubAllocation::blockIndex is public, so callers pack the bindless slot
// straight into shader-visible structs without a subRange() round-trip.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <donut/engine/DescriptorTableManager.h>
#include <nvrhi/nvrhi.h>

#include "rtxmg/utils/offset_allocator.h"

namespace rtxmg
{

class BufferSubAllocator;  // forward decl

class BufferSubAllocation
{
public:
    BufferSubAllocation() = default;

    operator bool() const { return allocation.offset != OffsetAllocator::Allocation::NO_SPACE; }

    // Bindless slot of this block's RawBuffer SRV; populated by subAllocate.
    uint32_t blockIndex = ~0u;

private:
    friend class BufferSubAllocator;

    // the allocation.offset is in units of BufferSubAllocator's minAlignment
    OffsetAllocator::Allocation allocation;

    // original requested allocation size
    // the OffsetAllocator's size may be bigger given its internal free space search
    uint32_t size{};

    // original requested alignment
    // This alignment may need to be applied when converting the allocation.offset back
    // to actual byte offset
    uint16_t alignmentMinusOne{};

    uint16_t block{};
#ifndef NDEBUG
    class BufferSubAllocator* allocator{};
#endif
};

// information about a range within a buffer
struct BufferRange
{
    nvrhi::IBuffer*          buffer       = nullptr;  // owning block buffer
    nvrhi::GpuVirtualAddress address      = 0;        // contains offset already
    uint32_t                 bindlessSlot = ~0u;      // descriptor heap slot of the block's RawBuffer SRV
    uint64_t                 offset       = 0;        // byte offset within buffer
    uint64_t                 range        = 0;        // byte size (== subAllocation.size)
};

// Allocates blocks of buffers that one can sub allocate from.
// If a requested allocation size is bigger than the block size, a dedicated
// block/buffer will be used.
class BufferSubAllocator
{
public:
    static constexpr uint32_t kMinAlignment     = 4;
    static constexpr uint32_t kMaxAlignment     = 1 << (sizeof(uint16_t) * 8);
    static constexpr uint32_t kMaxTotalBlocks   = 1 << (sizeof(uint16_t) * 8);
    static constexpr uint64_t kDefaultBlockSize = uint64_t(128) * 1024 * 1024;
    static constexpr uint32_t kDefaultAlignment = 16;

    BufferSubAllocator() = default;
    ~BufferSubAllocator();

    // Delete copy constructor and copy assignment operator
    BufferSubAllocator(const BufferSubAllocator&)            = delete;
    BufferSubAllocator& operator=(const BufferSubAllocator&) = delete;

    // Allow move constructor and move assignment operator
    BufferSubAllocator(BufferSubAllocator&& other) noexcept;
    BufferSubAllocator& operator=(BufferSubAllocator&& other) noexcept;

    struct InitInfo
    {
        nvrhi::IDevice*                        device          = nullptr;
        donut::engine::DescriptorTableManager* descriptorTable = nullptr;

        std::string debugName;

        // Create blocks as acceleration-structure storage instead of RawBuffer
        // SRVs (the cached-BLAS pool).  Such blocks get no bindless SRV —
        // bindlessSlot stays ~0u and callers use subRange().address.
        bool isAccelStructStorage = false;

        // must be power-of-two
        uint32_t minAlignment = kDefaultAlignment;

        // a single block's OffsetAllocator can track this many sub-allocations
        uint32_t perBlockAllocations = 128 * 1024;

        // size of each block
        uint64_t blockSize = kDefaultBlockSize;

        // 0 will default to blockSize * kMaxTotalBlocks
        uint64_t maxAllocatedSize = 0;

        // to avoid freeing and allocating blocks in succession
        bool keepLastBlock = true;
    };

    bool     init(const InitInfo& createInfo);
    void     deinit();
    // True once init() has succeeded and before deinit() — lets callers
    // lazy-init the allocator on first use without tracking a parallel flag.
    bool     isInitialized() const { return m_info.device != nullptr; }
    uint64_t getMaxAllocationSize() const { return m_state.maxAllocationSize; }

    // Block accessors, indexed by host block index ∈ [0, getBlockCount());
    // deallocated slots return nullptr / ~0u.
    uint32_t        getBlockCount()        const { return uint32_t(m_blocks.size()); }
    nvrhi::IBuffer* getBlockBuffer(uint32_t hostBlockIdx) const
    {
        return hostBlockIdx < m_blocks.size() ? m_blocks[hostBlockIdx].buffer.Get() : nullptr;
    }
    uint32_t        getBlockBindlessSlot(uint32_t hostBlockIdx) const
    {
        return hostBlockIdx < m_blocks.size() ? m_blocks[hostBlockIdx].bindlessSlot : ~0u;
    }
    nvrhi::GpuVirtualAddress getBlockGpuAddress(uint32_t hostBlockIdx) const
    {
        return (hostBlockIdx < m_blocks.size() && m_blocks[hostBlockIdx].buffer)
                   ? m_blocks[hostBlockIdx].buffer->getGpuVirtualAddress()
                   : 0;
    }

    struct Report
    {
        // sum of requests made by user
        uint64_t requestedSize{};
        // internal usage, can be greater than requestedSize
        uint64_t reservedSize{};
        // what is available within internal usage
        uint64_t freeSize{};
        // VRAM the block buffers actually occupy: a multiple of blockSize, plus
        // any dedicated over-sized blocks.
        uint64_t allocatedSize{};
    };

    // current report on memory consumption
    Report getReport() const;

    // sub allocate
    // alignment must fulfill kMinAlignment and kMaxAlignment
    // alignment is legal to be non-power-of-two, but must be divisible by kMinAlignment;
    //   the returned offsets will then be a multiple of the alignment
    // size must be <= getMaxAllocationSize()
    bool subAllocate(BufferSubAllocation& subAllocation, uint64_t size, uint32_t alignment = kDefaultAlignment);

    // Free sub allocation
    // Passing an invalid suballocation (bool(subAllocation) == false) is valid
    void subFree(BufferSubAllocation& subAllocation);

    // Get information about buffer/binding etc.
    // Passing an invalid suballocation (bool(subAllocation) == false) is valid
    // and will just return a zeroed output
    BufferRange subRange(const BufferSubAllocation& subAllocation) const;

protected:
    static constexpr uint32_t kInvalidBlockIndex = ~0u;

    bool createNewBuffer(uint32_t blockIndex, uint64_t size);
    void destroyBuffer(uint32_t blockIndex);

    uint32_t acquireBlockIndex();

    struct Block
    {
        // can be null for dedicated block that has only a single big allocation > m_info.blockSize
        std::unique_ptr<OffsetAllocator::Allocator> offsetAllocator;
        // can be null if block was fully deallocated
        nvrhi::BufferHandle             buffer;
        donut::engine::DescriptorHandle bindlessHandle;
        uint64_t                        bufferSize   = 0;
        uint32_t                        bindlessSlot = ~0u;
        // continuation of single linked list of blocks that were deallocated completely
        uint32_t nextFreeIndex = kInvalidBlockIndex;
        // continuation of double linked list of blocks that have OffsetAllocators
        uint32_t nextActiveIndex = kInvalidBlockIndex;
        uint32_t prevActiveIndex = kInvalidBlockIndex;
    };

    struct State
    {
        // adjusted size based on config
        uint64_t maxAllocationSize{};
        // adjusted size as the offset allocator operates in units of m_info.minAlignment
        uint32_t internalBlockUnits{};
        // adjusted max blocks based on m_info.maxAllocatedSize
        uint32_t maxBlocks{};

        // statistics
        uint64_t allocatedSize{};

        // single linked list of blocks that were deallocated completely (list head)
        uint32_t freeBlockIndex = kInvalidBlockIndex;

        // active blocks are blocks that have OffsetAllocators (i.e. not dedicated to a single allocation)
        uint32_t activeBlockCount = 0;

        // double linked list of blocks that are active (list head)
        uint32_t activeBlockIndex = kInvalidBlockIndex;
    };

    InitInfo           m_info;
    State              m_state;
    std::vector<Block> m_blocks;
};

}  // namespace rtxmg

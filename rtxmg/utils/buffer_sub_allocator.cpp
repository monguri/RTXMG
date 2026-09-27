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

// Port of nvpro_core2/nvvk::BufferSubAllocator; the nvrhi/donut adaptations are
// confined to createNewBuffer / destroyBuffer / subRange.

#include "rtxmg/utils/buffer_sub_allocator.h"

#include <cassert>

#include <donut/core/log.h>

namespace rtxmg
{

BufferSubAllocator::~BufferSubAllocator()
{
    // Owners hold this by value and may never call deinit() explicitly.
    deinit();
}

BufferSubAllocator::BufferSubAllocator(BufferSubAllocator&& other) noexcept
{
    // Whole-State swap: the block/free-list indices describe m_blocks and must
    // travel with it.
    std::swap(m_state, other.m_state);
    std::swap(m_blocks, other.m_blocks);
    std::swap(m_info, other.m_info);
}

BufferSubAllocator& BufferSubAllocator::operator=(BufferSubAllocator&& other) noexcept
{
    if (this != &other)
    {
        // Reclaim this allocator's blocks before taking over the other's.
        deinit();

        std::swap(m_state, other.m_state);
        std::swap(m_blocks, other.m_blocks);
        std::swap(m_info, other.m_info);
    }

    return *this;
}

bool BufferSubAllocator::init(const InitInfo& info)
{
    assert(m_info.device == nullptr);

    assert(info.device != nullptr);
    // Accel-struct blocks get no bindless SRV, so they need no descriptor table.
    assert(info.isAccelStructStorage || info.descriptorTable != nullptr);
    assert(info.minAlignment <= kMaxAlignment);
    assert(info.minAlignment >= kMinAlignment);

    // BufferSubAllocation::size is uint32_t, which bounds a single allocation.
    m_state.maxAllocationSize = (uint64_t(1) << (sizeof(BufferSubAllocation::size) * 8)) - 1;

    assert(info.blockSize <= m_state.maxAllocationSize);

    m_info = info;
    if (!m_info.maxAllocatedSize)
    {
        m_info.maxAllocatedSize = info.blockSize * kMaxTotalBlocks;
    }

    size_t maxBlocks = (m_info.maxAllocatedSize + m_info.blockSize - 1) / m_info.blockSize;
    assert(maxBlocks <= kMaxTotalBlocks);

    m_state.maxBlocks          = static_cast<uint32_t>(maxBlocks);
    m_state.internalBlockUnits = static_cast<uint32_t>((m_info.blockSize + m_info.minAlignment - 1) / m_info.minAlignment);

    if (m_info.keepLastBlock)
    {
        m_blocks.push_back({});
        Block& block = m_blocks.back();
        block.offsetAllocator =
            std::make_unique<OffsetAllocator::Allocator>(m_state.internalBlockUnits, m_info.perBlockAllocations);
        if (!createNewBuffer(0, uint64_t(m_state.internalBlockUnits) * m_info.minAlignment))
        {
            return false;
        }

        m_state.activeBlockCount = 1;
        m_state.activeBlockIndex = 0;
    }

    return true;
}

void BufferSubAllocator::deinit()
{
    if (!m_info.device)
        return;

    for (uint32_t i = 0; i < uint32_t(m_blocks.size()); ++i)
    {
        destroyBuffer(i);
    }

    m_info  = {};
    m_state = {};
    m_blocks.clear();
    m_blocks.shrink_to_fit();
}

BufferSubAllocator::Report BufferSubAllocator::getReport() const
{
    BufferSubAllocator::Report report;

    for (size_t i = 0; i < m_blocks.size(); i++)
    {
        const Block& block = m_blocks[i];

        const OffsetAllocator::Allocator* offsetAllocator = block.offsetAllocator.get();

        report.allocatedSize += block.bufferSize;

        if (offsetAllocator)
        {
            OffsetAllocator::StorageReport storageReport = offsetAllocator->storageReport();
            report.reservedSize += uint64_t(m_state.internalBlockUnits - storageReport.totalFreeSpace) * m_info.minAlignment;
            report.freeSize += uint64_t(storageReport.totalFreeSpace) * m_info.minAlignment;
        }
        else
        {
            // dedicated blocks
            report.reservedSize += block.bufferSize;
        }
    }

    report.requestedSize = m_state.allocatedSize;

    return report;
}

uint32_t BufferSubAllocator::acquireBlockIndex()
{
    uint32_t freeBlockIndex = m_state.freeBlockIndex;
    if (freeBlockIndex != kInvalidBlockIndex)
    {
        m_state.freeBlockIndex = m_blocks[m_state.freeBlockIndex].nextFreeIndex;
    }
    else
    {
        freeBlockIndex = uint32_t(m_blocks.size());
        m_blocks.push_back({});
    }

    return freeBlockIndex;
}

bool BufferSubAllocator::subAllocate(BufferSubAllocation& subAllocation, uint64_t size, uint32_t alignment)
{
    subAllocation = {};

    assert(alignment % kMinAlignment == 0);
    assert(alignment >= kMinAlignment && alignment <= kMaxAlignment);
    assert(size <= m_state.maxAllocationSize);

    bool alignmentIsPowerOfTwo = (alignment & (alignment - 1)) == 0;

    if (size + m_state.allocatedSize > m_info.maxAllocatedSize)
    {
        return false;
    }

    // Charged only on the success paths below: subFree is the only credit, so a
    // failed request that charged it would leak the budget for good.

    // if large use a dedicated block
    if (size >= m_info.blockSize)
    {
        if (m_state.freeBlockIndex == kInvalidBlockIndex && m_blocks.size() == size_t(m_state.maxBlocks))
        {
            return false;
        }

        // recycle a block or new one
        uint32_t freeBlockIndex = acquireBlockIndex();

        if (!alignmentIsPowerOfTwo)
        {
            // find largest power of 2 that fits into alignment
            uint32_t newAlignment = kMinAlignment;
            for (uint32_t searchAlignment = kMinAlignment; searchAlignment <= kMaxAlignment; searchAlignment *= 2)
            {
                if ((alignment & (searchAlignment - 1)) == 0)
                {
                    newAlignment = searchAlignment;
                }
                else
                {
                    break;
                }
            }
            alignment = newAlignment;
        }

        if (!createNewBuffer(freeBlockIndex, size))
        {
            return false;
        }

        subAllocation.allocation.offset   = 0;
        subAllocation.allocation.metadata = OffsetAllocator::Allocation::NO_SPACE;
        subAllocation.size                = static_cast<uint32_t>(size);
        subAllocation.alignmentMinusOne   = uint16_t(alignment - 1);
        subAllocation.block               = uint16_t(freeBlockIndex);
        subAllocation.blockIndex          = m_blocks[freeBlockIndex].bindlessSlot;
#ifndef NDEBUG
        subAllocation.allocator = this;
#endif

        // dedicated blocks are _not_ thrown into the active block list

        m_state.allocatedSize += size;
        return true;
    }


    // else try to find a sub allocation

    // adjust the size to account for local alignment
    uint64_t sizeAllocate = size;

    // for non power of two, always add extra space to return a proper offset
    if (!alignmentIsPowerOfTwo || alignment > m_info.minAlignment)
    {
        // adjust for requested alignment and add safety margin to size.
        // The offset returned from OffsetAllocator will only be aligned to m_info.minAlignment.
        // With the extra safety margin space, we can later adjust the returned offset to alignment,
        // see logic in `subRange`.
        sizeAllocate = (sizeAllocate + alignment - 1);
    }

    // offset allocator works in units of `m_info.minAlignment`
    uint32_t allocatorUnits = static_cast<uint32_t>((sizeAllocate + m_info.minAlignment - 1) / m_info.minAlignment);


    // iterate over active blocks to find allocation

    uint32_t activeBlockIndex = m_state.activeBlockIndex;

    while (activeBlockIndex != kInvalidBlockIndex)
    {
        Block& block = m_blocks[activeBlockIndex];

        // attempt to sub allocate from active blocks

        OffsetAllocator::Allocation allocation = block.offsetAllocator->allocate(allocatorUnits);

        if (allocation.offset != OffsetAllocator::Allocation::NO_SPACE)
        {
            subAllocation.allocation        = allocation;
            subAllocation.size              = static_cast<uint32_t>(size);
            subAllocation.alignmentMinusOne = uint16_t(alignment - 1);
            subAllocation.block             = uint16_t(activeBlockIndex);
            subAllocation.blockIndex        = block.bindlessSlot;
#ifndef NDEBUG
            subAllocation.allocator = this;
#endif

            m_state.allocatedSize += size;
            return true;
        }

        activeBlockIndex = block.nextActiveIndex;
    }

    // could not find anything

    // if we reached the limit for blocks, bail out
    if (m_state.freeBlockIndex == kInvalidBlockIndex && m_blocks.size() == size_t(m_state.maxBlocks))
    {
        return false;
    }

    {
        // add new block

        uint32_t freeBlockIndex = acquireBlockIndex();

        Block& block          = m_blocks[freeBlockIndex];
        block.offsetAllocator =
            std::make_unique<OffsetAllocator::Allocator>(m_state.internalBlockUnits, m_info.perBlockAllocations);
        if (!createNewBuffer(freeBlockIndex, uint64_t(m_state.internalBlockUnits) * m_info.minAlignment))
        {
            return false;
        }

        // insert block into active block list
        if (m_state.activeBlockIndex != kInvalidBlockIndex)
        {
            m_blocks[m_state.activeBlockIndex].prevActiveIndex = freeBlockIndex;
        }

        block.nextActiveIndex = m_state.activeBlockIndex;

        // make it new list head
        m_state.activeBlockIndex = freeBlockIndex;
        m_state.activeBlockCount++;


        // sub allocate from new block

        OffsetAllocator::Allocation allocation = block.offsetAllocator->allocate(allocatorUnits);

        if (allocation.offset != OffsetAllocator::Allocation::NO_SPACE)
        {
            subAllocation.allocation        = allocation;
            subAllocation.size              = static_cast<uint32_t>(size);
            subAllocation.alignmentMinusOne = uint16_t(alignment - 1);
            subAllocation.block             = uint16_t(freeBlockIndex);
            subAllocation.blockIndex        = block.bindlessSlot;
#ifndef NDEBUG
            subAllocation.allocator = this;
#endif

            m_state.allocatedSize += size;
            return true;
        }
        else
        {
            return false;
        }
    }
}

void BufferSubAllocator::subFree(BufferSubAllocation& subAllocation)
{
    // make it legal to pass unset ranges
    if (!subAllocation)
    {
        return;
    }

#ifndef NDEBUG
    assert(subAllocation.allocator == this);
#endif

    OffsetAllocator::Allocator* offsetAllocator = m_blocks[subAllocation.block].offsetAllocator.get();

    // dedicated blocks might not have an offset allocator
    if (offsetAllocator)
    {
        offsetAllocator->free(subAllocation.allocation);
    }

    m_state.allocatedSize -= subAllocation.size;

    // check if dedicated block or empty
    if (!offsetAllocator || offsetAllocator->storageReport().totalFreeSpace == m_state.internalBlockUnits)
    {
        // always free if dedicated
        // and maybe depending if we are the last one
        if (!offsetAllocator || (m_state.activeBlockCount > 1 || !m_info.keepLastBlock))
        {
            destroyBuffer(subAllocation.block);

            // blocks with OffsetAllocators are counted to active blocks
            if (offsetAllocator)
            {
                m_state.activeBlockCount--;

                // need to remove from linked list of active blocks

                uint32_t selfActiveIndex = subAllocation.block;
                uint32_t prevActiveIndex = m_blocks[selfActiveIndex].prevActiveIndex;
                uint32_t nextActiveIndex = m_blocks[selfActiveIndex].nextActiveIndex;
                if (prevActiveIndex != kInvalidBlockIndex)
                {
                    // set previous's next to self next
                    m_blocks[prevActiveIndex].nextActiveIndex = nextActiveIndex;
                }
                if (nextActiveIndex != kInvalidBlockIndex)
                {
                    // set next's previous to self previous
                    m_blocks[nextActiveIndex].prevActiveIndex = prevActiveIndex;
                }
                if (m_state.activeBlockIndex == selfActiveIndex)
                {
                    m_state.activeBlockIndex = nextActiveIndex;
                }
            }

            // nuke it completely
            m_blocks[subAllocation.block] = {};

            // chain into linked list of empty blocks
            m_blocks[subAllocation.block].nextFreeIndex = m_state.freeBlockIndex;
            m_state.freeBlockIndex                      = subAllocation.block;
        }
    }

    subAllocation = {};
}

BufferRange BufferSubAllocator::subRange(const BufferSubAllocation& subAllocation) const
{
    // make it legal to pass unset ranges
    if (!subAllocation)
    {
        return {};
    }

#ifndef NDEBUG
    assert(subAllocation.allocator == this);
#endif

    const Block& block = m_blocks[subAllocation.block];

    BufferRange info;
    info.buffer       = block.buffer.Get();
    info.bindlessSlot = block.bindlessSlot;
    info.range        = subAllocation.size;

    // OffsetAllocator's offset is in units of `m_info.minAlignment`
    info.offset = uint64_t(subAllocation.allocation.offset) * m_info.minAlignment;

    // The original requested alignment might have been greater than the minAlignment,
    // or might be non-power-of-two.
    // In that case we need to re-adjust the offset, which is safe to work as we
    // allocated a safety margin.
    uint32_t alignment = uint32_t(subAllocation.alignmentMinusOne) + 1;

    // allow non-power-of-two alignments
    uint64_t rest = info.offset % alignment;
    if (rest != 0)
    {
        info.offset += alignment - rest;
    }

    // apply offset to address
    info.address = info.buffer ? (info.buffer->getGpuVirtualAddress() + info.offset) : 0;

    return info;
}

bool BufferSubAllocator::createNewBuffer(uint32_t blockIndex, uint64_t size)
{
    Block& block = m_blocks[blockIndex];

    nvrhi::BufferDesc desc =
        nvrhi::BufferDesc()
            .setByteSize(size)
            .setFormat(nvrhi::Format::UNKNOWN)
            .setKeepInitialState(true)
            .setDebugName(m_info.debugName + "_block_" + std::to_string(blockIndex));

    if (m_info.isAccelStructStorage)
    {
        // D3D12 requires ALLOW_UNORDERED_ACCESS on any buffer that can be in the
        // RAYTRACING_ACCELERATION_STRUCTURE state, hence canHaveUAVs.
        desc.setCanHaveUAVs(true)
            .setIsAccelStructStorage(true)
            .setInitialState(nvrhi::ResourceStates::AccelStructWrite);
    }
    else
    {
        desc.setCanHaveRawViews(true).setInitialState(nvrhi::ResourceStates::ShaderResource);
    }

    block.buffer = m_info.device->createBuffer(desc);
    if (!block.buffer)
    {
        donut::log::error("BufferSubAllocator::createNewBuffer failed (size=%llu block=%u)",
                          static_cast<unsigned long long>(size), blockIndex);
        return false;
    }

    if (!m_info.isAccelStructStorage)
    {
        block.bindlessHandle =
            m_info.descriptorTable->CreateDescriptorHandle(nvrhi::BindingSetItem::RawBuffer_SRV(0, block.buffer));
        block.bindlessSlot = static_cast<uint32_t>(block.bindlessHandle.GetIndexInHeap());
    }
    block.bufferSize   = size;
    return true;
}

void BufferSubAllocator::destroyBuffer(uint32_t blockIndex)
{
    Block& block = m_blocks[blockIndex];
    // donut::DescriptorHandle releases the descriptor on destruction.
    block.bindlessHandle = {};
    block.buffer         = nullptr;
    block.bufferSize     = 0;
    block.bindlessSlot   = ~0u;
}

}  // namespace rtxmg

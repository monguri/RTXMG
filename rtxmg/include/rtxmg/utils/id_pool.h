/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

// The range algorithm below derives from Emil Persson's MakeID (v1.02,
// http://www.humus.name/3D/MakeID.h), which is public domain — the header above
// covers this port, not the original.  See notice.txt.

#pragma once

#include <cstdint>

namespace rtxmg {

// This class provides a way to create unique IDs out of a maximum pool.
// Useful to implement bindless texture index or similar allocators.
class IDPool
{
public:
    IDPool() = default;

    // number of elements in pool
    // poolSize must be >= 1
    // highest id is `poolSize-1`
    IDPool(uint32_t poolSize) { init(poolSize); }

    IDPool(const IDPool& other)            = delete;
    IDPool& operator=(const IDPool& other) = delete;

    IDPool(IDPool&& other) noexcept;
    IDPool& operator=(IDPool&& other) noexcept;

    ~IDPool() { deinit(); }

    // number of elements in pool
    // poolSize must be >= 1
    // highest id is `poolSize-1`
    void init(const uint32_t poolSize);
    void deinit();

    // operations return true on success

    // single ID
    bool createID(uint32_t& id);

    // consecutive IDs starting at returned id
    bool createRangeID(uint32_t& id, const uint32_t count);

    bool destroyID(const uint32_t id) { return destroyRangeID(id, 1); }
    bool destroyRangeID(const uint32_t id, const uint32_t count);
    void destroyAll();

    bool isRangeAvailable(uint32_t searchCount) const;

    void printRanges() const;
    void checkRanges() const;

private:
    struct Range
    {
        uint32_t first;
        uint32_t last;
    };

    Range*   m_ranges   = nullptr;  // Sorted array of ranges of free IDs
    uint32_t m_count    = 0;        // Number of ranges in list
    uint32_t m_capacity = 0;        // Total capacity of range list
    uint32_t m_maxID    = 0;        // Highest ID value
    uint32_t m_usedIDs  = 0;        // Number of IDs in use

    void insertRange(const uint32_t index);
    void destroyRange(const uint32_t index);
};

}  // namespace rtxmg

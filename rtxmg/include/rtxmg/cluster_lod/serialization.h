/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

// Pure pointer-arithmetic helpers for writing/reading the .nvsngeo cache format.
// No external dependencies — only <cstdint>, <cstring>, <cassert>, <span>.

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>
#include <span>

namespace serialization {

static constexpr uint64_t ALIGNMENT  = 16ULL;
static constexpr uint64_t ALIGN_MASK = ALIGNMENT - 1ULL;
static_assert(ALIGNMENT >= sizeof(uint64_t));

// Returns the number of bytes needed to serialise one span:
//   16-byte count block + data rounded up to 16-byte alignment.
template<typename T>
inline uint64_t GetCachedSize(const std::span<const T>& view)
{
    return ((view.size_bytes() + ALIGN_MASK) & ~ALIGN_MASK) + ALIGNMENT;
}

// Overload for mutable spans (e.g. from std::vector::span cast).
template<typename T>
inline uint64_t GetCachedSize(const std::span<T>& view)
{
    return GetCachedSize(std::span<const T>(view));
}

// Write `view` into raw memory at `dataAddress`, advance `dataAddress`.
// Sets `isValid = false` on overflow.
template<typename T>
inline void StoreAndAdvance(bool& isValid, uint64_t& dataAddress, uint64_t dataEnd,
                            const std::span<const T>& view)
{
    assert(dataAddress % ALIGNMENT == 0);

    if (isValid && dataAddress + GetCachedSize(view) <= dataEnd)
    {
        // 16-byte count block (zero-padded).
        union { uint64_t count; uint8_t countData[ALIGNMENT]; };
        std::memset(countData, 0, ALIGNMENT);
        count = view.size();

        std::memcpy(reinterpret_cast<void*>(dataAddress), countData, ALIGNMENT);
        dataAddress += ALIGNMENT;

        if (view.size())
        {
            std::memcpy(reinterpret_cast<void*>(dataAddress), view.data(), view.size_bytes());
            dataAddress += (view.size_bytes() + ALIGN_MASK) & ~ALIGN_MASK;
        }
    }
    else
    {
        isValid = false;
    }
}

// Read a span from raw memory at `dataAddress`, advance `dataAddress`.
// The returned span's data() points directly into the raw memory (zero-copy).
template<typename T>
inline void LoadAndAdvance(bool& isValid, uint64_t& dataAddress, uint64_t dataEnd,
                           std::span<const T>& view)
{
    assert(dataAddress % ALIGNMENT == 0);

    view = {};

    // The count block itself must lie inside the blob; a truncated file would
    // otherwise be read past its end just to size the span.
    if (!isValid || dataAddress + ALIGNMENT > dataEnd)
    {
        isValid = false;
        return;
    }

    const uint64_t count = *reinterpret_cast<const uint64_t*>(dataAddress);
    dataAddress += ALIGNMENT;

    // Divide rather than multiply: a corrupt count overflows sizeof(T) * count
    // and wraps the bound check into passing.
    if (count > (dataEnd - dataAddress) / sizeof(T))
    {
        isValid = false;
        return;
    }

    if (count)
        view = std::span<const T>(reinterpret_cast<const T*>(dataAddress), count);

    dataAddress += sizeof(T) * count;
    dataAddress  = (dataAddress + ALIGN_MASK) & ~ALIGN_MASK;
}

} // namespace serialization

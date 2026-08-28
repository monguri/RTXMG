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

// FNV-1a 64-bit hash — small, dependency-free content hashing for the
// cluster-LOD shard cache (naming shards by the hash of a geometry's source
// bytes, so identical content reuses a shard regardless of byte offset).

#pragma once

#include <cstddef>
#include <cstdint>

namespace rtxmg {

inline constexpr uint64_t kFnv1aOffsetBasis = 0xcbf29ce484222325ull;
inline constexpr uint64_t kFnv1aPrime       = 0x00000100000001b3ull;

// Hash `size` bytes at `data` into the running `seed` (FNV-1a).  Chain calls to
// hash several disjoint ranges into one digest.
inline uint64_t Fnv1a(const void* data, size_t size, uint64_t seed = kFnv1aOffsetBasis)
{
    const auto* p = static_cast<const uint8_t*>(data);
    uint64_t    h = seed;
    for (size_t i = 0; i < size; ++i)
    {
        h ^= p[i];
        h *= kFnv1aPrime;
    }
    return h;
}

// Mix a scalar value into the running hash (e.g. a count or length, so that two
// concatenated ranges hash differently from a single combined range).
template <typename T>
inline uint64_t Fnv1aValue(const T& value, uint64_t seed = kFnv1aOffsetBasis)
{
    return Fnv1a(&value, sizeof(T), seed);
}

} // namespace rtxmg

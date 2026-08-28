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

// A small dynamic-work parallel_for over donut's ThreadPool (which exposes only
// AddTask / WaitForTasks).  Spawns `workerCount` worker tasks that each pull
// item indices from a shared atomic cursor, so uneven per-item cost (e.g.
// baking a few huge meshes amongst many small ones) self-balances instead of
// stranding a worker.  Blocks until every item has been processed.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <donut/engine/ThreadPool.h>

namespace rtxmg {

// Calls fn(index, workerIdx) for each index in [0, count), distributed across
// `workerCount` workers of `pool`.  workerIdx is in [0, workerCount) and is
// stable for the duration of one worker — use it to index per-worker scratch.
// fn must be safe to call concurrently for distinct indices.  Runs serially in
// the calling thread when workerCount <= 1 or count <= 1 (no enqueue overhead).
template <typename Fn>
void ParallelFor(donut::engine::ThreadPool& pool, uint32_t workerCount, size_t count, Fn&& fn)
{
    if (count == 0)
        return;

    if (workerCount <= 1 || count == 1)
    {
        for (size_t i = 0; i < count; ++i)
            fn(i, uint32_t(0));
        return;
    }

    if (workerCount > count)
        workerCount = uint32_t(count);

    std::atomic<size_t> cursor{ 0 };

    for (uint32_t w = 0; w < workerCount; ++w)
    {
        pool.AddTask([w, count, &cursor, &fn]() {
            for (;;)
            {
                size_t i = cursor.fetch_add(1, std::memory_order_relaxed);
                if (i >= count)
                    break;
                fn(i, w);
            }
        });
    }

    pool.WaitForTasks();
}

} // namespace rtxmg

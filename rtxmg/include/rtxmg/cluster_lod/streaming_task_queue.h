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

#pragma once

#include <cstdint>
#include <queue>

#include <nvrhi/nvrhi.h>  // for nvrhi::EventQueryHandle

namespace rtxmg {

// Task-slot budget shared by every streaming sub-manager: each sizes its
// per-task ring by it, and the queue caps in-flight tasks at it.
inline constexpr uint32_t kStreamingMaxActiveTasks = 3;
inline constexpr uint32_t kInvalidTaskIndex        = ~0u;

//////////////////////////////////////////////////////////////////////////
//
// StreamingTaskQueue
//
// This is the central data structure to manage the lifetime of a task
// represented by a simple "taskIndex".
// We can test if a task has completed on the device and is available,
// furthermore we can pop such available tasks or push new ones.
//
// Each task is represented using a simple "taskIndex" (we recycle these).
//
// For a given task queue there can be only kStreamingMaxActiveTasks many
// tasks in-flight at any given time.
// We use busy waits till at least one slot is available top pop (see `ensureAcquisition`) to enforce this.
// That slot can the be re-cycled after it's release.
//
// ``` cpp
// // produce
// newTaskIndex = queue.AcquireTaskIndex();
// ... do stuff associating task's actual data with the index
// device->executeCommandList(cmd);
// queue.Push(newTaskIndex);
//
// // consume
// if (queue.CanPop(ensureAcquisition))
// {
//   completedTaskIndex = pop();
//   ... do stuff getting task's actual data using the index
//   queue.ReleaseTaskIndex(completedTaskIndex);
// }
//
// ```

class StreamingTaskQueue
{
public:
    static_assert(kStreamingMaxActiveTasks < 32);

    void Init(nvrhi::IDevice* device);
    void Deinit();

    uint32_t AcquireTaskIndex();
    void     ReleaseTaskIndex(uint32_t index);

    // Blocks on the head task's event query when `ensureAcquisition` and no slot
    // is free, so the caller is guaranteed an index every frame.
    bool CanPop(bool ensureAcquisition);

    void PushPending(uint32_t taskIndex);
    void SignalTask(uint32_t taskIndex);

    // Push a task after the command list that produced it has already been
    // submitted — setEventQuery fences against the queue's last submission.
    void Push(uint32_t taskIndex);

    uint32_t Pop();

private:
    struct Task
    {
        uint32_t taskIndex = kInvalidTaskIndex;
    };

    nvrhi::IDevice*         m_device = nullptr;
    nvrhi::EventQueryHandle m_eventQueries[kStreamingMaxActiveTasks];
    std::queue<Task>        m_taskQueue;
    uint32_t                m_availableTaskBits = (1u << kStreamingMaxActiveTasks) - 1u;
};

}  // namespace rtxmg

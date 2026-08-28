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

#include "rtxmg/cluster_lod/streaming_task_queue.h"

#include <cassert>

namespace rtxmg {

void StreamingTaskQueue::Init(nvrhi::IDevice* device)
{
    m_device            = device;
    m_availableTaskBits = (1u << kStreamingMaxActiveTasks) - 1u;
    for (uint32_t i = 0; i < kStreamingMaxActiveTasks; ++i)
    {
        m_eventQueries[i] = device->createEventQuery();
    }
}

void StreamingTaskQueue::Deinit()
{
    for (uint32_t i = 0; i < kStreamingMaxActiveTasks; ++i)
    {
        m_eventQueries[i] = nullptr;  // RefCountPtr releases
    }
    m_device = nullptr;
}

uint32_t StreamingTaskQueue::AcquireTaskIndex()
{
    // find available bit
    for (uint32_t i = 0; i < kStreamingMaxActiveTasks; ++i)
    {
        if (m_availableTaskBits & (1u << i))
        {
            m_availableTaskBits &= ~(1u << i);
            m_device->resetEventQuery(m_eventQueries[i]);
            return i;
        }
    }

    return kInvalidTaskIndex;
}

void StreamingTaskQueue::ReleaseTaskIndex(uint32_t index)
{
    assert((m_availableTaskBits & (1u << index)) == 0);
    m_availableTaskBits |= (1u << index);
}

bool StreamingTaskQueue::CanPop(bool ensureAcquisition)
{
    if (ensureAcquisition && !m_availableTaskBits && !m_taskQueue.empty())
    {
        // if there is no task bits available we must enforce a wait,
        // cause we must guarantee to have at least one available index
        // every frame
        m_device->waitEventQuery(m_eventQueries[m_taskQueue.front().taskIndex]);
    }

    return !m_taskQueue.empty() && m_device->pollEventQuery(m_eventQueries[m_taskQueue.front().taskIndex]);
}

void StreamingTaskQueue::PushPending(uint32_t taskIndex)
{
    m_taskQueue.push(Task{ .taskIndex = taskIndex });
}

void StreamingTaskQueue::SignalTask(uint32_t taskIndex)
{
    m_device->setEventQuery(m_eventQueries[taskIndex], nvrhi::CommandQueue::Graphics);
}

void StreamingTaskQueue::Push(uint32_t taskIndex)
{
    PushPending(taskIndex);
    SignalTask(taskIndex);
}

uint32_t StreamingTaskQueue::Pop()
{
    uint32_t taskIndex = m_taskQueue.front().taskIndex;
    assert(taskIndex != kInvalidTaskIndex);
    m_taskQueue.pop();
    return taskIndex;
}

}  // namespace rtxmg

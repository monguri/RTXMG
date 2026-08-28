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

// Thread-safe cluster-LOD bake progress, written by the bake workers and read by
// the render thread's loading bar.  A process-global singleton so the deep bake
// code and RenderSplashScreen reach it without threading a pointer through.

#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace rtxmg {

class BakeProgress
{
public:
    // Called once at the start of a bake/load phase.  `total` = geometries to
    // process (the slow set); `cached` = instant cache hits.  `label` is the
    // loading-screen caption for this phase (the progress bar is reused for
    // several load phases: bake, GPU metadata upload).
    void Begin(uint32_t total, uint32_t cached, uint32_t workerCount,
               const char* label = "Baking cluster LOD geometry")
    {
        m_total.store(total, std::memory_order_relaxed);
        m_cached.store(cached, std::memory_order_relaxed);
        m_done.store(0, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(m_mtx);
            m_inFlight.assign(workerCount, std::string());
            m_label = label;
        }
        m_active.store(true, std::memory_order_release);
    }

    // A worker started baking `name` (slot = its worker index).
    void SetInFlight(uint32_t worker, const std::string& name)
    {
        std::lock_guard<std::mutex> lock(m_mtx);
        if (worker < m_inFlight.size())
            m_inFlight[worker] = name;
    }

    // A worker finished one geometry.
    void CompleteOne(uint32_t worker)
    {
        m_done.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(m_mtx);
        if (worker < m_inFlight.size())
            m_inFlight[worker].clear();
    }

    void End() { m_active.store(false, std::memory_order_release); }

    // ---- readers (render thread) ----
    bool     IsActive() const { return m_active.load(std::memory_order_acquire); }
    uint32_t GetTotal()  const { return m_total.load(std::memory_order_relaxed); }
    uint32_t GetCached() const { return m_cached.load(std::memory_order_relaxed); }
    uint32_t GetDone()   const { return m_done.load(std::memory_order_relaxed); }

    // 0..1 fraction of the slow (to-bake) set completed (1 if nothing to bake).
    float GetFraction() const
    {
        const uint32_t t = GetTotal();
        return t == 0 ? 1.0f : float(GetDone()) / float(t);
    }

    // Loading-screen caption for the current phase (render thread).
    std::string GetLabel() const
    {
        std::lock_guard<std::mutex> lock(m_mtx);
        return m_label;
    }

    // Snapshot the non-empty in-flight mesh names (render thread).
    std::vector<std::string> GetInFlight() const
    {
        std::vector<std::string> out;
        std::lock_guard<std::mutex> lock(m_mtx);
        out.reserve(m_inFlight.size());
        for (const std::string& s : m_inFlight)
            if (!s.empty())
                out.push_back(s);
        return out;
    }

private:
    std::atomic<bool>        m_active{ false };
    std::atomic<uint32_t>    m_total{ 0 };
    std::atomic<uint32_t>    m_cached{ 0 };
    std::atomic<uint32_t>    m_done{ 0 };
    mutable std::mutex       m_mtx;
    std::vector<std::string> m_inFlight;  // per-worker current mesh name
    std::string              m_label;     // loading-screen caption for this phase
};

// Process-global bake progress (the bake writes it; the UI reads it).
inline BakeProgress& GetBakeProgress()
{
    static BakeProgress s_progress;
    return s_progress;
}

} // namespace rtxmg

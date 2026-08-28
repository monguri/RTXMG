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

// AdaptiveLodError — streaming-budget feedback on the LoD pixel error.  Raises
// the effective error while the streaming pools run hot, so the resident set
// fits the budgets, and walks it back down when there is headroom.  Driven by
// IClusterLodStreamingHooks::GetLoadFactor(), which is why this is streaming
// policy and not renderer policy.

#pragma once

#include <algorithm>

namespace rtxmg
{

class AdaptiveLodError
{
public:
    struct Config
    {
        // Weight of the newest load factor in the smoothed estimate.
        float smoothing    = 0.05f;
        // Deadband: outside [refineBelow, coarsenAbove] the error moves.  It is
        // what keeps the controller from oscillating.
        float coarsenAbove = 0.85f;
        float refineBelow  = 0.70f;
        // Per-frame multipliers.  Coarsening is the reaction to a budget that is
        // already overcommitted, so it is an order of magnitude faster than the
        // refine that gives detail back.
        float coarsenRate  = 1.02f;
        float refineRate   = 0.995f;
    };

    const Config& GetConfig() const { return m_config; }
    void SetConfig(const Config& config) { m_config = config; }

    // Fold this frame's streaming load factor into the smoothed estimate.
    void Observe(float loadFactor)
    {
        m_smoothedLoadFactor += (loadFactor - m_smoothedLoadFactor) * m_config.smoothing;
    }

    // One deadband step.  Returns the effective pixel error, never finer than
    // the error the user asked for.
    float Advance(float baseError)
    {
        m_effectiveError = std::max(baseError, m_effectiveError);
        if (m_smoothedLoadFactor > m_config.coarsenAbove)
            m_effectiveError *= m_config.coarsenRate;
        else if (m_smoothedLoadFactor < m_config.refineBelow)
            m_effectiveError *= m_config.refineRate;
        m_effectiveError = std::max(baseError, m_effectiveError);
        return m_effectiveError;
    }

    // Track the base error while the controller is off, so turning it on starts
    // from the current error rather than a stale adaptive value.
    void Reset(float baseError) { m_effectiveError = baseError; }

    float GetSmoothedLoadFactor() const { return m_smoothedLoadFactor; }
    float GetEffectiveError() const { return m_effectiveError; }

private:
    Config m_config             = {};
    float  m_smoothedLoadFactor = 0.0f;
    float  m_effectiveError     = 1.0f;
};

}  // namespace rtxmg

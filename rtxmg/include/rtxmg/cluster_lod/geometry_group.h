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

// The name of one streamable unit of geometry.  Its own header because both the
// streaming engine and the residency report on IClusterLodStreamingHooks speak
// it, and the hooks interface must not pull in the engine.

#pragma once

#include <cstdint>

namespace rtxmg
{

union GeometryGroup
{
    struct
    {
        uint32_t geometryID;
        uint32_t groupID;
    };
    uint64_t key;
};

}  // namespace rtxmg

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

#ifndef RTXMG_INSTANCE_DATA_H
#define RTXMG_INSTANCE_DATA_H

// Shared between C++ and HLSL via nvrhi/nvrhiHLSL.h
#include <nvrhi/nvrhiHLSL.h>

// Per-instance GPU data: 48 + 48 = 96 bytes (6 x 16-byte rows).
struct RTXMGInstanceData
{
    float3x4 transform;       // current frame object-to-world (column-major 3×4)
    float3x4 prevTransform;   // previous frame object-to-world (for motion vectors)
};

#ifdef __cplusplus
static_assert(sizeof(RTXMGInstanceData) == 96, "RTXMGInstanceData must be 96 bytes");
#endif

#endif // RTXMG_INSTANCE_DATA_H

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
//

#pragma once

#ifdef __cplusplus
#include <cstdint>
#endif

#include <nvrhi/nvrhiHLSL.h>

// Scratch device memory needed while filling clusters
struct TessellationCounters
{
    uint32_t clusters;
    uint32_t desiredClusters;
    uint32_t desiredVertices;
    uint32_t desiredTriangles;
    uint32_t desiredClasBlocks;

    // Pad for vulkan minStorageBufferOffsetAlignment = 16
    uint32_t pad[3];

#ifdef __cplusplus
    size_t DesiredClasBytes() const { return size_t(desiredClasBlocks) * nvrhi::rt::cluster::kClasByteAlignment; }
#endif
};

#ifdef __cplusplus
constexpr uint32_t kClusterCountByteOffset = offsetof(TessellationCounters, clusters);
#endif


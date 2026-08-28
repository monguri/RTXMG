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

#ifndef FILL_CLUSTERS_PARAMS_H // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define FILL_CLUSTERS_PARAMS_H

#include <nvrhi/nvrhiHLSL.h>
#include "rtxmg/subdivision/osd_ports/tmr/types.h"

// Number of clusters to calculate vertices for in a thread group
static const uint32_t kFillClustersVerticesWaves = 4;

// Number of lanes (threads) per wave for vertex cluster filling
static const uint32_t kFillClustersVerticesLanes = 32;

// We do cluster per x, y is the cluster UV evaluation points
static const uint32_t kFillClustersTexcoordsThreadsX = 32;

struct FillClustersParams
{
    uint32_t instanceIndex;
    uint32_t quantNBits;
    uint32_t isolationLevel;
    float globalDisplacementScale;

    uint32_t clusterPattern;
    uint32_t debugSurfaceIndex;
    uint32_t debugClusterIndex;
    uint32_t debugLaneIndex;
};

#endif // FILL_CLUSTERS_PARAMS_H
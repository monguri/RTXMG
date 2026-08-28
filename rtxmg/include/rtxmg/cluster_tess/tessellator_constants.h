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
#ifndef TESSELLATOR_CONSTANTS_H // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define TESSELLATOR_CONSTANTS_H

static const uint32_t kMaxApiClusterCount = 1 << 22;

// Max edge size per cluster/template
static const uint32_t kMaxClusterEdgeSegments = 11;

#ifndef __cplusplus
uint32_t GetTemplateIndex(uint16_t2 clusterSize)
{
    return (clusterSize.y - 1) * kMaxClusterEdgeSegments + (clusterSize.x - 1);
}

#endif

#endif // TESSELLATOR_CONSTANTS_H
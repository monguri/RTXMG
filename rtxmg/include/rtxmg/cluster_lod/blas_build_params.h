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

#include <nvrhi/nvrhiHLSL.h>

// Constant buffer for blas_reserve_clusters.hlsl and blas_insert_clusters.hlsl.
struct BlasBuildParams
{
    nvrhi::GpuVirtualAddress blasClasAddressesBaseVA;  // 8B  base VA of the CLAS-address pool
    uint32_t numRenderInstances;                        // 4B
    uint32_t patchCachedBlasCount;                      // 4B  BLAS caching: # cached-BLAS patches this frame (caching dispatches)
};

#ifdef __cplusplus
static_assert(sizeof(BlasBuildParams) == 16, "BlasBuildParams must be 16 bytes");
#endif


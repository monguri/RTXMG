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

#ifndef SCAN_SYSTEM_SHARED_H // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define SCAN_SYSTEM_SHARED_H

 /// Number of threads per block that are used for computing the horizontal prefix scans for a 2D buffer.
 /// To handle 8k textures this must be at least 128, otherwise a 3-level scan would be required.
#define PREFIX_SCAN_THREAD_BLOCK_SIZE 512

/// Number of image rows that each kernel invocation will handle during horizontal prefix scans.
#define PREFIX_SCAN_ROWS_PER_BLOCK 1

struct PrefixScanParams
{
    uint32_t elementCountX;
    uint32_t elementCountY;
    uint32_t outputWidth;
};

#endif // SCAN_SYSTEM_SHARED_H
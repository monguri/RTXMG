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

#ifndef HIZBUFFER_CONSTANTS_H // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define HIZBUFFER_CONSTANTS_H

// 9 LODs should support up to 8k monitor 
#define HIZ_MAX_LODS 9
#define HIZ_LOD0_TILE_SIZE 8u
#define HIZ_GROUP_SIZE 16

#endif // HIZBUFFER_CONSTANTS_H
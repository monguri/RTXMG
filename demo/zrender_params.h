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

#ifndef Z_RENDER_PARAMS_H // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define Z_RENDER_PARAMS_H

#ifdef __cplusplus
#include <donut/core/math/math.h>
using namespace donut::math;
#endif

struct ZRenderParams
{
    float3 eye;
    float pad;

    float3 U;
    int pad2;

    float3 V;
    int pad3;

    float3 W;
    float pad4;
};

struct ZRayPayload
{
    float hitT;
};


#endif // Z_RENDER_PARAMS_H

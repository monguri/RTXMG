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

#include "envmap/preprocess_envmap_params.h"
#include "rtxmg/utils/constants.h"

ConstantBuffer<PreprocessEnvMapParams> gPreprocessEnvMapParams : register(b0);

Texture2D<float4> gTextureColorIn : register(t0);

RWBuffer<float> gConditionalCdf : register(u1);
RWBuffer<float> gMarginalFunc : register(u2);

// clang-format off
[numthreads(1, 32, 1)]
[shader("compute")]
void main(uint2 dispatchThreadId : SV_DispatchThreadID)
// clang-format on
{
    if (dispatchThreadId.y >= gPreprocessEnvMapParams.envMapHeight)
    {
        return;
    }

    const float funcInt = gConditionalCdf[dispatchThreadId.y * (gPreprocessEnvMapParams.envMapWidth + 2) + gPreprocessEnvMapParams.envMapWidth + 1];
    gMarginalFunc[dispatchThreadId.y] = funcInt;
}


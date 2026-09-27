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

#include "envmap/scan_system_shared.h"

ConstantBuffer<PrefixScanParams> gPrefixScanParams : register(b0);

Buffer<float> input: register(t0);
RWBuffer<float> output: register(u0);

// clang-format off
[numthreads(1, 16, 1)]
[shader("compute")]
void main(uint2 dispatchThreadId : SV_DispatchThreadID)
// clang-format on
{
    int n = gPrefixScanParams.elementCountX;

    if (dispatchThreadId.y >= gPrefixScanParams.elementCountY || dispatchThreadId.x != 0)
    {
        return;
    }

    uint32_t outputOffset = dispatchThreadId.y * gPrefixScanParams.outputWidth;
    uint32_t inputOffset = dispatchThreadId.y * n;

    output[outputOffset + 0] = 0;
    float sum = 0;
    for (int i = 1; i <= n; ++i)
    {
        output[outputOffset + i] = output[outputOffset + i - 1] + input[inputOffset + i - 1] / n;
    }

    float funcInt = output[outputOffset + n];
    output[outputOffset + n + 1] = funcInt;
    if (funcInt == 0)
    {
        for (int i = 1; i <= n; ++i)
        {
            output[outputOffset + i] = float(i) / float(n);
        }
    }
    else
    {
        for (int i = 1; i <= n; ++i)
        {
            output[outputOffset + i] /= funcInt;
        }
    }
}

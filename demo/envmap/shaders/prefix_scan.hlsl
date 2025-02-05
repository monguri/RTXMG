/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
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

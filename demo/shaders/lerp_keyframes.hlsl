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

#include "lerp_keyframes_params.h"

StructuredBuffer<float3> kf0 : register(t0);
StructuredBuffer<float3> kf1 : register(t1);

RWStructuredBuffer<float3> dst : register(u0);

ConstantBuffer<LerpKeyFramesParams> g_lerpParams: register(b0);

[numthreads(32, 1, 1)]
void main(uint3 threadIdx : SV_DispatchThreadID)
{
    const uint32_t vertexIndex = threadIdx.x;
    if (vertexIndex >= g_lerpParams.numVertices)
        return;
    const float3 v0 = kf0[vertexIndex];
    const float3 v1 = kf1[vertexIndex];

    dst[vertexIndex] = lerp(v0, v1, g_lerpParams.animTime);
}

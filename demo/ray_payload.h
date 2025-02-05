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

#ifndef RAY_PAYLOAD_H // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define RAY_PAYLOAD_H

struct RayPayload
{
    uint instanceID;
    uint primitiveIndex;
    uint geometryIndex;
    float2 barycentrics;

    uint pathWeight; // RGBe9995 
    uint pathContribution; // RGBe9995
    uint bounce;
    uint multipurposeField; // can be re-used as the shuffled subpixel index, ray direction (PT)
    float3 rayOrigin; // for path tracing
    float pdf;
    uint seed;
    float hitT;
};

struct TestPayload
{
    int missed;
    uint instanceID;
    uint primitiveIndex;
    uint geometryIndex;
    float2 barycentrics;

    float3 rayDir;

    float3 color;
};

struct ShadowRayPayload
{
    int missed;
};

#endif // RAY_PAYLOAD_H
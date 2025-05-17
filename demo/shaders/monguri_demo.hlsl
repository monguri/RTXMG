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
#pragma pack_matrix(row_major)

#include "render_params.h"
#include <donut/shaders/bindless.h>

ConstantBuffer<RenderParams> g_RenderParams : register(b1);
ConstantBuffer<InstanceData> g_InstanceData : register(b1);

struct IAOutput
{
    float3 position : POSITION;
};

struct VSOutput
{
    float4 position : SV_Position;
};

VSOutput VS_WVP(IAOutput input)
{
    VSOutput output;
    float3 posWS = mul(g_InstanceData.transform, float4(input.position, 1.0f));
    float4 posCS = mul(g_RenderParams.viewProjectionMatrix, float4(posWS, 1.0f));
    output.position = posCS;
    return output;
}

float4 PS_Wireframe(VSOutput input)
{
    // ê^Ç¡çï
    return float4(0, 0, 0, 1);
}



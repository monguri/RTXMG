//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#ifndef GBUFFER_H  // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define GBUFFER_H

typedef float DepthFormat;
typedef float4 NormalFormat;
typedef float4 AlbedoFormat;
typedef float4 SpecularFormat;
typedef float SpecularHitTFormat;
typedef float RoughnessFormat;

static const uint32_t kInvalidInstanceId = ~0u;
static const uint32_t kInvalidSurfaceIndex = ~0u;

struct HitResult
{
    uint32_t instanceId;
    uint32_t surfaceIndex;
    float2 surfaceUV;
    float2 texcoord; // For displacement texture
};

#ifndef __cplusplus
HitResult DefaultHitResult()
{
    HitResult result;
    result.instanceId = kInvalidInstanceId;
    result.surfaceIndex = kInvalidSurfaceIndex;
    result.surfaceUV = float2(0.0f, 0.f);
    result.texcoord = float2(0.0f, 0.f);

    return result;
}
#endif

#endif // GBUFFER_H
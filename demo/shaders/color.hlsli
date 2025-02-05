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

#ifndef COLOR_HLSLI // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define COLOR_HLSLI 

inline
uint32_t ToRGBe9995(float3 Power)
{
    // All three numbers are packed into a shared exponent: e, which is the largest exponent + 1
    // Each element is assigned a signed mantissa in the range of [0,511]
    // x_i = 2^e * m_i / 512
    // Assumes no negatives
    const float MaxPower = max(Power.x, max(Power.y, Power.z));
    int UnbiasedExp2Int = (int)(asuint(MaxPower) >> 23) - 127 + 1;
    UnbiasedExp2Int = clamp(UnbiasedExp2Int, -16, 15);
    // Invert the exponent
    const float InverseExp2AndScale = abs(asfloat(((uint32_t)(127 + 9 - UnbiasedExp2Int)) << 23));
    const float3 pi = Power * InverseExp2AndScale;
    uint3  Fractional = uint3((uint32_t)pi.x, (uint32_t)pi.y, (uint32_t)pi.z);

    Fractional = min(Fractional, uint3(511, 511, 511));  // Prevents 512 which can occur due to round
    uint32_t Packed = ((uint32_t)UnbiasedExp2Int) << 27;
    Packed |= Fractional.z << 18;
    Packed |= Fractional.y << 9;
    Packed |= Fractional.x;
    return Packed;
}

inline
float3 FromRGBe9995(uint32_t i)
{
    // No negatives
    uint3 Fractional = 0;
    int PackedInt = (int)i;
    Fractional.x = PackedInt & 511;
    PackedInt >>= 9;
    Fractional.y = PackedInt & 511;
    PackedInt >>= 9;
    Fractional.z = PackedInt & 511;
    PackedInt >>= 9;
    const int UnbiasedExp2Int = PackedInt;    // After all the integer shifts, exponent is in 2s complement
    float SharedScaledPower = abs(asfloat((uint32_t)(UnbiasedExp2Int - 9 + 127) << 23));
    return SharedScaledPower * float3((float)Fractional.x, (float)Fractional.y, (float)Fractional.z);
}

#endif // COLOR_HLSLI 
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

#ifndef SHADER_DEBUG_H // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define SHADER_DEBUG_H

// Printf-style ring buffer for one predicated pixel / lane, dumped to the
// console by -dp and by the tessellator debug indices.  Purely diagnostic --
// the shipped right-click pick lives in pixel_pick.h -- so it can be compiled
// out with -D RTXMG_SHADER_DEBUG=OFF (CMake) without losing a feature.
//
// Measured on the DXIL path-tracer library (lib_6_6, all four permutations):
// =1 adds 169-182 instructions to ClosestHit and nothing anywhere else, and
// leaves every entry point peak live-value count identical, so it does not
// move the register budget the DXR pipeline takes from its worst entry point.
//
// =0 is diagnostic-only and NOT golden-gated: it drops the SHADER_DEBUG_INIT
// that motion_vectors.hlsl relies on to pin FP association, which moves the
// subdivision goldens (2.6 / 2.8).  See the comment at that call site.
#ifndef ENABLE_SHADER_DEBUG
#define ENABLE_SHADER_DEBUG 1
#endif

#ifdef __cplusplus
#include <ostream>
#else
#endif

 // Debug pixel
struct ShaderDebugElement
{
    enum PayloadType : uint
    {
        PayloadType_None,
        PayloadType_Uint,
        PayloadType_Uint2,
        PayloadType_Uint3,
        PayloadType_Uint4,
        PayloadType_Int,
        PayloadType_Int2,
        PayloadType_Int3,
        PayloadType_Int4,
        PayloadType_Float,
        PayloadType_Float2,
        PayloadType_Float3,
        PayloadType_Float4
    };

    uint4 uintData;
    float4 floatData;
    uint payloadType;
    uint lineNumber;
    uint2 pad0;


#ifdef __cplusplus
    static bool OutputLambda(std::ostream& ss, const ShaderDebugElement& e)
    {
        if (e.payloadType == ShaderDebugElement::PayloadType_None)
            return false;

        ss << "[Line:" << std::dec << e.lineNumber << "] ";

        if (e.payloadType >= ShaderDebugElement::PayloadType_Float &&
            e.payloadType <= ShaderDebugElement::PayloadType_Float4)
        {
            uint32_t numVectorElements = (e.payloadType - uint32_t(ShaderDebugElement::PayloadType_Float)) + 1;

            ss << std::setprecision(12) << e.floatData.data()[0];
            for (uint32_t i = 1; i < numVectorElements; i++)
                ss << ", " << e.floatData.data()[i];
        }
        else if (e.payloadType >= ShaderDebugElement::PayloadType_Int &&
            e.payloadType <= ShaderDebugElement::PayloadType_Int4)
        {
            uint32_t numVectorElements = (e.payloadType - uint32_t(ShaderDebugElement::PayloadType_Int)) + 1;
            ss << std::dec << static_cast<int>(e.uintData.data()[0]) << std::hex << "(0x" << e.uintData.data()[0] << ")";
            for (uint32_t i = 1; i < numVectorElements; i++)
                ss << ", " << std::dec << static_cast<int>(e.uintData.data()[i]) << std::hex << "(0x" << e.uintData.data()[i] << ")";
        }
        else if (e.payloadType >= ShaderDebugElement::PayloadType_Uint &&
            e.payloadType <= ShaderDebugElement::PayloadType_Uint4)
        {
            uint32_t numVectorElements = (e.payloadType - uint32_t(ShaderDebugElement::PayloadType_Uint)) + 1;
            ss << std::dec << e.uintData.data()[0] << std::hex << "(0x" << e.uintData.data()[0] << ")";
            for (uint32_t i = 1; i < numVectorElements; i++)
                ss << ", " << std::dec << e.uintData.data()[i] << std::hex << "(0x" << e.uintData.data()[i] << ")";
        }

        return true;
    }
#endif
};

#ifndef __cplusplus
#if ENABLE_SHADER_DEBUG

// The shader names its own buffer via SHADER_DEBUG_BUFFER and it is passed by
// parameter, never stored: in a raytracing library DXC leaves a stored
// RWStructuredBuffer as a Private pointer-to-StorageBuffer without declaring
// VariablePointersStorageBuffer, crashing the NV driver at pipeline creation.
static uint3 g_ShaderDebugPredicateID;
static uint3 g_ShaderDebugCurrentID;

uint ShaderDebugAllocateSlot(RWStructuredBuffer<ShaderDebugElement> output)
{
    uint bufferSize, bufferStride;
    output.GetDimensions(bufferSize, bufferStride);
    uint maxSize = bufferSize - 1;

    uint result;
    InterlockedAdd(output[0].payloadType, 1, result);
    return (result % maxSize) + 1;
}

void _ShaderDebug(RWStructuredBuffer<ShaderDebugElement> output, float4 value, uint lineNumber, uint payloadType)
{
    if (all(g_ShaderDebugPredicateID == g_ShaderDebugCurrentID))
    {
        ShaderDebugElement element = (ShaderDebugElement)0;
        element.payloadType = payloadType;
        element.lineNumber = lineNumber;
        element.floatData = value;
        element.uintData = 0;
        output[ShaderDebugAllocateSlot(output)] = element;
    }
}
void _ShaderDebug(RWStructuredBuffer<ShaderDebugElement> output, uint4 value, uint lineNumber, uint payloadType)
{
    if (all(g_ShaderDebugPredicateID == g_ShaderDebugCurrentID))
    {
        ShaderDebugElement element = (ShaderDebugElement)0;
        element.payloadType = payloadType;
        element.lineNumber = lineNumber;
        element.floatData = 0.f;
        element.uintData = value;
        output[ShaderDebugAllocateSlot(output)] = element;
    }
}

void ShaderDebug(RWStructuredBuffer<ShaderDebugElement> output, uint4 value, uint lineNumber)
{
    _ShaderDebug(output, value, lineNumber, ShaderDebugElement::PayloadType_Uint4);
}
void ShaderDebug(RWStructuredBuffer<ShaderDebugElement> output, uint3 value, uint lineNumber)
{
    _ShaderDebug(output, uint4(value, 0), lineNumber, ShaderDebugElement::PayloadType_Uint3);
}
void ShaderDebug(RWStructuredBuffer<ShaderDebugElement> output, uint2 value, uint lineNumber)
{
    _ShaderDebug(output, uint4(value, 0, 0), lineNumber, ShaderDebugElement::PayloadType_Uint2);
}
void ShaderDebug(RWStructuredBuffer<ShaderDebugElement> output, uint value, uint lineNumber)
{
    _ShaderDebug(output, uint4(value, 0, 0, 0), lineNumber, ShaderDebugElement::PayloadType_Uint);
}

void ShaderDebug(RWStructuredBuffer<ShaderDebugElement> output, int4 value, uint lineNumber)
{
    _ShaderDebug(output, value, lineNumber, ShaderDebugElement::PayloadType_Int4);
}
void ShaderDebug(RWStructuredBuffer<ShaderDebugElement> output, int3 value, uint lineNumber)
{
    _ShaderDebug(output, uint4(value, 0), lineNumber, ShaderDebugElement::PayloadType_Int3);
}
void ShaderDebug(RWStructuredBuffer<ShaderDebugElement> output, int2 value, uint lineNumber)
{
    _ShaderDebug(output, uint4(value, 0, 0), lineNumber, ShaderDebugElement::PayloadType_Int2);
}
void ShaderDebug(RWStructuredBuffer<ShaderDebugElement> output, int value, uint lineNumber)
{
    _ShaderDebug(output, uint4(value, 0, 0, 0), lineNumber, ShaderDebugElement::PayloadType_Int);
}

void ShaderDebug(RWStructuredBuffer<ShaderDebugElement> output, float4 value, uint lineNumber)
{
    _ShaderDebug(output, value, lineNumber, ShaderDebugElement::PayloadType_Float4);
}
void ShaderDebug(RWStructuredBuffer<ShaderDebugElement> output, float3 value, uint lineNumber)
{
    _ShaderDebug(output, float4(value, 0), lineNumber, ShaderDebugElement::PayloadType_Float3);
}
void ShaderDebug(RWStructuredBuffer<ShaderDebugElement> output, float2 value, uint lineNumber)
{
    _ShaderDebug(output, float4(value, 0, 0), lineNumber, ShaderDebugElement::PayloadType_Float2);
}
void ShaderDebug(RWStructuredBuffer<ShaderDebugElement> output, float value, uint lineNumber)
{
    _ShaderDebug(output, float4(value, 0, 0, 0), lineNumber, ShaderDebugElement::PayloadType_Float);
}

static void InitShaderDebugger(uint3 predicateID, uint3 currentID)
{
    g_ShaderDebugPredicateID = predicateID;
    g_ShaderDebugCurrentID = currentID;
}

static void InitShaderDebugger(uint2 predicateID, uint2 currentID)
{
    InitShaderDebugger(uint3(predicateID, 0), uint3(currentID, 0));
}

static void InitShaderDebugger(uint predicateID, uint currentID)
{
    InitShaderDebugger(uint3(predicateID, 0, 0), uint3(currentID, 0, 0));
}

#define SHADER_DEBUG(value) ShaderDebug(SHADER_DEBUG_BUFFER, value, __LINE__)
#define SHADER_DEBUG_INIT(predicateID, currentID) InitShaderDebugger(predicateID, currentID)

#else
#define SHADER_DEBUG(value)
#define SHADER_DEBUG_INIT(predicateID, currentID)
#endif

#endif // __cplusplus

#endif /* SHADER_DEBUG_H */
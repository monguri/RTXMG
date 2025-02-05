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

#ifndef PIXEL_DEBUG_H // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define PIXEL_DEBUG_H

#include "render_params.h"

#if ENABLE_PIXEL_DEBUG
 // Debug pixel
struct PixelDebugElement
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

    uint payloadType;
    uint lineNumber;

    uint4 uintData;
    float4 floatData;
};

#ifndef __cplusplus
struct PixelDebugger
{
    RWStructuredBuffer<PixelDebugElement> output;
    uint2 debugPixel;
    uint2 currentPixel;

    // don't have append support in donut yet, per-dispatch index
    uint outputSlot;

    void _PixelDebug(float4 value, uint lineNumber, uint payloadType)
    {
        if (all(debugPixel == currentPixel))
        {
            PixelDebugElement element;
            element.payloadType = payloadType;
            element.lineNumber = lineNumber;
            element.floatData = value;
            element.uintData = 0;
            output[outputSlot] = element;
            outputSlot++;
        }
    }
    void _PixelDebug(uint4 value, uint lineNumber, uint payloadType)
    {
        if (all(debugPixel == currentPixel))
        {
            PixelDebugElement element;
            element.payloadType = payloadType;
            element.lineNumber = lineNumber;
            element.floatData = 0.f;
            element.uintData = value;
            output[outputSlot] = element;
            outputSlot++;
        }
    }

    void PixelDebug(uint4 value, uint lineNumber)
    {
        _PixelDebug(value, lineNumber, PixelDebugElement::PayloadType_Uint4);
    }
    void PixelDebug(uint3 value, uint lineNumber)
    {
        _PixelDebug(uint4(value, 0), lineNumber, PixelDebugElement::PayloadType_Uint3);
    }
    void PixelDebug(uint2 value, uint lineNumber)
    {
        _PixelDebug(uint4(value, 0, 0), lineNumber, PixelDebugElement::PayloadType_Uint2);
    }
    void PixelDebug(uint value, uint lineNumber)
    {
        _PixelDebug(uint4(value, 0, 0, 0), lineNumber, PixelDebugElement::PayloadType_Uint);
    }

    void PixelDebug(int4 value, uint lineNumber)
    {
        _PixelDebug(value, lineNumber, PixelDebugElement::PayloadType_Int4);
    }
    void PixelDebug(int3 value, uint lineNumber)
    {
        _PixelDebug(uint4(value, 0), lineNumber, PixelDebugElement::PayloadType_Int3);
    }
    void PixelDebug(int2 value, uint lineNumber)
    {
        _PixelDebug(uint4(value, 0, 0), lineNumber, PixelDebugElement::PayloadType_Int2);
    }
    void PixelDebug(int value, uint lineNumber)
    {
        _PixelDebug(uint4(value, 0, 0, 0), lineNumber, PixelDebugElement::PayloadType_Int);
    }

    void PixelDebug(float4 value, uint lineNumber)
    {
        _PixelDebug(value, lineNumber, PixelDebugElement::PayloadType_Float4);
    }
    void PixelDebug(float3 value, uint lineNumber)
    {
        _PixelDebug(float4(value, 0), lineNumber, PixelDebugElement::PayloadType_Float3);
    }
    void PixelDebug(float2 value, uint lineNumber)
    {
        _PixelDebug(float4(value, 0, 0), lineNumber, PixelDebugElement::PayloadType_Float2);
    }
    void PixelDebug(float value, uint lineNumber)
    {
        _PixelDebug(float4(value, 0, 0, 0), lineNumber, PixelDebugElement::PayloadType_Float);
    }

    void Clear()
    {
        // Clear debug, assumes that m_size of debug buffer is smaller than x resolution
        uint outputSize;
        uint outputStride;
        output.GetDimensions(outputSize, outputStride);
        if (currentPixel.y == 0 && currentPixel.x < outputSize)
        {
            output[currentPixel.x].payloadType = PixelDebugElement::PayloadType_None;
        }
    }
};

static PixelDebugger g_PixelDebugger;

static void InitPixelDebugger(RWStructuredBuffer<PixelDebugElement> output, uint2 debugPixel, uint2 currentPixel, bool clear)
{
    g_PixelDebugger.output = output;
    g_PixelDebugger.debugPixel = debugPixel;
    g_PixelDebugger.currentPixel = currentPixel;
    g_PixelDebugger.outputSlot = 0;

    if (clear)
        g_PixelDebugger.Clear();
}

#define PIXEL_DEBUG(value) g_PixelDebugger.PixelDebug(value, __LINE__)
#define PIXEL_DEBUG_INIT(outputBuffer, debugPixel, currentPixel, clear) InitPixelDebugger(outputBuffer, debugPixel, currentPixel, clear)
#endif

#else
#define PIXEL_DEBUG(value) 
#define PIXEL_DEBUG_INIT(outputBuffer, debugPixel, currentPixel, clear)
#endif

#endif /* PIXEL_DEBUG_H */
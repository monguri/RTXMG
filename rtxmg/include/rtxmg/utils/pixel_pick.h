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

#ifndef PIXEL_PICK_H // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define PIXEL_PICK_H

// Right-click viewport pick: reports the instance / geometry / material under
// one pixel to the material readout and the Inspector's select-mesh-under-
// cursor.  It is a shipped feature, so it is deliberately NOT part of
// ENABLE_SHADER_DEBUG (shader_debug.h) - that ring is a diagnostic and can be
// compiled out on its own.
#ifndef ENABLE_PIXEL_PICK
#define ENABLE_PIXEL_PICK 1
#endif

struct PixelPickResult
{
    // 0 = nothing hit the picked pixel this frame.  Also the claim word: the
    // first hit along the path wins, so a bounce cannot overwrite the primary.
    uint32_t claimed;

    uint32_t instanceID;
    uint32_t surfaceID;   // subd: surfaceID; cluster-LOD: geometryID
    uint32_t materialID;
    // bit 0 clear: subd hit.  bit 0 set: cluster-LOD hit, bits 8+ = lodLevel.
    uint32_t tag;
};

#ifndef __cplusplus
#if ENABLE_PIXEL_PICK

// The shader names its own buffer via PIXEL_PICK_BUFFER and it is passed by
// parameter, never stored: in a raytracing library DXC leaves a stored
// RWStructuredBuffer as a Private pointer-to-StorageBuffer without declaring
// VariablePointersStorageBuffer, crashing the NV driver at pipeline creation.
static uint2 g_PixelPickTarget;
static uint2 g_PixelPickCurrent;

static void InitPixelPick(uint2 target, uint2 current)
{
    g_PixelPickTarget = target;
    g_PixelPickCurrent = current;
}

void PixelPickWrite(RWStructuredBuffer<PixelPickResult> output, uint instanceID, uint surfaceID, uint materialID, uint tag)
{
    if (all(g_PixelPickTarget == g_PixelPickCurrent))
    {
        uint previouslyClaimed;
        InterlockedAdd(output[0].claimed, 1u, previouslyClaimed);
        if (previouslyClaimed == 0u)
        {
            output[0].instanceID = instanceID;
            output[0].surfaceID = surfaceID;
            output[0].materialID = materialID;
            output[0].tag = tag;
        }
    }
}

#define PIXEL_PICK_INIT(target, current) InitPixelPick(target, current)
#define PIXEL_PICK_SUBD(instanceID, surfaceID, materialID) PixelPickWrite(PIXEL_PICK_BUFFER, instanceID, surfaceID, materialID, 0u)
#define PIXEL_PICK_CLUSTER_LOD(instanceID, geometryID, materialID, lodLevel) PixelPickWrite(PIXEL_PICK_BUFFER, instanceID, geometryID, materialID, 1u | (uint(lodLevel) << 8))

#else
#define PIXEL_PICK_INIT(target, current)
#define PIXEL_PICK_SUBD(instanceID, surfaceID, materialID)
#define PIXEL_PICK_CLUSTER_LOD(instanceID, geometryID, materialID, lodLevel)
#endif

#endif // __cplusplus

#endif /* PIXEL_PICK_H */

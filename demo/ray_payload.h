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

    // Alpha-blend surfaces (gltf alphaMode BLEND): ClusterLodAnyHit accumulates
    // emissive weighted by alpha and passes the ray through.  Occlusion needs
    // two passes: pass 1 (opaque) only flags FOUND, pass 2 re-traces bounded by
    // the opaque depth and ACCUMULATEs.  A sum and a product, so the result does
    // not depend on the unspecified anyhit invocation order.
    //
    // DEVIATION from gltf alphaMode BLEND: alpha occludes the opaque background
    // but is not applied between blend surfaces, which sum additively — an
    // anyhit gather has no depth order, so the alternative is an arbitrary one.
    // A high-alpha / low-emissive card therefore fails to dim emissive cards
    // behind it; that only shows on dark translucent BLEND geometry.
    uint  blendEmissive;         // RGBe9995 — additive emissive gathered along the ray (pass 2)
    float backgroundVisibility;  // prod(1-alpha); 1 = nothing crossed. Attenuates the opaque hit behind
    uint  blendState;            // bit0 ACCUMULATE (in, pass 2), bit1 FOUND (out, pass 1)
};

// RayPayload::blendState bits
#define RTXMG_BLEND_ACCUMULATE 1u // pass 2: accumulate emissive + pass through
#define RTXMG_BLEND_FOUND      2u // pass 1 output: an alphaMode BLEND surface was crossed

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
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

// group_codec.h — the cluster-group vertex codec: the arithmetic bit-packer
// CompressGroup writes and DecompressGroup reads back, plus the po2-grid
// TEXCOORD quantization both baked stores share.  Encoder and decoder sit in
// one translation unit because the bit layout is a contract between them —
// header word order, per-dimension shift/precision packing and per-cluster
// block ordering all have to agree exactly.

#pragma once

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>

#include "rtxmg/cluster_lod/baked_geometry.h"

// Po2-grid TEXCOORD quantization (ClusterEncoding): per axis, base = the
// cluster's min and step = 2^e, decoded as base + delta * step.  The po2 step
// makes that product exact, so the fp32 decode is bit-identical on host and
// GPU.  `deltas` receives 2 * vertexCount components, (dx, dy) per vertex.
// Returns false for clusters too small to amortize the 16-byte header, or whose
// UV range needs a step coarser than 2^-14 (>~0.13 texel at 4K); those stay raw
// float2.  Both the plain and the compressed store call this, so the two agree
// on which clusters carry ClusterEncoding::QuantizedTexN.
template <typename UvAt>
bool QuantizeClusterTexCoords(uint32_t vertexCount, UvAt&& uvAt,
                              float2& lo, float2& step, uint32_t* deltas)
{
    if (vertexCount <= 4)
        return false;

    lo        = {  FLT_MAX,  FLT_MAX };
    float2 hi = { -FLT_MAX, -FLT_MAX };
    for (uint32_t v = 0; v < vertexCount; v++)
    {
        const float2 uv = uvAt(v);
        lo = min(lo, uv);
        hi = max(hi, uv);
    }

    // Exact po2 walk rather than log2, to dodge rounding edge cases.
    auto stepExponent = [](float range) {
        int e = -24;  // finer is pointless (sub-ulp for |uv| ~ 1)
        while (range > 65535.f * std::ldexp(1.f, e))
            e++;
        return e;
    };
    const int ex = stepExponent(hi.x - lo.x);
    const int ey = stepExponent(hi.y - lo.y);
    if (ex > -14 || ey > -14)
        return false;

    step = { std::ldexp(1.f, ex), std::ldexp(1.f, ey) };
    for (uint32_t v = 0; v < vertexCount; v++)
    {
        const float2 uv = uvAt(v);
        const float  du = std::round((uv.x - lo.x) / step.x);
        const float  dv = std::round((uv.y - lo.y) / step.y);
        deltas[v * 2 + 0] = uint32_t(std::min(65535.f, std::max(0.f, du)));
        deltas[v * 2 + 1] = uint32_t(std::min(65535.f, std::max(0.f, dv)));
    }
    return true;
}

// The 16-byte header a quantized TEXCOORD block starts with, in both stores.
inline void StoreQuantizedTexHeader(float* dst, const float2& lo, const float2& step)
{
    dst[0] = lo.x;
    dst[1] = lo.y;
    dst[2] = step.x;
    dst[3] = step.y;
}

// ---------------------------------------------------------------------------
// CompressGroup — rewrite the vertex region of a group blob in the compressed
// layout, arithmetic-packing positions and texcoords (normals stay octahedral)
// and recording the uncompressed sizes on the GroupInfo.
// ---------------------------------------------------------------------------
void CompressGroup(GroupStorage&          dstStorage,
                   GroupInfo&             groupInfo,
                   const GeometryStorage& geometry,
                   const uint32_t*        vertexCacheLocal);

// ---------------------------------------------------------------------------
// DecompressGroup — expand a compressed group blob (GroupInfo with
// uncompressedSizeBytes != 0) into its uncompressed device layout.
//
// The uncompressed section (group header + clusters + generating-groups +
// bboxes + triangle data) is copied verbatim; per-cluster POSITIONS and
// TEXCOORDS are unpacked per the ClusterAttribute compression bits while
// normals are copied.  `dst` must be info.GetDeviceSize() bytes.  Assumes a
// compressed input — callers memcpy uncompressed groups themselves.
// ---------------------------------------------------------------------------
void DecompressGroup(const GroupInfo& info, const GroupView& src, void* dst, size_t dstSize);

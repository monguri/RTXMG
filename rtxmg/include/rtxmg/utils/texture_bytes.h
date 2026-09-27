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

#pragma once

#include <nvrhi/nvrhi.h>

#include <algorithm>
#include <cstdint>

namespace rtxmg {

// GPU footprint of a texture: the block-packed size of every mip of every array
// slice.  Derived from the nvrhi desc because a finalized texture has released
// its CPU-side blob and dataLayout.
inline uint64_t TextureGpuBytes(const nvrhi::TextureDesc& d)
{
    const nvrhi::FormatInfo& fi = nvrhi::getFormatInfo(d.format);
    const uint32_t blockSize = std::max<uint32_t>(1, fi.blockSize);
    uint64_t bytes = 0;
    for (uint32_t mip = 0; mip < std::max(1u, d.mipLevels); ++mip)
    {
        const uint64_t w  = std::max(1u, d.width  >> mip);
        const uint64_t h  = std::max(1u, d.height >> mip);
        const uint64_t dz = std::max(1u, d.depth  >> mip);
        bytes += ((w + blockSize - 1) / blockSize) * ((h + blockSize - 1) / blockSize)
               * fi.bytesPerBlock * dz;
    }
    return bytes * std::max(1u, d.arraySize);
}

inline uint64_t TextureGpuBytes(const nvrhi::ITexture* tex)
{
    return tex ? TextureGpuBytes(tex->getDesc()) : 0;
}

}  // namespace rtxmg

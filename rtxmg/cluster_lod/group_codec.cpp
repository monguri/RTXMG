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

#include "rtxmg/cluster_lod/group_codec.h"

#include <bit>
#include <cassert>
#include <cstring>
#include <limits>

#include "rtxmg/cluster_lod/baking/attribute_encoding.h"  // NormalPack / TangentPack

using namespace donut::math;

namespace
{
// Bit-level writer over a uint32_t array (mirror of InputBitStream below).
class OutputBitStream
{
public:
    void Init(size_t byteSize, uint32_t* data)
    {
        assert(byteSize % sizeof(uint32_t) == 0);
        m_data     = data;
        m_bitsSize = byteSize * 8;
        m_bitsPos  = 0;
    }

    void Write(uint32_t val, uint32_t bitCount)
    {
        assert(bitCount <= 32);
        assert(m_bitsPos + bitCount <= m_bitsSize);
        val &= (bitCount == 32) ? ~0u : ((1u << bitCount) - 1);

        size_t   idxLo = m_bitsPos / 32;
        size_t   idxHi = (m_bitsPos + bitCount - 1) / 32;
        uint32_t shift = uint32_t(m_bitsPos % 32);

        if (shift == 0)
            m_data[idxLo] = val;
        else
            m_data[idxLo] |= val << shift;

        if (shift + bitCount > 32)
            m_data[idxHi] = val >> (32 - shift);

        m_bitsPos += bitCount;
    }

    template <typename T>
    void Write(const T& tValue)
    {
        static_assert(sizeof(T) <= sizeof(uint32_t));
        union { uint32_t u32; T t; };
        u32 = 0;
        t   = tValue;
        Write(u32, sizeof(T) * 8);
    }

private:
    uint32_t* m_data     = nullptr;
    size_t    m_bitsSize = 0;
    size_t    m_bitsPos  = 0;
};

// Per-attribute arithmetic compressor: drops the low zero bits common to all
// deltas-from-min (lossless given the registered values), DIM components.
template <class T, uint32_t DIM>
class ArithmeticCompressor
{
public:
    ArithmeticCompressor()
    {
        for (uint32_t d = 0; d < DIM; d++)
        {
            m_lo[d]    = std::numeric_limits<T>::max();
            m_hi[d]    = std::numeric_limits<T>::min();
            m_masks[d] = 0;
        }
    }

    template <typename Tindices>
    void RegisterVertices(size_t count, const Tindices* indices, size_t vecSize, const T* vecBuffer, size_t stride)
    {
        m_count = count;
        for (size_t i = 0; i < count; i++)
        {
            size_t index = indices[i];
            assert(index < vecSize);
            (void)vecSize;
            const T* vec = &vecBuffer[index * stride];
            for (uint32_t d = 0; d < DIM; d++)
            {
                m_lo[d] = std::min(m_lo[d], vec[d]);
                m_hi[d] = std::max(m_hi[d], vec[d]);
            }
        }
        for (size_t i = 0; i < count; i++)
        {
            const T* vec = &vecBuffer[indices[i] * stride];
            for (uint32_t d = 0; d < DIM; d++)
                m_masks[d] |= (vec[d] - m_lo[d]);
        }
        ComputeVertexSize();
    }

    size_t GetOutputByteSize() const
    {
        size_t numDeltaBits = 0;
        for (uint32_t d = 0; d < DIM; d++)
            numDeltaBits += m_precisions[d];
        numDeltaBits *= m_count;
        // 16 (shifts) + 16 (precisions) + 32*DIM (base values) + delta bits.
        return sizeof(uint32_t) * ((16 + 16 + 32 * DIM + numDeltaBits + 31) / 32);
    }

    void BeginOutput(size_t byteSize, uint32_t* out)
    {
        assert(byteSize <= GetOutputByteSize());
        m_out.Init(byteSize, out);

        uint16_t outShifts = uint16_t(m_shifts[0]);
        uint16_t outPrec   = uint16_t(m_precisions[0] - 1);
        for (uint32_t d = 1; d < DIM; d++)
        {
            outShifts |= uint16_t(m_shifts[d] << (d * 5));
            outPrec   |= uint16_t((m_precisions[d] - 1) << (d * 5));
        }
        m_out.Write(outShifts);
        m_out.Write(outPrec);
        for (uint32_t d = 0; d < DIM; d++)
            m_out.Write(m_lo[d]);
    }

    template <typename Tindices>
    void OutputVertices(size_t count, const Tindices* indices, size_t vecSize, const T* vecBuffer, size_t stride)
    {
        (void)vecSize;
        for (size_t i = 0; i < count; i++)
        {
            const T* vec = &vecBuffer[indices[i] * stride];
            for (uint32_t d = 0; d < DIM; d++)
                m_out.Write((vec[d] - m_lo[d]) >> m_shifts[d], m_precisions[d]);
        }
    }

private:
    void ComputeVertexSize()
    {
        for (uint32_t d = 0; d < DIM; ++d)
        {
            if (m_masks[d] == 0)
            {
                m_shifts[d]     = 31;
                m_precisions[d] = 1;
            }
            else
            {
                m_shifts[d]            = std::countr_zero(m_masks[d]);
                const uint32_t range   = m_hi[d] - m_lo[d];
                m_precisions[d]        = std::max(int(std::bit_width(range >> m_shifts[d])), 1);
            }
        }
    }

    T              m_lo[DIM];
    T              m_hi[DIM];
    T              m_masks[DIM];
    size_t         m_count           = 0;
    int            m_shifts[DIM]     = {};
    int            m_precisions[DIM] = {};
    OutputBitStream m_out;
};

// Bit-level reader over a uint32_t array (matches the encoder's word layout).
class InputBitStream
{
public:
    InputBitStream() = default;

    void Init(size_t byteSize, const uint32_t* data)
    {
        assert(byteSize % sizeof(uint32_t) == 0);
        m_data     = data;
        m_bitsPos  = 0;
        m_bitsSize = byteSize * 8;
    }

    void Read(uint32_t* value, uint32_t bitCount)
    {
        assert(bitCount <= 32);
        assert(m_bitsPos + bitCount <= m_bitsSize);

        size_t   idxLo = m_bitsPos / 32;
        size_t   idxHi = (m_bitsPos + bitCount - 1) / 32;
        uint32_t shift = uint32_t(m_bitsPos % 32);

        union
        {
            uint64_t u64;
            uint32_t u32[2];
        };

        u32[0] = m_data[idxLo];
        u32[1] = m_data[idxHi];

        value[0] = uint32_t(u64 >> shift);
        value[0] &= bitCount == 32 ? ~0u : ((1u << bitCount) - 1);

        m_bitsPos += bitCount;
    }

    template <typename T>
    void Read(T& value)
    {
        static_assert(sizeof(T) <= sizeof(uint32_t));
        union
        {
            uint32_t u32;
            T        tValue;
        };
        Read(&u32, sizeof(T) * 8);
        value = tValue;
    }

    size_t GetBytesRead() const { return sizeof(uint32_t) * ((m_bitsPos + 31) / 32); }

private:
    const uint32_t* m_data     = nullptr;
    size_t          m_bitsSize = 0;
    size_t          m_bitsPos  = 0;
};

// Per-attribute arithmetic decompressor: header carries per-dimension shift +
// precision and a base value; each component is base + (deltaBits << shift).
template <class T, uint32_t DIM>
class ArithmeticDeCompressor
{
public:
    void Init(size_t byteSize, const uint32_t* data)
    {
        m_input.Init(byteSize, data);

        uint16_t outShifts;
        uint16_t outPrecs;
        m_input.Read(outShifts);
        m_input.Read(outPrecs);

        for (uint32_t d = 0; d < DIM; d++)
        {
            m_shifts[d]     = (outShifts >> (d * 5)) & 31;
            m_precisions[d] = ((outPrecs >> (d * 5)) & 31) + 1;
            m_input.Read(m_lo[d]);
        }
    }

    size_t ReadVertices(size_t count, T* output, size_t strideInElements)
    {
        for (size_t v = 0; v < count; v++)
        {
            T* vec = output + v * strideInElements;
            for (uint32_t d = 0; d < DIM; d++)
            {
                uint32_t deltaBits = 0;
                m_input.Read(&deltaBits, m_precisions[d]);
                vec[d] = m_lo[d] + (deltaBits << m_shifts[d]);
            }
        }
        return m_input.GetBytesRead();
    }

private:
    T              m_lo[DIM]         = {};
    int            m_shifts[DIM]     = {};
    int            m_precisions[DIM] = {};
    InputBitStream m_input;
};
}  // namespace

// Rewrites the vertex region StoreGroup's pass 1 just wrote with arithmetic-packed
// positions + texcoords (normals stay octahedral), and records the uncompressed
// sizes so GetDeviceSize() reports the expanded size.
void CompressGroup(GroupStorage&          dstStorage,
                   GroupInfo&             groupInfo,
                   const GeometryStorage& geometry,
                   const uint32_t*        vertexCacheLocal)
{
    const size_t attributeStride = geometry.vertexPositions.empty()
                                       ? 0u
                                       : geometry.vertexAttributes.size() / geometry.vertexPositions.size();

    // The compressor's vertex API is index-driven; the quantized delta words it
    // packs are already cluster-local, so they index through this identity.
    uint32_t identity[kMaxClusterVertices];
    for (uint32_t i = 0; i < kMaxClusterVertices; i++)
        identity[i] = i;

    uint32_t vertexOffset     = 0;
    uint32_t vertexDataOffset = 0;
    for (uint32_t c = 0; c < groupInfo.clusterCount; c++)
    {
        const uint32_t*    localVertices = vertexCacheLocal + vertexOffset;
        shaderio::Cluster& cluster       = dstStorage.clusters[c];
        const uint32_t     vertexCount   = cluster.vertexCountMinusOne + 1;

        // Hijack the triangle-index offset to store the compressed vertex-data
        // offset; DecompressGroup re-derives the real triangle offset.
        cluster.triangles = vertexDataOffset;

        // --- positions (vec3) ---
        {
            ArithmeticCompressor<uint32_t, 3> compressor;
            compressor.RegisterVertices(vertexCount, localVertices, geometry.vertexPositions.size(),
                                        reinterpret_cast<const uint32_t*>(geometry.vertexPositions.data()), 3);
            const size_t compressedSize = compressor.GetOutputByteSize();
            if (compressedSize >= sizeof(float) * 3 * vertexCount)
            {
                for (uint32_t v = 0; v < vertexCount; v++)
                    *reinterpret_cast<float3*>(&dstStorage.vertices[vertexDataOffset + v * 3]) =
                        geometry.vertexPositions[localVertices[v]];
                vertexDataOffset += 3 * vertexCount;
            }
            else
            {
                cluster.attributeBits |= shaderio::ClusterAttribute::CompressedVertexPos;
                compressor.BeginOutput(compressedSize,
                                       reinterpret_cast<uint32_t*>(&dstStorage.vertices[vertexDataOffset]));
                compressor.OutputVertices(vertexCount, localVertices, geometry.vertexPositions.size(),
                                          reinterpret_cast<const uint32_t*>(geometry.vertexPositions.data()), 3);
                vertexDataOffset += uint32_t(compressedSize / sizeof(uint32_t));
            }
        }

        // --- normals / tangents (octahedral-packed, not arithmetic-compressed) ---
        if (geometry.attributeBits & shaderio::ClusterAttribute::VertexNormal)
        {
            const bool hasTangent = (geometry.attributeBits & shaderio::ClusterAttribute::VertexTangent) != 0;
            for (uint32_t v = 0; v < vertexCount; v++)
            {
                const float* na = &geometry.vertexAttributes[localVertices[v] * attributeStride + geometry.attributeNormalOffset];
                float3 normal   = *reinterpret_cast<const float3*>(na);
                uint32_t encoded = shaderio::NormalPack(normal);
                if (hasTangent)
                {
                    const float* ta = &geometry.vertexAttributes[localVertices[v] * attributeStride + geometry.attributeTangentOffset];
                    float4 tangent  = *reinterpret_cast<const float4*>(ta);
                    encoded |= shaderio::TangentPack(normal, tangent) << ATTRENC_NORMAL_BITS;
                }
                *reinterpret_cast<uint32_t*>(&dstStorage.vertices[vertexDataOffset + v]) = encoded;
            }
            vertexDataOffset += vertexCount;
        }

        // --- texcoords (vec2) — TEX_0 then TEX_1, each independently ---
        for (uint32_t t = 0; t < 2; t++)
        {
            const uint32_t usedBit       = (t == 0) ? shaderio::ClusterAttribute::VertexTex0 : shaderio::ClusterAttribute::VertexTex1;
            const uint32_t compressedBit = (t == 0) ? shaderio::ClusterAttribute::CompressedVertexTex0
                                                    : shaderio::ClusterAttribute::CompressedVertexTex1;
            const uint32_t quantBit      = (t == 0) ? shaderio::ClusterEncoding::QuantizedTex0
                                                    : shaderio::ClusterEncoding::QuantizedTex1;
            const uint32_t texOff        = (t == 0) ? geometry.attributeTex0offset : geometry.attributeTex1offset;
            if (!(geometry.attributeBits & usedBit))
                continue;

            auto uvAt = [&](uint32_t v) {
                const float* ua = &geometry.vertexAttributes[localVertices[v] * attributeStride + texOff];
                return *reinterpret_cast<const float2*>(ua);
            };

            // Pass 1 already decided this channel's encoding; reproducing the
            // same quantization here keeps cluster.encodingBits honest, and the
            // 16-bit delta words pack better than raw float2s (no exponent).
            if (cluster.encodingBits & quantBit)
            {
                float2   lo{}, step{};
                uint32_t texDeltas[kMaxClusterVertices * 2];
                const bool quantized = QuantizeClusterTexCoords(vertexCount, uvAt, lo, step, texDeltas);
                assert(quantized && "encodingBits disagrees with QuantizeClusterTexCoords");
                (void)quantized;

                StoreQuantizedTexHeader(&dstStorage.vertices[vertexDataOffset], lo, step);
                vertexDataOffset += 4;

                ArithmeticCompressor<uint32_t, 2> compressor;
                compressor.RegisterVertices(vertexCount, identity, vertexCount, texDeltas, 2);
                const size_t compressedSize = compressor.GetOutputByteSize();
                if (compressedSize >= sizeof(uint32_t) * vertexCount)
                {
                    uint32_t* deltas =
                        reinterpret_cast<uint32_t*>(&dstStorage.vertices[vertexDataOffset]);
                    for (uint32_t v = 0; v < vertexCount; v++)
                        deltas[v] = texDeltas[v * 2] | (texDeltas[v * 2 + 1] << 16);
                    vertexDataOffset += vertexCount;
                }
                else
                {
                    cluster.attributeBits |= compressedBit;
                    compressor.BeginOutput(compressedSize,
                                           reinterpret_cast<uint32_t*>(&dstStorage.vertices[vertexDataOffset]));
                    compressor.OutputVertices(vertexCount, identity, vertexCount, texDeltas, 2);
                    vertexDataOffset += uint32_t(compressedSize / sizeof(uint32_t));
                }
                continue;
            }

            const uint32_t* texBase = reinterpret_cast<const uint32_t*>(geometry.vertexAttributes.data() + texOff);
            ArithmeticCompressor<uint32_t, 2> compressor;
            compressor.RegisterVertices(vertexCount, localVertices, geometry.vertexPositions.size(), texBase, attributeStride);
            const size_t compressedSize = compressor.GetOutputByteSize();
            if (compressedSize >= sizeof(float) * 2 * vertexCount)
            {
                for (uint32_t v = 0; v < vertexCount; v++)
                    *reinterpret_cast<float2*>(&dstStorage.vertices[vertexDataOffset + v * 2]) =
                        *reinterpret_cast<const float2*>(&geometry.vertexAttributes[localVertices[v] * attributeStride + texOff]);
                vertexDataOffset += 2 * vertexCount;
            }
            else
            {
                cluster.attributeBits |= compressedBit;
                compressor.BeginOutput(compressedSize,
                                       reinterpret_cast<uint32_t*>(&dstStorage.vertices[vertexDataOffset]));
                compressor.OutputVertices(vertexCount, localVertices, geometry.vertexPositions.size(), texBase, attributeStride);
                vertexDataOffset += uint32_t(compressedSize / sizeof(uint32_t));
            }
        }

        vertexOffset += vertexCount;
    }

    groupInfo.uncompressedSizeBytes       = groupInfo.sizeBytes;
    groupInfo.uncompressedVertexDataCount = groupInfo.vertexDataCount;
    groupInfo.vertexDataCount             = vertexDataOffset;
    groupInfo.sizeBytes                   = groupInfo.ComputeSize();
}

void DecompressGroup(const GroupInfo& info, const GroupView& groupSrc, void* dstWriteOnly, size_t dstSize)
{
    // The destination is sized + laid out for the uncompressed state.
    GroupInfo uncompressedInfo       = info;
    uncompressedInfo.sizeBytes       = info.uncompressedSizeBytes;
    uncompressedInfo.vertexDataCount = info.uncompressedVertexDataCount;

    GroupStorage groupDstWriteOnly(dstWriteOnly, uncompressedInfo);

    // Copy the uncompressed section verbatim (group header + clusters +
    // generating groups + bboxes + triangle indices); only vertex data differs.
    std::memcpy(dstWriteOnly, groupSrc.raw, info.ComputeUncompressedSectionSize());

    uint32_t trianglesDataOffset = 0;
    for (uint32_t c = 0; c < info.clusterCount; c++)
    {
        shaderio::Cluster&       clusterDstWriteOnly = groupDstWriteOnly.clusters[c];
        const shaderio::Cluster& clusterSrc          = groupSrc.clusters[c];
        const uint32_t           triangleCount       = clusterSrc.triangleCountMinusOne + 1;
        const uint32_t           vertexCount         = clusterSrc.vertexCountMinusOne + 1;

        // Destination vertex-data region for this cluster (uncompressed layout).
        uint32_t* dstData = groupDstWriteOnly.GetClusterLocalData(c, clusterSrc.vertices);
        // In the compressed blob the cluster's `triangles` offset points at the
        // compressed vertex stream (GetClusterIndices resolves it).
        const uint32_t* srcData = reinterpret_cast<const uint32_t*>(groupSrc.GetClusterIndices(c));

        // Re-point the cluster's triangle-index offset at the real index data in
        // the uncompressed layout.
        clusterDstWriteOnly.triangles =
            groupDstWriteOnly.GetClusterLocalOffset(c, groupDstWriteOnly.indices.data() + trianglesDataOffset);
        trianglesDataOffset += triangleCount * (clusterSrc.localMaterialID == shaderio::kPerTriangleMaterials ? 4u : 3u);

        uint32_t dstOffset = 0;

        // The baker and ClusterLodTex0ByteBase align the UV block on its
        // blob-relative offset, and a cluster's vertex slice is only 8-byte
        // aligned when every preceding slice happened to be — quantized UVs
        // make odd-vertex-count clusters 4 bytes long.  So align in the same
        // frame they do, not relative to this cluster.
        const size_t sliceByteBase = size_t(reinterpret_cast<const uint8_t*>(dstData)
                                            - static_cast<const uint8_t*>(dstWriteOnly));
        auto alignUvBlock = [&](uint32_t off) {
            return uint32_t((align_up(sliceByteBase + size_t(off) * 4u, 8u) - sliceByteBase) / 4u);
        };

        // positions (vec3)
        if (clusterSrc.attributeBits & shaderio::ClusterAttribute::CompressedVertexPos)
        {
            ptrdiff_t srcSize = ptrdiff_t(groupSrc.vertices.data() + groupSrc.vertices.size()) - ptrdiff_t(srcData);
            assert(srcSize >= 0);

            ArithmeticDeCompressor<uint32_t, 3> decompressor;
            decompressor.Init(size_t(srcSize), srcData);
            srcData += decompressor.ReadVertices(vertexCount, dstData + dstOffset, 3) / sizeof(uint32_t);
            dstOffset += 3 * vertexCount;
        }
        else
        {
            std::memcpy(dstData, srcData, 3u * sizeof(float) * vertexCount);
            srcData += 3 * vertexCount;
            dstOffset += 3 * vertexCount;
        }

        // normals (never compressed; one packed uint32 per vertex)
        if (clusterSrc.attributeBits & shaderio::ClusterAttribute::VertexNormal)
        {
            std::memcpy(dstData + dstOffset, srcData, sizeof(uint32_t) * vertexCount);
            srcData += vertexCount;
            dstOffset += vertexCount;
        }

        // texcoords (vec2) — TEX_0 then TEX_1, each independently compressed
        for (uint32_t t = 0; t < 2; t++)
        {
            const uint32_t usedBit       = t == 0 ? shaderio::ClusterAttribute::VertexTex0 : shaderio::ClusterAttribute::VertexTex1;
            const uint32_t compressedBit = t == 0 ? shaderio::ClusterAttribute::CompressedVertexTex0
                                                   : shaderio::ClusterAttribute::CompressedVertexTex1;
            const uint32_t quantBit      = t == 0 ? shaderio::ClusterEncoding::QuantizedTex0
                                                   : shaderio::ClusterEncoding::QuantizedTex1;

            // Po2-quantized channel: a verbatim 16-byte (base, step) header,
            // then one 16|16 delta word per vertex, packed or raw.
            if ((clusterSrc.attributeBits & usedBit) && (clusterSrc.encodingBits & quantBit))
            {
                dstOffset = alignUvBlock(dstOffset);
                std::memcpy(dstData + dstOffset, srcData, 4u * sizeof(float));
                srcData += 4;

                uint32_t* deltas = dstData + dstOffset + 4;
                if (clusterSrc.attributeBits & compressedBit)
                {
                    ptrdiff_t srcSize = ptrdiff_t(groupSrc.vertices.data() + groupSrc.vertices.size()) - ptrdiff_t(srcData);
                    assert(srcSize >= 0);

                    uint32_t pairs[kMaxClusterVertices * 2];
                    ArithmeticDeCompressor<uint32_t, 2> decompressor;
                    decompressor.Init(size_t(srcSize), srcData);
                    srcData += decompressor.ReadVertices(vertexCount, pairs, 2) / sizeof(uint32_t);
                    for (uint32_t v = 0; v < vertexCount; v++)
                        deltas[v] = pairs[v * 2] | (pairs[v * 2 + 1] << 16);
                }
                else
                {
                    std::memcpy(deltas, srcData, sizeof(uint32_t) * vertexCount);
                    srcData += vertexCount;
                }
                dstOffset += 4 + vertexCount;
                continue;
            }

            if ((clusterSrc.attributeBits & (usedBit | compressedBit)) == (usedBit | compressedBit))
            {
                dstOffset = alignUvBlock(dstOffset);

                ptrdiff_t srcSize = ptrdiff_t(groupSrc.vertices.data() + groupSrc.vertices.size()) - ptrdiff_t(srcData);
                assert(srcSize >= 0);

                ArithmeticDeCompressor<uint32_t, 2> decompressor;
                decompressor.Init(size_t(srcSize), srcData);
                srcData += decompressor.ReadVertices(vertexCount, dstData + dstOffset, 2) / sizeof(uint32_t);
                dstOffset += 2 * vertexCount;
            }
            else if (clusterSrc.attributeBits & usedBit)
            {
                dstOffset = alignUvBlock(dstOffset);
                std::memcpy(dstData + dstOffset, srcData, 2u * sizeof(float) * vertexCount);
                srcData += 2 * vertexCount;
                dstOffset += 2 * vertexCount;
            }
        }

        assert(size_t(dstData + dstOffset) <= size_t(dstWriteOnly) + dstSize);
        (void)dstSize;
    }
}

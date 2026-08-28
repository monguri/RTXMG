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

// .nvsngeo disk cache — read and write the cluster LOD geometry cache file.
//
// File layout:
//   [CacheFileHeader]          — magic, versions, BakerConfig
//   [Geometry 0 blob]          — GeometryBase | groupData | groupInfos |
//                                lodLevels | lodStats | lodNodes | lodNodeBboxes |
//                                localMaterialIDs
//   [Geometry 1 blob]          — same
//   ...
//   [Offset table]             — uint64_t[geometryCount*2 + 1] at end of file
//                                [offset₀, size₀, offset₁, size₁, ..., geometryCount]
//
// Each span is serialised as: 16-byte count header | aligned data (see serialization.h).

#pragma once

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "baked_geometry.h"
#include "baker.h"
#include "rtxmg/utils/file_mapping.h"

// ---------------------------------------------------------------------------
// CacheFileHeader
//
// First bytes of every .nvsngeo file. isValid() checks all versioned fields
// against compile-time constants to detect format mismatches.
// ---------------------------------------------------------------------------

struct CacheFileHeader
{
    // Zero the whole object so trailing padding doesn't leak into the on-disk
    // image, then re-initialise `header` — the memset wipes its default member
    // initialisers, and a header of all zeros makes isValid() compare two zeros
    // and pass for any format version.
    CacheFileHeader()
    {
        memset(this, 0, sizeof(*this));
        header = Header{};
    }

    bool IsValid() const
    {
        CacheFileHeader ref;
        return header.magic         == ref.header.magic
            && header.geoVersion    == ref.header.geoVersion
            && header.geoStructSize == ref.header.geoStructSize
            && header.configVersion == ref.header.configVersion
            && header.alignment     == ref.header.alignment;
    }

    struct Header
    {
        uint64_t magic         = 0x006f65676e73766eULL;  // "nvsngeo"
        // Bump whenever the baked geometry layout or content changes for EVERY
        // config, so stale caches are rejected rather than silently rendered.
        // A change confined to one config belongs in SemanticHash instead —
        // see BakerConfig::kCompressedLayoutVersion — or it re-bakes every warm
        // cache, including ones the change cannot affect.  19 = the LOD
        // hierarchy fix, which changes baked content on every config.
        uint32_t geoVersion    = 19;
        uint32_t geoStructSize = uint32_t(sizeof(GeometryView));
        // Bump when the meaning of configHash changes.  Deliberately not
        // sizeof(BakerConfig): the whole point of hashing is that adding a
        // field the bake ignores must not invalidate the cache.
        uint32_t configVersion = 3;
        uint32_t pad0          = 0;
        uint64_t alignment     = serialization::ALIGNMENT;
    };
    static_assert(sizeof(Header) % serialization::ALIGNMENT == 0,
                  "CacheFileHeader::Header must be serialization-aligned");

    Header   header;
    uint64_t configHash = 0;  // BakerConfig::SemanticHash()
    // Pads Header + configHash to 16-byte alignment.
    uint64_t pad        = 0;
};
static_assert(sizeof(CacheFileHeader) % serialization::ALIGNMENT == 0,
              "CacheFileHeader must be serialization-aligned");

// ---------------------------------------------------------------------------
// CacheFileView
//
// Provides zero-copy read access to a memory-mapped .nvsngeo file.
// The underlying FileReadMapping (or raw pointer) must outlive this object.
// ---------------------------------------------------------------------------

class CacheFileView
{
public:
    CacheFileView() = default;

    // True if a file has been successfully opened/initialised.
    bool IsValid() const { return m_dataSize != 0; }

    // Open and memory-map the file at `path`.  Returns true on success.
    // Owns the mapping — call deinit() or let the destructor close it.
    bool Init(const std::filesystem::path& path);

    // Initialise from an already-mapped buffer (e.g. RAM copy).
    // The caller owns the buffer lifetime.
    bool Init(uint64_t dataSize, const void* data);

    void Deinit();

    ~CacheFileView() { Deinit(); }

    // Non-copyable, movable.
    CacheFileView(const CacheFileView&)            = delete;
    CacheFileView& operator=(const CacheFileView&) = delete;
    CacheFileView(CacheFileView&&)                 = default;
    CacheFileView& operator=(CacheFileView&&)      = default;

    uint64_t GetGeometryCount() const { return m_geometryCount; }

    // BakerConfig::SemanticHash() of the config this cache was written with.
    uint64_t GetConfigHash() const;

    // Fill `view` with zero-copy spans into the mmap'd bytes for geometry `idx`.
    // Returns false if idx is out of range or data is corrupt.
    bool GetGeometryView(GeometryView& view, uint64_t idx) const;

private:
    template<typename T>
    const T* Ptr(uint64_t offset, uint64_t count = 1) const
    {
        assert(offset + sizeof(T) * count <= m_dataSize);
        return reinterpret_cast<const T*>(m_dataBytes + offset);
    }

    rtxmg::FileReadMapping m_mapping;       // owns mmap if opened via path
    uint64_t               m_dataSize      = 0;
    uint64_t               m_tableStart    = 0;
    const uint8_t*         m_dataBytes     = nullptr;
    uint64_t               m_geometryCount = 0;
};

// ---------------------------------------------------------------------------
// Per-geometry shard cache
//
// Alternative to the monolithic .nvsngeo: each unique geometry gets its own
// shard file named by the FNV-1a hash of its source bytes:
//     <gltf_dir>/_nvsngeocache/<hash16hex>.shard
// The folder is shared by every gltf in that directory (not keyed by filename)
// and the naming is content-based, so gltfs referencing the same meshes share
// shards and editing one source buffer rebakes only the affected geometry.  A
// shared manifest (index.bin) maps sourceKey -> {contentHash, sourceSize,
// sourceMtime} so loads validate on size+mtime without re-hashing every input.
//
// Shard file layout:  [ShardHeader][geometry blob (StoreCachedGeometry)]
// ---------------------------------------------------------------------------

struct alignas(serialization::ALIGNMENT) ShardHeader
{
    ShardHeader()
    {
        memset(this, 0, sizeof(*this));
        header = CacheFileHeader::Header{};
    }

    // Format validity: same versioned magic block as the monolith, so a geo/
    // config format bump invalidates shards too.
    bool IsValidFormat() const
    {
        CacheFileHeader::Header ref;
        return header.magic         == ref.magic
            && header.geoVersion    == ref.geoVersion
            && header.geoStructSize == ref.geoStructSize
            && header.configVersion == ref.configVersion
            && header.alignment     == ref.alignment;
    }

    CacheFileHeader::Header header;             // shared versioned magic block
    uint64_t               configHash  = 0;     // BakerConfig::SemanticHash() of the bake
    uint64_t               sourceHash  = 0;     // FNV-1a of the geometry's source bytes
    uint64_t               sourceSize  = 0;     // cheap gate: summed source-buffer file sizes
    int64_t                sourceMtime = 0;     // cheap gate: max source-buffer mtime
    uint64_t               blobSize    = 0;     // bytes of the trailing geometry blob
    // alignas(ALIGNMENT) rounds sizeof up to a 16-byte multiple so the blob
    // following it in the file starts 16-aligned, as store/LoadCachedGeometry
    // require.
};
static_assert(sizeof(ShardHeader) % serialization::ALIGNMENT == 0,
              "ShardHeader must be serialization-aligned so the trailing blob is aligned");

// ---------------------------------------------------------------------------
// ShardCache — owns the memory-mapped shard files backing zero-copy
// GeometryViews.  Must outlive any GeometryView it produces.
// ---------------------------------------------------------------------------

class ShardCache
{
public:
    ShardCache()                              = default;
    ShardCache(const ShardCache&)             = delete;
    ShardCache& operator=(const ShardCache&)  = delete;
    ShardCache(ShardCache&&)                  = default;
    ShardCache& operator=(ShardCache&&)       = default;

    // Map (or reuse an already-mapped) shard at `path`, validate format +
    // BakerConfig, and fill `outView` with zero-copy spans into the mapping.
    // Returns false (leaving outView untouched) if missing/corrupt/mismatched.
    bool LoadGeometryView(const std::filesystem::path& path,
                          const BakerConfig&           config,
                          GeometryView&                outView);

    bool   IsEmpty()        const { return m_mappings.empty(); }

    // Read just the ShardHeader of `path` without retaining a mapping.
    // Returns false if missing / too small / format-mismatched.
    static bool ReadHeader(const std::filesystem::path& path, ShardHeader& outHeader);

private:
    std::unordered_map<std::string, size_t> m_pathToIndex;
    std::vector<rtxmg::FileReadMapping>     m_mappings;
};

// ---------------------------------------------------------------------------
// Free functions — write
// ---------------------------------------------------------------------------

// Serialise one GeometryView into `data` (must be GetCachedSize() bytes).
// Returns true on success.
bool StoreCachedGeometry(const GeometryView& view, uint64_t dataSize, void* data);

// Deserialise one geometry blob — zero-copy spans into `data` (which must be
// 16-byte aligned and hold `dataSize` bytes).  Returns true on success.
bool LoadCachedGeometry(GeometryView& view, uint64_t dataSize, const void* data);

// Serialise `view` to a shard file atomically (write to a temp sibling then
// rename into place), stamping `config` + source identity into the header.
bool SaveShardAtomic(const std::filesystem::path& path,
                     const GeometryView&          view,
                     const BakerConfig&           config,
                     uint64_t                     sourceHash,
                     uint64_t                     sourceSize,
                     int64_t                      sourceMtime);

// Write a complete .nvsngeo file for all views in `geometries`.
// `config` is written into the header so readers can validate parameters.
// Returns true on success.
bool SaveCache(const std::filesystem::path&        path,
               const std::vector<GeometryView>&    geometries,
               const BakerConfig&                  config);


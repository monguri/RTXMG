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

// Memory-mapped cgltf file I/O for the cluster-LOD baker.
//
// cgltf's default file I/O reads every .bin / .glb chunk fully into heap, a
// second copy of all geometry.  These callbacks map them instead, bounding
// physical RAM by the bake's working set.
//
// Usage:
//   rtxmg::FileMappingList mappings;
//   cgltf_options options    = {};
//   options.file.read        = rtxmg::CgltfReadMapped;
//   options.file.release     = rtxmg::CgltfReleaseMapped;
//   options.file.user_data   = &mappings;
//   // ... cgltf_parse_file / cgltf_load_buffers ...
//
// Declare the FileMappingList BEFORE the cgltf_data owner so it is destroyed
// AFTER it (cgltf_free releases each buffer through CgltfReleaseMapped, which
// must still find the list alive).

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <cgltf.h>

#include "rtxmg/utils/file_mapping.h"

namespace rtxmg {

// Refcounted set of read-only memory-mapped buffer files keyed by path.
struct FileMappingList
{
    struct Entry
    {
        FileReadMapping mapping;
        int64_t         refCount = 1;
    };

    // When non-empty, only paths present here are mapped; any other requested
    // path resolves to (data=nullptr, size=0) — used to fault in just the
    // buffers needed by stale geometries (see the per-geometry shard cache).
    std::unordered_set<std::string> subsetNames;

    std::unordered_map<std::string, Entry>       nameToMapping;
    std::unordered_map<const void*, std::string> dataToName;

    // Each bake worker parses its own cgltf_data, so Open/Close run concurrently.
    std::mutex mutex;

    // Open (or bump the refcount of) the mapping for `path`.  On success sets
    // *size/*data and returns true.  A subset miss also returns true with
    // *data=nullptr / *size=0 (cgltf treats that as an empty, unused buffer).
    bool Open(const char* path, size_t* size, void** data);

    // Drop one reference to the mapping backing `data`; unmaps at zero.
    void Close(void* data);

    ~FileMappingList();
};

// cgltf_options.file.read / .release callbacks.  user_data must point at a
// FileMappingList.
cgltf_result CgltfReadMapped(const cgltf_memory_options* memoryOptions,
                             const cgltf_file_options*   fileOptions,
                             const char*                 path,
                             cgltf_size*                 size,
                             void**                      data);

void CgltfReleaseMapped(const cgltf_memory_options* memoryOptions,
                        const cgltf_file_options*   fileOptions,
                        void*                       data);

} // namespace rtxmg

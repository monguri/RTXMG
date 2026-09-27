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

#include "rtxmg/cluster_lod/baking/gltf_file_mapping.h"

#include <cassert>

namespace rtxmg {

bool FileMappingList::Open(const char* path, size_t* size, void** data)
{
    std::lock_guard<std::mutex> lock(mutex);
    std::string                 pathStr(path);

    // Subset filter: when active, buffers outside the set are reported as empty
    // so cgltf skips them (they belong to geometries served from cache).
    if (!subsetNames.empty() && subsetNames.find(pathStr) == subsetNames.end())
    {
        *data = nullptr;
        *size = 0;
        return true;
    }

    auto it = nameToMapping.find(pathStr);
    if (it != nameToMapping.end())
    {
        *data = const_cast<void*>(it->second.mapping.Data());
        *size = it->second.mapping.Size();
        it->second.refCount++;
        return true;
    }

    Entry entry;
    if (entry.mapping.Open(pathStr))
    {
        const void* mappingData = entry.mapping.Data();
        *data                   = const_cast<void*>(mappingData);
        *size                   = entry.mapping.Size();
        dataToName.insert({ mappingData, pathStr });
        nameToMapping.insert({ pathStr, std::move(entry) });
        return true;
    }

    return false;
}

void FileMappingList::Close(void* data)
{
    std::lock_guard<std::mutex> lock(mutex);

    auto itName = dataToName.find(data);
    if (itName == dataToName.end())
        return;  // subset-miss (null) buffer, or already released

    auto itMapping = nameToMapping.find(itName->second);
    if (itMapping != nameToMapping.end())
    {
        if (--itMapping->second.refCount == 0)
        {
            nameToMapping.erase(itMapping);
            dataToName.erase(itName);
        }
    }
}

FileMappingList::~FileMappingList()
{
    assert(nameToMapping.empty() && dataToName.empty()
           && "FileMappingList destroyed with open mappings — declare it before the cgltf_data owner");
}

cgltf_result CgltfReadMapped(const cgltf_memory_options* /*memoryOptions*/,
                             const cgltf_file_options* fileOptions,
                             const char*               path,
                             cgltf_size*               size,
                             void**                    data)
{
    auto* mappings = static_cast<FileMappingList*>(fileOptions->user_data);
    return mappings->Open(path, size, data) ? cgltf_result_success
                                            : cgltf_result_io_error;
}

void CgltfReleaseMapped(const cgltf_memory_options* /*memoryOptions*/,
                        const cgltf_file_options* fileOptions,
                        void*                     data)
{
    auto* mappings = static_cast<FileMappingList*>(fileOptions->user_data);
    mappings->Close(data);
}

} // namespace rtxmg

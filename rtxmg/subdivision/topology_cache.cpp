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
//
// clang-format off

#include "rtxmg/subdivision/topology_cache.h"
#include "rtxmg/subdivision/topology_map.h"

#include <opensubdiv/tmr/topologyMap.h>

// clang-format on

using namespace OpenSubdiv;

union Key
{
    Tmr::TopologyMap::Traits traits;
    uint8_t value;
};

TopologyCache::TopologyCache(TopologyCache::Options const& opts)
    : options(opts)
{
}

TopologyMap& TopologyCache::get(uint8_t traits)
{
    std::lock_guard lock(m_mtx);

    Key key{
        .value = traits,
    };

    if (auto it = m_topologyMaps.find(key.value); it != m_topologyMaps.end())
        return *it->second;

    auto aTopologyMap = std::make_unique<Tmr::TopologyMap>(
        key.traits, Tmr::TopologyMap::Options(uint8_t(m_topologyMaps.size())));

    auto [it, done] = m_topologyMaps.emplace(
        key.value, std::make_unique<TopologyMap>(std::move(aTopologyMap)));

    return *it->second;
}

bool TopologyCache::Empty() const
{
    std::lock_guard lock(m_mtx);
    return m_topologyMaps.empty();
}

size_t TopologyCache::Size() const
{
    std::lock_guard lock(m_mtx);
    return m_topologyMaps.size();
}

void TopologyCache::Clear()
{
    std::lock_guard lock(m_mtx);
    return m_topologyMaps.clear();
}

// note: all hashing must be completed before hoisting maps into device memory !
std::vector<std::unique_ptr<TopologyMap const>>
TopologyCache::InitDeviceData(std::shared_ptr<donut::engine::DescriptorTableManager> descriptorTable,
    nvrhi::ICommandList* commandList, bool keepHostData)
{
    std::lock_guard lock(m_mtx);

    std::vector<std::unique_ptr<TopologyMap const>> topologyMaps;

    topologyMaps.reserve(m_topologyMaps.size());

    for (auto& it : m_topologyMaps)
    {
        it.second->InitDeviceData(descriptorTable, commandList, keepHostData);

        topologyMaps.emplace_back(std::move(it.second));
    }
    return topologyMaps;
}

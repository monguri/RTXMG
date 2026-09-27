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
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#include "rtxmg/subdivision/topology_map.h"
// clang-format on

// Thread-safe cache for Tmr::TopologyMap
//
// The topology cache collects a set of topology maps used to hash the topology
// of subD surfaces in a scene based on 'traits' (subdivision rules). Once all
// the meshes have been parsed, the topology maps can be serialized and hoisted
// in device memory.
//
// note: ownership of the host-side transient Tmr::TopologyyMaps is passed on to
// the device container (::TopologyMap) so that we can still support a CPU code
// path.
//

class TopologyCache
{
public:
    struct Options
    {
        // see Tmr::SubdivisionPlanBuilder::Options for details
        uint8_t const isoLevelSharp = 6;
        uint8_t const isoLevelSmooth = 3;
        bool const useTerminalNodes = false;
    } const options;

    TopologyCache(Options const& options);

    TopologyMap& get(uint8_t traits);

    TopologyMap const& Get(uint8_t traits) const { return this->Get(traits); }

    bool Empty() const;

    size_t Size() const;

    void Clear();

    // note: all hashing must be completed before hoisting maps into device memory
    // !
    std::vector<std::unique_ptr<TopologyMap const>>
        InitDeviceData(std::shared_ptr<donut::engine::DescriptorTableManager> descriptorTable, 
            nvrhi::ICommandList* commandList, bool keepHostData = true);

private:
    mutable std::mutex m_mtx;

    std::map<uint8_t, std::unique_ptr<TopologyMap>> m_topologyMaps;
};

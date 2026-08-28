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

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include <cassert>

#include <donut/core/math/math.h>

using namespace donut::math;

class MayaLogger
{

public:
    static std::unique_ptr<MayaLogger> Create(char const* m_filepath);

    ~MayaLogger();

    struct Descriptor
    {
        std::string nodeName;
        std::string nodePath;
    };

    // particles
    struct ParticleDescriptor : Descriptor
    {

        uint32_t renderType = 3;

        std::vector<float3> positions;
        std::vector<float3> velocities;
        std::vector<float3> colors;

        uint32_t pointSize = 2;
    };
    void CreateParticles(ParticleDescriptor const& desc);

private:

    std::string m_filepath;
    FILE* m_fp = nullptr;
};

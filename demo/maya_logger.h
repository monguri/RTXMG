
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

//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// clang-format off

#pragma once

#include "rtxmg_demo.h"
#include "rtxmg/cluster_builder/tessellator_config.h"
#include "rtxmg/scene/scene.h"

#include <array>
#include <string>
#include <json/json.h>

#include <donut/core/math/math.h>

using namespace donut::math;

// clang-format on

// All state & values controllable from the command-line
// used to initialize the application on launch.
//
// note: much of this state should be broken up into sub-sections
// note: currently the application owns this struct and can override
//       some of its values interactively (which is less than ideal).

// Things that can be modified by a scene.json file
struct SceneArgs
{
    std::array<std::string, TextureType::TEXTURE_TYPE_COUNT> textures;

    float envmapAzimuth = 0.f;
    float envmapElevation = 0.f;
    float envmapIntensity = 1.f;

    ColorMode colorMode = ColorMode::BASE_COLOR;
    ShadingMode shadingMode = ShadingMode::PT;
    TonemapOperator tonemapOperator = TonemapOperator::Aces;

    float wireframeThickness = .1f;

    int spp = 1;
    int ptMaxBounces = 2;

    float firefliesClamp = 0.f;
    float exposure = 1.f;

    float dispScale = 1.f;
    float dispBias = 0.f;

    float roughnessOverride = 0.f;

    bool enableWireframe = false;
    bool enableDenoiser = true;
};

struct Args : SceneArgs
{
    // convenience accessors for legacy code
    using SceneArgs::colorMode;
    using SceneArgs::dispBias;
    using SceneArgs::dispScale;
    using SceneArgs::enableWireframe;
    using SceneArgs::envmapAzimuth;
    using SceneArgs::envmapElevation;
    using SceneArgs::envmapIntensity;
    using SceneArgs::exposure;
    using SceneArgs::firefliesClamp;
    using SceneArgs::ptMaxBounces;
    using SceneArgs::shadingMode;
    using SceneArgs::spp;
    using SceneArgs::wireframeThickness;

    SceneArgs& sceneArgs() { return *this; }

    int width = 1920;
    int height = 1080;
    bool resolutionSetByCmdLine = false;

    std::string outfile;
    std::string meshInputFile = std::string{};
    std::string camString;

    uint4 edgeSegments = { 8, 8, 8, 8 };

    unsigned char quantNBits{ 0 };

    bool enableFrustumVisibility = true;
    bool enableBackfaceVisibility = true;
    bool enableHiZVisibility = true;
    bool updateTessCamera = true;
    bool enableVertexNormals = false;

    TessellatorConfig::MemorySettings tessMemorySettings;
    TessellatorConfig::VisibilityMode visMode = TessellatorConfig::VisibilityMode::VIS_LIMIT_EDGES;
    TessellatorConfig::AdaptiveTessellationMode tessMode = TessellatorConfig::AdaptiveTessellationMode::SPHERICAL_PROJECTION;

    // Note: the defaults here are intended for TMR
    int isoLevelSharp = 6;
    int isoLevelSmooth = 3;
    uint32_t globalIsolationLevel = TessellatorConfig::kMaxIsolationLevel;

    float fineTessellationRate = TessellatorConfig::kDefaultFineTessellationRate;
    float coarseTessellationRate = TessellatorConfig::kDefaultCoarseTessellationRate;
    ClusterPattern clusterPattern = ClusterPattern::SLANTED;
        
    float3 missColor = { .75, .75, .75 };

    bool debug = false;
    bool gpuValidation = false;
    bool aftermath = false;
    bool enableStreamlineLog = false;    
    bool enableAccelBuildLogging = false;
    bool enableTimeView = false;
    bool startMaximized = false;

    void Parse(int argc, char const* const* argv);
};
Args& operator << (Args& args, const Json::Value& node);

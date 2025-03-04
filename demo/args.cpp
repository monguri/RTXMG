//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
//

// clang-format off

#include "args.h"
#include "rtxmg_demo.h"

#include <assert.h>
#include <cstdio>
#include <functional>
#include <map>
#include <numbers>
#include <stdexcept>

#include "rtxmg/scene/json.h"

// clang-format on

static void printUsageAndExit(const char* argv0,
    const std::string& token = {})
{
    // clang-format off

    static char const* msg =
        "Usage  : &s [options]\n"
        "Options: \n"
        "  -h                      | --help                  Print this usage message\n"
        "  -at <mode>              | --adaptiveTessellation  Mode is: [uniform | world | sphere]\n"
        "                          | --aftermath             Enable Aftermath for GPU Crash debugging (not compatible with --debug)"
        "  -a <true|false>         | --amplification         Tessellation amplification (default true).\n"
        "  -bbt <t|c>              | --bvhBuildType          [t (triangles) | c (clusters)]\n"
        "  -cc <mode>              | --clusterColor          Mode is: [base | level (isolation) | uv | bid (block id) | tid (tri id) | \n"
        "                                                    cid (cluster id) | n (normal) | bNum (# blocks) | tArea (utri area)]\n"
        "  -ctr <f>                | --coarseTessellationRate Coarse adaptive edge sampling rate\n"
        "  -d                      | --debug                 Enable D3D12 Debug Layer\n"
        "  -gd                     | --gpudebug              Enable D3D12 GPU Validation\n"
        "  -ds <float>             | --displacementScale     Displacement scale along normal (relative to the largest extent of the object's AABB)\n"
        "  -ed <true|false>        | --enableDenoiser        Enable Denoiser\n"
        "  -envmap <envmap.exr>    |                         Specify the environment map to use\n"
        "  -es <e0> <e1> <e2> <e3> | --edgeSegments          Cluster edge m_size for all four edges (default 5 5 5 5)\n"
        "  -f <filename>           | --file                  Specify file for image output\n"
        "  -ftr <f>                | --fineTessellationRate  Fine adaptive edge sampling rate\n"
        "  -isoLevelSharp <int>                              Max isolation level near sharp features such as creases (default 6)\n"
        "  -isoLevelSmooth <int>                             Max isolation level near smooth features such as extraordinary vertices (default 3)\n"
        "  -ll <n>                 | --loglevel              Set logCallbackLevel (default: 2)\n"
        "                          | --logAccelBuild         Log each buffer of the acceleration build (slow!)\n"
        "                          | --maxClusters           Memory budget Set the max num of clustersverts\n"
        "                          | --maxVerts              Memory budget Set the max num of verts\n"
        "                          | --maxClasMB             Memory budget Set the max num of clas memory in MB\n"
        "  -mrl <int>              | --maxRefinementLevel    Legacy: same as isoLevelSharp\n"
        "  -mc <r> <g> <b>         | --missColor             Miss color\n"
        "  -mf <filepath>          | --meshInputFile         Read .obj or scene file\n"
        "  -p \"[eye][at][up]fov\"   | --cameraPos             Camera pose\n"
        "  -ptmb <n>               | --ptMaxBounces          Max PT bounces\n"
        "  -qbits <n>              | --quantizeBits          Quantize vertex positions by n bits [0,32)\n"
        "  -qf <true|false>        | --quadFiltering         Filter out 1x1 clusters during metric tessellation phase\n"
        "  -res <w> <h>            | --resolution            Set image dimensions to <w>x<h> (default 768 768)\n"
        "  -sm [prim_rays|ao|pt]   | --shadingMode           primary rays or AO or path tracing\n"
        "  -spp <n>                                          Number of samples per pixel (need to use a perfect square number, default 1)\n"
        "  -tv <true|false>        | --timeview              Enable Timeview (default false)\n"
        "  -wf <true|false>        | --wireframe             Set wireframe (default true)\n"
        "  -wm                     | --windowMaximized       Start window maximized (default false)\n";

    // clang-format on

    std::fprintf(stderr, msg, argv0);

    if (token.find("Unknown option") != std::string::npos)
        std::fprintf(stderr, "\n****** %s ******", token.c_str());
    else if (!token.empty())
        std::fprintf(stderr, "\n****** Invalid usage of '%s' ******",
            token.c_str());
    exit(1);
}

static bool parseBooleanArg(char const* token,
    char const* value, bool defaultValue)
{
    if (std::strncmp(value, "true", 4) == 0)
        return true;
    if (std::strncmp(value, "false", 5) == 0)
        return false;
    printUsageAndExit("rtxmg_demo", token);
    return defaultValue;
}

void Args::Parse(int argc, char const* const* argv)
{
    static std::map<std::string, ShadingMode> shadingModes{
        {"prim_rays", ShadingMode::PRIMARY_RAYS},
        {"ao", ShadingMode::AO},
        {"pt", ShadingMode::PT} };

    static std::map<std::string, ColorMode> colorModes{
        {"base", ColorMode::BASE_COLOR},
        {"uv", ColorMode::COLOR_BY_CLUSTER_UV},        
        {"cid", ColorMode::COLOR_BY_CLUSTER_ID},
        {"tid", ColorMode::COLOR_BY_MICROTRI_ID},
        {"n", ColorMode::COLOR_BY_NORMAL},
        {"texcoord", ColorMode::COLOR_BY_TEXCOORD},
        {"mat", ColorMode::COLOR_BY_MATERIAL},
        {"tArea", ColorMode::COLOR_BY_MICROTRI_AREA} };

    static const std::map<std::string, TextureType> TextureTypes = {
        { "-envmap", TextureType::ENVMAP }
    };

    static std::map <std::string, TessellatorConfig::AdaptiveTessellationMode> adaptiveTessellationModes = {
        {"uniform", TessellatorConfig::AdaptiveTessellationMode::UNIFORM},
        {"sphere", TessellatorConfig::AdaptiveTessellationMode::SPHERICAL_PROJECTION},
        {"world", TessellatorConfig::AdaptiveTessellationMode::WORLD_SPACE_EDGE_LENGTH}
    };

    auto parseEnum = [&argv]<typename T>(std::string const& arg, std::map<std::string, T> const& enumModes)
    {
        if (const auto it = enumModes.find(arg); it != enumModes.end())
            return it->second;
        else
            printUsageAndExit(argv[0], arg);
        return T(0);
    };

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);

        auto parseArgValues = [&argc, &argv, &i, &arg](int n,
            std::function<void()> func)
            {
                if (i >= argc - n)
                    printUsageAndExit(argv[0], argv[i]);
                func();
            };

        if (arg == "--debug" || arg == "-d")
        {
            debug = true;
        }
        else if (arg == "--gpudebug" || arg == "-gd")
        {
            gpuValidation = true;
        }
        else if (arg == "--aftermath")
        {
            aftermath = true;
        }
        else if (arg == "--sllog")
        {
            enableStreamlineLog = true;
        }
        else if (arg == "--help" || arg == "-h")
        {
            printUsageAndExit(argv[0]);
        }
        else if (arg == "--file" || arg == "-f")
        {
            parseArgValues(1, [&]() { outfile = argv[++i]; });
        }
        else if (arg == "--timeview" || arg == "-tv")
        {
            parseArgValues(1, [&]() { enableTimeView = parseBooleanArg(arg.c_str(), argv[++i], enableTimeView); });
        }
        else if (arg == "-p" || arg == "--cameraPos")
        {
            parseArgValues(1, [&]() { camString = std::string(argv[++i]); });
        }
        else if (arg == "-res" || arg == "--resolution")
        {
            parseArgValues(2, [&]()
                {
                    width = atoi(argv[++i]);
                    height = atoi(argv[++i]);
                });
            resolutionSetByCmdLine = true;
        }
        else if (arg == "--wireframe" || arg == "-wf")
        {
            parseArgValues(1, [&]()
                {
                    enableWireframe =
                        parseBooleanArg(arg.c_str(), argv[++i], enableWireframe);
                });
        }
        else if (arg == "--displacementScale" || arg == "-ds")
        {
            parseArgValues(1, [&]() { dispScale = (float)atof(argv[++i]); });
        }
        else if (arg == "--adaptiveTessellation" || arg == "-at")
        {
            parseArgValues(
                1, [&]() { tessMode = parseEnum(argv[++i], adaptiveTessellationModes); });
        }
        else if (arg == "--edgeSegments" || arg == "-es")
        {
            parseArgValues(4, [&]()
            {
                edgeSegments.x = static_cast<uint32_t>(atoi(argv[++i]));
                edgeSegments.y = static_cast<uint32_t>(atoi(argv[++i]));
                edgeSegments.z = static_cast<uint32_t>(atoi(argv[++i]));
                edgeSegments.w = static_cast<uint32_t>(atoi(argv[++i]));
            });
        }
        else if (arg == "--missColor" || arg == "-mc")
        {
            parseArgValues(3, [&]()
                {
                    missColor = { (float)atof(argv[++i]), (float)atof(argv[++i]),
                                 (float)atof(argv[++i]) };
                });
        }
        else if (arg == "--ptMaxBounces" || arg == "-ptmb")
        {
            parseArgValues(1, [&]() { ptMaxBounces = atoi(argv[++i]); });
        }
        else if (arg == "--meshInputFile" || arg == "-mf")
        {
            parseArgValues(1, [&]() { meshInputFile = argv[++i]; });
        }
        else if (arg == "--maxRefinementLevel" || arg == "-mrl" ||
            arg == "-isoLevelSharp")
        {
            parseArgValues(1, [&]() { isoLevelSharp = atoi(argv[++i]); });
        }
        else if (arg == "-isoLevelSmooth")
        {
            parseArgValues(1, [&]() { isoLevelSmooth = atoi(argv[++i]); });
        }
        else if (arg == "-envmap")
        {
            try
            {
                textures[parseEnum(arg, TextureTypes)] = argv[++i];
            }
            catch (std::exception& e)
            {
                std::fprintf(stderr, "Invalid option in '%s %s' : %s\n", arg.c_str(), argv[i], e.what());
                exit(1);
            }
        }
        else if (arg == "--shadingMode" || arg == "-sm")
        {
            parseArgValues(
                1, [&]() { shadingMode = parseEnum(argv[++i], shadingModes); });
        }
        else if (arg == "-spp")
        {
            parseArgValues(1, [&]()
                {
                    spp = atoi(argv[++i]);
                    const int sqrt_spp =
                        static_cast<int>(std::sqrt(static_cast<float>(spp)));
                    if (sqrt_spp * sqrt_spp !=
                        spp) // check if spp is a perfect sqare number
                        printUsageAndExit(argv[0], arg);
                });
        }
        else if (arg == "--logAccelBuild")
        {
            enableAccelBuildLogging = true;
        }
        else if (arg == "--vertBufferMB")
        {
            parseArgValues(1, [&]() { tessMemorySettings.vertexBufferBytes = size_t(atoi(argv[++i])) << 20ull; });
        }
        else if (arg == "--maxClusters")
        {
            parseArgValues(1, [&]() { tessMemorySettings.maxClusters = atoi(argv[++i]); });
        }
        else if (arg == "--clasBufferMB")
        {
            parseArgValues(1, [&]() { tessMemorySettings.clasBufferBytes = size_t(atoi(argv[++i])) << 20ull; });
        }
        else if (arg == "--fineTessellationRate" || arg == "-ftr")
        {
            parseArgValues(1, [&]() { fineTessellationRate = (float)atof(argv[++i]); });
        }
        else if (arg == "--coarseTessellationRate" || arg == "-ctr")
        {
            parseArgValues(1, [&]() { coarseTessellationRate = (float)atof(argv[++i]); });
        }
        else if (arg == "--enableDenoiser" || arg == "-ed")
        {
            enableDenoiser = true;
        }
        else if (arg == "--windowMaximized" || arg == "-wm")
        {
            startMaximized = true;
        }
        else
            printUsageAndExit(argv[0], std::string("Unknown option: ") + argv[i]);
    }
}


auto parseJsonEnum = []<typename T, size_t N>(const Json::Value & node,
    const std::array<const char*, N> &enums, T & result) constexpr
{
    uint8_t index = 0;
    for (const char* e : enums)
    {
        if (std::strncmp(node.asString().c_str(), e, std::strlen(e)) == 0)
        {
            result = T(index);
            break;
        }
        ++index;
    }
};

Args& operator << (Args& args, const Json::Value& node)
{
    if (const auto& value = node["envmap rotation"]; value.isDouble())
    {
        value >> args.envmapAzimuth;
        args.envmapAzimuth = (args.envmapAzimuth / 180.f) * float(std::numbers::pi);
    }

    if (const auto& value = node["envmap elevation"]; value.isDouble())
    {
        value >> args.envmapElevation;
        args.envmapElevation = (args.envmapElevation / 180.f) * float(std::numbers::pi);
    }
    if (const auto& value = node["envmap intensity"]; value.isDouble())
    {
        value >> args.envmapIntensity;
    }
    if (const auto& value = node["shading mode"]; value.isString())
        parseJsonEnum(value, kShadingModeNames, args.shadingMode);

    if (const auto& value = node["color mode"]; value.isString())
    {
        parseJsonEnum(value, kColorModeNames, args.colorMode);
    }

    if (const auto& value = node["spp"]; value.isIntegral())
        value >> args.spp;
    if (const auto& value = node["max bounces"]; value.isIntegral())
        value >> args.ptMaxBounces;

    if (const auto& value = node["firefly max intensity"]; value.isDouble())
        value >> args.firefliesClamp;

    if (const auto& value = node["exposure"]; value.isDouble())
        value >> args.exposure;

    if (const auto& value = node["tonemap operator"]; value.isString())
        parseJsonEnum(value, kToneMapOperatorNames, args.tonemapOperator);

    // displacement
    if (const auto& value = node["displacementScale"]; value.isDouble())
        value >> args.dispScale;

    if (const auto& value = node["displacementBias"]; value.isDouble())
    {
        value >> args.dispBias;
        donut::log::warning("Scene arg displacementBias is not supported, ignoring");
    }

    if (const auto& value = node["wireframe"]; value.isBool())
        value >> args.enableWireframe;

    if (const auto& value = node["wireframe thickness"]; value.isDouble())
        value >> args.wireframeThickness;

    if (const auto& value = node["enableDenoiser"]; value.isBool())
        value >> args.enableDenoiser;

    return args;
}
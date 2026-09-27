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

#include "shot_list.h"

#include "rtxmg/scene/json.h"

#include <donut/core/log.h>

#include <json/json.h>

#include <map>

// clang-format on

namespace fs = std::filesystem;
using donut::log::warning;

namespace
{

// Same spellings the -sm / -cm flags accept, so a shot list reads like the CLI.
const std::map<std::string, ShadingMode> kShadingModes{
    { "prim_rays", ShadingMode::PRIMARY_RAYS }, { "primary", ShadingMode::PRIMARY_RAYS },
    { "ao",        ShadingMode::AO },
    { "pt",        ShadingMode::PT },
};

// Same spellings --dlssMode accepts.
const std::map<std::string, donut::app::StreamlineInterface::DLSSMode> kDlssModes{
    { "off",         donut::app::StreamlineInterface::DLSSMode::eOff },
    { "ultra",       donut::app::StreamlineInterface::DLSSMode::eUltraPerformance },
    { "performance", donut::app::StreamlineInterface::DLSSMode::eMaxPerformance },
    { "balanced",    donut::app::StreamlineInterface::DLSSMode::eBalanced },
    { "quality",     donut::app::StreamlineInterface::DLSSMode::eMaxQuality },
    { "DLAA",        donut::app::StreamlineInterface::DLSSMode::eDLAA },
    { "dlaa",        donut::app::StreamlineInterface::DLSSMode::eDLAA },
};

const std::map<std::string, ColorMode> kColorModes{
    { "base",       ColorMode::BASE_COLOR },        { "basecolor", ColorMode::BASE_COLOR },
    { "n",          ColorMode::COLOR_BY_SHADING_NORMAL },
    { "normal",     ColorMode::COLOR_BY_SHADING_NORMAL },
    { "texcoord",   ColorMode::COLOR_BY_TEXCOORD },
    { "uv",         ColorMode::COLOR_BY_CLUSTER_UV },
    { "mat",        ColorMode::COLOR_BY_MATERIAL },
    { "cid",        ColorMode::COLOR_BY_CLUSTER_ID },
    { "tid",        ColorMode::COLOR_BY_MICROTRI_ID },
    { "tArea",      ColorMode::COLOR_BY_MICROTRI_AREA },
    { "lod",        ColorMode::COLOR_BY_LOD_LEVEL },
    { "group",      ColorMode::COLOR_BY_CLUSTER_GROUP },
    { "blas",       ColorMode::COLOR_BY_BLAS_SOURCE },
    { "blascached", ColorMode::COLOR_BY_BLAS_CACHED },
};

// Reverse lookups for the filename tag.  First spelling wins, so the tag is
// stable even though several aliases parse to the same enum.
std::string NameOf(ShadingMode m)
{
    switch (m)
    {
    case ShadingMode::PRIMARY_RAYS: return "primary";
    case ShadingMode::AO:           return "ao";
    default:                        return "pt";
    }
}

std::string NameOf(ColorMode m)
{
    // "n" would win the map order but reads as noise in a filename.
    if (m == ColorMode::COLOR_BY_SHADING_NORMAL)
        return "normal";
    for (const auto& [name, value] : kColorModes)
        if (value == m)
            return name;
    return "base";
}

}  // anonymous namespace

const char* DlssModeName(donut::app::StreamlineInterface::DLSSMode mode)
{
    using DLSSMode = donut::app::StreamlineInterface::DLSSMode;
    switch (mode)
    {
    case DLSSMode::eMaxPerformance:   return "performance";
    case DLSSMode::eBalanced:         return "balanced";
    case DLSSMode::eMaxQuality:       return "quality";
    case DLSSMode::eUltraPerformance: return "ultra";
    case DLSSMode::eUltraQuality:     return "ultraquality";
    case DLSSMode::eDLAA:             return "DLAA";
    default:                          return "off";
    }
}

bool Shot::ShowsColorMode() const
{
    // AO writes baseColor * visibility, so a colour mode survives to the image
    // there too -- but tagging plain AO "ao-base" would rename existing goldens.
    return shadingMode == ShadingMode::PRIMARY_RAYS ||
           (shadingMode == ShadingMode::AO && colorMode != ColorMode::BASE_COLOR);
}

std::string Shot::Tag() const
{
    std::string tag = NameOf(shadingMode);
    if (ShowsColorMode())
        tag += "-" + NameOf(colorMode);
    if (wireframe)
        tag += "-wf";
    return tag;
}

bool LoadShotList(const fs::path& path, std::vector<ShotEntry>& entries)
{
    Json::Value root;
    try
    {
        root = readFile(path);
    }
    catch (const std::exception& e)
    {
        warning("--shot-list: %s", e.what());
        return false;
    }

    if (!root.isArray() || root.empty())
    {
        warning("--shot-list '%s': expected a non-empty array of capture points",
                path.generic_string().c_str());
        return false;
    }

    for (Json::ArrayIndex i = 0; i < root.size(); ++i)
    {
        const Json::Value& node = root[i];
        ShotEntry          entry;

        entry.label = read<std::string>(node["label"], "");
        if (entry.label.empty())
        {
            warning("--shot-list entry %u: missing 'label'", i);
            return false;
        }

        entry.camera = read<std::string>(node["camera"], "");
        if (entry.camera.empty())
        {
            warning("--shot-list '%s': missing 'camera'", entry.label.c_str());
            return false;
        }

        // Deliberately no default: auto-exposure drifts with scene luminance, so
        // an entry that inherited it would produce a golden that moves on its own.
        if (!node["exposure"].isNumeric())
        {
            warning("--shot-list '%s': missing 'exposure'.  Every entry must pin it "
                    "explicitly -- auto-exposure is temporal and would drift the golden.",
                    entry.label.c_str());
            return false;
        }
        entry.exposure = node["exposure"].asFloat();

        if (const Json::Value& settle = node["settle"]; settle.isNumeric())
            entry.settleFrames = settle.asInt();
        else if (settle.isString() && settle.asString() != "auto")
        {
            warning("--shot-list '%s': 'settle' must be a frame count or \"auto\"",
                    entry.label.c_str());
            return false;
        }
        entry.settleCap = read<uint32_t>(node["settleCap"], entry.settleCap);

        // Absent = inherit the CLI value.  Present but malformed is an error
        // rather than a silent no-op: a sweep whose settings never applied
        // produces N identical entries that look like a real result.
        if (const Json::Value& dlss = node["dlssMode"]; dlss.isString())
        {
            const auto it = kDlssModes.find(dlss.asString());
            if (it == kDlssModes.end())
            {
                warning("--shot-list '%s': unknown dlssMode '%s'",
                        entry.label.c_str(), dlss.asString().c_str());
                return false;
            }
            entry.dlssMode    = it->second;
            entry.dlssModeSet = true;
        }
        else if (!dlss.isNull())
        {
            warning("--shot-list '%s': 'dlssMode' must be a string", entry.label.c_str());
            return false;
        }

        if (const Json::Value& lpe = node["lodPixelError"]; lpe.isNumeric())
        {
            entry.lodPixelError = lpe.asFloat();
            if (entry.lodPixelError <= 0.f)
            {
                warning("--shot-list '%s': 'lodPixelError' must be > 0", entry.label.c_str());
                return false;
            }
        }
        else if (!lpe.isNull())
        {
            warning("--shot-list '%s': 'lodPixelError' must be a number", entry.label.c_str());
            return false;
        }

        if (const Json::Value& res = node["resolution"]; res.isArray())
        {
            if (res.size() != 2 || !res[0].isIntegral() || !res[1].isIntegral() ||
                res[0].asInt() <= 0 || res[1].asInt() <= 0)
            {
                warning("--shot-list '%s': 'resolution' must be [width, height]",
                        entry.label.c_str());
                return false;
            }
            entry.outputWidth  = res[0].asInt();
            entry.outputHeight = res[1].asInt();
        }
        else if (!res.isNull())
        {
            warning("--shot-list '%s': 'resolution' must be [width, height]", entry.label.c_str());
            return false;
        }

        if (const Json::Value& adaptive = node["adaptiveLodError"]; adaptive.isBool())
            entry.adaptiveLodError = adaptive.asBool() ? 1 : 0;
        else if (!adaptive.isNull())
        {
            warning("--shot-list '%s': 'adaptiveLodError' must be true or false",
                    entry.label.c_str());
            return false;
        }

        if (const Json::Value& nms = node["normalMapShading"]; nms.isBool())
            entry.normalMapShading = nms.asBool() ? 1 : 0;
        else if (!nms.isNull())
        {
            warning("--shot-list '%s': 'normalMapShading' must be true or false",
                    entry.label.c_str());
            return false;
        }

        const Json::Value& shots = node["shots"];
        if (!shots.isArray() || shots.empty())
        {
            warning("--shot-list '%s': 'shots' must be a non-empty array", entry.label.c_str());
            return false;
        }

        for (const Json::Value& shotNode : shots)
        {
            Shot              shot;
            const std::string mode = read<std::string>(shotNode["mode"], "pt");
            const auto        modeIt = kShadingModes.find(mode);
            if (modeIt == kShadingModes.end())
            {
                warning("--shot-list '%s': unknown mode '%s'", entry.label.c_str(), mode.c_str());
                return false;
            }
            shot.shadingMode = modeIt->second;

            if (const std::string cm = read<std::string>(shotNode["colorMode"], ""); !cm.empty())
            {
                const auto cmIt = kColorModes.find(cm);
                if (cmIt == kColorModes.end())
                {
                    warning("--shot-list '%s': unknown colorMode '%s'",
                            entry.label.c_str(), cm.c_str());
                    return false;
                }
                shot.colorMode = cmIt->second;
            }

            if (const Json::Value& wf = shotNode["wireframe"]; wf.isBool())
                shot.wireframe = wf.asBool();
            else if (!wf.isNull())
            {
                warning("--shot-list '%s': 'wireframe' must be true or false",
                        entry.label.c_str());
                return false;
            }

            // A primary-ray debug mode has no stochastic sampling to converge;
            // AO takes one visibility sample a frame and the path tracer needs a
            // long accumulation.  None of the counts covers the denoiser's own
            // re-accumulation floor.
            const uint32_t defaultFrames = shot.shadingMode == ShadingMode::PT ? 128u
                                         : shot.shadingMode == ShadingMode::AO ? 64u
                                                                               : 8u;
            shot.frames   = read<uint32_t>(shotNode["frames"], defaultFrames);
            shot.exposure = read<float>(shotNode["exposure"],
                                        shot.ShowsColorMode() ? 1.f : entry.exposure);
            entry.shots.push_back(shot);
        }

        entries.push_back(std::move(entry));
    }

    return true;
}

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

#include "rtxmg_demo.h"

#include <donut/app/StreamlineInterface.h>

#include <filesystem>
#include <string>
#include <vector>

// clang-format on

// --shot-list <file.json>: a list of capture points walked in one process, so a
// scene load is amortized across every golden image it produces rather than
// paid once per screenshot.
//
// [
//   {
//     "label":    "board-closeup",
//     "camera":   "[1.2 0.8 -0.4][0 0.1 0][0 1 0]45",
//     "exposure": 1.5,
//     "settle":   "auto",
//     "dlssMode": "quality",
//     "lodPixelError": 0.5,
//     "normalMapShading": true,
//     "shots":    [ { "mode": "pt", "frames": 128 },
//                   { "mode": "primary", "colorMode": "base" },
//                   { "mode": "ao", "colorMode": "normal", "wireframe": true } ]
//   }
// ]
//
// The setting keys are optional and inherit the run's CLI values when absent,
// so a settings sweep is N entries that differ only in those.  A shot may carry
// its own "exposure"; primary-ray modes default to 1.0 rather than the entry's.

struct Shot
{
    ShadingMode shadingMode = ShadingMode::PT;
    ColorMode   colorMode   = ColorMode::BASE_COLOR;

    // Absolute, not inherited from -wf: a shot list states its shots in full,
    // the same way it does mode and colour mode.
    bool        wireframe   = false;

    // Frames to accumulate after applying the mode, before the capture.  A
    // path-traced shot needs many; a primary-ray debug mode converges at once.
    uint32_t    frames      = 0;  // 0 = per-mode default

    // Resolved at parse time from the entry's exposure: a primary-ray debug mode
    // decodes flat values rather than luminance, so a beauty exposure tuned for
    // the path tracer just blows it out.
    float       exposure    = 1.f;

    // True when the colour mode reaches the image unshaded, which is what makes
    // it worth naming in the tag and worth capturing at exposure 1.  The path
    // tracer runs it through the BRDF, so there it is not a distinct look.
    bool ShowsColorMode() const;

    // "<label>.<tag>", the screenshot stem.
    std::string Tag() const;
};

struct ShotEntry
{
    std::string       label;
    std::string       camera;    // same syntax as -p / --cameraPos
    float             exposure = 1.f;
    std::vector<Shot> shots;

    // Settings applied at the camera cut, before the settle, so residency
    // converges under the settings the shot is captured with.  eOff is itself a
    // mode, so the flag carries "unset" rather than a sentinel value.
    bool                                      dlssModeSet = false;
    donut::app::StreamlineInterface::DLSSMode dlssMode =
        donut::app::StreamlineInterface::DLSSMode::eOff;
    float   lodPixelError    = 0.f;  // 0 = inherit -lpe
    int32_t adaptiveLodError = -1;   // -1 = inherit --adaptive-error, else 0/1
    // Shading only; loading normal maps is a scene-reload setting, so a list
    // that toggles this has to run with --normalmaps for either state to mean
    // anything.  -1 = inherit --normalmapshading, else 0/1.
    int32_t normalMapShading = -1;

    // Output resolution, i.e. the window's client size.  0 = leave it alone.
    // Ignored in fullscreen, where the display drives it -- so a matrix that
    // spans resolutions runs windowed for the small ones and -fs for native.
    int32_t outputWidth  = 0;
    int32_t outputHeight = 0;

    // Frames to wait for streaming residency after the camera cut.  -1 means
    // watch the residency counters go flat instead of using a fixed count.
    int32_t settleFrames = -1;
    // Upper bound on the "auto" wait, so a scene that never converges still
    // finishes rather than hanging the run.
    uint32_t settleCap = 1500;
};

// Parses `path`.  Returns false and logs on a malformed file, an unknown mode,
// or a missing camera/exposure -- auto-exposure is a temporal feedback loop, so
// an entry that inherits it would drift for reasons unrelated to any change.
bool LoadShotList(const std::filesystem::path& path, std::vector<ShotEntry>& entries);

// The --dlssMode spelling of `mode`, for the stats dump.
const char* DlssModeName(donut::app::StreamlineInterface::DLSSMode mode);

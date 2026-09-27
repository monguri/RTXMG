/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <imgui_internal.h>
#include <implot.h>
#include <implot_internal.h>

#include <cassert>
#include <cmath>
#include <type_traits>
#include <variant>

#include <donut/core/math/math.h>

#include "rtxmg/profiler/gui.h"
#include "rtxmg/utils/formatters.h"

using namespace donut::math;


// clang-format on

// expects a [0, 1] normalized value 
static ImVec4 heatmapColor( float value )
{
    static float3 colors[] = { {0.f, 1.f, 0.f}, { 1., 1.f, 0.f}, { 1.f, 0.f, 0.f } };

    uint8_t i0 = 0;
    uint8_t i1 = 0;
    float m_fp = 0.f;

    if( value <= 0.f )
        return ImVec4( .5f, .5f, .5f, 1.f );
    else if( value >= 1.f )
        i0 = i1 = (uint8_t) std::size( colors ) - 1;
    else
    {
        m_fp = value * ( std::size( colors ) - 1 );
        i0 = (uint8_t)std::floor( m_fp );
        i1 = i0 + 1;
        m_fp = m_fp - float( i0 );
    }

    float3 c = colors[i0] + m_fp * ( colors[i1] - colors[i0] );
    return ImVec4( c.x, c.y, c.z, 1.f );
}

void ProfilerGUI::BuildControllerUI( ImFont* iconicFont, ImPlotContext *plotContext )
{
    ImGui::SetNextWindowPos(controllerWindow.pos, ImGuiCond_Always, controllerWindow.pivot);
    ImGui::SetNextWindowSize(controllerWindow.size);
    ImGui::SetNextWindowBgAlpha(.65f);

    ImGui::Begin("ProfilerController", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar);

    ImVec2 itemSize = ImGui::GetItemRectSize();
    char buf[50];
    if (clusterLodTrisValid)
    {
        // Cluster-LOD: unique is the CLAS-deduped footprint, total the instanced sum.
        char bufU[32];
        char bufT[32];
        HumanFormatter(static_cast<double>(clusterLodUniqueTris), bufU, sizeof(bufU));
        HumanFormatter(static_cast<double>(clusterLodTotalTris),  bufT, sizeof(bufT));
        ImGui::Text("Uniq Tris %s", bufU);
        ImGui::Text("Total Tris %s", bufT);
    }
    else if (HumanFormatter(static_cast<double>(desiredTris), buf, sizeof(buf)))
        ImGui::Text("Tris %s", buf);
    else
        ImGui::Text("Too many !");

    // The Profiler-window toggle lives in the top-left sidebar; this controller
    // window is just the always-on Tris / FPS HUD.
    if (fps >= 0)
        ImGui::Text("FPS   % 5d", fps);
    else
        ImGui::Text("FPS    ----");

    (void)iconicFont;

    controllerWindow.size = ImGui::GetWindowSize();

    ImGui::End();

    ImPlot::SetCurrentContext(plotContext);
}

void ProfilerGUI::BuildFrequencySelectorUI()
{
    Profiler& profiler = Profiler::Get();

    int rate = 0;

    if (profiler.recordingFrequency >= 120)
        rate = 5;
    else if (profiler.recordingFrequency >= 60)
        rate = 4;
    else if (profiler.recordingFrequency >= 30)
        rate = 3;
    else if (profiler.recordingFrequency >= 10)
        rate = 2;
    else if (profiler.recordingFrequency >= 1)
        rate = 1;

    ImVec2 size = ImGui::GetWindowSize();

    ImGui::SameLine(size[0] - (64 + 10));
    ImGui::PushItemWidth(64);
    if (ImGui::Combo("##SamplingFrequency", &rate, "---Hz\0001Hz\00010Hz\00030Hz\00060Hz\000120Hz\0"))
    {
        switch (rate)
        {
            case 0: profiler.recordingFrequency = -1; break;
            case 1: profiler.recordingFrequency = 1; break;
            case 2: profiler.recordingFrequency = 10; break;
            case 3: profiler.recordingFrequency = 30; break;
            case 4: profiler.recordingFrequency = 60; break;
            case 5: profiler.recordingFrequency = 120; break;
        }
    }
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Profiling rate: frequency (in Hertz) at which samples are recorded each second\n"
            "or unconstrained records every frame.");
}


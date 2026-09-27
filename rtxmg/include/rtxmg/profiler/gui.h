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

#pragma once

#include "rtxmg/profiler/profiler.h"
#include <imgui_internal.h>
#include "donut/core/math/math.h"

struct ImPlotContext;
class UserInterface;

// clang-format on

class ProfilerGUI
{
  public:

    // if fps >= 0 displays value in profiler controller window
    int fps = -1;

    // if ntris > 0 displays value in profiler controller window
    uint32_t desiredTris = 0;
    uint32_t allocatedTris = 0;

    // Cluster-LOD per-frame TLAS triangles (TRACK_RENDER_STATS): the CLAS-deduped
    // footprint and the instanced sum, replacing the cluster_tess tris count.
    bool     clusterLodTrisValid  = false;
    uint64_t clusterLodUniqueTris = 0;
    uint64_t clusterLodTotalTris  = 0;
    uint32_t desiredClusters = 0;
    uint32_t allocatedClusters = 0;

    struct ControllerWindow
    {
        ImVec2    pos = ImVec2(0, 0);
        ImVec2    pivot = ImVec2(0, 0);
        ImVec2    size = ImVec2(115, 0);
    } controllerWindow;

    struct ProfilerWindow
    {
        ImVec2    pos = ImVec2(0, 0);
        ImVec2    pivot = ImVec2(1, 0);
        ImVec2    size = ImVec2(0, 0);
        ImVec2    screenLayoutSize = ImVec2(0, 0);
    } profilerWindow;

    bool displayGraphWindow = true;

  public:
    // Active Profiler tab, persisted to imgui.ini via the RTXMG settings handler.
    // A non-empty requestedTab force-selects that tab, then is cleared.
    std::string activeTab;
    std::string requestedTab;

    template <typename... SamplerGroup>
    void BuildUI( ImFont *iconicFont, ImPlotContext *plotContext, SamplerGroup&... groups );

  private:

    void BuildControllerUI( ImFont *iconicFont, ImPlotContext *plotContext );
    void BuildFrequencySelectorUI();
};

inline ImVec2 MakeImVec2(const dm::float2& v)
{
    return ImVec2{ v.x, v.y };
}

inline ImVec2 MakeImVec2(const dm::int2& v)
{
    return ImVec2{ float(v.x), float(v.y) };
}

inline dm::float2 MakeFloat2(const ImVec2& v)
{
    return dm::float2{ v.x, v.y };
}

inline void SetConstrainedWindowPos(const char *windowName, ImVec2 windowPos, const ImVec2& windowPivot, const ImVec2& screenSize)
{
    ImGuiCond cond = ImGuiCond_FirstUseEver;    
    ImGuiWindow* window = ImGui::FindWindowByName(windowName);

    // Bound the window position to be on screen by a margin
    const float kMinOnscreenLength = 20.0f;
    if (window)
    {
        const dm::float2 kMinOnscreenSize = { kMinOnscreenLength, kMinOnscreenLength };
        dm::float2 currentWindowPos = MakeFloat2(window->Pos);
        dm::float2 currentWindowSize = MakeFloat2(window->Size);
        dm::box2 windowRect{ currentWindowPos, currentWindowPos + currentWindowSize };
        dm::box2 screenLayoutRect{ kMinOnscreenSize, MakeFloat2(screenSize) - kMinOnscreenSize };
        
        if (!screenLayoutRect.intersects(windowRect))
        {
            cond = ImGuiCond_Always;
            dm::float2 minCornerAdjustment = -min(windowRect.m_maxs - screenLayoutRect.m_mins, dm::float2::zero());
            dm::float2 maxCornerAdjustment = -max(windowRect.m_mins - screenLayoutRect.m_maxs, dm::float2::zero());
            dm::float2 adjustment = minCornerAdjustment + maxCornerAdjustment;
            windowRect = windowRect.translate(adjustment);

            windowPos = MakeImVec2(windowRect.m_mins + MakeFloat2(windowPivot) * currentWindowSize);
        }
    }
    ImGui::SetNextWindowPos(windowPos, cond, windowPivot);
}

template <typename... SamplerGroup>
inline void ProfilerGUI::BuildUI( ImFont *iconicFont, ImPlotContext *context, SamplerGroup&... groups )
{
    BuildControllerUI(iconicFont, context);

    if (displayGraphWindow)
    {
        const char* kWindowName = "Profiler";
        SetConstrainedWindowPos(kWindowName, profilerWindow.pos, profilerWindow.pivot, profilerWindow.screenLayoutSize);
        ImGui::SetNextWindowSize(profilerWindow.size, ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowCollapsed(false, ImGuiCond_FirstUseEver); // shown expanded by default
        ImGui::SetNextWindowBgAlpha(.65f);

        if (ImGui::Begin(kWindowName, &displayGraphWindow, ImGuiWindowFlags_None))
        {
            BuildFrequencySelectorUI();

            if (ImGui::BeginTabBar("MyTabBar", ImGuiTabBarFlags_Reorderable))
            {
                ImVec2 tabSize = profilerWindow.size;
                (
                    [&] {
                        // A tab whose data the current scene can't produce is hidden,
                        // not empty (e.g. Streaming on a scene with no cluster-LOD).
                        if (!groups.TabEnabled())
                            return;
                        // Force-select the tab restored from imgui.ini, for one frame.
                        ImGuiTabItemFlags tabFlags = (!requestedTab.empty() && requestedTab == groups.name)
                                                         ? ImGuiTabItemFlags_SetSelected
                                                         : ImGuiTabItemFlags_None;
                        if( ImGui::BeginTabItem( groups.name.c_str(), nullptr, tabFlags ) )
                        {
                            activeTab = groups.name;  // track open tab for save
                            groups.BuildUI( iconicFont, context );
                            ImGui::EndTabItem();
                        }
                    }(),
                    ... );
                ImGui::EndTabBar();
                requestedTab.clear();  // applied — the user can switch freely again
            }
        }
        ImGui::End();
    }
}

//
// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#include "rtxmg/profiler/profiler.h"
#include <imgui.h>

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
    uint32_t desiredClusters = 0;
    uint32_t allocatedClusters = 0;

    struct ControllerWindow
    {
        ImVec2    pos = ImVec2(0, 0);
        ImGuiCond cond = ImGuiCond(0);
        ImVec2    pivot = ImVec2(0, 0);
        ImVec2    size = ImVec2(115, 0);
    } controllerWindow;

    struct ProfilingWindow
    {
        ImVec2    pos = ImVec2(0, 0);
        ImGuiCond cond = ImGuiCond_FirstUseEver;
        ImVec2    pivot = ImVec2(1, 0);
        ImVec2    size = ImVec2(800, 250);
        bool      resetLayout = false;
    } profilingWindow;

    bool displayGraphWindow = true;

  public:
    template <typename... SamplerGroup>
    void BuildUI( ImFont *iconicFont, ImPlotContext *plotContext, SamplerGroup const&... groups );

  private:

    void BuildControllerUI( ImFont *iconicFont, ImPlotContext *plotContext );
    void BuildFrequencySelectorUI();
};

template <typename... SamplerGroup>
inline void ProfilerGUI::BuildUI( ImFont *iconicFont, ImPlotContext *context, SamplerGroup const&... groups )
{
    // Synchronize with device streams so timers can be polled safely by UI elements
    Profiler::Get().FrameSync();

    BuildControllerUI(iconicFont, context);

    if (displayGraphWindow)
    {
        ImGui::SetNextWindowCollapsed(true, ImGuiCond_FirstUseEver); // Collapse the window by default
        if (profilingWindow.resetLayout)
        {
            ImGui::SetNextWindowPos(profilingWindow.pos, profilingWindow.cond, profilingWindow.pivot);
            ImGui::SetNextWindowSize(profilingWindow.size, profilingWindow.cond);
        }
        ImGui::SetNextWindowBgAlpha(.65f);

        if (ImGui::Begin("Profiler", &displayGraphWindow, ImGuiWindowFlags_None))
        {
            BuildFrequencySelectorUI();

            if (ImGui::BeginTabBar("MyTabBar", ImGuiTabBarFlags_Reorderable))
            {
                ImVec2 tabSize = profilingWindow.size;
                (
                    [&] {
                        if( ImGui::BeginTabItem( groups.name.c_str() ) )
                        {
                            groups.BuildUI( iconicFont, context );
                            ImGui::EndTabItem();
                        }
                    }(),
                    ... );
                ImGui::EndTabBar();
            }
        }
        ImGui::End();
    }
}

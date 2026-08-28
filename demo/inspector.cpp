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
#include "inspector.h"

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <string>
#include <vector>

#include <imgui.h>

#include "implot.h"
#include "rtxmg_demo_app.h"

#include "rtxmg/profiler/statistics.h"
#include "rtxmg/utils/formatters.h"
#include "rtxmg/utils/pixel_pick.h"

constexpr ImGuiTableFlags kFlags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                   ImGuiTableFlags_SizingStretchProp;

static void TextHuman(double v)
{
    char buf[64];
    HumanFormatter(v, buf, sizeof(buf));
    ImGui::TextUnformatted(buf);
}

static void TextMem(double v)
{
    char buf[64];
    MemoryFormatter(v, buf, sizeof(buf));
    ImGui::TextUnformatted(buf);
}

void GeometryInspector::Draw(bool& show)
{
#if ENABLE_PIXEL_PICK
    // A fresh right-click pick opens the Inspector.
    {
        const auto& pick = m_app.GetRenderer().GetPixelPick();
        if (pick.valid && pick.sequence != m_handledPickSeq)
            show = true;
    }
#endif

    if (!show)
        return;

    const RTXMGScene& scene = m_app.GetScene();
    const auto& clusterLodGeoms  = scene.GetClusterLodGeometries();
    const auto& subdStats  = stats::evaluatorSamplers.surfaceTableStats;
    m_geoms = &clusterLodGeoms;

#if ENABLE_PIXEL_PICK
    // Consume picks here, outside ImGui::Begin, so selection stays live even
    // when the window is collapsed (ImGui::Begin returns false when collapsed,
    // which would otherwise block every pick).
    ConsumePick();
#endif

    // Drive the viewport highlight.  A viewport pick narrows it to the picked
    // instance and LOD level, a LOD-row click to that level across all
    // instances, and a geometry-row selection highlights everything.
    {
        const bool haveSel = m_selectedGeom != ~0u && m_highlightSelection;
        m_app.GetRenderer().SetSelectedClusterLodGeometry(
            haveSel ? int(m_selectedGeom) : -1,
            (haveSel && m_pickedInstance != ~0u) ? int(m_pickedInstance) : -1,
            haveSel ? m_selectedLod : -1);
        const bool haveSubdSel = m_selectedSubd != ~0u && m_highlightSelection;
        m_app.GetRenderer().SetSelectedSubdMesh(haveSubdSel ? int(m_selectedSubd) : -1);
    }

    ImGui::SetNextWindowSize(ImVec2(760.f, 480.f), ImGuiCond_FirstUseEver);
    // The window itself never scrolls; the table and the subdivision child
    // scroll internally, so their headers and captions stay visible.
    if (ImGui::Begin("Inspector", &show,
                     ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse))
    {
        // Darker than the default amber so the white overlay text stays legible.
        ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.48f, 0.34f, 0.02f, 1.f));

        // ---- Highlight Selection checkbox ----------------------------------------
        ImGui::Checkbox("Highlight Selection", &m_highlightSelection);
        ImGui::Spacing();

        // Setup cluster-LOD streaming state.
        if (!clusterLodGeoms.empty())
        {
            ClusterLodResources* resources = m_app.GetRenderer().GetClusterLodResources();
            m_streamingHooks = resources ? resources->GetStreamingHooks() : nullptr;
            if (m_lodDynHooks != (const void*)m_streamingHooks)
                m_residency = {};
            m_residencyGroupsRebuilt =
                m_streamingHooks && m_streamingHooks->GetResidencyReport(m_residency);

            m_haveResidency  = m_streamingHooks != nullptr;
            m_stripPositions = m_residency.stripResidentPositions;
            m_stripNormals   = m_residency.stripResidentNormals;
            m_haveClas       = !m_residency.residentClasBytes.empty();

            RefreshStaticTotals(scene);
        }

        if (!clusterLodGeoms.empty())
        {
            RefreshResidency();
        }

        // ---- Selected Mesh collapsible ------------------------------------------
        if (m_selectedGeom != ~0u || m_selectedSubd != ~0u)
        {
            if (ImGui::CollapsingHeader("Selected Mesh", ImGuiTreeNodeFlags_DefaultOpen))
            {
                if (!clusterLodGeoms.empty() && m_selectedGeom != ~0u)
                {
                    DrawSelectionSummary(scene);
                }
                else if (!subdStats.empty() && m_selectedSubd != ~0u &&
                         m_selectedSubd < uint32_t(subdStats.size()))
                {
                    const auto& s = subdStats[m_selectedSubd];
                    const auto& subdInsts = scene.GetSubdMeshInstances();
                    if (m_pickedInstance != ~0u && m_pickedInstance < subdInsts.size())
                    {
                        const auto& node = subdInsts[m_pickedInstance].node;
                        ImGui::Text("Instance: %u %s", m_pickedInstance,
                                    node ? node->name.c_str() : "");
                    }
                    ImGui::Text("Mesh: %u '%s'", m_selectedSubd,
                                s.name.empty() ? "(unnamed)" : s.name.c_str());
                    if (m_pickedInstance != ~0u)
                        ImGui::Text("Material: %u %s", m_pickedMaterialID,
                                    m_pickedMaterialName.c_str());
                    s.BuildDetailUI(m_iconicFont, m_implot, m_selectedSubd);
                }
            }
            ImGui::Spacing();
        }

        if (!clusterLodGeoms.empty())
        {
            DrawClodTable(!subdStats.empty() && m_subdHeaderOpen);
            ImGui::Spacing();
        }

        if (!subdStats.empty())
            DrawSubdTable();

        if (clusterLodGeoms.empty() && subdStats.empty())
            ImGui::TextDisabled("No geometry loaded.");

        ImGui::PopStyleColor();
    }
    ImGui::End();
}

void GeometryInspector::RefreshStaticTotals(const RTXMGScene& scene)
{
    const auto& clusterLodGeoms = *m_geoms;

    // One-time static sweep for this scene: per-(geometry, LOD) totals +
    // a display name from the first instance referencing each geometry
    // (instances carry the glTF node/mesh name).
    const void* groupInfosKey = clusterLodGeoms[0].groupInfos.data();
    if (m_geomsKey != (const void*)clusterLodGeoms.data() ||
        m_groupInfosKey != groupInfosKey || m_geomCount != clusterLodGeoms.size() ||
        m_staticStripPositions != m_stripPositions || m_staticStripNormals != m_stripNormals)
    {
        m_staticStripPositions = m_stripPositions;
        m_staticStripNormals   = m_stripNormals;
        m_geomsKey      = clusterLodGeoms.data();
        m_groupInfosKey = groupInfosKey;
        m_geomCount     = clusterLodGeoms.size();

        m_lodOffset.assign(clusterLodGeoms.size() + 1, 0);
        for (size_t i = 0; i < clusterLodGeoms.size(); ++i)
            m_lodOffset[i + 1] = m_lodOffset[i] +
                uint32_t(std::min<size_t>(clusterLodGeoms[i].lodLevelsCount, clusterLodGeoms[i].lodLevels.size()));

        m_lodStatic.assign(m_lodOffset.back(), {});
        for (size_t i = 0; i < clusterLodGeoms.size(); ++i)
        {
            const auto& g = clusterLodGeoms[i];
            const uint32_t lodCount = m_lodOffset[i + 1] - m_lodOffset[i];
            for (uint32_t L = 0; L < lodCount && L < g.lodStats.size(); ++L)
            {
                // Baked per-LOD totals (LodStats).  Summing the groups
                // here instead would walk groupData, which is an mmap of
                // the shard cache: one page fault per group, scene-wide.
                const LodStats& src = g.lodStats[L];
                auto& st = m_lodStatic[m_lodOffset[i] + L];
                st.totGroups   = src.totGroups;
                st.totClusters = src.totClusters;
                st.totTris     = src.totTris;
                st.totBytes    = src.totBytes;
                st.totPosBytes = src.posBytes;
                st.totNrmBytes = src.nrmBytes;
                st.totUvBytes  = src.uvBytes;
                st.quantUv     = src.quantUv != 0;
                st.compressed  = src.compressed != 0;

                // The coarsest LOD is the always-resident low-detail
                // prefix, stored UNSTRIPPED in its own persistent buffer;
                // deeper LODs stream in stripped.  Subtracting the
                // stripped channels ignores the per-cluster realignment
                // ResidentGroupDeviceBytes() does, so this is close but
                // not byte-exact.
                const bool coarsest = (L + 1 == lodCount);
                uint64_t   dev      = src.totDeviceBytes;
                if (!coarsest)
                {
                    if (m_stripPositions) dev -= std::min(dev, src.posBytes);
                    if (m_stripNormals)   dev -= std::min(dev, src.nrmBytes);
                }
                st.totDeviceBytes = dev;
            }
        }

        m_names.assign(clusterLodGeoms.size(), {});
        m_instCount.assign(clusterLodGeoms.size(), 0);
        for (const auto& inst : scene.GetClusterLodInstances())
        {
            if (inst.geometryID >= clusterLodGeoms.size())
                continue;
            m_instCount[inst.geometryID]++;
            if (m_names[inst.geometryID].empty() && !inst.name.empty())
                m_names[inst.geometryID] = inst.name;
        }
        // Mesh dedup lets differently named nodes share one geometry, so
        // mark rows whose name is one representative among N.
        for (size_t i = 0; i < m_names.size(); ++i)
            if (m_instCount[i] > 1 && !m_names[i].empty())
                m_names[i] += " (x" + std::to_string(m_instCount[i]) + ")";

        m_openGeom.assign(clusterLodGeoms.size(), 0);
        // Force the residency aggregation to rebuild against the new scene.
        m_lodDyn.clear();
        m_lodDynHooks = nullptr;
        m_sortOrder.clear();
        m_selectedGeom     = ~0u;
        m_selectedLod      = -1;
        m_pickedInstance   = ~0u;
        m_scrollToSelected = false;
    }
}

#if ENABLE_PIXEL_PICK
void GeometryInspector::ConsumePick()
{
    const auto& pick = m_app.GetRenderer().GetPixelPick();
    if (!pick.valid || pick.sequence == m_handledPickSeq)
        return;
    m_handledPickSeq = pick.sequence;

    if (pick.isClusterLod && m_geoms && pick.surfaceID < uint32_t(m_geoms->size()))
    {
        m_selectedSubd = ~0u;
        if (pick.surfaceID == m_selectedGeom)
        {
            m_selectedGeom     = ~0u;
            m_selectedLod      = -1;
            m_pickedInstance   = ~0u;
        }
        else
        {
            m_selectedGeom       = pick.surfaceID;
            m_selectedLod        = int32_t(pick.lodLevel);
            m_pickedInstance     = pick.instanceID;
            m_pickedMaterialID   = pick.materialID;
            m_pickedMaterialName = pick.name;
            if (m_selectedGeom < m_openGeom.size())
                m_openGeom[m_selectedGeom] = 1;
            m_scrollToSelected = true;
        }
    }
    else if (!pick.isClusterLod)
    {
        const auto& subdInstances = m_app.GetScene().GetSubdMeshInstances();
        m_selectedGeom   = ~0u;
        m_selectedLod    = -1;
        m_pickedInstance = ~0u;
        if (pick.instanceID < uint32_t(subdInstances.size()))
        {
            uint32_t meshID = subdInstances[pick.instanceID].meshID;
            if (meshID == m_selectedSubd)
            {
                m_selectedSubd = ~0u;
            }
            else
            {
                m_selectedSubd       = meshID;
                m_pickedInstance     = pick.instanceID;
                m_pickedMaterialID   = pick.materialID;
                m_pickedMaterialName = pick.name;
            }
        }
    }
}
#endif

void GeometryInspector::RefreshResidency()
{
    const auto& clusterLodGeoms = *m_geoms;

    // Residency aggregation from the streaming resident-group list, each
    // entry resolving its LOD via GroupInfo::lodLevel.  Only recomputed
    // when the report rebuilt that list, so a converged scene skips the
    // sweep.  On the preload path the render code below substitutes the
    // static totals instead — every group is resident there.
    m_residencyRefreshed = m_lodDyn.size() != m_lodStatic.size() ||
                           m_lodDynHooks != (const void*)m_streamingHooks ||
                           m_residencyGroupsRebuilt;
    if (m_residencyRefreshed)
    {
        m_lodDynHooks = m_streamingHooks;
        m_lodDyn.assign(m_lodStatic.size(), {});
    }
    if (m_residencyRefreshed && m_haveResidency)
    {
        auto flatLodOf = [&](const rtxmg::GeometryGroup& gg) -> uint32_t
        {
            if (gg.geometryID >= clusterLodGeoms.size() ||
                gg.groupID >= clusterLodGeoms[gg.geometryID].groupInfos.size())
                return ~0u;
            const uint32_t flat = m_lodOffset[gg.geometryID] +
                                  clusterLodGeoms[gg.geometryID].groupInfos[gg.groupID].lodLevel;
            return flat < m_lodOffset[gg.geometryID + 1] ? flat : ~0u;
        };

        // Pinned groups are never removed, so the always-resident
        // low-detail prefix always leads the resident list and the
        // pinned-vs-streamable split is just an index compare.
        const std::vector<rtxmg::GeometryGroup>& groups = m_residency.residentGroups;
        const size_t pinnedCount = m_residency.pinnedGroupsCount;
        for (size_t n = 0; n < groups.size(); ++n)
        {
            const rtxmg::GeometryGroup& gg = groups[n];
            const uint32_t flat = flatLodOf(gg);
            if (flat == ~0u)
                continue;
            const auto& gi = clusterLodGeoms[gg.geometryID].groupInfos[gg.groupID];
            const auto& st = m_lodStatic[flat];
            auto&       d  = m_lodDyn[flat];
            d.resGroups++; d.resClusters += gi.clusterCount;
            d.resTris += gi.triangleCount; d.resBytes += gi.sizeBytes;

            // Per-attribute bytes prorated from the LOD's baked totals by
            // this group's share of the LOD's stored bytes.  The exact
            // figure is in the cluster headers, but this loop reruns on
            // every residency change, and groupData is an mmap of the
            // shard cache: walking them faults a page per resident group
            // per frame.  Sums back to the static total once the LOD is
            // fully resident.
            const double share = st.totBytes ? double(gi.sizeBytes) / double(st.totBytes) : 0.0;
            auto         part  = [share](uint64_t tot) { return uint64_t(double(tot) * share); };
            const uint64_t pos = part(st.totPosBytes);
            const uint64_t nrm = part(st.totNrmBytes);

            // Matches the Memory-tab geometry pool, minus the per-geometry
            // hierarchy nodes (navigation data, not group memory).
            uint64_t devBytes = gi.GetDeviceSize();
            if (n < pinnedCount)
            {
                // The pinned low-detail prefix is stored unstripped.
                d.pinnedDeviceBytes += devBytes;
                d.lowDetailClusters += gi.clusterCount;
                d.hasLowDetail = true;
            }
            else
            {
                if (m_stripPositions) devBytes -= std::min(devBytes, pos);
                if (m_stripNormals)   devBytes -= std::min(devBytes, nrm);
                d.streamResClusters += gi.clusterCount;
            }
            d.resDeviceBytes += devBytes;
            // Per-attribute resident: positions/normals collapse to 0 when
            // stripped (fetched from AS / facet), UVs stay resident.  The
            // low-detail root's retained pos/nrm land in Total, not here.
            d.resPosBytes += m_stripPositions ? 0 : pos;
            d.resNrmBytes += m_stripNormals   ? 0 : nrm;
            d.resUvBytes  += part(st.totUvBytes);
        }
    }
}

void GeometryInspector::DrawSelectionSummary(const RTXMGScene& scene)
{
    const auto& clusterLodGeoms = *m_geoms;

    // The row label is only the FIRST instance's name and mesh dedup lets
    // differently named nodes share a geometry, so a viewport pick also
    // names the node actually clicked and its resolved material.
    // (ASCII only: the ImGui font renders em-dashes as '?'.)
    if (m_selectedGeom != ~0u && m_selectedGeom < clusterLodGeoms.size())
    {
        const char* rowName =
            (m_selectedGeom < m_names.size() && !m_names[m_selectedGeom].empty())
                ? m_names[m_selectedGeom].c_str() : "(unnamed)";
        char lodStr[32];
        if (m_selectedLod >= 0)
            snprintf(lodStr, sizeof(lodStr), "LOD-%d only", m_selectedLod);
        else
            snprintf(lodStr, sizeof(lodStr), "all LODs");
        const auto& clusterLodInsts = scene.GetClusterLodInstances();
        if (m_pickedInstance != ~0u && m_pickedInstance < clusterLodInsts.size())
        {
            ImGui::Text("Instance: %u %s", m_pickedInstance,
                        clusterLodInsts[m_pickedInstance].name.c_str());
            ImGui::Text("Mesh: %u '%s'  %s", m_selectedGeom, rowName, lodStr);
            ImGui::Text("Material: %u %s", m_pickedMaterialID,
                        m_pickedMaterialName.c_str());
        }
        else
        {
            const uint32_t nInst = m_selectedGeom < m_instCount.size()
                                       ? m_instCount[m_selectedGeom] : 1u;
            ImGui::TextDisabled("Instance: all %u", nInst);
            ImGui::Text("Mesh: %u '%s'  %s", m_selectedGeom, rowName, lodStr);
        }
    }
    else
    {
        ImGui::TextDisabled("No selection. Right-click a mesh in the viewport or click a "
                            "table row to highlight it.");
    }

    // ---- Live Resident-vs-Disk breakdown of the selection -------------
    // Sums m_lodDyn over the selected geometry's LODs, so the figures
    // track loads/unloads live.
    if (m_selectedGeom != ~0u && m_selectedGeom < clusterLodGeoms.size())
    {
        const uint32_t gi0 = m_lodOffset[m_selectedGeom];
        const uint32_t gi1 = m_lodOffset[m_selectedGeom + 1];
        uint64_t resPos=0, resNrm=0, resUv=0, resDev=0;
        uint64_t dskPos=0, dskNrm=0, dskUv=0, dskTot=0, totDev=0;
        uint32_t resGrp=0, totGrp=0, resCls=0, totCls=0;
        uint64_t resTri=0, totTri=0;
        uint64_t resClas=0;  // resident CLAS bytes over the selected LODs
        bool hasNrm=false, quantUv=false, compressed=false;
        for (uint32_t f = gi0; f < gi1; ++f)
        {
            const uint32_t L = f - gi0;
            if (m_selectedLod >= 0 && uint32_t(m_selectedLod) != L) continue;
            const auto& st = m_lodStatic[f];
            dskPos += st.totPosBytes; dskNrm += st.totNrmBytes; dskUv += st.totUvBytes;
            dskTot += st.totBytes;  // on-disk stored size
            totDev += st.totDeviceBytes; totGrp += st.totGroups; totCls += st.totClusters;
            totTri += st.totTris;
            hasNrm |= (st.totNrmBytes > 0); quantUv |= st.quantUv; compressed |= st.compressed;
            if (m_haveClas)
                resClas += ResidentClasBytes(m_selectedGeom, L);
            if (m_haveResidency)
            {
                const auto& dy = m_lodDyn[f];
                resPos += dy.resPosBytes; resNrm += dy.resNrmBytes; resUv += dy.resUvBytes;
                resDev += dy.resDeviceBytes; resGrp += dy.resGroups; resCls += dy.resClusters;
                resTri += dy.resTris;
            }
            else  // preload: everything resident
            {
                resPos += st.totPosBytes; resNrm += st.totNrmBytes; resUv += st.totUvBytes;
                resDev += st.totDeviceBytes; resGrp += st.totGroups; resCls += st.totClusters;
                resTri += st.totTris;
            }
        }

        // The denominator is the fully-resident size, NOT disk: disk isn't
        // loaded anywhere, so it gets its own plain-value column.  A
        // stripped channel has max-resident 0, so print just the resident
        // value instead of a redundant "0 B / 0 B".
        auto residBar = [&](double res, double maxRes, bool asMem)
        {
            auto fmt = [&](double v, char* out) { if (asMem) MemoryFormatter(v, out, 32); else HumanFormatter(v, out, 32); };
            char a[32]; fmt(res, a);
            char ov[80];
            if (maxRes > 0.0)
            {
                char b[32]; fmt(maxRes, b);
                snprintf(ov, sizeof(ov), "%s / %s", a, b);
            }
            else
                snprintf(ov, sizeof(ov), "%s", a);
            ImGui::ProgressBar(maxRes > 0.0 ? float(res / maxRes) : 0.f, ImVec2(-FLT_MIN, 0.f), ov);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Fill = resident / max-resident (residency %%).");
        };

        // Max-resident (fully-loaded device size) per channel: stripped
        // channels can never be resident, so their bar denominator is 0.
        const double maxResPos = m_stripPositions ? 0.0 : double(dskPos);
        const double maxResNrm = m_stripNormals   ? 0.0 : double(dskNrm);
        const double maxResUv  = double(dskUv);

        ImGui::SeparatorText("Mesh Memory (selection)");
        if (ImGui::BeginTable("##cluster_lod_attr", 3, kFlags))
        {
            ImGui::TableSetupColumn("Channel", ImGuiTableColumnFlags_WidthStretch, 2.f);
            ImGui::TableSetupColumn("Resident", ImGuiTableColumnFlags_WidthStretch, 2.f);
            ImGui::TableSetupColumn("Disk", ImGuiTableColumnFlags_WidthStretch, 1.f);
            ImGui::TableHeadersRow();
            auto arow = [&](const char* label, double res, double maxRes, double dsk,
                            const char* tip = nullptr)
            {
                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::TextUnformatted(label);
                if (tip && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tip);
                ImGui::TableNextColumn(); residBar(res, maxRes, true);
                ImGui::TableNextColumn(); TextMem(dsk);
                if (compressed && ImGui::IsItemHovered())
                    ImGui::SetTooltip("Uncompressed layout; groups are arithmetic-compressed\n"
                                      "on disk (smaller stored size).");
            };
            arow(m_stripPositions ? "Positions (fetched from AS)" : "Positions (f32x3)",
                 double(resPos), maxResPos, double(dskPos));
            if (hasNrm)
                arow(m_stripNormals ? "Normals (dropped - facet shading)" : "Normals (packed 4B/vtx)",
                     double(resNrm), maxResNrm, double(dskNrm));
            arow(quantUv ? "UVs (po2-grid quantized 4B/vtx)" : "UVs (raw f32x2 8B/vtx)",
                 double(resUv), maxResUv, double(dskUv));
            arow("Total (device)", double(resDev), double(totDev), double(dskTot),
                 "Matches the master table's Resident column. The channel rows above\n"
                 "are vertex data only; this also includes headers, indices, bboxes\n"
                 "and the unstripped low-detail root.");
            ImGui::EndTable();
        }

        ImGui::SeparatorText("Residency (selection)");
        if (ImGui::BeginTable("##cluster_lod_resid", 2, kFlags))
        {
            ImGui::TableSetupColumn("Item", ImGuiTableColumnFlags_WidthStretch, 1.f);
            ImGui::TableSetupColumn("Resident / Total", ImGuiTableColumnFlags_WidthStretch, 2.f);
            ImGui::TableHeadersRow();
            auto crow = [&](const char* label, double res, double tot)
            {
                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::TextUnformatted(label);
                ImGui::TableNextColumn();
                char a[32], b[32];
                HumanFormatter(res, a, sizeof(a)); HumanFormatter(tot, b, sizeof(b));
                char ov[80]; snprintf(ov, sizeof(ov), "%s / %s", a, b);
                ImGui::ProgressBar(tot > 0.0 ? float(res / tot) : 0.f, ImVec2(-FLT_MIN, 0.f), ov);
            };
            crow("Groups",    double(resGrp), double(totGrp));
            crow("Clusters",  double(resCls), double(totCls));
            crow("Triangles", double(resTri), double(totTri));
            // CLAS exists only for resident clusters, so there is no baked
            // "full CLAS size" to divide by: the fill is cluster residency
            // and the overlay the actual resident bytes ("n/a" until the
            // first readback lands; persistent allocator only).
            {
                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::TextUnformatted("CLAS (built)");
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Cluster-level acceleration structures, built on the GPU for\n"
                                      "resident clusters only. Fill = resident clusters / total\n"
                                      "(the fraction of this selection's CLAS that is built);\n"
                                      "overlay = resident CLAS bytes.");
                ImGui::TableNextColumn();
                const float frac = totCls > 0 ? float(double(resCls) / double(totCls)) : 0.f;
                char ov[32];
                if (m_haveClas) MemoryFormatter(double(resClas), ov, sizeof(ov));
                else          snprintf(ov, sizeof(ov), "n/a");
                ImGui::ProgressBar(frac, ImVec2(-FLT_MIN, 0.f), ov);
            }
            ImGui::EndTable();
        }
    }
}

void GeometryInspector::DrawClodTable(bool haveSubd)
{
    const auto& clusterLodGeoms = *m_geoms;

    // Resident/total cell ("R / T", memory or human-count formatted).
    auto rt = [&](double res, double tot, bool asMem)
    {
        char a[32], b[32];
        if (asMem) { MemoryFormatter(res, a, sizeof(a)); MemoryFormatter(tot, b, sizeof(b)); }
        else       { HumanFormatter (res, a, sizeof(a)); HumanFormatter (tot, b, sizeof(b)); }
        ImGui::Text("%s / %s", a, b);
    };
    // The fraction is resident-pool BYTES, but whether a row counts as
    // fully pinned is decided by the streamable CLUSTER count — that only
    // drives the "pinned"/"cached" label and the gray tint.  No semantic
    // coloring: red/green would read as cost/quality.
    auto residencyBar = [&](double resDevBytes, double totDevBytes,
                            uint32_t streamTotClusters, bool anyPinned, bool cached)
    {
        const ImVec4 gray(0.30f, 0.30f, 0.33f, 1.f);
        const bool   pinnedRow = (streamTotClusters == 0);
        float frac = totDevBytes > 0.0 ? float(resDevBytes / totDevBytes)
                                       : (anyPinned ? 1.f : 0.f);
        char ov[16];
        if (pinnedRow)
            snprintf(ov, sizeof(ov), cached ? "cached" : (anyPinned ? "pinned" : "-"));
        else
            snprintf(ov, sizeof(ov), "%.0f%%", frac * 100.f);
        if (pinnedRow && anyPinned) ImGui::PushStyleColor(ImGuiCol_PlotHistogram, gray);
        ImGui::ProgressBar(frac, ImVec2(-FLT_MIN, 0.f), ov);
        if (pinnedRow && anyPinned) ImGui::PopStyleColor();
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Fill = resident device memory fraction.\n"
                              "Gray = pinned (always resident): a \"cached\" BLAS level\n"
                              "or the low-detail prefix.");
    };


    if (!ImGui::CollapsingHeader("Cluster LOD Meshes", ImGuiTreeNodeFlags_DefaultOpen))
        return;
    // LOD-hierarchy nav bytes are in the Memory-tab Geometry pool but
    // belong to no group, so Resident Mem falls short of the pool by
    // exactly this.  Shown so the two reconcile.
    uint64_t navBytes = 0;
    for (const auto& g : clusterLodGeoms)
        navBytes += g.lodNodes.size_bytes() + g.lodNodeBboxes.size_bytes();
    {
        char nb[32]; MemoryFormatter(double(navBytes), nb, sizeof(nb));
        ImGui::TextDisabled("Memory-tab Geometry = Resident Mem (groups) + %s LOD hierarchy "
                            "(nodes/bboxes, not group data).", nb);
    }
    // Takes the rest of the window, minus a share for the subd section
    // when one is present.
    const float clusterLodTableH = std::max(
        ImGui::GetTextLineHeightWithSpacing() * 4.f,
        haveSubd ? ImGui::GetContentRegionAvail().y * 0.55f
                 : ImGui::GetContentRegionAvail().y);
    if (ImGui::BeginTable("##cluster_lod_geom", 8,
                          kFlags | ImGuiTableFlags_Sortable | ImGuiTableFlags_ScrollY,
                          ImVec2(0.f, clusterLodTableH)))
    {
        ImGui::TableSetupScrollFreeze(0, 2);  // header row + totals row
        // Numeric columns sort by their RESIDENT value (the "R" of
        // "R / T"), descending-first; default sort = resident device
        // memory, so the biggest live consumers lead the table.
        ImGui::TableSetupColumn("Mesh", ImGuiTableColumnFlags_PreferSortAscending);
        ImGui::TableSetupColumn("Resident", ImGuiTableColumnFlags_PreferSortDescending);
        // Actual resident-pool bytes (positions/normals stripped, UVs
        // quantized) — reconciles with the Memory tab's Geometry pool.
        ImGui::TableSetupColumn("Resident Mem", ImGuiTableColumnFlags_PreferSortDescending |
                                                ImGuiTableColumnFlags_DefaultSort);
        // On-disk stored size (gi.sizeBytes): compression-aware.
        ImGui::TableSetupColumn("Disk Size", ImGuiTableColumnFlags_PreferSortDescending);
        ImGui::TableSetupColumn("CLAS", ImGuiTableColumnFlags_PreferSortDescending);
        ImGui::TableSetupColumn("Groups", ImGuiTableColumnFlags_PreferSortDescending);
        ImGui::TableSetupColumn("Clusters", ImGuiTableColumnFlags_PreferSortDescending);
        ImGui::TableSetupColumn("Triangles", ImGuiTableColumnFlags_PreferSortDescending);
        // Headers submitted by hand rather than via TableHeadersRow() so each
        // one can carry its own tooltip.
        {
            static const char* kColTips[8] = {
                "Expand a row for per-LOD residency.", nullptr,
                "Actual resident-pool bytes (positions/normals stripped, UVs quantized).",
                "On-disk stored size (compression-aware).",
                nullptr, nullptr, nullptr, nullptr };
            ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
            for (int c = 0; c < 8; ++c)
            {
                if (!ImGui::TableSetColumnIndex(c))
                    continue;
                ImGui::TableHeader(ImGui::TableGetColumnName(c));
                if (kColTips[c] && ImGui::IsItemHovered())
                    ImGui::SetTooltip("%s", kColTips[c]);
            }
        }

        struct LodAgg { uint32_t resGroups=0, totGroups=0, resClusters=0, totClusters=0;
                        uint32_t sRes=0, sTot=0, pinnedClusters=0;  // streamable / pinned split (bar)
                        uint64_t resTris=0, totTris=0, resBytes=0, totBytes=0, clasBytes=0;
                        uint64_t resDeviceBytes=0, totDeviceBytes=0, pinnedDeviceBytes=0;
                        bool     cached=false; };  // a cached-BLAS level pins this row

        // A cached BLAS at cachedLevel also covers every COARSER level
        // (higher index) — a coarser desired LOD is served by the
        // more-detailed cached BLAS — so those levels never stream.
        auto lodAgg = [&](uint32_t i, uint32_t L, uint32_t cachedLevel) -> LodAgg
        {
            const auto& st = m_lodStatic[m_lodOffset[i] + L];
            LodAgg a;
            a.totGroups = st.totGroups; a.totClusters = st.totClusters;
            a.totTris   = st.totTris;   a.totBytes    = st.totBytes;
            a.totDeviceBytes = st.totDeviceBytes;
            a.clasBytes = ResidentClasBytes(i, L);
            if (!m_haveResidency)
            {
                // Preload: every group resident and pinned.
                a.resGroups = a.totGroups; a.resClusters = a.totClusters;
                a.resTris   = a.totTris;   a.resBytes    = a.totBytes;
                a.resDeviceBytes = st.totDeviceBytes; a.pinnedDeviceBytes = st.totDeviceBytes;
                a.pinnedClusters = a.totClusters;
                return a;
            }
            const auto& dyn = m_lodDyn[m_lodOffset[i] + L];
            a.resGroups = dyn.resGroups; a.resClusters = dyn.resClusters;
            a.resTris   = dyn.resTris;   a.resBytes    = dyn.resBytes;
            a.resDeviceBytes = dyn.resDeviceBytes; a.pinnedDeviceBytes = dyn.pinnedDeviceBytes;
            const bool cachedCovered =
                (cachedLevel != uint32_t(shaderio::kTraversalInvalidLodLevel)) && (L >= cachedLevel);
            // Pinned: always-resident low-detail prefix, or covered by
            // this geometry's cached BLAS.
            a.pinnedClusters = cachedCovered ? st.totClusters : dyn.lowDetailClusters;
            a.sTot = st.totClusters - a.pinnedClusters;
            a.sRes = cachedCovered ? 0 : dyn.streamResClusters;
            // Label as "cached" only when caching is the reason it's
            // pinned; the persistent low-detail level stays "pinned".
            a.cached = cachedCovered && !dyn.hasLowDetail;
            return a;
        };
        // Groups partition groupInfos by lodLevel, so summing the LODs
        // yields exact geometry totals.
        auto geomAgg = [&](uint32_t i) -> LodAgg
        {
            const uint32_t cachedLevel = CachedBlasLevel(i);
            LodAgg sum;
            for (uint32_t L = 0; L < m_lodOffset[i + 1] - m_lodOffset[i]; ++L)
            {
                const LodAgg a = lodAgg(i, L, cachedLevel);
                sum.resGroups += a.resGroups; sum.totGroups += a.totGroups;
                sum.resClusters += a.resClusters; sum.totClusters += a.totClusters;
                sum.sRes += a.sRes; sum.sTot += a.sTot; sum.pinnedClusters += a.pinnedClusters;
                sum.resTris += a.resTris; sum.totTris += a.totTris;
                sum.resBytes += a.resBytes; sum.totBytes += a.totBytes;
                sum.resDeviceBytes += a.resDeviceBytes; sum.totDeviceBytes += a.totDeviceBytes;
                sum.pinnedDeviceBytes += a.pinnedDeviceBytes;
                sum.clasBytes += a.clasBytes;
                sum.cached |= a.cached;
            }
            return sum;
        };

        auto lodColumns = [&](const LodAgg& a)   // all columns after the name
        {
            ImGui::TableNextColumn(); residencyBar(double(a.resDeviceBytes), double(a.totDeviceBytes),
                                                   a.sTot, a.pinnedClusters > 0, a.cached);
            ImGui::TableNextColumn(); rt(double(a.resDeviceBytes), double(a.totDeviceBytes), true);
            // Disk Size has no "resident" numerator: it is a fixed asset
            // size, and disk data isn't loaded anywhere.
            ImGui::TableNextColumn(); TextMem(double(a.totBytes));
            ImGui::TableNextColumn(); if (m_haveClas) TextMem(double(a.clasBytes)); else ImGui::TextDisabled("-");
            ImGui::TableNextColumn(); rt(double(a.resGroups),   double(a.totGroups),   false);
            ImGui::TableNextColumn(); rt(double(a.resClusters), double(a.totClusters), false);
            ImGui::TableNextColumn(); rt(double(a.resTris),     double(a.totTris),     false);
        };

        // Sort keys derive from the same aggregates the cells render, so
        // re-sorting is only needed when the spec changes or the
        // residency aggregates refreshed.
        if (m_selectedGeom != ~0u && m_selectedGeom >= clusterLodGeoms.size())
        {
            m_selectedGeom     = ~0u;
            m_selectedLod      = -1;
            m_pickedInstance   = ~0u;
            m_scrollToSelected = false;
        }

        ImGuiTableSortSpecs* sortSpecs = ImGui::TableGetSortSpecs();
        if (m_sortOrder.size() != clusterLodGeoms.size())
        {
            m_sortOrder.resize(clusterLodGeoms.size());
            for (uint32_t i = 0; i < uint32_t(clusterLodGeoms.size()); ++i)
                m_sortOrder[i] = i;
            if (sortSpecs)
                sortSpecs->SpecsDirty = true;
        }
        if (sortSpecs && (sortSpecs->SpecsDirty || m_residencyRefreshed) && sortSpecs->SpecsCount > 0)
        {
            const ImGuiTableColumnSortSpecs& spec = sortSpecs->Specs[0];
            const bool asc = spec.SortDirection == ImGuiSortDirection_Ascending;
            if (spec.ColumnIndex == 0)
            {
                std::stable_sort(m_sortOrder.begin(), m_sortOrder.end(),
                    [&](uint32_t a, uint32_t b)
                    {
                        const int c = m_names[a].compare(m_names[b]);
                        if (c != 0) return asc ? (c < 0) : (c > 0);
                        return asc ? (a < b) : (a > b);  // unnamed / ties: index order
                    });
            }
            else
            {
                std::vector<double> keys(clusterLodGeoms.size());
                for (uint32_t i = 0; i < uint32_t(clusterLodGeoms.size()); ++i)
                {
                    const LodAgg a = geomAgg(i);
                    switch (spec.ColumnIndex)
                    {
                        case 1:  keys[i] = a.totDeviceBytes ? double(a.resDeviceBytes) / double(a.totDeviceBytes)
                                                            : (a.pinnedClusters ? 1.0 : 0.0); break;
                        case 2:  keys[i] = double(a.resDeviceBytes); break;
                        case 3:  keys[i] = double(a.totBytes);    break;  // Disk Size (total)
                        case 4:  keys[i] = double(a.clasBytes);   break;
                        case 5:  keys[i] = double(a.resGroups);   break;
                        case 6:  keys[i] = double(a.resClusters); break;
                        default: keys[i] = double(a.resTris);     break;
                    }
                }
                std::stable_sort(m_sortOrder.begin(), m_sortOrder.end(),
                    [&](uint32_t a, uint32_t b)
                    { return asc ? (keys[a] < keys[b]) : (keys[a] > keys[b]); });
            }
            sortSpecs->SpecsDirty = false;
        }

        // ---- Grand-totals row.  Submitted OUTSIDE the clipper as the
        // first data row so the ScrollY freeze keeps it visible.  These
        // sums should line up with the Memory / Streaming tab stats.
        {
            LodAgg grand;
            for (uint32_t i = 0; i < uint32_t(clusterLodGeoms.size()); ++i)
            {
                const LodAgg s = geomAgg(i);
                grand.resGroups += s.resGroups; grand.totGroups += s.totGroups;
                grand.resClusters += s.resClusters; grand.totClusters += s.totClusters;
                grand.resTris += s.resTris; grand.totTris += s.totTris;
                grand.resBytes += s.resBytes; grand.totBytes += s.totBytes;
                grand.resDeviceBytes += s.resDeviceBytes; grand.totDeviceBytes += s.totDeviceBytes;
                grand.clasBytes += s.clasBytes;
            }
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::TextUnformatted("Total");
            ImGui::TableNextColumn();
            {
                const float frac = grand.totDeviceBytes
                    ? float(double(grand.resDeviceBytes) / double(grand.totDeviceBytes)) : 0.f;
                char ov[16]; snprintf(ov, sizeof(ov), "%.0f%%", frac * 100.f);
                ImGui::ProgressBar(frac, ImVec2(-FLT_MIN, 0.f), ov);
            }
            ImGui::TableNextColumn(); rt(double(grand.resDeviceBytes), double(grand.totDeviceBytes), true);
            ImGui::TableNextColumn(); TextMem(double(grand.totBytes));  // Disk Size (no resident numerator)
            ImGui::TableNextColumn(); if (m_haveClas) TextMem(double(grand.clasBytes)); else ImGui::TextDisabled("-");
            ImGui::TableNextColumn(); rt(double(grand.resGroups),   double(grand.totGroups),   false);
            ImGui::TableNextColumn(); rt(double(grand.resClusters), double(grand.totClusters), false);
            ImGui::TableNextColumn(); rt(double(grand.resTris),     double(grand.totTris),     false);
        }

        // Flattened row list for ImGuiListClipper: one row per geometry
        // plus, for expanded ones, a row per LOD.  Every row is
        // frame-height (each has a ProgressBar cell), which the clipper's
        // uniform-height assumption requires.  A row toggled this frame
        // is reflected next frame.
        struct RowRef { uint32_t geom; int32_t lod; };  // lod: -1 geometry, >=0 LOD
        std::vector<RowRef> rows;
        rows.reserve(clusterLodGeoms.size());
        int selectedRow = -1;
        for (const uint32_t i : m_sortOrder)
        {
            // Scroll target: the picked LOD row when one is selected
            // and visible, else the geometry row.
            if (i == m_selectedGeom && (m_selectedLod < 0 || !m_openGeom[i]))
                selectedRow = int(rows.size());
            rows.push_back({ i, -1 });
            if (m_openGeom[i])
                for (uint32_t L = 0; L < m_lodOffset[i + 1] - m_lodOffset[i]; ++L)
                {
                    if (i == m_selectedGeom && int32_t(L) == m_selectedLod)
                        selectedRow = int(rows.size());
                    rows.push_back({ i, int32_t(L) });
                }
        }

        ImGuiListClipper clipper;
        clipper.Begin(int(rows.size()));
        // Force the picked row through the clipper so SetScrollHereY
        // can target it even while it's scrolled out of view.
        if (m_scrollToSelected && selectedRow >= 0)
            clipper.IncludeItemByIndex(selectedRow);
        while (clipper.Step())
        {
            for (int n = clipper.DisplayStart; n < clipper.DisplayEnd; ++n)
            {
                const RowRef row = rows[n];
                if (row.lod == -1)
                {
                    const uint32_t i = row.geom;
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::PushID(int(i));
                    // NoTreePushOnOpen: the per-LOD rows are emitted as
                    // independent clipped rows, not nested under this node.
                    ImGui::SetNextItemOpen(m_openGeom[i] != 0);
                    const ImGuiTreeNodeFlags nodeFlags =
                        ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_NoTreePushOnOpen |
                        (i == m_selectedGeom ? ImGuiTreeNodeFlags_Selected : 0);
                    const char* nm = m_names[i].empty() ? nullptr : m_names[i].c_str();
                    const bool open = nm ? ImGui::TreeNodeEx("##g", nodeFlags, "%s", nm)
                                         : ImGui::TreeNodeEx("##g", nodeFlags, "Geometry %u", i);
                    if (ImGui::IsItemClicked())
                    {
                        // Clicking the selected row again deselects.
                        const bool wasSelected = (m_selectedGeom == i && m_selectedLod < 0);
                        m_selectedGeom   = wasSelected ? ~0u : i;
                        m_selectedLod    = -1;
                        m_pickedInstance = ~0u;
                    }
                    if (m_scrollToSelected && i == m_selectedGeom &&
                        (m_selectedLod < 0 || !m_openGeom[i]))
                    {
                        ImGui::SetScrollHereY(0.35f);
                        m_scrollToSelected = false;
                    }
                    ImGui::PopID();
                    m_openGeom[i] = open ? 1 : 0;
                    lodColumns(geomAgg(i));
                }
                else
                {
                    ImGui::TableNextRow();
                    const bool selRow = (row.geom == m_selectedGeom &&
                                         row.lod == m_selectedLod);
                    if (selRow)
                        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0,
                                               ImGui::GetColorU32(ImGuiCol_Header));
                    ImGui::TableNextColumn();
                    ImGui::Indent();
                    // Narrows the highlight to this LOD across all
                    // instances; re-clicking falls back to the geometry.
                    {
                        char lodLbl[16];
                        snprintf(lodLbl, sizeof(lodLbl), "LOD %d", row.lod);
                        ImGui::PushID(n);
                        if (ImGui::Selectable(lodLbl, false))
                        {
                            m_selectedGeom   = row.geom;
                            m_selectedLod    = selRow ? -1 : row.lod;
                            m_pickedInstance = ~0u;
                        }
                        ImGui::PopID();
                    }
                    ImGui::Unindent();
                    if (m_scrollToSelected && selRow)
                    {
                        ImGui::SetScrollHereY(0.35f);
                        m_scrollToSelected = false;
                    }
                    lodColumns(lodAgg(row.geom, uint32_t(row.lod), CachedBlasLevel(row.geom)));
                }
            }
        }
        ImGui::EndTable();
    }
}

void GeometryInspector::DrawSubdTable()
{
    const auto& subdStats = stats::evaluatorSamplers.surfaceTableStats;

    m_subdHeaderOpen = ImGui::CollapsingHeader("Subdivision Meshes", ImGuiTreeNodeFlags_DefaultOpen);
    if (!m_subdHeaderOpen)
        return;
    ImPlot::SetCurrentContext(m_implot);

    if (m_openSubd.size() != subdStats.size())
        m_openSubd.assign(subdStats.size(), 0);

    ImGui::BeginChild("##subd_section", ImVec2(0.f, 0.f));

    if (ImGui::BeginTable("##subd_geom", 10, kFlags))
    {
        ImGui::TableSetupColumn("Mesh");
        ImGui::TableSetupColumn("Memory");
        ImGui::TableSetupColumn("Surfaces");
        ImGui::TableSetupColumn("BSpline");
        ImGui::TableSetupColumn("Regular");
        ImGui::TableSetupColumn("Smooth");
        ImGui::TableSetupColumn("Sharp");
        ImGui::TableSetupColumn("Holes");
        ImGui::TableSetupColumn("Val max");
        ImGui::TableSetupColumn("Shp max");
        ImGui::TableHeadersRow();

        ImGuiListClipper clipper;
        clipper.Begin(int(subdStats.size()));
        while (clipper.Step())
        {
            for (int n = clipper.DisplayStart; n < clipper.DisplayEnd; ++n)
            {
                const uint32_t i = uint32_t(n);
                const auto& s = subdStats[i];
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::PushID((int)i);
                const bool selected = (m_selectedSubd == i);
                if (ImGui::Selectable(s.name.empty() ? "(unnamed)" : s.name.c_str(),
                                      selected, ImGuiSelectableFlags_SpanAllColumns))
                {
                    m_selectedSubd = selected ? ~0u : i;
                    m_selectedGeom = ~0u;
                    m_selectedLod  = -1;
                    m_pickedInstance = ~0u;
                }
                ImGui::PopID();
                ImGui::TableNextColumn(); TextMem(double(s.byteSize));
                ImGui::TableNextColumn(); TextHuman(double(s.surfaceCount));
                ImGui::TableNextColumn(); TextHuman(double(s.bsplineSurfaceCount));
                ImGui::TableNextColumn(); TextHuman(double(s.regularSurfaceCount));
                ImGui::TableNextColumn(); TextHuman(double(s.isolationSurfaceCount));
                ImGui::TableNextColumn(); TextHuman(double(s.sharpSurfaceCount));
                ImGui::TableNextColumn(); TextHuman(double(s.holesCount));
                ImGui::TableNextColumn(); ImGui::Text("%u", s.maxValence);
                ImGui::TableNextColumn(); ImGui::Text("%.2f", s.sharpnessMax);
            }
        }
        ImGui::EndTable();
    }

    ImGui::EndChild();  // ##subd_section
}

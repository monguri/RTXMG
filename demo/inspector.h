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
#include <string>
#include <vector>

#include "rtxmg/cluster_lod/streaming_hooks.h"
#include "rtxmg/scene/scene.h"
#include "rtxmg/utils/pixel_pick.h"

struct ImFont;
struct ImPlotContext;
class RTXMGDemoApp;

// ---------------------------------------------------------------------------
// Geometry Inspector — a toggleable window listing every geometry mesh in the
// scene, with type-dependent columns (subdivision surface vs cluster-LOD) and
// per-LOD residency detail.  All data is host-side; no GPU readback.
//
// City-scale scenes have thousands of geometries and ~1M groups, so nothing
// here may cost O(total groups) per frame:
//   * The immutable per-(geometry, LOD) totals and display names are swept once
//     per scene, keyed on the geometry array identity.
//   * Residency aggregates come from the streaming resident-group list —
//     O(resident) — and only on frames where the residency epoch moved.  The
//     pinned-vs-streamable split is positional (the always-resident low-detail
//     prefix leads that list), not a per-group set lookup.
//   * Rows go through ImGuiListClipper, which needs each tree row's open state
//     without submitting it — hence expansion lives here rather than in ImGui's
//     per-window tree-node storage.
// ---------------------------------------------------------------------------
class GeometryInspector
{
public:
    // `implot` itself is unusable as an identifier here: FetchImplot.cmake's
    // add_compile_definitions leaks it as a macro.
    GeometryInspector(RTXMGDemoApp& app, ImFont* iconicFont, ImPlotContext* implotContext)
        : m_app(app), m_iconicFont(iconicFont), m_implot(implotContext)
    {
    }

    // The viewport highlight and the pick handler run even while hidden; `show`
    // is set by a fresh viewport pick and cleared by the window's close button.
    void Draw(bool& show);

private:
    void RefreshStaticTotals(const RTXMGScene& scene);
#if ENABLE_PIXEL_PICK
    void ConsumePick();
#endif
    void RefreshResidency();
    void DrawSelectionSummary(const RTXMGScene& scene);
    void DrawClodTable(bool haveSubd);
    void DrawSubdTable();

    // 0 until the backend's first CLAS-sizes readback lands, and on any backend
    // that has no persistent CLAS allocator.
    uint64_t ResidentClasBytes(uint32_t geometryID, uint32_t lodLevel) const
    {
        return (geometryID < m_residency.residentClasBytes.size() &&
                lodLevel < shaderio::kMaxLodLevels)
                   ? m_residency.residentClasBytes[geometryID][lodLevel]
                   : 0ull;
    }
    uint32_t CachedBlasLevel(uint32_t geometryID) const
    {
        return geometryID < m_residency.cachedBlasLevels.size()
                   ? m_residency.cachedBlasLevels[geometryID]
                   : uint32_t(shaderio::kTraversalInvalidLodLevel);
    }

    RTXMGDemoApp&  m_app;
    ImFont*        m_iconicFont = nullptr;
    ImPlotContext* m_implot     = nullptr;

    // Per-frame context, republished at the top of every cluster-LOD section
    // draw and read by the Refresh/Draw steps below it.
    const std::vector<GeometryView>* m_geoms          = nullptr;
    IClusterLodStreamingHooks*       m_streamingHooks = nullptr;
    // The residency snapshot this frame draws from; kept across frames so its
    // allocations are reused and the backend can skip the O(resident) sweep.
    IClusterLodStreamingHooks::ResidencyReport m_residency;
    bool m_residencyGroupsRebuilt = false;
    bool m_haveResidency       = false;
    bool m_stripPositions      = false;
    bool m_stripNormals        = false;
    bool m_haveClas            = false;
    bool m_residencyRefreshed  = false;

    // Identity of the scene's geometry array the static caches were built
    // from; the caches are rebuilt when any component changes (scene switch).
    const void* m_geomsKey      = nullptr;
    const void* m_groupInfosKey = nullptr;
    size_t      m_geomCount     = 0;

    // Immutable per-(geometry, LOD) totals, flattened; lodOffset[i] indexes
    // geometry i's first LOD entry (geomCount + 1 entries).
    struct LodStatic
    {
        uint32_t totGroups = 0, totClusters = 0;
        uint64_t totTris = 0, totBytes = 0;      // totBytes = disk stored size (gi.sizeBytes)
        uint64_t totDeviceBytes = 0;             // resident-pool size (positions/normals
                                                 // stripped, UVs quantized) if fully resident
        // Per-attribute disk (uncompressed baked layout), summed over all groups
        // of this LOD — the denominators for the per-channel residency bars.
        uint64_t totPosBytes = 0, totNrmBytes = 0, totUvBytes = 0;
        bool     quantUv = false;     // any cluster uses po2-grid quantized UVs
        bool     compressed = false;  // any group is arithmetic-compressed on disk
    };
    std::vector<LodStatic>   m_lodStatic;
    std::vector<uint32_t>    m_lodOffset;
    // Strip config the static totDeviceBytes was built under; a runtime toggle
    // (e.g. Vertex Normals) changes the resident-pool sizing, so the static
    // sweep must rebuild when either flag moves.
    bool  m_staticStripPositions = false;
    bool  m_staticStripNormals   = false;
    // Per-geometry display name ("" = none): the FIRST instance's node name,
    // suffixed with " (xN)" when N > 1 instances share the geometry — the
    // importer dedups meshes by accessor identity, so several differently
    // named glTF nodes/meshes can legitimately map to one geometry.
    std::vector<std::string> m_names;
    std::vector<uint32_t>    m_instCount;  // instances referencing each geometry

    // Residency aggregates, same layout as lodStatic.  Members, not locals, so
    // the allocations are reused; lodDynHooks records which backend they were
    // built from, so a rebuilt one cannot reuse an old epoch value.
    struct LodDyn
    {
        uint32_t resGroups = 0, resClusters = 0;
        uint64_t resTris = 0, resBytes = 0;        // resBytes = disk stored size of resident groups
        uint64_t resDeviceBytes = 0;               // actual resident-pool bytes (stripped)
        uint64_t pinnedDeviceBytes = 0;            // resident-pool bytes of the pinned prefix
        // Per-attribute RESIDENT bytes.  Positions/normals report 0 while
        // stripped; what the unstripped low-detail root keeps is folded into the
        // Total's structural remainder rather than shown as a channel.
        uint64_t resPosBytes = 0, resNrmBytes = 0, resUvBytes = 0;
        uint32_t streamResClusters = 0;  // resident clusters not pinned low-detail
        uint32_t lowDetailClusters = 0;  // clusters in the always-resident prefix
        bool     hasLowDetail = false;
    };
    std::vector<LodDyn> m_lodDyn;
    const void*         m_lodDynHooks = nullptr;

    // Tree-row expansion (cluster-LOD geometry rows / subd surface rows).
    std::vector<uint8_t> m_openGeom;
    std::vector<uint8_t> m_openSubd;

    // Geometry-row display order (indices into clusterLodGeoms) per the table's
    // active header sort spec; re-sorted when the spec changes or the
    // residency aggregates refresh (keys are residency-dependent).
    std::vector<uint32_t> m_sortOrder;

    // Selected geometry row (viewport right-click or row click) and, for
    // viewport picks, the hit cluster's LOD row within it (-1 = geometry-level).
    // handledPickSeq is the last PixelPick::sequence consumed.
    uint32_t m_selectedGeom     = ~0u;
    int32_t  m_selectedLod      = -1;
    // Viewport picks only: the clicked instance + its resolved material
    // (shown in the selection caption; cleared by row-click selection).
    uint32_t    m_pickedInstance   = ~0u;
    uint32_t    m_pickedMaterialID = ~0u;
    std::string m_pickedMaterialName;
    bool     m_scrollToSelected = false;
    uint32_t m_handledPickSeq   = 0;
    bool     m_highlightSelection = true;
    uint32_t m_selectedSubd       = ~0u;
    bool     m_subdHeaderOpen     = true;
};

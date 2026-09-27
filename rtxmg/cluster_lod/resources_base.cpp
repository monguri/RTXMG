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

#include "rtxmg/cluster_lod/resources_base.h"

#include <donut/core/log.h>
#include <donut/engine/DescriptorTableManager.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using namespace donut::log;

ClusterLodPrebuiltGeometryMetadata ClusterLodPrebuiltGeometryMetadata::Build(
    const GeometryView&                     geom,
    donut::engine::DescriptorTableManager*  descriptorTable,
    nvrhi::IDevice*                         device,
    nvrhi::ICommandList*                    commandList)
{
    ClusterLodPrebuiltGeometryMetadata out;
    BaseGeometry&                base = out.base;

    // ---- LOD tree buffers ------------------------------------------------
    {
        const size_t numNodes     = geom.lodNodes.size();
        const size_t numLodLevels = geom.lodLevelsCount;

        base.lodLevels.Create(numLodLevels, "ClusterLodLodLevels", device);
        base.lodNodes.Create(numNodes,       "ClusterLodLodNodes",  device);
        base.lodNodeBboxes.Create(numNodes,  "ClusterLodLodNodeBboxes", device);

        commandList->writeBuffer(base.lodLevels,     geom.lodLevels.data(),     geom.lodLevels.size_bytes());
        commandList->writeBuffer(base.lodNodes,      geom.lodNodes.data(),      geom.lodNodes.size_bytes());
        commandList->writeBuffer(base.lodNodeBboxes, geom.lodNodeBboxes.data(), geom.lodNodeBboxes.size_bytes());
    }

    // Measure LOD node tree depth (root at index 0, leaves are isGroup=1).
    // The multipass traversal runs one traversal_run dispatch per depth.
    if (!geom.lodNodes.empty())
    {
        std::vector<std::pair<uint32_t, uint32_t>> stack;
        stack.emplace_back(0u, 1u);
        uint32_t geomDepth = 1;
        while (!stack.empty())
        {
            auto [nodeIdx, depth] = stack.back();
            stack.pop_back();
            if (nodeIdx >= geom.lodNodes.size())
                continue;
            geomDepth = std::max(geomDepth, depth);
            const shaderio::Node& nd = geom.lodNodes[nodeIdx];
            if (nd.nodeRange.isGroup != 0)
                continue;
            uint32_t childCount  = nd.nodeRange.childCountMinusOne + 1;
            uint32_t childOffset = nd.nodeRange.childOffset;
            for (uint32_t i = 0; i < childCount; ++i)
                stack.emplace_back(childOffset + i, depth + 1);
        }
        out.nodeTreeDepth = geomDepth;
    }

    // ---- Per-group address table -------------------------------------------
    {
        const size_t numGroups = geom.groupInfos.size();

        base.streamingGroupAddresses.Create(std::max(numGroups, size_t(1)),
                                            "ClusterLodStreamingGroupAddresses", device);

        // Every group starts non-resident (invalid blob SRV); derived classes
        // patch real addresses as groups stream in (preload fills all at Init).
        std::vector<shaderio::GroupAddress> groupAddresses(
            numGroups, shaderio::GroupAddress{ shaderio::kStreamingInvalidSrvIndex, 0u });

        if (numGroups > 0)
            commandList->writeBuffer(base.streamingGroupAddresses, groupAddresses.data(),
                                     numGroups * sizeof(shaderio::GroupAddress));

        // Culling reads the node-level lodNodeBboxes; flattening a per-cluster
        // bbox table on top of that cost seconds of scene load at city scale.
    }

    // ---- Bindless registrations ---------------------------------------------
    {
        auto registerSRV = [&](nvrhi::IBuffer* buf,
                               donut::engine::DescriptorHandle& outHandle) -> uint32_t
        {
            outHandle = descriptorTable->CreateDescriptorHandle(
                nvrhi::BindingSetItem::StructuredBuffer_SRV(0, buf));
            return static_cast<uint32_t>(outHandle.GetIndexInHeap());
        };
        auto registerUAV = [&](nvrhi::IBuffer* buf,
                               donut::engine::DescriptorHandle& outHandle) -> uint32_t
        {
            outHandle = descriptorTable->CreateDescriptorHandle(
                nvrhi::BindingSetItem::StructuredBuffer_UAV(0, buf));
            return static_cast<uint32_t>(outHandle.GetIndexInHeap());
        };

        out.lodLevelsSRV      = registerSRV(base.lodLevels,     base.lodLevelsSRVHandle);
        out.nodesSRV          = registerSRV(base.lodNodes,      base.nodesSRVHandle);
        out.nodeBboxesSRV     = registerSRV(base.lodNodeBboxes, base.nodeBboxesSRVHandle);
        out.groupAddressesSRV = registerSRV(base.streamingGroupAddresses,
                                            base.streamingGroupAddressesSRVHandle);
        out.groupAddressesUAV = registerUAV(base.streamingGroupAddresses,
                                            base.streamingGroupAddressesUAVHandle);
    }

    return out;
}

void ClusterLodResourcesBase::UploadGeometryMetadata(
    size_t                                  geomIndex,
    const GeometryView&                     geom,
    BaseGeometry&                           outBase,
    shaderio::Geometry&                     outShaderGeom,
    uint32_t                                instancesOffset,
    uint32_t                                instancesCount,
    donut::engine::DescriptorTableManager*  descriptorTable,
    nvrhi::IDevice*                         device,
    nvrhi::ICommandList*                    commandList)
{
    // Reuse scene-side metadata built on the load thread rather than rebuilding
    // it on the render thread; placeholder entries fall through to Build().
    const ClusterLodPrebuiltGeometryMetadata* pre =
        (m_prebuiltMetadata && geomIndex < m_prebuiltMetadata->size() &&
         (*m_prebuiltMetadata)[geomIndex].IsValid())
            ? &(*m_prebuiltMetadata)[geomIndex]
            : nullptr;

    uint32_t lodLevelsSRV, nodesSRV, nodeBboxesSRV, groupAddressesSRV, groupAddressesUAV;
    uint32_t nodeTreeDepth;
    if (pre)
    {
        outBase.lodLevels               = pre->base.lodLevels;
        outBase.lodNodes                = pre->base.lodNodes;
        outBase.lodNodeBboxes           = pre->base.lodNodeBboxes;
        outBase.streamingGroupAddresses = pre->base.streamingGroupAddresses;
        // Descriptor handles stay scene-owned; outBase's handles remain empty.
        lodLevelsSRV      = pre->lodLevelsSRV;
        nodesSRV          = pre->nodesSRV;
        nodeBboxesSRV     = pre->nodeBboxesSRV;
        groupAddressesSRV = pre->groupAddressesSRV;
        groupAddressesUAV = pre->groupAddressesUAV;
        nodeTreeDepth     = pre->nodeTreeDepth;
    }
    else
    {
        ClusterLodPrebuiltGeometryMetadata built =
            ClusterLodPrebuiltGeometryMetadata::Build(geom, descriptorTable, device, commandList);
        lodLevelsSRV      = built.lodLevelsSRV;
        nodesSRV          = built.nodesSRV;
        nodeBboxesSRV     = built.nodeBboxesSRV;
        groupAddressesSRV = built.groupAddressesSRV;
        groupAddressesUAV = built.groupAddressesUAV;
        nodeTreeDepth     = built.nodeTreeDepth;
        outBase           = std::move(built.base);  // handles owned by outBase
    }

    m_maxNodeTreeDepth = std::max(m_maxNodeTreeDepth, nodeTreeDepth);

    // ---- Debug topology dump ------------------------------------------------
    {
        if (m_debugClusterLod)
        {
            info("ClusterLod topology geom[%zu]", geomIndex);
            info("  nodes=%zu levels=%zu groups=%zu clusters=%u treeDepth=%u",
                 geom.lodNodes.size(), geom.lodLevelsCount, geom.groupInfos.size(),
                 geom.totalClustersCount,
                 !geom.lodNodes.empty() ? nodeTreeDepth : 0u);

            for (size_t l = 0; l < geom.lodLevelsCount; ++l)
            {
                const shaderio::LodLevel& lv = geom.lodLevels[l];
                info("  L%zu: groups=[%u..%u) clusters=[%u..%u) minBSR=%.4f minMQE=%.4f",
                     l,
                     lv.groupOffset, lv.groupOffset + lv.groupCount,
                     lv.clusterOffset, lv.clusterOffset + lv.clusterCount,
                     lv.minBoundingSphereRadius, lv.minMaxQuadricError);
            }

            if (!geom.lodNodes.empty())
            {
                const shaderio::Node& root = geom.lodNodes[0];
                uint32_t rootIsGroup = root.nodeRange.isGroup;
                uint32_t rootOff     = root.nodeRange.childOffset;
                uint32_t rootCount   = root.nodeRange.childCountMinusOne + 1u;
                info("  root: isGroup=%u childOffset=%u childCount=%u metric.error=%.6g sphere=(%.3f,%.3f,%.3f,r=%.3f)",
                     rootIsGroup, rootOff, rootCount,
                     root.traversalMetric.maxQuadricError,
                     root.traversalMetric.boundingSphereX,
                     root.traversalMetric.boundingSphereY,
                     root.traversalMetric.boundingSphereZ,
                     root.traversalMetric.boundingSphereRadius);
                if (rootIsGroup == 0u)
                {
                    for (uint32_t c = 0; c < rootCount && c < 16u; ++c)
                    {
                        uint32_t ci = rootOff + c;
                        if (ci >= geom.lodNodes.size()) break;
                        const shaderio::Node& ch = geom.lodNodes[ci];
                        uint32_t chIsGroup = ch.nodeRange.isGroup;
                        info("    root.child[%u] idx=%u isGroup=%u %s=%u %s=%u metric.error=%.6g r=%.3f",
                             c, ci, chIsGroup,
                             chIsGroup ? "groupIndex"    : "childOffset",
                             chIsGroup ? (uint32_t)ch.groupRange.groupIndex              : (uint32_t)ch.nodeRange.childOffset,
                             chIsGroup ? "clusterCount"  : "childCount",
                             chIsGroup ? (uint32_t)ch.groupRange.groupClusterCountMinusOne + 1u
                                        : (uint32_t)ch.nodeRange.childCountMinusOne + 1u,
                             ch.traversalMetric.maxQuadricError,
                             ch.traversalMetric.boundingSphereRadius);
                    }
                }
            }

            size_t logGroups = std::min<size_t>(geom.groupInfos.size(), 4u);
            for (size_t gi = 0; gi < logGroups; ++gi)
            {
                const GroupInfo& gInfo = geom.groupInfos[gi];
                GroupView gView(geom.groupData, gInfo);
                const shaderio::Group& gHdr = *gView.group;
                info("  group[%zu]: clusterCount=%u lodLevel=%u clusterResidentID(hdr)=%u metric.error=%.6g r=%.3f",
                     gi, (uint32_t)gInfo.clusterCount, (uint32_t)gHdr.lodLevel, (uint32_t)gHdr.clusterResidentID,
                     gHdr.traversalMetric.maxQuadricError,
                     gHdr.traversalMetric.boundingSphereRadius);
            }
        }
    }

    // ---- Fill BASE fields of shaderio::Geometry -----------------------------
    {
        // The importer drops geometries with no LOD hierarchy, so this can only
        // trip if one reached the scene by another route.
        assert(geom.lodLevelsCount != 0 && geom.lodLevelsCount <= geom.lodLevels.size());
        const shaderio::LodLevel& lastLevel =
            geom.lodLevels[geom.lodLevelsCount - 1];
        // The low-detail path makes one always-resident group per geometry, so a
        // multi-group root is unsupported (a root group with >1 cluster is fine).
        // Warn, don't assert: one bad geometry shouldn't abort the scene load.
        if (lastLevel.groupCount != 1)
        {
            warning("ClusterLod geom[%zu]: LOD root has %u groups (expected 1), "
                         "clusters=%u — only the first root group is used as low-detail "
                         "(coarsest LoD may be incomplete). Likely a baking issue.",
                         geomIndex, lastLevel.groupCount, lastLevel.clusterCount);
        }

        outShaderGeom = {};
        outShaderGeom.instancesOffset    = instancesOffset;
        outShaderGeom.instancesCount     = instancesCount;
        outShaderGeom.lodLevelsCount     = static_cast<uint32_t>(geom.lodLevelsCount);
        outShaderGeom.lowDetailClusterID = lastLevel.clusterOffset;
        outShaderGeom.lowDetailTriangles =
            static_cast<uint16_t>(geom.groupInfos[lastLevel.groupOffset].triangleCount);
        outShaderGeom.lowDetailBlasAddress = 0;    // filled by derived class' CLAS-build
        outShaderGeom.cachedBlasLodLevel   = shaderio::kTraversalInvalidLodLevel;
        outShaderGeom.cachedBlasAddress    = 0;
        outShaderGeom.cachedBlasTriangles  = 0;  // render-stats; set per-frame by stream_update_scene
        outShaderGeom.cachedBlasClusters   = 0;
        outShaderGeom.bbox                 = geom.bbox;

        outShaderGeom.lodLevelsSRV   = lodLevelsSRV;
        outShaderGeom.nodesSRV       = nodesSRV;
        outShaderGeom.nodeBboxesSRV  = nodeBboxesSRV;

        outShaderGeom.streamingGroupAddressesSRV = groupAddressesSRV;
        outShaderGeom.streamingGroupAddressesUAV = groupAddressesUAV;

        // Read the scene's offset for this geometry rather than re-deriving it, so
        // there is no unenforced contract that this runs in the same order as the
        // loop that packs the flat buffer.
        outShaderGeom.materialBaseID       = m_clusterLodMaterialBaseID;
        outShaderGeom.localMaterialsOffset =
            (m_clusterLodLocalMaterialsOffsets && geomIndex < m_clusterLodLocalMaterialsOffsets->size())
                ? (*m_clusterLodLocalMaterialsOffsets)[geomIndex]
                : 0u;
        // count 0 (--nomat) takes ResolveMaterialIDFromLocal's "no indirection"
        // branch, collapsing every geometry onto the default-gray materialBaseID.
        outShaderGeom.localMaterialsCount  = m_enableMaterials
                                                 ? static_cast<uint32_t>(geom.localMaterialIDs.size())
                                                 : 0u;
    }
}

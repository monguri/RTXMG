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

/*
  Shader Description
  ==================

  Single-thread utility shader dispatched between traversal passes to clamp
  counters, advance the multipass window, and refresh indirect-dispatch args.
  Every clamp also records its pre-clamp demand in a desired* counter, which is
  the only place overflow can be observed once the clamp has run.

  BuildSetup::TraversalRun (mode 1):
    Called ONCE between traversal_init and the first traversal_run dispatch.
    Clamps traversalNodeWriteCounter to maxTraversalInfos and initialises the
    multipass range:
      traversalNodeStart = 0
      traversalNodeEnd   = clamped write counter
      traversalGroupStart/End = 0
    Computes indirectDispatchNodes so the first traversal_run pass dispatches
    one thread per seeded root.

  BuildSetup::TraversalRunPassCombined (mode 2):
    Called after the final traversal_run pass. Same as
    TraversalRunPassNodesOnly for the node window, plus snapshots the
    group window
      traversalGroupStart = previous End
      traversalGroupEnd   = current write counter
    so the subsequent traversal_run_groups dispatch processes all accumulated
    leaf groups.

  BuildSetup::TraversalRunPassNodesOnly (mode 3):
    Called after a non-final traversal_run pass. Bumps the node window to
    consume the nodes just appended (traversalNodeStart = previous End,
    traversalNodeEnd = current write counter) and clears the group window so
    the paired traversal_run_groups dispatch is a no-op on this pass.

  BuildSetup::BlasInsertion (mode 5):
    Called after traversal_run_groups. Clamps renderClusterCounter to
    maxRenderClusters, writes the clamped count into numRenderedClusters,
    and fills indirectDispatchBlasInsertion for blas_insert_clusters.
*/

#pragma pack_matrix(row_major)

#include "traversal_common.hlsli"
#include <donut/shaders/binding_helpers.hlsli>

// shaderio::BuildSetup mode IDs are defined in shaderio.h (shared with C++).

// Setup mode push constant. b0 is SceneBuildingConstants in traversal_common.
// DECLARE_PUSH_CONSTANTS emits [[vk::push_constant]] on Vulkan (a real push-
// constant range, which nvrhi's PushConstants layout item maps to) and a plain
// register(b1) cbuffer on D3D12.  The static alias keeps the bare `setup` use
// sites working.
struct SetupPushData
{
    uint setup;
};
DECLARE_PUSH_CONSTANTS(SetupPushData, g_Push, 1, 0);
static const shaderio::BuildSetup setup = (shaderio::BuildSetup)g_Push.setup;

[numthreads(1, 1, 1)]
void main()
{
    if (setup == shaderio::BuildSetup::TraversalRun)
    {
        // Stash the raw demand before the in-place clamp erases it.  Not behind
        // TRACK_RENDER_STATS: the harness gates on this via --dump-stats.
        uint rawNodes  = u_Counters[0].traversalNodeWriteCounter;
        u_Counters[0].desiredTraversalNodes = rawNodes;

        uint nodeCount = min(rawNodes, g_Constants.maxTraversalInfos);
        u_Counters[0].traversalNodeWriteCounter = nodeCount;

        // Multipass: first pass reads [0, nodeCount); group window starts empty.
        u_Counters[0].traversalPass        = 0u;
        u_Counters[0].traversalNodeStart   = 0u;
        u_Counters[0].traversalNodeEnd     = nodeCount;
        u_Counters[0].traversalGroupStart  = 0u;
        u_Counters[0].traversalGroupEnd    = 0u;

        uint nodeGrid = (nodeCount + shaderio::kTraversalRunThreads - 1u) / shaderio::kTraversalRunThreads;
        u_Counters[0].indirectDispatchNodesX = nodeGrid;
        u_Counters[0].indirectDispatchNodesY = 1u;
        u_Counters[0].indirectDispatchNodesZ = 1u;

        // Group dispatch is a no-op on pass 0; TraversalRunPassCombined fills it in.
        u_Counters[0].indirectDispatchGroupsX = 0u;
        u_Counters[0].indirectDispatchGroupsY = 1u;
        u_Counters[0].indirectDispatchGroupsZ = 1u;
    }
    else if (setup == shaderio::BuildSetup::TraversalRunPassNodesOnly
          || setup == shaderio::BuildSetup::TraversalRunPassCombined)
    {
        // Advance to the next pass: read slices that were produced by the previous dispatch.
        uint pass = u_Counters[0].traversalPass + 1u;
        u_Counters[0].traversalPass = pass;

        // Peak raw demand across the passes.  The write counters keep counting
        // past the cap here, but TraversalRun clamped the node one in place, so
        // the running max is what preserves a first-pass overflow.
        u_Counters[0].desiredTraversalNodes  = max(u_Counters[0].desiredTraversalNodes,
                                                   u_Counters[0].traversalNodeWriteCounter);
        u_Counters[0].desiredTraversalGroups = max(u_Counters[0].desiredTraversalGroups,
                                                   u_Counters[0].traversalGroupWriteCounter);

        uint nodeStart = min(u_Counters[0].traversalNodeEnd,         g_Constants.maxTraversalInfos);
        uint nodeEnd   = min(u_Counters[0].traversalNodeWriteCounter, g_Constants.maxTraversalInfos);
        u_Counters[0].traversalNodeStart = nodeStart;
        u_Counters[0].traversalNodeEnd   = nodeEnd;
        uint nodeCount = nodeEnd - nodeStart;

        uint groupCount = 0u;
        if (setup == shaderio::BuildSetup::TraversalRunPassCombined)
        {
            uint groupStart = min(u_Counters[0].traversalGroupEnd,         g_Constants.maxTraversalInfos);
            uint groupEnd   = min(u_Counters[0].traversalGroupWriteCounter, g_Constants.maxTraversalInfos);
            u_Counters[0].traversalGroupStart = groupStart;
            u_Counters[0].traversalGroupEnd   = groupEnd;
            groupCount = groupEnd - groupStart;
        }
        else
        {
            // TraversalRunPassNodesOnly: no group dispatch this pass.
            u_Counters[0].traversalGroupStart = 0u;
            u_Counters[0].traversalGroupEnd   = 0u;
        }

        uint nodeGrid  = (nodeCount  + shaderio::kTraversalRunThreads    - 1u) / shaderio::kTraversalRunThreads;
        uint groupGrid = (groupCount + shaderio::kTraversalGroupsThreads - 1u) / shaderio::kTraversalGroupsThreads;

        u_Counters[0].indirectDispatchNodesX = nodeGrid;
        u_Counters[0].indirectDispatchNodesY = 1u;
        u_Counters[0].indirectDispatchNodesZ = 1u;

        u_Counters[0].indirectDispatchGroupsX = groupGrid;
        u_Counters[0].indirectDispatchGroupsY = 1u;
        u_Counters[0].indirectDispatchGroupsZ = 1u;
    }
    else if (setup == shaderio::BuildSetup::BlasInsertion)
    {
        uint rawCount = u_Counters[0].renderClusterCounter;
        uint clamped  = min(rawCount, g_Constants.maxRenderClusters);

        // The raw (pre-clamp) demand and the effective cap let the host detect
        // overflow (rawCount > maxRenderClusters => clusters dropped this
        // frame).  Not behind TRACK_RENDER_STATS: it is an overflow signal the
        // harness gates on, not a display counter.
        u_Counters[0].desiredRenderClusters     = rawCount;
        u_Counters[0].effectiveMaxRenderClusters = g_Constants.maxRenderClusters;

        u_Counters[0].numRenderedClusters   = clamped;
        u_Counters[0].renderClusterCounter  = clamped;

        // Indirect dispatch for the BLAS-insertion pass:
        // one thread group per kBlasInsertThreads clusters.
        uint numGroups = (clamped + shaderio::kBlasInsertThreads - 1u) / shaderio::kBlasInsertThreads;
        u_Counters[0].indirectDispatchBlasInsertionX = numGroups;
        u_Counters[0].indirectDispatchBlasInsertionY = 1u;
        u_Counters[0].indirectDispatchBlasInsertionZ = 1u;
    }
}

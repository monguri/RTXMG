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
  Only used with USE_BLAS_SHARING.

  Classifies the LoD range [lodLevelMin, lodLevelMax] of each instance and
  accumulates it into the geometry's LoD histograms.  Also seeds each instance
  to use the geometry's pre-built low-detail BLAS (may be overridden later in
  instance_assign_blas).  One thread per instance.

  Follow-up pass: blas_elect_sharing_provider.hlsl.

  Notes:
  * geometry.nodes / geometry.lodLevels are read via bindless StructuredBuffer
    at ResourceDescriptorHeap[NonUniformResourceIndex(...SRV)].
  * The LoD histograms are accumulated with InterlockedAdd / InterlockedMax on
    the u_GeometryHistograms UAV.
  * Per-instance results are written to u_InstanceBuildInfos / u_InstanceBlasAddrs
    (see traversal_common.hlsli).
  * geometryLodLevelMax + geometryID are not stored here; both are derived
    (geometry.lodLevelsCount-1, RenderInstance.geometryID) in
    traversal_init_blas_sharing.
  * oViewPos uses RenderInstance.worldMatrixI + g_Constants.viewPos for the
    object-space min-sphere far-push.
  * Frustum culling is a runtime switch (g_Constants.useCulling / useHardCull);
    HiZ occlusion is the CLUSTER_LOD_HIZ_OCCLUSION compile permutation.
*/

#pragma pack_matrix(row_major)

#include "rtxmg/cluster_lod/shaderio.h"
#include "feature_gates.hlsli"
#include "rtxmg/cluster_lod/shaders/traversal_common.hlsli"
#include "rtxmg/cluster_lod/shaders/culling.hlsli"  // frustum culling: IntersectFrustum

// HiZ occlusion pyramid (register space 1) — see traversal_init.hlsl.
#if CLUSTER_LOD_HIZ_OCCLUSION
Texture2D<float> t_ClusterLodHiZ[HIZ_MAX_LODS] : register(t0, space1);
SamplerState     s_ClusterLodHiZSampler        : register(s0, space1);
#endif

// A geometry participates in BLAS sharing only when it has enough instances to
// amortize a shared build — >=1 with caching on, so the cached BLAS can still
// be elected, >=2 otherwise.
bool TestForBlasSharing(shaderio::Geometry geometry)
{
#if USE_BLAS_CACHING
    return geometry.instancesCount >= 1u;
#else
    return geometry.instancesCount >= 2u;
#endif
}

[numthreads(shaderio::kInstanceClassifyLodThreads, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID)
{
    uint instanceID   = gid.x;
    uint instanceLoad = min(g_Constants.numRenderInstances - 1u, instanceID);
    bool isValid      = instanceID == instanceLoad;

    shaderio::RenderInstance instance = t_Instances[instanceLoad];
    uint geometryID = instance.geometryID;
    shaderio::Geometry geometry = t_Geometries[geometryID];

    // Mirrors traversal_init's visibility test on the sharing path: off-screen
    // instances are tagged not-visible so traversal_init_blas_sharing can skip
    // traversing them and traversal_run can attenuate their LoD.  Runs on the
    // clamped instanceLoad data, hence harmless on padding threads.
    bool isVisible = true;
    if (g_Constants.useCulling)
    {
        float4 clipMin, clipMax;
        bool   clipValid;
        bool   inFrustum = IntersectFrustum(g_Constants.cullViewProjMatrix,
                                            geometry.bbox.lo, geometry.bbox.hi,
                                            instance.worldMatrix,
                                            clipMin, clipMax, clipValid);
        bool bigEnough = IntersectSize(clipMin, clipMax, 1.0f, g_Constants.viewportf.xy);
#if CLUSTER_LOD_HIZ_OCCLUSION
        bool notOccluded = IntersectHiz(clipMin, clipMax, t_ClusterLodHiZ, s_ClusterLodHiZSampler,
                                        g_Constants.hizNumLODs, g_Constants.hizInvSize,
                                        g_Constants.viewportf.xy);
#else
        bool notOccluded = true;
#endif
        isVisible = inFrustum && (!clipValid || (bigEnough && notOccluded));
    }
    uint visibilityState = isVisible ? shaderio::InstanceVisibility::Visible : 0u;

    // isValid-guarded because instanceID is unclamped: a padding thread would
    // index out of bounds.
    if (isValid && g_Constants.useCulling)
        u_InstanceVisibility[instanceID] = visibilityState;

    if (isValid)
    {
        StructuredBuffer<shaderio::Node> nodes =
            ResourceDescriptorHeap[NonUniformResourceIndex(geometry.nodesSRV)];
        StructuredBuffer<shaderio::LodLevel> lodLevels =
            ResourceDescriptorHeap[NonUniformResourceIndex(geometry.lodLevelsSRV)];

        // setup evaluation of lod metric
        float3x4 worldMatrix  = instance.worldMatrix;
        float3x4 worldMatrixI = instance.worldMatrixI;
        float    uniformScale = ComputeUniformScale(worldMatrix);

        // A soft-culled instance classifies at coarser LoD, so it is more likely
        // to share or cache.  errorScale feeds the TestForTraversal calls below.
        float errorScale = 1.0f;
        if (g_Constants.useCulling && !g_Constants.useHardCull
            && visibilityState == 0u)
            errorScale = g_Constants.culledErrorScale;

        float4x4 transform = mul(g_Constants.traversalViewMatrix, ToFloat4x4(worldMatrix));
        // camera position in object space (for the min-sphere far-push below)
        float3   oViewPos  = mul(worldMatrixI, float4(g_Constants.viewPos, 1.0f));

        // The geometry's root node contains one child node per lod level.
        uint rootNodePacked     = nodes[0].packed;
        uint childOffset        = Node_nodeChildOffset(rootNodePacked);

        bool geometryUsesBlasSharing = TestForBlasSharing(geometry);
        uint geometryLodLevelMax     = geometry.lodLevelsCount - 1u;

        bool findMin     = true;
        uint lodLevelMin = 0u;
        uint lodLevelMax = geometryLodLevelMax;

        // A hard-culled instance is never traversed (it renders from its
        // persistent low-detail BLAS), so skip its LoD-range classification and
        // histogram: it keeps the default range and doesn't skew the sharing
        // election.  Soft-culled instances still classify, just coarser.
        const bool hardCull = g_Constants.useCulling && g_Constants.useHardCull;
        if (isVisible || !hardCull)
        {
            // lodLevelMin = finest (high detail) level this instance reaches;
            // lodLevelMax = coarsest (low detail) level it still needs.
            // An instance may span several lod levels depending on distance and
            // orientation:  camera -> [lodLevelMin .... lodLevelMax]
            for (uint lodLevel = 0u; lodLevel < geometry.lodLevelsCount; lodLevel++)
            {
                shaderio::Node childNode = nodes[childOffset + lodLevel];
                shaderio::TraversalMetric traversalMetric = childNode.traversalMetric;

                // Max-sphere test (offline-accumulated maximum sphere of the
                // level's groups): the first level that is "coarse enough"
                // marks the highest detail actually rendered.
                if (findMin && TestForTraversal(transform, uniformScale, traversalMetric, errorScale))
                {
                    findMin     = false;
                    lodLevelMin = lodLevel;
                }

                // For sharing we also need lodLevelMax: the smallest possible
                // sphere for the level, pushed to the furthest point within the
                // max sphere.  If even that is coarse enough, no group could
                // first transition at a higher level, so this is the last
                // active level.
                if (geometryUsesBlasSharing && !findMin)
                {
                    float3 oSpherePos = float3(traversalMetric.boundingSphereX,
                                               traversalMetric.boundingSphereY,
                                               traversalMetric.boundingSphereZ);
                    float3 oViewDir   = normalize(oSpherePos - oViewPos);

                    oSpherePos += oViewDir * (traversalMetric.boundingSphereRadius
                                              - lodLevels[lodLevel].minBoundingSphereRadius);

                    traversalMetric.boundingSphereX      = oSpherePos.x;
                    traversalMetric.boundingSphereY      = oSpherePos.y;
                    traversalMetric.boundingSphereZ      = oSpherePos.z;
                    traversalMetric.boundingSphereRadius = lodLevels[lodLevel].minBoundingSphereRadius;
                    traversalMetric.maxQuadricError      = lodLevels[lodLevel].minMaxQuadricError;

                    if (TestForTraversal(transform, uniformScale, traversalMetric, errorScale))
                    {
                        lodLevelMax = lodLevel;
                        break;
                    }
                }
            }

            if (visibilityState == 0u && g_Constants.sharingPushCulled != 0u)
            {
                // Push invisible instances out by one level so they are more
                // likely to share another instance's BLAS.
                lodLevelMin = min(lodLevelMin + 1u, lodLevelMax);
            }

            // If the finest level used is the geometry's coarsest available
            // level, the instance only uses the low-detail / pre-built BLAS.
            bool lowestDetailOnly = lodLevelMin == geometryLodLevelMax;

            if (lowestDetailOnly)
            {
                // low-detail BLAS only — nothing to add to the histograms
            }
            else if (geometryUsesBlasSharing)
            {
                uint dummy;
                InterlockedAdd(u_GeometryHistograms[geometryID].lodLevelMinHistogram[lodLevelMin], 1u, dummy);
                InterlockedAdd(u_GeometryHistograms[geometryID].lodLevelMaxHistogram[lodLevelMax], 1u, dummy);

                // For each lodLevelMax bucket, elect a "stable" provider: the
                // instance with the highest lodLevelMin (least detail).  Pack
                // lodLevelMin in the top 5 bits, instanceID in the low 27.
                uint packedLodInstance = (lodLevelMin << 27) | (instanceID & 0x7FFFFFFu);
                InterlockedMax(u_GeometryHistograms[geometryID].lodLevelMaxPackedInstance[lodLevelMax],
                               packedLodInstance, dummy);
            }
        }

        // Drives the per-instance decision in traversal_init_blas_sharing.
        u_InstanceBuildInfos[instanceID].lodLevelMin = lodLevelMin;
        u_InstanceBuildInfos[instanceID].lodLevelMax = lodLevelMax;

        // Always initialize the instance to the pre-built low-detail BLAS so it
        // is renderable; may be overridden in instance_assign_blas.
        u_InstanceBlasAddrs[instanceID] = geometry.lowDetailBlasAddress;
    }
}

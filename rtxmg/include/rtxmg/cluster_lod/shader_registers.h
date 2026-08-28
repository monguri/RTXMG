/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// ---------------------------------------------------------------------------
// Canonical register slots for the cluster-LoD build passes (ClusterLodPass and
// ClusterLodBlasPass, six binding layouts between them).  A buffer keeps the
// same register in every layout that binds it, so no two kernels disagree about
// what lives on a slot.  A buffer bound both ways gets one UAV slot and one SRV
// slot -- those are separate HLSL register spaces.
//
// The HLSL declarations and the C++ BindingLayout/BindingSet items both read
// these names, which is what keeps the two sides from drifting: a register that
// moves without its binding-set entry is a compile error, not silent garbage on
// one backend.
//
// ClusterLodStreaming is NOT covered: it has one unified layout of its own,
// documented in streaming.cpp, already canonical within itself.  Buffers it
// shares with traversal (resident groups, the SceneStreaming aggregate, active
// groups, group IDs, the unload ring, geometries) therefore still sit on
// different slots in the two families.
//
// Constant buffers are not part of the map: b0 is "this pass's constants"
// (SceneBuildingConstants for traversal, BlasBuildParams for the BLAS pass) and
// b1 is traversal_setup's push constant.
// ---------------------------------------------------------------------------

#ifndef __cplusplus
// Two levels so the slot argument is macro-expanded before the ## paste.
#define CLOD_REG_CAT_(prefix, slot) prefix##slot
#define CLOD_REG_CAT(prefix, slot)  CLOD_REG_CAT_(prefix, slot)
#define CLOD_SRV(slot) register(CLOD_REG_CAT(t, slot))
#define CLOD_UAV(slot) register(CLOD_REG_CAT(u, slot))
#endif

// ---- UAVs -----------------------------------------------------------------
#define CLOD_U_COUNTERS                 0   // SceneBuildingCounters
#define CLOD_U_TRAVERSAL_NODE_Q         1
#define CLOD_U_TRAVERSAL_GROUP_Q        2
#define CLOD_U_RENDER_CLUSTERS          3
#define CLOD_U_INSTANCE_BUILD_INFOS     4
#define CLOD_U_INSTANCE_BLAS_ADDRS      5
#define CLOD_U_BLAS_ARGS                6   // cluster::IndirectArgs, one per BLAS build
#define CLOD_U_RESIDENT_GROUPS          7
#define CLOD_U_STREAMING                8   // SceneStreaming aggregate
#define CLOD_U_LOAD_GEOMETRY_GROUPS     9
#define CLOD_U_GEOMETRY_BUILD_INFOS    10
#define CLOD_U_GEOMETRY_HISTOGRAMS     11
#define CLOD_U_INSTANCE_VISIBILITY     12
#define CLOD_U_PER_INSTANCE_TRIANGLES  13
#define CLOD_U_UNIQUE_SEEN_CLUSTERS    14
#define CLOD_U_ACTIVE_GROUPS           15
#define CLOD_U_GROUP_IDS               16
#define CLOD_U_UNLOAD_GEOMETRY_GROUPS  17
#define CLOD_U_BLAS_CLAS_ADDRS         18
#define CLOD_U_CACHED_BLAS_SRC         19
#define CLOD_U_CACHED_BLAS_DST         20
#define CLOD_U_PER_GEOM_SEEN           21

// ---- SRVs -----------------------------------------------------------------
#define CLOD_T_GEOMETRIES               0
#define CLOD_T_RENDER_INSTANCES         1
#define CLOD_T_COUNTERS                 2
#define CLOD_T_RENDER_CLUSTERS          3
#define CLOD_T_RESIDENT_CLAS_ADDRS      4
#define CLOD_T_INSTANCE_BUILD_INFOS     5
#define CLOD_T_BLAS_ADDRESSES           6
#define CLOD_T_GEOMETRY_PATCHES         7
#define CLOD_T_GEOMETRY_BUILD_INFOS     8
#define CLOD_T_PER_INSTANCE_TRIANGLES   9

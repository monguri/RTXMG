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
#pragma pack_matrix(row_major)

#include <nvrhi/nvrhiHLSL.h>
#include "rtxmg/cluster_tess/fill_instance_descs_params.h"

ConstantBuffer<FillInstanceDescsParams> g_Params : register(b0);

StructuredBuffer<nvrhi::GpuVirtualAddress> t_BlasAddresses : register(t0);
RWStructuredBuffer<nvrhi::rt::IndirectInstanceDesc> u_InstanceDescs : register(u0);

[numthreads(kFillInstanceDescsThreads, 1, 1)]
void main(uint3 threadIdx : SV_DispatchThreadID)
{
    uint32_t instanceIndex = threadIdx.x;
    if (instanceIndex >= g_Params.numInstances)
        return;

    nvrhi::GpuVirtualAddress blasAddress = t_BlasAddresses[instanceIndex];

    uint32_t outIndex = g_Params.instanceOffset + instanceIndex;
    nvrhi::rt::IndirectInstanceDesc instanceDesc = u_InstanceDescs[outIndex];
    instanceDesc.blasDeviceAddress = blasAddress;
    u_InstanceDescs[outIndex] = instanceDesc;
}
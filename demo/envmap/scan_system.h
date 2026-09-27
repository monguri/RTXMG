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

#include <nvrhi/nvrhi.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/CommonRenderPasses.h>

#include "scan_system_shared.h"

struct ScanSystem
{
    nvrhi::BufferHandle m_prefixScanParams;
    nvrhi::ShaderHandle m_prefixScan;
    nvrhi::ComputePipelineHandle m_prefixScanPSO;
    nvrhi::BindingLayoutHandle m_prefixScanBSL;

    void Init(std::shared_ptr<donut::engine::ShaderFactory> shaderFactory, nvrhi::IDevice* device);
    void PrefixScan(nvrhi::IBuffer* inputBuffer, nvrhi::IBuffer* outputBuffer, int inputWidth, int inputHeight, nvrhi::ICommandList* commandList);
};

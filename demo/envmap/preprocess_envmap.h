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

#include "rtxmg/utils/buffer.h"

struct PreprocessEnvMapShaders
{
    nvrhi::ComputePipelineHandle m_computeConditionalPSO;
    nvrhi::ComputePipelineHandle m_computeMarginalPSO;

    nvrhi::BindingLayoutHandle m_bindingLayout;
};

struct PreprocessEnvMapResources
{
    RTXMGBuffer<float> m_conditionalFunc;
    RTXMGBuffer<float> m_conditionalCdf;
    RTXMGBuffer<float> m_marginalFunc;
    RTXMGBuffer<float> m_marginalCdf;
    nvrhi::BufferHandle m_params;
    nvrhi::SamplerHandle m_sampler;
};
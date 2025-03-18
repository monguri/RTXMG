/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
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
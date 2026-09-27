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

#ifndef BLIT_PARAMS_H // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define BLIT_PARAMS_H

#include "rtxmg_demo.h"

#ifdef __cplusplus
#include <donut/core/math/math.h>
using namespace donut::math;
#endif

struct BlitParams
{
    BlitDecodeMode m_blitDecodeMode;
    TonemapOperator m_tonemapOperator;
    float m_exposure;
    float m_zNear;
    
    float m_zFar;
    float m_separator;
    // When enabled, the blit divides m_autoExposureScale (donut's exposure
    // target) by the adapted scene luminance and folds that into m_exposure.
    uint m_autoExposureEnabled;
    float m_autoExposureScale;

#ifdef __cplusplus
    BlitParams()
        : m_blitDecodeMode(BlitDecodeMode::None)
        , m_tonemapOperator(TonemapOperator::Linear)
        , m_exposure(1.0f)
        , m_zNear(1.0f)
        , m_zFar(100.f)
        , m_separator(0.5f)
        , m_autoExposureEnabled(0)
        , m_autoExposureScale(0.707f)
    {
    }
#endif
};

#endif // BLIT_PARAMS_H

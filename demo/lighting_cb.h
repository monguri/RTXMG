/*
 * SPDX-FileCopyrightText: Copyright (c) 2014-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef LIGHTING_CB_H // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define LIGHTING_CB_H

#include <donut/shaders/light_cb.h>
#include <donut/shaders/view_cb.h>

struct LightingConstants
{
    float4 ambientColor;

    LightConstants light;
    PlanarViewConstants view;
};

#endif // LIGHTING_CB_H
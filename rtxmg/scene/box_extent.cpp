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
//


#include "rtxmg/scene/box_extent.h"

using namespace donut::math;

float MaxBoxExtent(const box3& aabb)
{
    float3 diagonal = aabb.diagonal();
    return max(max(diagonal.x, diagonal.y), diagonal.z);
}

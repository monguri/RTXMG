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

#ifndef RTXMG_BOX3_H // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define RTXMG_BOX3_H

struct Box3
{
    // Alignment for constant buffer
    float3 m_min; 
    float pad0;
    float3 m_max; 
    float pad1;

    void Init()
    {
        m_min = float3(1e37f, 1e37f, 1e37f);
        m_max = float3(-1e37f, -1e37f, -1e37f);
        pad0 = 0; pad1 = 0;
    }

    void Init(float3 v0, float3 v1, float3 v2)
    {
        m_min = min(v0, min(v1, v2));
        m_max = max(v0, max(v1, v2));
    }

    void Include(float3 p)
    {
        m_min = min(m_min, p);
        m_max = max(m_max, p);
    }

    float3 Extent()
    {
        return m_max - m_min;
    }

    bool Valid()
    {
        return m_min.x <= m_max.x &&
            m_min.y <= m_max.y &&
            m_min.z <= m_max.z;
    }

#ifdef __cplusplus
    Box3(const donut::math::box3& b)
    {
        m_min = b.m_mins;
        m_max = b.m_maxs;
    }

    Box3()
    {
        Init();
    }
#endif
};

#if defined(__cplusplus)
static_assert(sizeof(Box3) % 16 == 0);
#elif defined(TARGET_D3D12)
_Static_assert(sizeof(Box3) % 16 == 0, "Must be 16 byte aligned for constant buffer");
#endif

#endif // RTXMG_BOX3_H
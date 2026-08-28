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

#ifndef RTXMG_CONSTANTS_H // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define RTXMG_CONSTANTS_H

/* scalar functions used in vector functions */
#ifndef M_PIf
#define M_PIf       3.14159265358979323846f
#endif
#ifndef M_PI_2f
#define M_PI_2f     1.57079632679489661923f
#endif
#ifndef M_1_PIf
#define M_1_PIf     0.318309886183790671538f
#endif

#ifndef FLT_MIN
#define FLT_MIN         1.175494351e-38
#endif 

#define PI_OVER_2 (.5f * M_PIf)
#define PI_OVER_4 (.25f * M_PIf)
#define TWO_PI (2.f * M_PIf)
#define INV_2PI (.5f * M_1_PIf)

#define DEBUG_SURFACE -1

#endif/* RTXMG_CONSTANTS_H */
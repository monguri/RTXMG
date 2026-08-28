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

/*-------------------------------------------------------------------------------------------------
Contains functions for aligning numbers to power-of-two boundaries.
-------------------------------------------------------------------------------------------------*/

#pragma once

#include <stddef.h>  // for size_t

namespace rtxmg {
/*-------------------------------------------------------------------------------------------------
# Function `is_aligned<integral>(x, a)`
Returns whether `x` is a multiple of `a`. `a` must be a power of two.
-------------------------------------------------------------------------------------------------*/
template <class integral>
constexpr bool is_aligned(integral x, size_t a) noexcept
{
    return (x & (integral(a) - 1)) == 0;
}

/*-------------------------------------------------------------------------------------------------
# Function `align_up<integral>(x, a)`
Rounds `x` up to a multiple of `a`. `a` must be a power of two.
-------------------------------------------------------------------------------------------------*/
template <class integral>
constexpr integral align_up(integral x, size_t a) noexcept
{
    return integral((x + (integral(a) - 1)) & ~integral(a - 1));
}

/*-------------------------------------------------------------------------------------------------
# Function `align_down<integral>(x, a)`
Rounds `x` down to a multiple of `a`. `a` must be a power of two.
-------------------------------------------------------------------------------------------------*/
template <class integral>
constexpr integral align_down(integral x, size_t a) noexcept
{
    return integral(x & ~integral(a - 1));
}
}  // namespace rtxmg

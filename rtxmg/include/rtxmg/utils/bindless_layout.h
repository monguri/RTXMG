/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace rtxmg
{
// Global bindless (SM6.6 ResourceDescriptorHeap) descriptor capacity.  Vulkan
// requires a pipeline's set layout to be identically defined (descriptorCount
// AND stage flags) to the set allocated from the shared DescriptorTableManager
// table, so every layout bound with it must use this exact capacity; D3D12
// ignores it.  Keep in sync with the renderer's ReserveCapacity call.
constexpr uint32_t kGlobalBindlessCapacity = 1u << 16;  // 65536

// Canonical global bindless layout desc — use everywhere so capacity /
// visibility / type can't drift between call sites.
inline nvrhi::BindlessLayoutDesc MakeGlobalBindlessLayoutDesc()
{
    nvrhi::BindlessLayoutDesc desc;
    desc.visibility  = nvrhi::ShaderType::All;
    desc.firstSlot   = 0;
    desc.maxCapacity = kGlobalBindlessCapacity;
    desc.layoutType  = nvrhi::BindlessLayoutDesc::LayoutType::MutableSrvUavCbv;
    return desc;
}
}  // namespace rtxmg

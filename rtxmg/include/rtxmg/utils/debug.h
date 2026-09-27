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

#ifndef RTXMG_DEBUG_H // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define RTXMG_DEBUG_H 

#include "rtxmg/utils/constants.h"
#include "rtxmg/utils/buffer.h"

int GetUniqueFileIndex(const char* baseName, const char* extension);
void WriteTexToCSV(nvrhi::ICommandList* commandList, nvrhi::ITexture* tex, char const filename[]);
void WriteBufferToCSV(nvrhi::ICommandList* commandList, RTXMGBuffer<float>& buf, char const filename[], int width, int height);

#define logassert(condition, message, ...) if (!(condition)) { donut::log::fatal(message, ##__VA_ARGS__); assert(false); }

#endif /* RTXMG_DEBUG_H */
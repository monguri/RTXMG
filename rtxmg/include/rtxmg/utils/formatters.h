/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

 // Follows the ImPlotFormatter signature
int HumanFormatter(double value, char* buff, int bufsize, void* = nullptr);
int MetricFormatter(double value, char* buff, int bufsize, void* data);
int MegabytesFormatter(double value, char* buff, int bufsize, void* = nullptr);
int MemoryFormatter(double value, char* buff, int bufsize, void* = nullptr);

template<typename T>
inline int HumanFormatter(T value, char* buff, int bufsize, void* = nullptr)
{
    return HumanFormatter(static_cast<double>(value), buff, bufsize);
}

template<typename T>
inline int MetricFormatter(T value, char* buff, int bufsize, void* = nullptr)
{
    return MetricFormatter(static_cast<double>(value), buff, bufsize);
}

template<typename T>
inline int MegabytesFormatter(T value, char* buff, int bufsize, void* = nullptr)
{
    return MegabytesFormatter(static_cast<double>(value), buff, bufsize);
}

template<typename T>
inline int MemoryFormatter(T value, char* buff, int bufsize, void* = nullptr)
{
    return MemoryFormatter(static_cast<double>(value), buff, bufsize);
}

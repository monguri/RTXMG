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
// clang-format off

#include <vector>

// clang-format off

template <typename T>
struct segmented_vector
{
    std::vector<T>        elements;      // element vector
    std::vector<uint32_t> offsets{ 0 };  // offset vector first element 0
    std::vector<uint32_t> sizes;         // segment sizes

    template <typename U>  // a container
    void Append(const U& segment)
    {
        sizes.push_back(static_cast<uint32_t>(segment.size()));
        offsets.push_back(offsets.back() + sizes.back());
        elements.insert(elements.end(), segment.begin(), segment.end());
    }

    void Append(const T* a_elements, uint32_t n_elements)
    {
        sizes.push_back(n_elements);
        offsets.push_back(offsets.back() + sizes.back());
        elements.insert(elements.end(), &a_elements[0], &a_elements[n_elements]);
    }

    void Reserve(size_t n)
    {
        offsets.reserve(n);
        sizes.reserve(n);
    }

    T* Data() { return elements.data(); }
    size_t Size() { return elements.size(); }
};


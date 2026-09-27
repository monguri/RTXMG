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


#include <cstring>
#include <fstream>
#include <sstream>
#include <filesystem>

#include <iostream>

int GetUniqueFileIndex(const char* baseName, const char* extension)
{
    // Avoid overwriting an existing screenshot : scan the default output directory
    // for existing files with pattern 'screenshot_xxxx.bmp' to find the highest index.
    int index = -1;
    namespace fs = std::filesystem;
    for (auto it : fs::directory_iterator(fs::current_path()))
    {
        if (it.path().extension() != extension)
            continue;
        std::string filename = it.path().filename().generic_string();
        if (std::strstr(filename.c_str(), baseName) != filename.c_str())
            continue;
        int existingIndex = std::atoi(filename.c_str() + strlen(baseName));
        index = std::max(index, existingIndex);
    }
    return index + 1;
}

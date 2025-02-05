/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef SCAN_SYSTEM_SHARED_H // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define SCAN_SYSTEM_SHARED_H

 /// Number of threads per block that are used for computing the horizontal prefix scans for a 2D buffer.
 /// To handle 8k textures this must be at least 128, otherwise a 3-level scan would be required.
#define PREFIX_SCAN_THREAD_BLOCK_SIZE 512

/// Number of image rows that each kernel invocation will handle during horizontal prefix scans.
#define PREFIX_SCAN_ROWS_PER_BLOCK 1

struct PrefixScanParams
{
    uint32_t elementCountX;
    uint32_t elementCountY;
    uint32_t outputWidth;
};

#endif // SCAN_SYSTEM_SHARED_H
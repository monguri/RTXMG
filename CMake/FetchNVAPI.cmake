#
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
include(FetchContent)

set(NVAPI_FETCH_URL "https://github.com/NVIDIA/nvapi.git" CACHE STRING "Url to nvapi git repo to fetch")
set(NVAPI_FETCH_TAG "ce6d2a183f9559f717e82b80333966d19edb9c8c" CACHE STRING "Tag of nvapi git repo")
set(NVAPI_FETCH_DIR "" CACHE STRING "Directory to fetch streamline to, empty uses build directory default")

include(FetchContent)
FetchContent_Declare(
    nvapi
    GIT_REPOSITORY ${NVAPI_FETCH_URL}
    GIT_TAG ${NVAPI_FETCH_TAG}
    SOURCE_DIR ${NVAPI_FETCH_DIR}
)
FetchContent_MakeAvailable(nvapi)

message(STATUS "Updating nvapi from ${NVAPI_FETCH_URL}, tag ${NVAPI_FETCH_TAG}, into folder ${nvapi_SOURCE_DIR}")
set(NVAPI_SEARCH_PATHS "${nvapi_SOURCE_DIR}")
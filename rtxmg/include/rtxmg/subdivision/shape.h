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

#pragma once

#include <donut/core/math/math.h>

#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <opensubdiv/far/types.h>

enum Scheme { kBilinear = 0, kCatmark, kLoop };

struct Shape
{
    // Note: increment the version to trigger cache rebuild.
    static constexpr std::chrono::system_clock::duration version{ 6 };

    // full(er) spec here: http://paulbourke.net/dataformats/mtl/
    // and here: https://en.wikipedia.org/wiki/Wavefront_.obj_file
    struct material
    {
        std::string name;

        float ka[3] = { 0.f, 0.f, 0.f }; // ambient
        float kd[3] = { .8f, .8f, .8f }; // diffuse
        float ks[3] = { 0.f, 0.f, 0.f }; // specular
        float ke[3] = { 0.f, 0.f, 0.f }; // emissive
        float ns = 0.f;                // specular exponent
        float ni = 0.f;        // optical density (1.0=no refraction, glass=1.5)
        float sharpness = 0.f; // reflection sharpness
        float tf[3] = { 0.f, 0.f, 0.f }; // transmission filter
        float d = 0.f;                 // dissolve factor (1.0 = opaque)
        int illum = 4;                 // illumination model
        float bm = 1.f;                // bump multipler
        float bb = 0.f;                // bump bias

        std::string map_ka;   // ambient
        std::string map_kd;   // diffuse
        std::string map_ks;   // specular
        std::string map_bump; // bump

        // MTL extensions (exported by Blender & others)
        float Pr = 0.8f;     // roughness
        float Pm = 0.f;     // metallic
        float Ps = 0.f;     // sheen
        float Pc = 0.f;     // clearcoat thickness
        float Pcr = 0.f;    // clearcoat roughness
        float Ke = 0.f;     // emissive
        float aniso = 0.f;  // anisotropy
        float anisor = 0.f; // anisotropy rotation

        std::string map_ke;  // emissive
        std::string map_pr;  // roughness
        std::string map_pm;   // metalness
        std::string map_rma; // roughness / metalness / ambient occlusion
        std::string map_orm; // alt version of rma

        unsigned int udim = 0;
    };

    int FindMaterial(char const* name);

    std::string mtllib;
    std::vector<unsigned short> mtlbind;

    // contiguous set of face that share the same material
    struct Subshape
    {
        size_t startFaceIndex = 0;
        int mtlBind = -1;
    };

    std::vector<Subshape> subshapes;
    std::vector<uint16_t> faceToSubshapeIndex;
    std::vector<std::unique_ptr<material>> mtls;

    struct tag
    {
        std::string name;
        std::vector<int> intargs;
        std::vector<float> floatargs;
        std::vector<std::string> stringargs;

        static bool ParseTag(char const* stream, tag* t);
        std::string GenTag() const;
    };
    std::vector<tag> tags;

    // read from/write to cache
    void WriteShape(const std::string& objFile) const;
    bool ReadShape(const std::string& objFile);

    uint32_t GetNumVertices() const { return (int)verts.size(); }
    uint32_t GetNumFaces() const { return (int)nvertsPerFace.size(); }
    int GetFVarWidth() const { return HasUV() ? 2 : 0; }
    bool HasUV() const { return !(uvs.empty() || faceuvs.empty()); }

    std::filesystem::path filepath;

    std::vector<int> nvertsPerFace;
    std::vector<int> faceverts;
    std::vector<int> faceuvs;
    std::vector<int> facenormals;

    Scheme scheme = kCatmark;

    bool isLeftHanded = false;

    // Vertex attributes
    std::vector<donut::math::float3> verts;
    std::vector<donut::math::float2> uvs;
    std::vector<donut::math::float3> normals; // unused for subd's

    donut::math::box3 aabb;

    // catmark cube shape
    static std::unique_ptr<Shape> DefaultShape();

    // load a single obj file
    static std::unique_ptr<Shape>
        LoadObjFile(const std::filesystem::path& pathStr, bool parseMaterials = true, bool requireUVs = true);

    std::string                   capslib;
// TODO: Capsules
//     CapsuleCache                  capsules;
};

// Parse a .mtl into its material list. Exposed separately from LoadObjFile so a
// texture pre-pass can enumerate an OBJ's maps without parsing its geometry.
std::vector<std::unique_ptr<Shape::material>> ParseMtllib(const char* filepath);

// ParseMtllib, then expand any <UDIM>/<UVTILE> maps into one material per tile
// present on disk, so every map is a concrete path. `basepath` is the directory
// the map paths are relative to (the OBJ's, as LoadObjFile uses). A tiled map
// with no tiles on disk contributes no materials.
std::vector<std::unique_ptr<Shape::material>> ParseMtllibResolved(
    const std::filesystem::path& mtlPath, const std::filesystem::path& basepath);

// Read the `mtllib` name an .obj declares, without parsing the rest of the file.
// Empty if the file declares none.
std::string ReadObjMtllibName(const std::filesystem::path& objPath);

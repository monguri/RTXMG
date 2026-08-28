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

#include "rtxmg/scene/material_library.h"

#include "rtxmg/subdivision/shape.h"
#include "rtxmg/subdivision/subdivision_surface.h"

#include <donut/core/log.h>

#include <algorithm>
#include <filesystem>
#include <map>
#include <string>
#include <tuple>
#include <utility>

namespace fs = std::filesystem;

using namespace donut;

namespace rtxmg
{

void MaterialLibrary::BuildSubdMaterials(
    std::span<const std::unique_ptr<SubdivisionSurface>> subdMeshes,
    const TextureLoader& textureLoader, engine::ThreadPool* threadPool)
{
    std::map<std::tuple<fs::path, std::string, uint32_t>, uint32_t> sceneMatCache;

    for (const std::unique_ptr<SubdivisionSurface>& subdMesh : subdMeshes)
    {
        auto shape = subdMesh->GetShape();

        // MTL has no encoding convention, so every slot defers to the DDS header.
        auto addTexture = [&textureLoader, threadPool](const fs::path& shapePath, const std::string& mtlLib,
                                 const std::string& texPath) -> std::shared_ptr<LoadedTexture>
        {
            if (texPath.empty())
                return nullptr;
            fs::path fp = (((shapePath.parent_path() / mtlLib)).parent_path() / texPath).lexically_normal();
            return textureLoader.Load(fp, threadPool, engine::SRGBMode::FromFile);
        };

        const fs::path resolvedMtllib =
            (shape->filepath.parent_path() / shape->mtllib).lexically_normal();

        std::vector<uint32_t> mtlBindToSceneIndex(shape->mtls.size(), UINT32_MAX);
        std::vector<uint32_t> meshSubshapeIndices;
        meshSubshapeIndices.reserve(shape->subshapes.size());

        for (uint32_t subshapeIdx = 0; subshapeIdx < (uint32_t)shape->subshapes.size(); ++subshapeIdx)
        {
            const auto& subshape = shape->subshapes[subshapeIdx];
            const uint32_t mtlBind = subshape.mtlBind;

            if (mtlBindToSceneIndex[mtlBind] == UINT32_MAX)
            {
                const auto& mtl = shape->mtls[mtlBind];
                auto key = std::make_tuple(resolvedMtllib, mtl->name, mtl->udim);
                auto it = sceneMatCache.find(key);
                if (it != sceneMatCache.end())
                {
                    mtlBindToSceneIndex[mtlBind] = it->second;
                }
                else
                {
                    auto mat = std::make_shared<RTXMGMaterial>();
                    mat->name                        = mtl->udim ? mtl->name + "." + std::to_string(mtl->udim) : mtl->name;
                    mat->baseOrDiffuseColor          = mtl->kd;
                    mat->emissiveColor               = mtl->ke;
                    mat->roughness                   = mtl->Pr;
                    mat->metalness                   = mtl->Pm;
                    mat->baseOrDiffuseTexture        = addTexture(shape->filepath, shape->mtllib, mtl->map_kd);
                    mat->metalnessTexture            = addTexture(shape->filepath, shape->mtllib, mtl->map_pm);
                    mat->roughnessTexture            = addTexture(shape->filepath, shape->mtllib, mtl->map_pr);
                    mat->specularF0Texture           = addTexture(shape->filepath, shape->mtllib, mtl->map_ks);
                    mat->normalOrDisplacementTexture = addTexture(shape->filepath, shape->mtllib, mtl->map_bump);
                    mat->normalOrDisplacementScale   = mtl->bm;
                    mat->isDisplacementMap           = !mtl->map_bump.empty();

                    const uint32_t sceneIdx = static_cast<uint32_t>(m_materials.size());
                    m_materials.push_back(mat);
                    sceneMatCache[key] = sceneIdx;
                    mtlBindToSceneIndex[mtlBind] = sceneIdx;
                }
            }

            const uint32_t sceneIdx = mtlBindToSceneIndex[mtlBind];
            const auto& mat = m_materials[sceneIdx];
            if (mat->isDisplacementMap && mat->normalOrDisplacementTexture)
                subdMesh->m_hasDisplacementMaterial = true;

            meshSubshapeIndices.push_back(sceneIdx);
        }
        m_subdSubshapeMaterialIndices.push_back(std::move(meshSubshapeIndices));
    }
}

void MaterialLibrary::BuildClusterLodMaterials(
    std::span<const ClusterLodMaterialDesc> descs,
    const TextureLoader& textureLoader, engine::ThreadPool* threadPool)
{
    // Under --nomat the whole set is replaced by one default-gray material, which
    // every cluster then resolves to via the localMaterialsCount=0 override that
    // the cluster-LoD resource init applies.
    m_clusterLodMaterialBaseID = static_cast<uint32_t>(m_materials.size());
    if (!m_enableMaterials)
    {
        auto defaultMat = std::make_shared<RTXMGMaterial>();
        defaultMat->name               = "rtxmg_nomat_default_gray";
        defaultMat->baseOrDiffuseColor = float3(0.5f, 0.5f, 0.5f);
        defaultMat->opacity            = 1.0f;
        defaultMat->metalness          = 0.0f;
        defaultMat->roughness          = 1.0f;
        defaultMat->alphaCutoff        = 0.0f;
        defaultMat->doubleSided        = false;
        defaultMat->isAlphaMasked      = false;
        m_materials.push_back(defaultMat);
        return;
    }

    uint32_t normalMapCount = 0;
    for (const ClusterLodMaterialDesc& desc : descs)
    {
        if (!desc.normalTexturePath.empty())
            ++normalMapCount;

        auto mat = std::make_shared<RTXMGMaterial>();
        mat->name               = desc.name;
        mat->baseOrDiffuseColor = float3(desc.baseColorFactor.x, desc.baseColorFactor.y, desc.baseColorFactor.z);
        mat->opacity            = desc.baseColorFactor.w;
        mat->metalness          = desc.metalness;
        mat->roughness          = desc.roughness;
        mat->emissiveColor      = desc.emissiveColor * desc.emissiveIntensity;
        mat->alphaCutoff        = desc.alphaCutoff;
        mat->doubleSided        = desc.doubleSided;
        mat->isAlphaMasked      = (desc.alphaModeGltf == 1);
        mat->isAlphaBlend       = (desc.alphaModeGltf == 2);
        mat->isTransmissive     = (desc.transmissionFactor > 0.f);
        mat->transmissionFactor = desc.transmissionFactor;
        mat->ior                = desc.ior;

        mat->baseOrDiffuseTexture = textureLoader.Load(desc.baseColorTexturePath, threadPool, engine::SRGBMode::ForceSRGB);
        // GLTF packs metalness (.b) and roughness (.g) in one texture — share the handle.
        mat->metalnessTexture     = textureLoader.LoadMetallicRoughness(desc.metallicRoughnessTexturePath, threadPool);
        mat->roughnessTexture     = mat->metalnessTexture;
        mat->emissiveTexture      = textureLoader.Load(desc.emissiveTexturePath, threadPool, engine::SRGBMode::ForceSRGB);
        // These slots must stay in sync with what ApplyKtxTextureBudget accounts
        // for (rtxmg/scene/texture_budget.cpp), including the normal map below.
        if (m_enableNormalMaps)
        {
            // Tangent-space, so never sRGB; the shader reconstructs Z, so a
            // two-channel file needs no component-mapping override here.
            mat->normalOrDisplacementTexture = textureLoader.Load(desc.normalTexturePath, threadPool,
                                                                  engine::SRGBMode::ForceLinear);
            mat->normalOrDisplacementScale   = desc.normalTextureScale;
            mat->isDisplacementMap           = false;
        }

        m_materials.push_back(mat);
    }

    if (normalMapCount)
        log::info("Cluster-LoD materials: %u of %zu declare a normal map (%s).",
                  normalMapCount, descs.size(),
                  m_enableNormalMaps ? "loaded" : "dropped, --normalmaps to load");
}

void MaterialLibrary::BuildMaterialBuffer(nvrhi::ICommandList* commandList)
{
    if (m_materials.empty())
        return;

    std::vector<RTXMGMaterialConstants> cpuMaterials;
    cpuMaterials.reserve(m_materials.size());

    uint32_t matIdx = 0;
    for (const auto& mat : m_materials)
    {
        RTXMGMaterialConstants mc{};
        if (mat)
        {
            mat->FillMaterialConstants(mc);
            mc.materialID = static_cast<int>(matIdx);
        }
        cpuMaterials.push_back(mc);
        ++matIdx;
    }

    const size_t byteSize = cpuMaterials.size() * sizeof(RTXMGMaterialConstants);
    if (!m_materialBuffer || m_materialBuffer->getDesc().byteSize != byteSize)
    {
        nvrhi::BufferDesc desc;
        desc.byteSize     = byteSize;
        desc.structStride = sizeof(RTXMGMaterialConstants);
        desc.debugName    = "RTXMGMaterialBuffer";
        desc.initialState = nvrhi::ResourceStates::ShaderResource;
        desc.keepInitialState = true;
        m_materialBuffer = m_device->createBuffer(desc);
    }
    commandList->writeBuffer(m_materialBuffer, cpuMaterials.data(), byteSize);
}

void MaterialLibrary::BuildClusterLodLocalMaterialIDs(
    std::span<const GeometryStorage> storages, nvrhi::ICommandList* commandList)
{
    m_clusterLodLocalMaterialsOffsets.assign(storages.size(), 0u);
    m_clusterLodLocalMaterialsCounts .assign(storages.size(), 0u);

    std::vector<uint32_t> flat;
    size_t totalCount = 0;
    for (const auto& s : storages)
        totalCount += s.localMaterialIDs.size();
    flat.reserve(std::max(totalCount, size_t(1)));

    uint32_t cursor = 0;
    for (size_t g = 0; g < storages.size(); ++g)
    {
        const auto& storage = storages[g];
        m_clusterLodLocalMaterialsOffsets[g] = cursor;
        m_clusterLodLocalMaterialsCounts [g] = static_cast<uint32_t>(storage.localMaterialIDs.size());
        for (uint32_t gid : storage.localMaterialIDs)
            flat.push_back(gid);
        cursor += static_cast<uint32_t>(storage.localMaterialIDs.size());
    }
    if (flat.empty())
        flat.push_back(0u);  // 1-element padding so the SRV is non-empty.

    const size_t bufBytes = flat.size() * sizeof(uint32_t);
    if (!m_clusterLodLocalMaterialIDsBuffer ||
        m_clusterLodLocalMaterialIDsBuffer->getDesc().byteSize != bufBytes)
    {
        nvrhi::BufferDesc d;
        d.byteSize         = bufBytes;
        d.structStride     = sizeof(uint32_t);
        d.debugName        = "ClusterLodLocalMaterialIDs";
        d.initialState     = nvrhi::ResourceStates::ShaderResource;
        d.keepInitialState = true;
        m_clusterLodLocalMaterialIDsBuffer = m_device->createBuffer(d);
    }
    commandList->writeBuffer(m_clusterLodLocalMaterialIDsBuffer, flat.data(), bufBytes);
}

} // namespace rtxmg

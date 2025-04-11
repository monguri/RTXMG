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

#pragma pack_matrix(row_major)

#include "rtxmg/utils/shader_debug.h"

#include "render_params.h"
#include "lighting_cb.h"
#include <donut/shaders/bindless.h>
#include <donut/shaders/lighting.hlsli>
#include <donut/shaders/packing.hlsli>
#include <donut/shaders/scene_material.hlsli>
#include <donut/shaders/surface.hlsli>
#include <donut/shaders/utils.hlsli>
#include <donut/shaders/binding_helpers.hlsli>

#include "rtxmg/cluster_builder/cluster.h"
#include "rtxmg/hiz/hiz_buffer_constants.h"
#include "rtxmg/utils/constants.h"

#include "ray_payload.h"
#include "color.hlsli"
#include "gbuffer.h"
#include "brdf.hlsli"
#include "utils.hlsli"

#define CONCAT(a,b) a##b
#define CONCAT_UAV(x) CONCAT(u,x)
#define NV_SHADER_EXTN_SLOT CONCAT_UAV(RTXMG_NVAPI_SHADER_EXT_SLOT)
#define NV_SHADER_EXTN_REGISTER_SPACE space0
#include "nvHLSLExtns.h"

#include "envmap/shaders/envmap.hlsli"

ConstantBuffer<LightingConstants> g_Const : register(b0);
ConstantBuffer<RenderParams> g_RenderParams : register(b1);

RWTexture2D<float4> u_Accum     : register(u0);

// GBuffer
RWTexture2D<DepthFormat>    u_Depth         : register(u1);
RWTexture2D<NormalFormat>   u_Normal        : register(u2);
RWTexture2D<AlbedoFormat>   u_Albedo        : register(u3);
RWTexture2D<SpecularFormat> u_Specular      : register(u4);
RWTexture2D<SpecularHitTFormat> u_SpecularHitT  : register(u5);
RWTexture2D<RoughnessFormat>    u_Roughness     : register(u6);
RWStructuredBuffer<HitResult>   u_HitResult     : register(u7);

#if ENABLE_DUMP_FLOAT
RWTexture2D<float4> u_DebugTex1 : register(u8);
RWTexture2D<float4> u_DebugTex2 : register(u9);
RWTexture2D<float4> u_DebugTex3 : register(u10);
RWTexture2D<float4> u_DebugTex4 : register(u11);
#endif

#if ENABLE_SHADER_DEBUG
RWStructuredBuffer<ShaderDebugElement> u_PixelDebug : register(u12);
#endif

RWBuffer<uint32_t>  u_TimeviewBuffer : register(u13);


RaytracingAccelerationStructure SceneBVH : register(t0);
StructuredBuffer<InstanceData> t_InstanceData : register(t1);
StructuredBuffer<GeometryData> t_GeometryData : register(t2);
StructuredBuffer<MaterialConstants> t_MaterialConstants : register(t3);
Texture2D<float4> t_EnvMap : register(t4);
StructuredBuffer<float> t_EnvMapConditionalCDF: register(t5);
StructuredBuffer<float> t_EnvMapMarginalCDF : register(t6);
StructuredBuffer<float> t_EnvMapConditionalFunc: register(t7);
StructuredBuffer<float> t_EnvMapMarginalFunc : register(t8);
StructuredBuffer<ClusterShadingData> t_ClusterShadingData : register(t9);
StructuredBuffer<float3> t_ClusterVertexPositions : register(t10);
StructuredBuffer<SubdInstance> t_SubdInstances : register(t11);

SamplerState s_MaterialSampler : register(s0);

VK_BINDING(0, 1) ByteAddressBuffer t_BindlessBuffers[] : register(t0, space1);
VK_BINDING(1, 1) Texture2D t_BindlessTextures[] : register(t0, space2);

#include "self_intersection_avoidance.hlsli"

// Cluster look up
uint16_t2 ClusterGetEdgeSize(uint32_t clusterId)
{
    ClusterShadingData clusterShadingData = t_ClusterShadingData[clusterId];
    return uint16_t2(clusterShadingData.m_clusterSizeX, clusterShadingData.m_clusterSizeY);
}

uint3 ClusterGetVertexIndices(uint32_t primId)
{
    const uint32_t      clusterId = NvRtGetClusterID();
    const uint16_t      triID = (uint16_t)primId;

    // vertex quad ordering: 
    // 23
    // 01
    // triangle ordering: left edge first -- 032+013 (diagonal:03) or 012+132 (diagonal:12)
    // 21 .5    or   2. 54
    // 0. 34    or   01 .3
    // vx,vy are row-major vertex indices in range [0..sx][0..sy] sx,sy are cluster edge m_size
    // if vx,vy are the lower left corner vtx idxs, then diagonal:03 == ((vx & 1) == (vy & 1))

    uint16_t2 clusterEdgeSize = ClusterGetEdgeSize(clusterId);

    const uint16_t qs = clusterEdgeSize.x;      // quad stride
    const uint16_t vs = clusterEdgeSize.x + 1;  // vert stride
    const uint16_t qid = triID >> 1;             // quad id
    const uint16_t qx = qid % qs;               // quad x
    const uint16_t qy = qid / qs;               // quad y
    const uint16_t vid = qy * vs + qx;           // lower-left vertex id
    const bool    diag03 = ((qx & 1) == (qy & 1));       // is diag 0-3 (true) or 1-2 (false)

    const uint16_t df = uint16_t(diag03) << 1 | uint16_t(triID & 1);

    uint3 indices;
    switch (df)
    {
    case 0b00: indices = uint3(vid, vid + 1, vid + vs); break;
    case 0b01: indices = uint3(vid + 1, vid + 1 + vs, vid + vs); break;
    case 0b10: indices = uint3(vid, vid + 1 + vs, vid + vs); break;
    case 0b11: indices = uint3(vid, vid + 1, vid + 1 + vs); break;
    }

    return indices;
}

inline uint16_t2 Index2D(uint32_t indexLinear, uint16_t lineStride)
{
    return uint16_t2(uint16_t(indexLinear % lineStride), uint16_t(indexLinear / lineStride));
}

// Given a cluster triangle id, find the uv coordinates in the parametric surface
// that generated the triangle's three corners.
//
inline void GetSurfaceUV(out float2 uvs[3], ClusterShadingData clusterShadingData, uint primId)
{
    const uint3    uMajorVtxIDs = ClusterGetVertexIndices(primId);

    const uint16_t2 clusterSize = uint16_t2(clusterShadingData.m_clusterSizeX, clusterShadingData.m_clusterSizeY);
    const uint16_t2 clusterOffset = clusterShadingData.m_clusterOffset;
    const uint16_t4 edgeSegments = clusterShadingData.m_edgeSegments;

    const GridSampler sampler = { edgeSegments };

    // offset local i,j index to surface index
    uint16_t2 vertexIndex2d = Index2D(uMajorVtxIDs.x, clusterSize.x + 1) + clusterOffset;
    uvs[0] = sampler.UV(vertexIndex2d, (ClusterPattern)g_RenderParams.clusterPattern);
    vertexIndex2d = Index2D(uMajorVtxIDs.y, clusterSize.x + 1) + clusterOffset;
    uvs[1] = sampler.UV(vertexIndex2d, (ClusterPattern)g_RenderParams.clusterPattern);
    vertexIndex2d = Index2D(uMajorVtxIDs.z, clusterSize.x + 1) + clusterOffset;
    uvs[2] = sampler.UV(vertexIndex2d, (ClusterPattern)g_RenderParams.clusterPattern);
}

struct IntersectionRecord
{
    float3 p;             // world space intersection point
    float3 n;             // world space shading normal
    float3 gn;            // world space geometry normal
    float2 texcoord;      // user-assigned texcoord from base mesh
    float3 barycentrics;  // barycentrics
    float3 distToEdge;
    float  hitT;

    // Gbuffer output needed for motion vecs
    uint32_t surfaceIndex;
    float2   surfaceUV;

    MaterialSample ms;
};

void SetupPrimaryRay(uint2 pixelPosition, float2 subPixelJitter, out float3 rayOrigin, out float3 rayDirection)
{
    float2 d = ((float2(pixelPosition) + 0.5f + subPixelJitter) *
        g_RenderParams.camera.dimsInv) *
        2.f -
        1.f;

    d *= float2(1, -1);

    RayDesc ray;
    rayOrigin = g_RenderParams.eye;
    rayDirection = normalize(d.x * g_RenderParams.U + d.y * g_RenderParams.V +
        g_RenderParams.W);
}

RayDesc SetupShadowRay(float3 surfacePos, float3 L)
{
    RayDesc ray;
    ray.Origin = surfacePos - WorldRayDirection() * 0.001;
    ray.Direction = L;
    ray.TMin = 0;
    ray.TMax = 1.#INF;
    return ray;
}

[shader("miss")] void ShadowMiss(inout ShadowRayPayload payload : SV_RayPayload)
{
    payload.missed = true;
}

bool IsOccluded(float3 worldPos, float3 towardsLight)
{
    ShadowRayPayload shadowPayload = (ShadowRayPayload)0;
    shadowPayload.missed = false;

    RayDesc shadowRay = SetupShadowRay(worldPos, towardsLight);

    TraceRay(SceneBVH,
        RAY_FLAG_NONE,
        0xFF,
        1, // shadow hit group
        0,
        1, // shadow miss shader
        shadowRay, shadowPayload);

    return !shadowPayload.missed;
}

enum GeometryAttributes
{
    GeomAttr_Position = 0x01,
    GeomAttr_TexCoord = 0x02,
    GeomAttr_Normal = 0x04,
    GeomAttr_Tangents = 0x08,

    GeomAttr_All = 0x0F
};

struct GeometrySample
{
    InstanceData instance;
    MaterialConstants material;

    float3 vertexPositions[3];
    float2 vertexTexcoords[3];

    float3 barycentrics;
    float2 texcoord;
    float3x4 objectToWorld;
    float3x4 worldToObject;

    uint geometryIndex;
    uint clusterId;
    uint surfaceIndex;
    float2 surfaceUV;
};

GeometrySample
GetGeometryFromHit(RayPayload payload)
{
    GeometrySample gs = (GeometrySample)0;

    InstanceData instance = t_InstanceData[payload.instanceID];

    uint geometryIndex = instance.firstGeometryIndex + payload.geometryIndex;

    GeometryData geometry = t_GeometryData[geometryIndex];

    gs.instance = instance;
    gs.geometryIndex = geometryIndex;
    gs.objectToWorld = instance.transform;
    gs.material = t_MaterialConstants[geometry.materialIndex];
    gs.clusterId = ~0u;

    float3x3 w2oRotation = transpose((float3x3)gs.objectToWorld);
    float3 w2oTranslation = -mul(w2oRotation, float3(gs.objectToWorld[0][3], gs.objectToWorld[1][3], gs.objectToWorld[2][3]));
    gs.worldToObject = float3x4(
        w2oRotation[0], w2oTranslation.x,
        w2oRotation[1], w2oTranslation.y,
        w2oRotation[2], w2oTranslation.z);

    gs.barycentrics.yz = payload.barycentrics;
    gs.barycentrics.x = 1.0 - (gs.barycentrics.y + gs.barycentrics.z);


    // Look up cluster geometry data

    uint32_t clusterId = NvRtGetClusterID();
    gs.clusterId = clusterId;
    ClusterShadingData clusterShadingData = t_ClusterShadingData[clusterId];

    uint3 localVtxIndices = ClusterGetVertexIndices(payload.primitiveIndex);
    uint3 globalVtxIndices = localVtxIndices + clusterShadingData.m_vertexOffset;

    // Load vertex positions.
    gs.vertexPositions[0] = t_ClusterVertexPositions[globalVtxIndices[0]];
    gs.vertexPositions[1] = t_ClusterVertexPositions[globalVtxIndices[1]];
    gs.vertexPositions[2] = t_ClusterVertexPositions[globalVtxIndices[2]];

    // Texcoords
    // Bilinear texcoords
    float2 uvs[3];
    GetSurfaceUV(uvs, clusterShadingData, payload.primitiveIndex);
    float2 uv = gs.barycentrics.x * uvs[0] + gs.barycentrics.y * uvs[1] + gs.barycentrics.z * uvs[2];

    gs.surfaceUV = uv;
    gs.surfaceIndex = clusterShadingData.m_surfaceId;

    SHADER_DEBUG(gs.surfaceIndex);

    // bilerp from 4 corner attributes
    const float u = uv.x;
    const float v = uv.y;

    gs.texcoord = clusterShadingData.m_texcoords[0] * (1.0f - u) * (1.0f - v)
        + clusterShadingData.m_texcoords[1] * u * (1.0f - v)
        + clusterShadingData.m_texcoords[2] * u * v
        + clusterShadingData.m_texcoords[3] * (1.0f - u) * v;

    return gs;
}

void GetClipPoints(out float3 outClipPoints[3], in GeometrySample gs)
{
    float3 worldSpacePositions[3];
    worldSpacePositions[0] =
        mul(gs.instance.transform, float4(gs.vertexPositions[0], 1.0)).xyz;
    worldSpacePositions[1] =
        mul(gs.instance.transform, float4(gs.vertexPositions[1], 1.0)).xyz;
    worldSpacePositions[2] =
        mul(gs.instance.transform, float4(gs.vertexPositions[2], 1.0)).xyz;

    float4 projectedPoints[3];
    projectedPoints[0] = mul(g_RenderParams.viewProjectionMatrix,
        float4(worldSpacePositions[0], 1.0));
    projectedPoints[1] = mul(g_RenderParams.viewProjectionMatrix,
        float4(worldSpacePositions[1], 1.0));
    projectedPoints[2] = mul(g_RenderParams.viewProjectionMatrix,
        float4(worldSpacePositions[2], 1.0));

    outClipPoints[0] = projectedPoints[0].xyz / projectedPoints[0].w;
    outClipPoints[1] = projectedPoints[1].xyz / projectedPoints[1].w;
    outClipPoints[2] = projectedPoints[2].xyz / projectedPoints[2].w;
}

MaterialSample RTXMG_EvaluateSceneMaterial(GeometrySample gs, float3 geometryNormal, MaterialTextureSample textures)
{
    MaterialSample result = DefaultMaterialSample();
    result.roughness = 1;

    ColorMode colorMode = g_RenderParams.colorMode;

    if (colorMode == ColorMode::COLOR_BY_NORMAL)
    {
        result.baseColor = 0.5f * (float3(1, 1, 1) + geometryNormal);
        result.diffuseAlbedo = 0.5f * (float3(1, 1, 1) + geometryNormal);
    }
    else if (colorMode == ColorMode::COLOR_BY_TOPOLOGY)
    {
        uint32_t clusterId = NvRtGetClusterID();
        uint32_t surfaceId = t_ClusterShadingData[clusterId].m_surfaceId;

        SubdInstance subdInstance = t_SubdInstances[InstanceID()];
        StructuredBuffer<uint16_t> topologyQuality = ResourceDescriptorHeap[NonUniformResourceIndex(subdInstance.topologyQualityBindlessIndex)];
        uint16_t surfaceValue = topologyQuality[surfaceId];

        float value = float(surfaceValue) / 255.f;

        result.baseColor = lerp(float3(0.f, 1.f, 0.f), float3(1.f, 0.f, 0.f), value);
    }
    else if (colorMode == ColorMode::COLOR_BY_TEXCOORD)
    {
        result.baseColor = float3(frac(gs.texcoord), 0);
        result.diffuseAlbedo = float3(frac(gs.texcoord), 0);
    }
    else if (colorMode == ColorMode::COLOR_BY_MATERIAL)
    {
        float3 hashedColor = UintToColor(gs.material.materialID);
        result.baseColor = hashedColor;
        result.diffuseAlbedo = hashedColor;
    }
    else if (colorMode == ColorMode::COLOR_BY_GEOMETRY_INDEX)
    {
        float3 hashedColor = UintToColor(gs.material.materialID);
        result.baseColor = hashedColor;
        result.diffuseAlbedo = hashedColor;
    }
    else if (colorMode == ColorMode::BASE_COLOR)
    {
        if (g_RenderParams.shadingMode == ShadingMode::AO)
        {
            result.baseColor = 0.8;
        }
        else
        {
            result.baseColor = lerp(gs.material.baseOrDiffuseColor.rgb, textures.baseOrDiffuse.rgb, textures.baseOrDiffuse.a);

            result.roughness = gs.material.roughness;
            if (g_RenderParams.roughnessOverride > 0.f)
            {
                result.roughness = g_RenderParams.roughnessOverride;
            }
            else if (gs.material.metalRoughOrSpecularTextureIndex >= 0)
            {
                result.roughness = textures.metalRoughOrSpecular.r;
            }
            result.roughness = max(result.roughness, 1e-4f);

            result.metalness = gs.material.metalness;
            result.hasMetalRoughParams = true;
            // Compute the BRDF inputs for the metal-rough model
            // https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#metal-brdf-and-dielectric-brdf
            if (g_RenderParams.shadingMode == ShadingMode::PT)
            {
                // the metalness map is arriving as "emissive" to re-use donut materials
                // and specular maps arrive in the occlusion texture.
                if (gs.material.occlusionTextureIndex >= 0)
                {
                    result.metalness = -1.f;
                    result.specularF0 = textures.occlusion.rgb;
                }
                else if (gs.material.emissiveTextureIndex >= 0)
                {
                    result.metalness = textures.emissive.r;
                }
            }
            else
            {
                // direct rays will use the built in donut model.
                result.diffuseAlbedo = lerp(result.baseColor * (1.0 - c_DielectricSpecular), 0.0, result.metalness);
                result.specularF0 = lerp(c_DielectricSpecular, result.baseColor.rgb, result.metalness);
            }
        }
    }
    else if (colorMode == ColorMode::COLOR_BY_CLUSTER_ID)
    {
        ClusterShadingData clusterShadingData = t_ClusterShadingData[gs.clusterId];
        uint linearClusterOffset = (clusterShadingData.m_clusterOffset.y * clusterShadingData.m_clusterSizeX) + clusterShadingData.m_clusterOffset.x;

        uint hash = 0;
        hash = MurmurAdd(hash, clusterShadingData.m_surfaceId);
        hash = MurmurAdd(hash, linearClusterOffset);
        float3 hashedColor = UintToColor(hash);
        result.baseColor = hashedColor;
        result.diffuseAlbedo = hashedColor;
    }
    else if (colorMode == ColorMode::COLOR_BY_MICROTRI_ID)
    {
        uint primitiveIndex = PrimitiveIndex();

        ClusterShadingData clusterShadingData = t_ClusterShadingData[gs.clusterId];
        uint linearClusterOffset = (clusterShadingData.m_clusterOffset.y * clusterShadingData.m_clusterSizeX) + clusterShadingData.m_clusterOffset.x;

        uint hash = 0;
        hash = MurmurAdd(hash, clusterShadingData.m_surfaceId);
        hash = MurmurAdd(hash, linearClusterOffset);
        hash = MurmurAdd(hash, primitiveIndex);
        float3 hashedColor = UintToColor(hash);
        result.baseColor = hashedColor;
        result.diffuseAlbedo = hashedColor;
    }
    else if (colorMode == ColorMode::COLOR_BY_SURFACE_INDEX)
    {
        float3 hashedColor = UintToColor(gs.surfaceIndex);
        result.baseColor = hashedColor;
        result.diffuseAlbedo = hashedColor;
    }
    else if (colorMode == ColorMode::COLOR_BY_CLUSTER_UV)
    {
        result.baseColor = float3(gs.surfaceUV, 0);
        result.diffuseAlbedo = float3(gs.surfaceUV, 0);
    }
    else if (colorMode == ColorMode::COLOR_BY_MICROTRI_AREA)
    {
        float3 clipPoints[3];
        GetClipPoints(clipPoints, gs);

        const float uTriScreenArea = .5f * length(cross(clipPoints[0] - clipPoints[1], clipPoints[2] - clipPoints[0]));
        const float uTriAreaInPixels = g_RenderParams.camera.dims.x * g_RenderParams.camera.dims.y * uTriScreenArea / 4.f;  // the area of the screen is 4.f since the screen vertices range from [-1, 1]

        const float normUTriAreaInPixels = Remap(clamp(uTriAreaInPixels, 0.f, 2.f), 0.f, 2.f, 1.0f, 0.0f);
        result.baseColor = Temperature(normUTriAreaInPixels);
        result.diffuseAlbedo = result.baseColor;
    }

    result.occlusion = 1.0;

    // if you need to highlight a particular surface
    if (gs.surfaceIndex == DEBUG_SURFACE)
    {
        result.baseColor = float3(1, 0, 0);
    }

    return result;
}

MaterialTextureSample RTXMG_DefaultMaterialTextures()
{
    MaterialTextureSample values;
    values.baseOrDiffuse = float4(1.0, 1.0, 1.0, 0.0); // fully transparent texture
    values.metalRoughOrSpecular = float4(0.8, 0, 0, 0);
    values.emissive = float4(0.0, 0.0, 0.0, 0.0); // no metal default
    return values;
}

MaterialSample
SampleGeometryMaterial(GeometrySample gs,
    float3 geometryNormal,
    float2 texGradX,
    float2 texGradY,
    float mipLevel, // <-- Use a compile time constant for mipLevel, < 0 for aniso filtering
    SamplerState materialSampler)
{
    MaterialTextureSample textures = RTXMG_DefaultMaterialTextures();

    if ((gs.material.baseOrDiffuseTextureIndex >= 0) &&
        (gs.material.flags & MaterialFlags_UseBaseOrDiffuseTexture) != 0)
    {
        Texture2D diffuseTexture = t_BindlessTextures[NonUniformResourceIndex(
            gs.material.baseOrDiffuseTextureIndex)];

        if (mipLevel >= 0)
            textures.baseOrDiffuse =
            diffuseTexture.SampleLevel(materialSampler, gs.texcoord, mipLevel);
        else
        {
            textures.baseOrDiffuse = diffuseTexture.SampleGrad(
                materialSampler, gs.texcoord, texGradX, texGradY);
        }
    }

    if ((gs.material.metalRoughOrSpecularTextureIndex >= 0) &&
        (gs.material.flags & MaterialFlags_UseMetalRoughOrSpecularTexture) != 0)
    {
        Texture2D specularTexture = t_BindlessTextures[NonUniformResourceIndex(
            gs.material.metalRoughOrSpecularTextureIndex)];

        if (mipLevel >= 0)
            textures.metalRoughOrSpecular =
            specularTexture.SampleLevel(materialSampler, gs.texcoord, mipLevel);
        else
            textures.metalRoughOrSpecular = specularTexture.SampleGrad(
                materialSampler, gs.texcoord, texGradX, texGradY);
    }

    if ((gs.material.emissiveTextureIndex >= 0) &&
        (gs.material.flags & MaterialFlags_UseEmissiveTexture) != 0)
    {
        Texture2D emissiveTexture = t_BindlessTextures[NonUniformResourceIndex(
            gs.material.emissiveTextureIndex)];

        if (mipLevel >= 0)
            textures.emissive =
            emissiveTexture.SampleLevel(materialSampler, gs.texcoord, mipLevel);
        else
            textures.emissive = emissiveTexture.SampleGrad(
                materialSampler, gs.texcoord, texGradX, texGradY);
    }
    if ((gs.material.occlusionTextureIndex >= 0) &&
        (gs.material.flags & MaterialFlags_UseOcclusionTexture) != 0)
    {
        Texture2D occlusionTexture = t_BindlessTextures[NonUniformResourceIndex(
            gs.material.occlusionTextureIndex)];

        if (mipLevel >= 0)
            textures.occlusion =
            occlusionTexture.SampleLevel(materialSampler, gs.texcoord, mipLevel);
        else
            textures.occlusion = occlusionTexture.SampleGrad(
                materialSampler, gs.texcoord, texGradX, texGradY);
    }

    return RTXMG_EvaluateSceneMaterial(gs, geometryNormal, textures);
}

MaterialSample GetMaterialSample(GeometrySample gs, float3 geometryNormal)
{
    uint2 pixelPosition = DispatchRaysIndex().xy;

    float2 noJitter = float2(0.f, 0.f);

    float3 ray0Origin, ray0Direction;
    float3 rayXOrigin, rayXDirection;
    float3 rayYOrigin, rayYDirection;

    SetupPrimaryRay(pixelPosition, noJitter, ray0Origin, ray0Direction);
    SetupPrimaryRay(pixelPosition + uint2(1, 0), noJitter, rayXOrigin, rayXDirection);
    SetupPrimaryRay(pixelPosition + uint2(0, 1), noJitter, rayYOrigin, rayYDirection);
    float3 worldSpacePositions[3];
    worldSpacePositions[0] =
        mul(gs.instance.transform, float4(gs.vertexPositions[0], 1.0)).xyz;
    worldSpacePositions[1] =
        mul(gs.instance.transform, float4(gs.vertexPositions[1], 1.0)).xyz;
    worldSpacePositions[2] =
        mul(gs.instance.transform, float4(gs.vertexPositions[2], 1.0)).xyz;
    float3 bary_0 = computeRayIntersectionBarycentrics(
        worldSpacePositions, ray0Origin, ray0Direction);
    float3 bary_x = computeRayIntersectionBarycentrics(
        worldSpacePositions, rayXOrigin, rayXDirection);
    float3 bary_y = computeRayIntersectionBarycentrics(
        worldSpacePositions, rayYOrigin, rayYDirection);
    float2 texCoord0 = interpolate(gs.vertexTexcoords, bary_0);
    float2 texCoordX = interpolate(gs.vertexTexcoords, bary_x);
    float2 texCoordY = interpolate(gs.vertexTexcoords, bary_y);
    float2 texGradX = texCoordX - texCoord0;
    float2 texGradY = texCoordY - texCoord0;

    MaterialSample ms = SampleGeometryMaterial(gs, geometryNormal, texGradX, texGradY, -1, s_MaterialSampler);

    return ms;
}

IntersectionRecord GetIntersectionRecord(RayPayload payload)
{
    IntersectionRecord ir = (IntersectionRecord)0;
    ir.hitT = RayTCurrent();

    GeometrySample gs = GetGeometryFromHit(payload);

    float3 objP, objN, wldP;
    float wldOffset;

    SafeSpawnPoint(objP, wldP, objN, ir.gn, wldOffset,
        gs.vertexPositions[0], gs.vertexPositions[1], gs.vertexPositions[2],
        gs.barycentrics.yz, gs.objectToWorld, gs.worldToObject);

    if (dot(ir.gn, WorldRayDirection()) > 0.0f)
    {
        ir.gn = -ir.gn;
    }

    ir.p = SafeSpawnPoint(wldP, ir.gn, wldOffset);

    ir.n = ir.gn;
    ir.texcoord = gs.texcoord;
    ir.barycentrics = gs.barycentrics;
    ir.surfaceIndex = gs.surfaceIndex;
    ir.surfaceUV = gs.surfaceUV;

    ir.ms = GetMaterialSample(gs, ir.gn);
    ir.ms.geometryNormal = ir.n;
    ir.ms.shadingNormal = ir.n;

    if (g_RenderParams.enableWireframe)
    {
        float3 clipPoints[3];
        GetClipPoints(clipPoints, gs);

        float3 e_01 = clipPoints[0] - clipPoints[1];
        float3 e_12 = clipPoints[1] - clipPoints[2];
        float3 e_20 = clipPoints[2] - clipPoints[0];

        float area = .5f * length(cross(e_01, e_20));

        ir.distToEdge = 2.f * area * gs.barycentrics;

        ir.distToEdge.x /= length(e_12);
        ir.distToEdge.y /= length(e_20);
        ir.distToEdge.z /= length(e_01);
    }
    return ir;
}

float3 ShadeSurface(IntersectionRecord ir)
{
    float3 diffuseTerm = 0, specularTerm = 0;

    if (!IsOccluded(ir.p, -g_Const.light.direction))
    {
        ShadeSurface(g_Const.light, ir.ms, ir.p, WorldRayDirection(), diffuseTerm,
                       specularTerm);
    }

    return (diffuseTerm + specularTerm +
        ir.ms.diffuseAlbedo * g_Const.ambientColor.rgb);
}

struct Attributes
{
    float2 uv;
};

[shader("miss")] void Miss(inout RayPayload payload
    : SV_RayPayload)
{
    bool hasEnvMap = g_RenderParams.hasEnvironmentMap;

    float3 d = WorldRayDirection();
    float2 u = convertDirToTexCoords(d, g_RenderParams.envmapRotation);

    float3 lightContribution;
    if (g_RenderParams.shadingMode == ShadingMode::PT)
    {
        if (hasEnvMap)
        {
            lightContribution = envMapEvaluate(u, t_EnvMap, g_RenderParams.envmapIntensity, s_MaterialSampler);
        }
        else
        {
            lightContribution = ((float3(d.y, d.y, d.y) + 1.f) / 2.f) * g_RenderParams.missColor;
        }
    }
    else
    {
        lightContribution = g_RenderParams.missColor;
    }

    if (g_RenderParams.shadingMode == ShadingMode::PT)
    {
        float brdfPdf = payload.pdf;
        uint bounce = payload.bounce;

        if (bounce > 0)
        {
            // Calculate pdf if texture environment map is present and when not present,
            // uniformly sample the hemisphere for the gradient environment map
            const float lightPdf = hasEnvMap ? envMapPdf(u, t_EnvMap, t_EnvMapConditionalFunc, t_EnvMapMarginalCDF, s_MaterialSampler) : 1.f / (4.f * M_PIf);
            const float misWeight = PowerHeuristic(1.f, brdfPdf, 1.f, lightPdf);
            const float3 pathWeight = FromRGBe9995(payload.pathWeight);
            lightContribution *= pathWeight * misWeight;
        }
        else
        {
            if (g_RenderParams.enableEnvmapHeatmap)
            {
                float pdf = hasEnvMap ? envMapPdf(u, t_EnvMap, t_EnvMapConditionalFunc, t_EnvMapMarginalCDF, s_MaterialSampler) : 0.0f;
                lightContribution = hasEnvMap ? Temperature(pdf) : Temperature((d.y + 1.f) / 2.f);
            }
        }
        payload.pathContribution = ToRGBe9995(lightContribution);
    }
    else
    {
        payload.pathWeight = ToRGBe9995(lightContribution);
    }
    payload.hitT = 1.#INF;
    payload.instanceID = ~0u;

    bool clearGBuffer = g_RenderParams.denoiserMode != DenoiserMode::None;
    if (clearGBuffer)
    {
        // Write no hit
        uint2 dispatchDims = DispatchRaysDimensions().xy;
        uint2 dispatchPixel = DispatchRaysIndex().xy;
        uint dispatchIndex = dispatchPixel.x + dispatchDims.x * dispatchPixel.y;

        if (g_RenderParams.shadingMode != ShadingMode::PT || payload.bounce == 0)
        {
            u_HitResult[dispatchIndex] = DefaultHitResult();

            // Clear gbuffer
            // using linear depth
            u_Depth[dispatchPixel] = g_RenderParams.zFar;
            u_Normal[dispatchPixel] = float3(0.0f, 0.0f, 0.0f);
            u_Albedo[dispatchPixel] = float3(0.0f, 0.0f, 0.0f);
            u_Specular[dispatchPixel] = float3(0.0f, 0.0f, 0.0f);
            u_Roughness[dispatchPixel] = 0.0f;
        }

        // If PT mode, then if we miss on bounce 0 or 1, then clear to zFar
        // If non-PT mode, then only bounce 0, clear to ZFar
        if (payload.bounce <= 1)
        {
            u_SpecularHitT[dispatchPixel] = g_RenderParams.zFar;
        }
    }
}

float WireframeWeight(IntersectionRecord ir)
{
    float thickness = g_RenderParams.wireframeThickness * 1e-4f;
    float smoothness = 1e-7f;
    float3 b = ir.barycentrics * ir.distToEdge;

    float minBary = min(min(b.x, b.y), b.z);
    return smoothstep(thickness, thickness + smoothness, minBary);
}

float3 AOSample(const float3 normal, const float2 u)
{
    const Onb onb = MakeOnb(normal);
    float3    dir = CosineSampleHemisphere(u);
    dir = onb.ToWorld(dir);
    return normalize(dir);
}

float3 SampleDirect(MaterialSample material, float3 p, float3 gN, float3 N, float3 V, inout uint32_t seed)
{
    float2 u = float2(Rnd(seed), Rnd(seed));

    float lightPdf = 1.f;
    float3 envMapColor;
    float3 L = envMapImportanceSample(u, g_RenderParams.envmapRotationInv,
        lightPdf, envMapColor, t_EnvMap, g_RenderParams.envmapIntensity, t_EnvMapConditionalFunc, t_EnvMapMarginalFunc, t_EnvMapConditionalCDF, t_EnvMapMarginalCDF, s_MaterialSampler);
    float3 lightContribution = 0.f;
    if (IsOccluded(p, L))
    {
        return lightContribution;
    }
    lightContribution = envMapColor;
    float brdfPdf = 1.f;
    const float3 brdfWeight = BRDFEval(material, gN, N, V, L, brdfPdf);
    const float misWeight = PowerHeuristic(1.f, lightPdf, 1.f, brdfPdf);
    // brdfWeight includes scaling by dot( N, L )
    float3 result = lightContribution * brdfWeight * misWeight / lightPdf;
    return result;
}

[shader("closesthit")]void ClosestHit(inout RayPayload payload
    : SV_RayPayload, in Attributes attrib
    : SV_IntersectionAttributes)
{
    SHADER_DEBUG_INIT(u_PixelDebug, g_RenderParams.debugPixel, DispatchRaysIndex().xy);

    payload.instanceID = InstanceID();
    payload.primitiveIndex = PrimitiveIndex();
    payload.geometryIndex = GeometryIndex();
    payload.barycentrics = attrib.uv;
    payload.hitT = RayTCurrent();

    SHADER_DEBUG(uint4(payload.instanceID, payload.primitiveIndex, payload.geometryIndex, NvRtGetClusterID()));
    SHADER_DEBUG(float3(payload.barycentrics, payload.hitT));

    IntersectionRecord ir = GetIntersectionRecord(payload);

    float3 pathWeight = 1.f;

    float wfWeight = g_RenderParams.enableWireframe ? WireframeWeight(ir) : 1.0f;

    if (g_RenderParams.denoiserMode != DenoiserMode::None)
    {
        uint2 dispatchDims = DispatchRaysDimensions().xy;
        uint2 dispatchPixel = DispatchRaysIndex().xy;
        uint dispatchIndex = dispatchPixel.x + dispatchDims.x * dispatchPixel.y;

        const bool writePrimarySurfaceGBuffer = (g_RenderParams.shadingMode != ShadingMode::PT || payload.bounce == 0);

        if (writePrimarySurfaceGBuffer)
        {
            HitResult hitResult;
            hitResult.instanceId = payload.instanceID;
            hitResult.surfaceIndex = ir.surfaceIndex;
            hitResult.surfaceUV = ir.surfaceUV;
            hitResult.texcoord = ir.texcoord;
            u_HitResult[dispatchIndex] = hitResult;

            float depth = dot(normalize(g_RenderParams.W), ir.p - g_RenderParams.eye);
            u_Depth[dispatchPixel] = depth;
            u_Normal[dispatchPixel] = ir.n;

            float3 V = -WorldRayDirection();
            FresnelBlend brdf = MakeFresnelBlend(ir.ms.baseColor, ir.ms.specularF0, ir.ms.metalness, ir.ms.roughness);
            u_Albedo[dispatchPixel] = brdf.m_diffuse.m_albedo * wfWeight;
            float3 spec = BRDFEnvApprox(brdf, ir.n, V);
            u_Specular[dispatchPixel] = spec * wfWeight;
            u_Roughness[dispatchPixel] = ir.ms.roughness;
        }

        if (g_RenderParams.shadingMode != ShadingMode::PT)
        {
            // Non-PT mode clear to far Z
            u_SpecularHitT[dispatchPixel] = g_RenderParams.zFar;
        }
        else if (payload.bounce == 1)
        {
            // We want to write the hit distance from primary surface to specular hit
            u_SpecularHitT[dispatchPixel] = ir.hitT;
        }
    }

    if (wfWeight == 0.f)
    {
        payload.pathWeight = 0;
        return;
    }


    if (g_RenderParams.shadingMode == ShadingMode::PRIMARY_RAYS)
    {
        pathWeight = ir.ms.baseColor;
        payload.pathWeight = ToRGBe9995(pathWeight);
    }
    else if (g_RenderParams.shadingMode == ShadingMode::AO)
    {
        uint spp = g_RenderParams.denoiserMode != DenoiserMode::None ? 1 : g_RenderParams.spp;

        uint32_t seed = payload.seed;
        float2 subPixelJitter = RandomStrat(payload.multipurposeField, sqrt(spp), seed);
        float3 L = AOSample(ir.n, subPixelJitter);
        bool occluded = IsOccluded(ir.p, L);
        pathWeight *= ir.ms.baseColor * (occluded ? 0.f : 1.f);
        payload.pathWeight = ToRGBe9995(pathWeight);
        payload.seed = seed;
    }
    else if (g_RenderParams.shadingMode == ShadingMode::PT)
    {
        uint32_t seed = payload.seed;
        float3 V = -WorldRayDirection();

        float3 pw = FromRGBe9995(payload.pathWeight);
        pathWeight *= pw;
        float3 lightContribution = 0;
        float samplePdf = payload.pdf;

        if (g_RenderParams.hasEnvironmentMap)
        {
            float3 directLighting = SampleDirect(ir.ms, ir.p, ir.gn, ir.n, V, seed);
            lightContribution = pathWeight * directLighting;
        }

        if (payload.bounce < g_RenderParams.ptMaxBounces - 1)
        {
            // No need to do this on the final hit, since we won't trace 
            // another ray anyway
            float3 L = 0;

            float3 indirectWeight;

            indirectWeight = BRDFSample(ir.ms, ir.gn, ir.n, V, L, samplePdf, seed);
            pathWeight *= indirectWeight;

            payload.pathWeight = ToRGBe9995(pathWeight);
            payload.multipurposeField = PackNormalizedVector(L);
            payload.rayOrigin = ir.p;

            payload.pdf = samplePdf;
            payload.seed = seed;
        }
        payload.pathContribution = ToRGBe9995(lightContribution);
    }
}

void TraceRadiancePT(uint           bounce,
                     inout uint     seed,
                     const uint32_t subPixelIndex,
                     inout float3   rayOrigin,
                     inout float3   rayDirection,
                     inout float3   pathWeight,
                     inout float3   pathContribution,
                     inout float    pdf,
                     out float      hitT)
{
    RayPayload payload = (RayPayload)0;
    payload.instanceID = ~0u;
    payload.pathWeight = ToRGBe9995(pathWeight);
    payload.bounce = bounce;
    payload.multipurposeField = subPixelIndex;
    payload.pathContribution = ToRGBe9995(pathContribution);
    payload.pdf = pdf;
    payload.seed = seed;

    RayDesc ray;
    ray.Origin = rayOrigin;
    ray.Direction = rayDirection;
    ray.TMin = 0;
    ray.TMax = 1.#INF;

    TraceRay(SceneBVH,
        RAY_FLAG_NONE,
        0xFF,
        0, // which hit group to use
        0,
        0, // which miss shader to use (0 = regular, 1 = shadow)
        ray, payload);

    pathWeight = FromRGBe9995(payload.pathWeight);
    hitT = payload.hitT;
    seed = payload.seed;
    rayDirection = UnpackNormalizedVector(payload.multipurposeField); // multipurposeField gets re-used for normal
    rayOrigin = payload.rayOrigin;
    float3 pc = FromRGBe9995(payload.pathContribution);
    if (g_RenderParams.denoiserMode == DenoiserMode::None)
    {
        pathContribution += bounce > 0 ? FireflyFiltering(pc, g_RenderParams.fireflyMaxIntensity) : pc;
    }
    else
    {
        pathContribution += pc;
    }
    pdf = payload.pdf;
}

void TraceRadiancePR(float3 rayOrigin, float3 rayDirection, out float3 pathWeight, out float hitT)
{
    RayPayload payload = (RayPayload)0;
    payload.instanceID = ~0u;

    RayDesc ray;
    ray.Origin = rayOrigin;
    ray.Direction = rayDirection;
    ray.TMin = 0;
    ray.TMax = 1.#INF;

    TraceRay(SceneBVH,
        RAY_FLAG_NONE,
        0xFF,
        0, // which hit group to use
        0,
        0, // which miss shader to use (0 = regular, 1 = shadow)
        ray, payload);

    pathWeight = FromRGBe9995(payload.pathWeight);
    hitT = payload.hitT;
}

void TraceRadianceAO(inout uint32_t seed,
                    const uint32_t         subPixelIndex,
                    float3                 rayOrigin,
                    float3                 rayDirection,
                    out float3 pathWeight,
                    out float hitT)
{
    RayPayload payload = (RayPayload)0;
    payload.instanceID = ~0u;
    payload.multipurposeField = subPixelIndex;
    payload.seed = seed;

    RayDesc ray;
    ray.Origin = rayOrigin;
    ray.Direction = rayDirection;
    ray.TMin = 0;
    ray.TMax = 1.#INF;

    TraceRay(SceneBVH,
               RAY_FLAG_NONE,
               0xFF,
               0, // which hit group to use
               0,
               0, // which miss shader to use (0 = regular, 1 = shadow)
               ray, payload);

    pathWeight = FromRGBe9995(payload.pathWeight);
    hitT = payload.hitT;
    seed = payload.seed;
}

uint TimeDiff(uint startTime, uint endTime)
{
    // Account for (at most one) overflow
    return endTime >= startTime ? (endTime - startTime) : (~0u - (startTime - endTime));
}

[shader("raygeneration")]void RayGen()
{
    SHADER_DEBUG_INIT(u_PixelDebug, g_RenderParams.debugPixel, DispatchRaysIndex().xy);

    uint startTime = NvGetSpecial(NV_SPECIALOP_GLOBAL_TIMER_LO);
    DUMP_FLOAT4(1, float4(0, 0, 0, 1)); // Clear debug buffer
    DUMP_FLOAT4(2, float4(0, 0, 0, 1)); // Clear debug buffer
    DUMP_FLOAT4(3, float4(0, 0, 0, 1)); // Clear debug buffer
    DUMP_FLOAT4(4, float4(0, 0, 0, 1)); // Clear debug buffer

    uint2 pixelPosition = DispatchRaysIndex().xy;
    uint imageIndex = GetImageIndex();

    float3 result = 0;
    float3 diffuseResult = 0;
    float3 specularResult = 0;
    unsigned int seed = TEA(16, imageIndex, g_RenderParams.subFrameIndex);

    float3 rayOrigin, rayDirection;

    float hitT = 1.#INF;

    uint spp = g_RenderParams.denoiserMode != DenoiserMode::None ? 1 : g_RenderParams.spp;

    const uint32_t shflIdxRndOffset = Rnd(seed) * spp;
    const int strataCount = sqrt(spp);

    for (int i = 0; i < spp; i++)
    {
        float2 subPixelJitter = g_RenderParams.denoiserMode != DenoiserMode::None ? g_RenderParams.jitter :
            (RandomStrat(i, strataCount, seed) - 0.5f);
        int subpixelIndex2nd = (i + shflIdxRndOffset) % spp;

        SetupPrimaryRay(pixelPosition, subPixelJitter, rayOrigin, rayDirection);

        float3 pathWeight = 1.f;

        if (g_RenderParams.shadingMode == ShadingMode::PT)
        {
            float3 pathContribution = 0.f;
            float pdf = 0.f;

            for (uint32_t bounce = 0; bounce < g_RenderParams.ptMaxBounces; bounce++)
            {
                TraceRadiancePT(bounce, seed, subpixelIndex2nd,
                    rayOrigin, rayDirection, pathWeight, pathContribution, pdf, hitT);

                if (isinf(hitT) || !any(pathWeight))
                {
                    break;
                }

                if (g_RenderParams.denoiserMode == DenoiserMode::None)
                {
                    // Russian Roulette 
                    if (bounce > 1)
                    {
                        float rrProbability = min(0.95f, Luminance(pathWeight));
                        if (rrProbability < Rnd(seed))
                            break;
                        else
                            pathWeight /= rrProbability;
                    }
                }
            }
            result += pathContribution;
        }
        else if (g_RenderParams.shadingMode == ShadingMode::AO)
        {
            TraceRadianceAO(seed, subpixelIndex2nd, rayOrigin, rayDirection, pathWeight, hitT);
            result += pathWeight;
        }
        else
        {
            TraceRadiancePR(rayOrigin, rayDirection, pathWeight, hitT);
            result += pathWeight;
        }
    }
    result /= spp;

    float4 accumVal;
    if (g_RenderParams.denoiserMode != DenoiserMode::None)
    {
        accumVal = float4(result, 1.f);
    }
    else
    {
        accumVal = u_Accum[pixelPosition];
        accumVal = lerp(accumVal, float4(result, 1.f), 1.f / float(g_RenderParams.subFrameIndex + 1));
    }

    if (g_RenderParams.enableTimeView)
    {
        uint endTime = NvGetSpecial(NV_SPECIALOP_GLOBAL_TIMER_LO);
        uint deltaTime = TimeDiff(startTime, endTime);

        if (g_RenderParams.subFrameIndex == 0)
        {
            if (pixelPosition.x == 0 && pixelPosition.y == 0)
            {
                u_TimeviewBuffer[0] = 0xffffffff;
                u_TimeviewBuffer[1] = 0;
            }
        }
        else if (g_RenderParams.subFrameIndex == 1)
        {
            int orig;
            InterlockedMin(u_TimeviewBuffer[0], deltaTime, orig);
            InterlockedMax(u_TimeviewBuffer[1], deltaTime, orig);
        }
        else
        {
            uint minValue = u_TimeviewBuffer[0];
            uint maxValue = u_TimeviewBuffer[1];

            float3 result = Temperature(((float)deltaTime) / ((float)maxValue - minValue));
            accumVal = u_Accum[pixelPosition];
            accumVal = lerp(accumVal, float4(result, 1.f), 1.f / float(g_RenderParams.subFrameIndex - 1));
        }
    }

    u_Accum[pixelPosition] = accumVal;
}

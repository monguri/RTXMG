//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef RTXMG_BRDF_HLSLI // using instead of "#pragma once" due to https://github.com/microsoft/DirectXShaderCompiler/issues/3943
#define RTXMG_BRDF_HLSLI

#include <donut/shaders/lighting.hlsli>
#include "utils.hlsli"

#define MIN_DIELECTRICS_F0     0.04f
#define INV_MIN_DIELECTRICS_F0 (1.f / MIN_DIELECTRICS_F0);

// Attenuates F90 for very low F0 values
// Source: "An efficient and Physically Plausible Real-Time Shading Model" in ShaderX7 by Schuler
// Also see section "Overbright highlights" in Hoffman's 2010 "Crafting Physically Motivated Shading Models for Game Development" for discussion
// IMPORTANT: Note that when F0 is calculated using metalness, it's value is never less than MIN_DIELECTRICS_F0, and therefore,
// this adjustment has no effect. To be effective, F0 must be authored separately, or calculated in different way. See main text for discussion.
float ShadowedF90(float3 F0)
{
    // This scalar value is somewhat arbitrary, Schuler used 60 in his article. In here, we derive it from MIN_DIELECTRICS_F0 so
    // that it takes effect for any reflectance lower than least reflective dielectrics
    //const float t = 60.0f;
    const float t = INV_MIN_DIELECTRICS_F0;
    return min(1.0f, t * Luminance(F0));
}


template <typename T>
inline T Sqr(T v) {
    return v * v;
}

template <typename T>
inline float LengthSquared( T n )
{
    return dot( n, n );
}

inline float AbsDot( float3 v1, float3 v2 )
{
    return clamp(abs( dot( v1, v2 ) ), .00001f, 1.f);
}

inline float CosTheta( float3 w )
{
    return w.z;
}
inline float Cos2Theta( float3 w )
{
    return w.z * w.z;
}
inline float AbsCosTheta( float3 w )
{
    return clamp(abs( w.z ), .00001f, 1.f);
}
inline float Sin2Theta( float3 w )
{
    return max( 0.00001f, 1.0f - Cos2Theta( w ) );
}
inline float SinTheta( float3 w )
{
    return sqrt( Sin2Theta( w ) );
}
inline float TanTheta( float3 w )
{
    return SinTheta( w ) / CosTheta( w );
}
inline float Tan2Theta( float3 w )
{
    return Sin2Theta( w ) / Cos2Theta( w );
}
inline float CosPhi( float3 w )
{
    const float sinTheta = SinTheta( w );
    return ( sinTheta == 0.0f ) ? 1.0f : clamp( w.x / sinTheta, -1.0f, 1.0f );
}
inline float SinPhi( float3 w )
{
    const float sinTheta = SinTheta( w );
    return ( sinTheta == 0.0f ) ? 0.0f : clamp( w.y / sinTheta, -1.0f, 1.0f );
}
inline float Cos2Phi( float3 w )
{
    return CosPhi( w ) * CosPhi( w );
}
inline float Sin2Phi( float3 w )
{
    return SinPhi( w ) * SinPhi( w );
}
inline float CosDPhi( float3 wa, float3 wb )
{
    return clamp( ( wa.x * wb.x + wa.y * wb.y ) / sqrt( ( wa.x * wa.x + wa.y * wa.y ) * ( wb.x * wb.x + wb.y * wb.y ) ),
                  -1.0f, 1.0f );
}

inline bool SameHemisphere( float3 a, float3 b )
{
    return a.z * b.z > 0.00001f;
}

inline float3 Reflect( float3 wo, float3 n )
{
    return -1.0f * wo + 2.0f * dot( wo, n ) * n;
}


// Schlick's approximation to Fresnel term
// f90 should be 1.0, except for the trick used by Schuler (see 'ShadowedF90' function)
inline float3 EvalFresnelSchlick( float3 f0, float f90, float NdotS )
{
    return f0 + ( f90 - f0 ) * pow( 1.0f - NdotS, 5.0f );
}

inline float3 EvalFresnel( float3 f0, float f90, float NdotS )
{
    // Default is Schlick's approximation
    return EvalFresnelSchlick( f0, f90, NdotS );
}

inline float3 BaseColorToDiffuseReflectance(float3 baseColor, float metalness)
{
    return baseColor * (1.0f - metalness);
}

float3 BaseColorToSpecularF0(float3 baseColor, float metalness)
{
    return lerp(float3(MIN_DIELECTRICS_F0, MIN_DIELECTRICS_F0, MIN_DIELECTRICS_F0), baseColor, metalness);
}

class TrowbridgeReitzDistribution
{
    float alphaX, alphaY;
    inline float D(float3 wm)
    {
        float tan2Theta = Tan2Theta(wm);
        if (isinf(tan2Theta))
            return 0.f;
        float cos4Theta = Sqr(Cos2Theta(wm));
        if (cos4Theta < 1e-16f)
            return 0.f;
        float e = tan2Theta * (Sqr(CosPhi(wm) / alphaX) + Sqr(SinPhi(wm) / alphaY));
        return 1.f / (M_PIf * alphaX * alphaY * cos4Theta * Sqr(1.f + e));
    }

    bool EffectivelySmooth() { return max(alphaX, alphaY) < 1e-3f; }

    float G1(float3 w) { return 1.f / (1.f + Lambda(w)); }
 
    float Lambda( float3 w )
    {
        float tan2Theta = Tan2Theta( w );
        if( isinf( tan2Theta ) )
            return 0;
        float alpha2 = Sqr( CosPhi( w ) * alphaX ) + Sqr( SinPhi( w ) * alphaY );
        return ( sqrt( 1.f + alpha2 * tan2Theta ) - 1.f ) / 2.f;
    }

    float G(float3 wo, float3 wi) { return 1.f / (1.f + Lambda(wo) + Lambda(wi)); }
    
    float D(float3 w, float3 wm)
    {
        return G1(w) / AbsCosTheta(w) * D(wm) * AbsDot(w, wm);
    }
    

    float Pdf(float3 w, float3 wm)
    {
        return D(w, wm);
    }

    float3 Sample_wh(float3 w, float2 u) {
        // Transform w to hemispherical configuration
        float3 wh = normalize(float3(alphaX * w.x, alphaY * w.y, w.z));
        if (wh.z < 0.f)
            wh = -wh;

	    float lensq = wh.x * wh.x + wh.y * wh.y;
        // Find orthonormal basis for visible normal sampling
        float3  T1    = lensq > 0.f ? float3( -wh.y, wh.x, 0.f ) * rsqrt( lensq ) : float3( 1.f, 0.f, 0.f );
        float3  T2    = cross( wh, T1 );

        float r   = sqrt( u.x );
        float phi = TWO_PI * u.y;
        float t1  = r * cos( phi );
        float t2  = r * sin( phi );
        float s   = 0.5f * ( 1.0f + wh.z );
        t2        = ( 1.0f - s ) * sqrt( 1.0f - t1 * t1 ) + s * t2;
        // Section 4.3: reprojection onto hemisphere
        float3 nh = t1 * T1 + t2 * T2 + sqrt( max( 0.0f, 1.0f - t1 * t1 - t2 * t2 ) ) * wh;
        return normalize( float3(alphaX * nh.x, alphaY * nh.y, max(0.00001f, nh.z)));
    }


    float RoughnessToAlpha(float roughness) { return Sqr(roughness); }

    void Regularize() {
        if (alphaX < 0.3f)
            alphaX = clamp(2.f * alphaX, 0.1f, 0.3f);
        if (alphaY < 0.3f)
            alphaY = clamp(2.f * alphaY, 0.1f, 0.3f);
    }
};

TrowbridgeReitzDistribution MakeTrowbridgeReitzDistribution(float ax, float ay)
{
    TrowbridgeReitzDistribution ret;
    ret.alphaX = ax*ax;
    ret.alphaY = ay*ay;
    
    if( !ret.EffectivelySmooth() )
    {
        // If one direction has some roughness, then the other can't
        // have zero (or very low) roughness; the computation of |e| in
        // D() blows up in that case.
        ret.alphaX = max( ret.alphaX, 1e-4f );
        ret.alphaY = max( ret.alphaY, 1e-4f );
    }
    return ret;
}

class DiffuseReflection
{
    float3 m_albedo;
    
    float3 Eval( const float3 wo, const float3 wi ) 
    {
        return !SameHemisphere( wo, wi ) ? 0.f : m_albedo * M_1_PIf;
    }
    float Pdf( const float3 wo, const float3 wi )
    {
        return SameHemisphere( wo, wi ) ? AbsCosTheta( wi ) * M_1_PIf : 0.0f;
    }

    float3 Sample_f( const float3 wo, out float3 wi, const float2 u, out float pdf )
    {
        // Cosine-sample the hemisphere, flipping the direction if necessary
        // default left handed since we are in bxdf coordinate system
        wi  = CosineSampleHemisphere( u );
        pdf = Pdf( wo, wi );
        return Eval( wo, wi );
    }
};


DiffuseReflection MakeDiffuseReflection(float3 baseColor, float metalness)
{
    DiffuseReflection ret;
    if (metalness >= 0.0)
    {
        ret.m_albedo = BaseColorToDiffuseReflectance(baseColor, metalness);
    }
    else
    {
        ret.m_albedo = baseColor;
    }
    return ret;
}

class MicrofacetReflection
{
    float3 m_f0;
    float m_f90;
    TrowbridgeReitzDistribution m_distribution;
    
    float3 SchlickFresnel( float cosTheta )
    {
        return EvalFresnel( m_f0, m_f90, cosTheta );
    }

    float3 Eval( float3 wo, float3 wi )
    {
        if( !SameHemisphere( wo, wi ) )
        {
            return 0.f;
        }
        const float cosThetaO = AbsCosTheta( wo );
        const float cosThetaI = AbsCosTheta( wi );
        // Handle degenerate cases for microfacet reflection
        if( cosThetaI <= 0.0f || cosThetaO <= 0.0f )
        {
            return 0.f;
        }
        float3 wh = normalize( wi + wo );
        wh *= sign( wh.z );
        const float3 F = SchlickFresnel( dot( wo, wh ) );
        const float3 result =
            F * m_distribution.D( wh ) * m_distribution.G( wo, wi ) / ( 4.0f * cosThetaI * cosThetaO );
        return result;
    }

    float Pdf( float3 wo, float3 wi )
    {
        if( !SameHemisphere( wo, wi ) )
        {
            return 0.f;
        }
        const float cosThetaO = AbsCosTheta( wo );
        const float cosThetaI = AbsCosTheta( wi );
        // Handle degenerate cases for microfacet reflection
        if( cosThetaI <= 0.0f || cosThetaO <= 0.0f )
        {
            return 0.f;
        }
        const float3 wh = normalize( wo + wi );
        return m_distribution.Pdf( wo, wh ) / ( 4.0f * dot( wo, wh ) );
    }

    float3 Sample_f( float3 wo, out float3 wi, float2 u, out float pdf )
    {
        const float3 wh = m_distribution.Sample_wh( wo, u );
        wi              = Reflect( wo, wh );
        if( !SameHemisphere( wo, wi ) )
        {
            pdf = 0.f;
            return 0.f;
        }
        float cosTheta_o = AbsCosTheta(wo), cosTheta_i = AbsCosTheta(wi);
        if( cosTheta_i <= 0 || cosTheta_o <= 0 )
        {
            pdf = 0.f;
            return 0.f;
        }
        pdf = m_distribution.Pdf( wo, wh ) / ( 4.0f * dot( wo, wh ) );
        const float3 F = SchlickFresnel( dot( wo, wh ) );
        const float D = m_distribution.D( wh );
        const float G = m_distribution.G(wo, wi);
        const float3 result =
            F * D * G / ( 4.0f * cosTheta_i * cosTheta_o );
        return result;
    }
};

MicrofacetReflection MakeMicrofacetReflection(float3 baseColor, float3 f0, float metalness, float roughness)
{
    MicrofacetReflection ret;
    ret.m_f0 = metalness >= 0.0 ? BaseColorToSpecularF0(baseColor, metalness) : f0;
    ret.m_f90 = ShadowedF90(ret.m_f0);
    ret.m_distribution = MakeTrowbridgeReitzDistribution(roughness, roughness);
    return ret;
}

class FresnelBlend
{
    DiffuseReflection m_diffuse;
    MicrofacetReflection m_microfacet;
    
    float Pdf(float3 wo, float3 wi)
    {
        if( !SameHemisphere( wo, wi ) )
            return 0.f;
        const float3 wh       = normalize( wo + wi );
        const float  F        = getSpecularProbability( dot( wi, wh ) );
        const float  diffuse  = m_diffuse.Pdf( wo, wi );
        const float  specular = m_microfacet.Pdf( wo, wi );
        return ( 1.f - F ) * diffuse + F * specular;
    }
    
    float getSpecularProbability(float cosTheta)
    {                  
        const float3 specularF0 = Luminance(m_microfacet.m_f0);
        float diffuseReflectance = Luminance(m_diffuse.m_albedo);
        float Fresnel = saturate(Luminance(EvalFresnel(specularF0, m_microfacet.m_f90, max(0.0f, cosTheta))));
        // Approximate relative contribution of BRDFs using the Fresnel term
        float specular = Fresnel;
        float diffuse = diffuseReflectance * (1.0f - Fresnel);

        // Return probability of selecting specular BRDF over diffuse BRDF
        float p = (specular / max(0.0001f, (specular + diffuse)));

        // Clamp probability to avoid undersampling of less prominent BRDF
        return clamp(p, 0.1f, 0.9f);
    }

    float3 Eval( float3 wo, float3 wi )
    {
        if( !SameHemisphere( wo, wi ) )
            return 0.f;
        const float3 wh = normalize( wo + wi );
        const float3 F        = m_microfacet.SchlickFresnel( dot( wi, wh ) );
        const float3 diffuse = m_diffuse.Eval(wo,wi);
        const float3 specular = m_microfacet.Eval(wo,wi);

        return ( float3( 1.f, 1.f, 1.f ) - F ) * diffuse + specular;
    }
    
    float3 Sample_f( float3 wo, out float3 wi, float2 u, out float pdf, inout uint32_t seed )
    {
        const float specularPdf = getSpecularProbability( CosTheta( wo ) );
        float3      weight      = 1.f;
        
        if( Rnd( seed ) < specularPdf )
        {
            weight *= m_microfacet.Sample_f( wo, wi, u, pdf ) / specularPdf;
        }
        else
        {
            weight *= m_diffuse.Sample_f( wo, wi, u, pdf ) / ( 1.f - specularPdf );
        }
        return weight;
    }
    
    float3 Sample_f_forceDiffuse(float3 wo, out float3 wi, float2 u, out float pdf, inout uint32_t seed)
    {
        const float specularPdf = getSpecularProbability(CosTheta(wo));
        return m_diffuse.Sample_f(wo, wi, u, pdf) / (1.f - specularPdf);
    }

    float3 Sample_f_forceSpecular(float3 wo, out float3 wi, float2 u, out float pdf, inout uint32_t seed)
    {
        const float specularPdf = getSpecularProbability(CosTheta(wo));
        return m_microfacet.Sample_f(wo, wi, u, pdf) / specularPdf;
    }

};

FresnelBlend MakeFresnelBlend(float3 baseColor, float3 f0, float metalness, float roughness)
{
    FresnelBlend ret;
    ret.m_diffuse = MakeDiffuseReflection(baseColor, metalness);
    ret.m_microfacet = MakeMicrofacetReflection(baseColor, f0, metalness, roughness);
    return ret;
}

float3 BRDFEval(MaterialSample material, float3 gN, float3 N, float3 V, float3 L, out float pdf )
{
    if( dot( gN, V ) < 0.f || dot( gN, L ) < 0.f )
    {
        pdf = 0.f;
        return 0;
    }
    
    FresnelBlend brdf = MakeFresnelBlend(material.baseColor, material.specularF0, material.metalness, material.roughness);

    const Onb    onb = MakeOnb( N );
    const float3 wo     = onb.ToLocal( V );
    const float3 wi     = onb.ToLocal( L );
    pdf                 = brdf.Pdf( wo, wi );
    const float3 weight = brdf.Eval( wo, wi ) * AbsCosTheta( wi );
    if (isnan(pdf) || any(isnan(weight)))
    {
        pdf = 0.f;
        return 0;
    }
    return weight;
}


float3 BRDFSample(MaterialSample material, float3 gN, float3 N, float3 V, out float3 L, out float pdf, inout uint32_t seed)
{
    if (dot(gN, V) < 0.f || dot(gN, L) < 0.f)
    {
        pdf = 0.f;
        return 0.f;
    }
    pdf = 0.f;
    
    FresnelBlend brdf = MakeFresnelBlend(material.baseColor, material.specularF0, material.metalness, material.roughness);

    const Onb onb = MakeOnb(N);
    const float3 wo = onb.ToLocal(V);
    
    float3 wi = 0.f;
    const float2 u = float2(Rnd(seed), Rnd(seed));
    const float3 weight = brdf.Sample_f(wo, wi, u, pdf, seed) * AbsCosTheta(wi) / pdf;
    if (isnan(pdf) || any(isnan(weight)))
    {
        pdf = 0.f;
        return 0.f;
    }
    L = onb.ToWorld(wi);
    if (dot(L, gN) < 0.f)
    {
        pdf = 0.f;
        return 0.f;
    }
    return weight;
}


float3 BRDFSampleDiffuse(MaterialSample material, float3 gN, float3 N, float3 V, out float3 L, out float pdf, inout uint32_t seed)
{
    if (dot(gN, V) < 0.f || dot(gN, L) < 0.f)
    {
        pdf = 0.f;
        return 0.f;
    }
    pdf = 0.f;
    
    FresnelBlend brdf = MakeFresnelBlend(material.baseColor, material.specularF0, material.metalness, material.roughness);

    const Onb onb = MakeOnb(N);
    const float3 wo = onb.ToLocal(V);
    
    float3 wi = 0.f;
    const float2 u = float2(Rnd(seed), Rnd(seed));
    const float3 weight = brdf.Sample_f_forceDiffuse(wo, wi, u, pdf, seed) * AbsCosTheta(wi) / pdf;
    if (isnan(pdf) || any(isnan(weight)))
    {
        pdf = 0.f;
        return 0.f;
    }
    L = onb.ToWorld(wi);
    if (dot(L, gN) < 0.f)
    {
        pdf = 0.f;
        return 0.f;
    }
    return weight;
}


float3 BRDFSampleSpecular(MaterialSample material, float3 gN, float3 N, float3 V, out float3 L, out float pdf, inout uint32_t seed)
{
    if (dot(gN, V) < 0.f || dot(gN, L) < 0.f)
    {
        pdf = 0.f;
        return 0.f;
    }
    pdf = 0.f;
    
    FresnelBlend brdf = MakeFresnelBlend(material.baseColor, material.specularF0, material.metalness, material.roughness);

    const Onb onb = MakeOnb(N);
    const float3 wo = onb.ToLocal(V);
    
    float3 wi = 0.f;
    const float2 u = float2(Rnd(seed), Rnd(seed));
    const float3 weight = brdf.Sample_f_forceSpecular(wo, wi, u, pdf, seed) * AbsCosTheta(wi) / pdf;
    if (isnan(pdf) || any(isnan(weight)))
    {
        pdf = 0.f;
        return 0.f;
    }
    L = onb.ToWorld(wi);
    if (dot(L, gN) < 0.f)
    {
        pdf = 0.f;
        return 0.f;
    }
    return weight;
}

// Return approximated preintegrated specular term which assumes that light is aligned with normal.
// This provides the denoiser's expected input for the specular buffer.
//
// From [Ray Tracing Gems, Chapter 32]
// const float3 vIBLSpecularTerm = EnvBRDFApprox(specularColor, roughness * roughness, dot(N, V));
float3 BRDFEnvApprox(FresnelBlend brdf, float3 N, float3 V)
{ 
    // specularColor is the reflectance from a direction parallel to the normal.
    float3 specularColor = brdf.m_microfacet.m_f0;
        
    // alpha is the square of the linear roughness in the GGX model.
    // This approximation only supports isotropic alpha, use max alpha.
    float alpha = max(brdf.m_microfacet.m_distribution.alphaX, brdf.m_microfacet.m_distribution.alphaY);
        
    float NoV = abs(dot(N, V));
    float4 X;
    X.x = 1.f;
    X.y = NoV;
    X.z = NoV * NoV;
    X.w = NoV * X.z;
 
    float4 Y;
    Y.x = 1.f;
    Y.y = alpha;
    Y.z = alpha * alpha;
    Y.w = alpha * Y.z;
 
    float2x2 M1 = float2x2(0.99044f, -1.28514f, 1.29678f, -0.755907f);
    float3x3 M2 = float3x3(1.f, 2.92338f, 59.4188f, 20.3225f, -27.0302f, 222.592f, 121.563f, 626.13f, 316.627f);
 
    float2x2 M3 = float2x2(0.0365463f, 3.32707, 9.0632f, -9.04756);
    float3x3 M4 = float3x3(1.f, 3.59685f, -1.36772f, 9.04401f, -16.3174f, 9.22949f, 5.56589f, 19.7886f, -20.2123f);
 
    float bias = dot(mul(M1, X.xy), Y.xy) * rcp(dot(mul(M2, X.xyw), Y.xyw));
    float scale = dot(mul(M3, X.xy), Y.xy) * rcp(dot(mul(M4, X.xzw), Y.xyw));
 
    return mad(specularColor, max(0, scale), max(0, bias));
}

#endif // RTXMG_BRDF_HLSLI
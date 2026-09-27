/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <vector>

// clang-format on

// A generic data sampler with basic statistics functionality

// default running avg window = 1 second @ 60Hz
template <typename T, size_t _size = 60>
struct Sampler : public std::array<T, _size>
{
    std::string name;

    // values tracked in running circular buffer
    T samples_sum = T( 0 );

    // values tracked since most recent reset
    size_t samples_count = 0;
    T      latest        = {};
    T      total         = T( 0 );
    T      min           = std::numeric_limits<T>::max();
    T      max           = std::numeric_limits<T>::lowest();

    void PushBack( T sample );

    void Reset();

    // Over the retained window, not the whole history: only the last _size
    // samples are still around to sort.
    T Median() const { return Percentile( 0.5 ); }
    T Percentile( double p ) const
    {
        const size_t n = std::min( _size, samples_count );
        if( n == 0 )
            return T( 0 );
        std::vector<T> sorted( this->begin(), this->begin() + n );
        const size_t   k = std::min( n - 1, size_t( p * double( n ) ) );
        std::nth_element( sorted.begin(), sorted.begin() + k, sorted.end() );
        return sorted[k];
    }
    T Average() const { return static_cast<T>( double( total ) / double( samples_count ) ); }
    T RunningAverage() const
    {
        return static_cast<T>( double( samples_sum ) / double( std::min( samples_count, _size ) ) );
    }

    // current position in circular buffer
    uint32_t Offset() const { return static_cast<uint32_t>( samples_count % _size ); }

    void Print()
    {
        size_t n = std::min( _size, samples_count );
        for(size_t i = 0; i < n; i++ )
        {
            std::printf( "Benchmark: frame %d time %.4f ms\n", (int)i, (*this)[i]);
        }
    }
};

template <typename T, size_t _size>
inline void Sampler<T, _size>::PushBack( T sample )
{
    latest = sample;
    total += latest;
    min = std::min( latest, min );
    max = std::max( latest, max );

    samples_sum += sample;

    if( samples_count < _size )
        ( *this )[samples_count++] = sample;
    else
    {
        T& oldest = ( *this )[samples_count++ % _size];
        samples_sum -= oldest;
        oldest = sample;
    }
}

template <typename T, size_t _size>
inline void Sampler<T, _size>::Reset()
{
    samples_count = 0;
    latest        = {};
    samples_sum   = T( 0 );
    min           = std::numeric_limits<T>::max();
    max           = std::numeric_limits<T>::lowest();
#if !defined( NDEBUG )
    this->fill( T( 0 ) );
#endif
}
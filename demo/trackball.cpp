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
//

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>


#include "trackball.h"

enum class KeyboardControls : uint8_t
{
    MoveUp = 0,
    MoveDown,
    MoveLeft,
    MoveRight,
    MoveForward,
    MoveBackward,
    RollLeft,
    RollRight,
    SpeedUp,
    SlowDown,
    OrbitMode,
    Count
};

static std::array<bool, int(KeyboardControls::Count)> keyboardState = { false };

enum class MouseButtons : uint8_t { Left = 0, Middle, Right, Count };

static std::array<bool, int(MouseButtons::Count)> mouseButtonsState = { false };

void Trackball::KeyboardUpdate(int key, int code, int action, int mods)
{
    static const std::array<std::pair<int, int>, 18> keyboardMap = { {
        {GLFW_KEY_Q, int(KeyboardControls::MoveDown)},
        {GLFW_KEY_E, int(KeyboardControls::MoveUp)},
        {GLFW_KEY_A, int(KeyboardControls::MoveLeft)},
        {GLFW_KEY_D, int(KeyboardControls::MoveRight)},
        //        { GLFW_KEY_DOWN,          int(KeyboardControls::MoveDown) },
        //        { GLFW_KEY_UP,            int(KeyboardControls::MoveUp) },
        //        { GLFW_KEY_LEFT,          int(KeyboardControls::MoveLeft) },
        //        { GLFW_KEY_RIGHT,         int(KeyboardControls::MoveRight) },
        {GLFW_KEY_W, int(KeyboardControls::MoveForward)},
        {GLFW_KEY_S, int(KeyboardControls::MoveBackward)},
        {GLFW_KEY_Z, int(KeyboardControls::RollLeft)},
        {GLFW_KEY_X, int(KeyboardControls::RollRight)},
        {GLFW_KEY_LEFT_SHIFT, int(KeyboardControls::SpeedUp)},
        {GLFW_KEY_RIGHT_SHIFT, int(KeyboardControls::SpeedUp)},
        {GLFW_KEY_LEFT_CONTROL, int(KeyboardControls::SlowDown)},
        {GLFW_KEY_RIGHT_CONTROL, int(KeyboardControls::SlowDown)},
        {GLFW_KEY_LEFT_ALT, int(KeyboardControls::OrbitMode)},
        {GLFW_KEY_RIGHT_ALT, int(KeyboardControls::OrbitMode)},
    } };

    auto it = std::find_if(
        keyboardMap.begin(), keyboardMap.end(),
        [&key](std::pair<int, int> control) { return control.first == key; });

    if (it != keyboardMap.end())
        keyboardState[it->second] = (action == GLFW_PRESS || action == GLFW_REPEAT);
}

void Trackball::MouseButtonUpdate(int button, int action, int)
{
    static const std::array<std::pair<int, int>, 3> mouseButtonsdMap = { {
        {GLFW_MOUSE_BUTTON_LEFT, int(MouseButtons::Left)},
        {GLFW_MOUSE_BUTTON_MIDDLE, int(MouseButtons::Middle)},
        {GLFW_MOUSE_BUTTON_RIGHT, int(MouseButtons::Right)},
    } };

    assert(action == GLFW_PRESS || action == GLFW_RELEASE);

    auto it = std::find_if(mouseButtonsdMap.begin(), mouseButtonsdMap.end(),
        [&button](std::pair<int, int> control)
        {
            return control.first == button;
        });

    if (it != mouseButtonsdMap.end())
    {
        mouseButtonsState[it->second] = action == GLFW_PRESS;
        if (action == GLFW_RELEASE)
            m_performTracking = false;
    }
}

void Trackball::Animate(float deltaT)
{
    if (!m_roamMode)
        return;

    float moveSpeed = deltaT * m_moveSpeed;
    float rollSpeed = deltaT * m_rollSpeed;

    if (keyboardState[int(KeyboardControls::SpeedUp)])
    {
        moveSpeed *= 3.f;
        rollSpeed *= 3.f;
    }

    if (keyboardState[int(KeyboardControls::SlowDown)])
    {
        moveSpeed *= .1f;
        rollSpeed *= .1f;
    }

    auto const& basis = m_camera->GetBasis();

    if (keyboardState[int(KeyboardControls::MoveForward)])
        m_camera->Translate(m_camera->GetDirection() * moveSpeed);
    if (keyboardState[int(KeyboardControls::MoveBackward)])
        m_camera->Translate(-m_camera->GetDirection() * moveSpeed);

    if (keyboardState[int(KeyboardControls::MoveLeft)])
        m_camera->Translate(-normalize(basis[0]) * moveSpeed);
    if (keyboardState[int(KeyboardControls::MoveRight)])
        m_camera->Translate(normalize(basis[0]) * moveSpeed);

    if (keyboardState[int(KeyboardControls::MoveUp)])
        m_camera->Translate(normalize(basis[1]) * moveSpeed);
    if (keyboardState[int(KeyboardControls::MoveDown)])
        m_camera->Translate(-normalize(basis[1]) * moveSpeed);

    if (keyboardState[int(KeyboardControls::RollLeft)])
        m_camera->Roll(rollSpeed);
    if (keyboardState[int(KeyboardControls::RollRight)])
        m_camera->Roll(-rollSpeed);
}

void Trackball::MouseTrackingUpdate(int2 pos, int2 canvasSize)
{
    if (!m_performTracking)
    {
        ReinitOrientationFromCamera();
        m_performTracking = true;
        return;
    }

    m_delta = pos - m_prevPos;
    m_prevPos = pos;
    m_pos = pos;

    UpdateCamera(pos, canvasSize);
}

bool Trackball::MouseWheelUpdate(int dir)
{
    Zoom(dir);
    return true;
}

float3 Trackball::GetCameraDirection() const
{
    // use lat/long for view definition
    float3 localDir;
    localDir.x = cos(m_latitude) * sin(m_longitude);
    localDir.y = cos(m_latitude) * cos(m_longitude);
    localDir.z = sin(m_latitude);

    return m_u * localDir.x + m_v * localDir.y + m_w * localDir.z;
}

void Trackball::ApplyGimbalLock()
{
    if (!m_gimbalLock)
    {
        ReinitOrientationFromCamera();
        if (m_camera->HasChanged())
            m_camera->SetUp(m_w);
    }
}

static inline float3 getUnitVector(int2 pos, int2 canvas)
{
    float3 ret;
    ret.x = 2.0f * float(pos.x) / float(canvas.x) - 1.0f;
    ret.y = 2.0f * float(pos.y) / float(canvas.y) - 1.0f;
    ret.z = 0.0f;
    float length = ret.x * ret.x + ret.y * ret.y;
    if (length <= 1)
    {
        ret.z = sqrt(1.0f - length);
        length += ret.z * ret.z;
    }
    else
    {
        float norm = 1.0f / sqrt(length);
        ret.x *= norm;
        ret.y *= norm;
    }
    return ret;
}

void Trackball::UpdateCamera(int2 pos, int2 canvas)
{
    if (m_roamMode == false || keyboardState[int(KeyboardControls::OrbitMode)])
    {
        if (mouseButtonsState[int(MouseButtons::Left)])
        {
            m_latitude = radians(std::min(89.0f, std::max(-89.0f, degrees(m_latitude) + 0.5f * m_delta.y)));
            m_longitude = radians(fmod(degrees(m_longitude) - 0.5f * m_delta.x, 360.0f));

            float3 dirWS = GetCameraDirection();

            m_camera->SetEye(m_camera->GetLookat() + dirWS * m_cameraEyeLookatDistance);

            ApplyGimbalLock();
        }
        else if (mouseButtonsState[int(MouseButtons::Middle)])
        {
            float2 delta = { float(m_delta.x) / float(canvas.x),
                            float(-m_delta.y) / float(canvas.y) };
            m_camera->Pan(delta);
        }
        else if (mouseButtonsState[int(MouseButtons::Right)])
        {
            float factor = float(m_delta.x) / float(canvas.x) +
                float(m_delta.y) / float(canvas.y);
            constexpr float const dollySpeed = 2.f;
            m_camera->Dolly(1.f - dollySpeed * factor);
        }
    }
    else if (m_roamMode == true)
    {
        if (mouseButtonsState[int(MouseButtons::Left)])
        {
            m_latitude = radians(std::min(
                89.0f, std::max(-89.0f, degrees(m_latitude) + 0.5f * m_delta.y)));
            m_longitude =
                radians(fmod(degrees(m_longitude) - 0.5f * m_delta.x, 360.0f));

            float3 dirWS = GetCameraDirection();

            m_camera->SetLookat(m_camera->GetEye() -
                dirWS * m_cameraEyeLookatDistance);

            ApplyGimbalLock();
        }
    }
}

void Trackball::SetReferenceFrame(const float3& u, const float3& v,
    const float3& w)
{
    m_u = u;
    m_v = v;
    m_w = w;
    float3 dirWS = -m_camera->GetDirection();
    float3 dirLocal;
    dirLocal.x = dot(dirWS, u);
    dirLocal.y = dot(dirWS, v);
    dirLocal.z = dot(dirWS, w);
    m_longitude = atan2(dirLocal.x, dirLocal.y);
    m_latitude = asin(dirLocal.z);
}

void Trackball::Zoom(int direction)
{
    float zoom = (direction > 0) ? 1 / m_zoomMultiplier : m_zoomMultiplier;
    m_cameraEyeLookatDistance *= zoom;
    const float3& lookat = m_camera->GetLookat();
    const float3& eye = m_camera->GetEye();
    m_camera->SetEye(lookat + (eye - lookat) * zoom);
}

void Trackball::ReinitOrientationFromCamera()
{
    auto const& basis = m_camera->GetBasis();
    m_u = normalize(basis[0]);
    m_v = normalize(basis[1]);
    m_w = -normalize(basis[2]);
    std::swap(m_v, m_w);
    m_latitude = 0.0f;
    m_longitude = 0.0f;
    m_cameraEyeLookatDistance =
        length(m_camera->GetLookat() - m_camera->GetEye());
}

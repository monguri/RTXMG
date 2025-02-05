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

#pragma once

#include <array>

#include <donut/core/math/math.h>

#include "rtxmg/scene/camera.h"

using namespace donut::math;

class Camera;

class Trackball
{
public:
    void KeyboardUpdate(int key, int code, int action, int mods);
    void MouseButtonUpdate(int button, int action, int mods);
    void MouseTrackingUpdate(int2 pos, int2 canvasSize);
    bool MouseWheelUpdate(int dir);

    void Animate(float deltaT);

    void Zoom(int direction);

    float MoveSpeed() const { return m_moveSpeed; }
    void SetMoveSpeed(float speed) { m_moveSpeed = speed; }

    float RollSpeed() const { return m_rollSpeed; }
    void SetRollSpeed(float speed) { m_rollSpeed = speed; }

    // Set the camera that will be changed according to user input.
    // Warning, this also initializes the reference frame of the trackball from
    // the camera. The reference frame defines the orbit's singularity.
    inline void SetCamera(Camera* camera)
    {
        m_camera = camera;
        ReinitOrientationFromCamera();
    }
    inline const Camera* CurrentCamera() const { return m_camera; }

    // Setting the gimbal lock to 'on' will fix the reference frame (i.e., the
    // singularity of the trackball). In most cases this is preferred. For free
    // scene exploration the gimbal lock can be turned off, which causes the
    // trackball's reference frame to be Update on every camera Update (adopted
    // from the camera).
    bool GimbalLock() const { return m_gimbalLock; }
    void SetGimbalLock(bool val) { m_gimbalLock = val; }

    // Adopts the reference frame from the camera.
    // Note that the reference frame of the camera usually has a different 'up'
    // than the 'up' of the camera. Though, typically, it is desired that the
    // trackball's reference frame aligns with the actual up of the camera.
    void ReinitOrientationFromCamera();

    // Specify the frame of the orbit that the camera is orbiting around.
    // The important bit is the 'up' of that frame as this is defines the
    // singularity. Here, 'up' is the 'w' component. Typically you want the up of
    // the reference frame to align with the up of the camera. However, to be able
    // to really freely move around, you can also constantly Update the reference
    // frame of the trackball. This can be done by calling
    // reinitOrientationFromCamera(). In most cases it is not required though (set
    // the frame/up once, leave it as is).
    void SetReferenceFrame(const float3& u, const float3& v, const float3& w);

    // In 'roam' mode, the mouse moves the camera in first-person mode and the
    // 'alt' key must be held to move in third-person mode (orbit). 'roam' mode
    // off disables first person movement (keyboard & mouse), but orbiting no
    // longer requires holding the 'alt' key.
    void SetRoamMode(bool roam) { m_roamMode = roam; }
    bool GetRoamMode() const { return m_roamMode; }

private:
    float3 GetCameraDirection() const;

    void ApplyGimbalLock();

    void UpdateCamera(int2 pos, int2 canvasSize);

private:
    bool m_roamMode = true;
    bool m_gimbalLock = true;
    Camera* m_camera = nullptr;
    float m_cameraEyeLookatDistance = 0.0f;
    float m_zoomMultiplier = 1.1f;
    float m_moveSpeed = 1.0f;
    float m_rollSpeed = 180.f / 5.f;

    float m_latitude = 0.0f;  // in radians
    float m_longitude = 0.0f; // in radians

    // mouse tracking
    bool m_performTracking = false;
    int2 m_pos = { 0, 0 };
    int2 m_prevPos = { 0, 0 };
    int2 m_delta = { 0, 0 };

    // trackball computes camera orientation (eye, lookat) using
    // latitude/longitude with respect to this frame local frame for trackball
    float3 m_u = { 0.0f, 0.0f, 0.0f };
    float3 m_v = { 0.0f, 0.0f, 0.0f };
    float3 m_w = { 0.0f, 0.0f, 0.0f };
};

#pragma once

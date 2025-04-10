# RTX Mega Geometry SDK Change Log

## 0.9.2

Performance Improvements 

Test Scene: amy_kitchenset.scene.json (default camera: 79M microtriangles) 
Hardware: RTX 5090 @ 4K Render Resolution (r572.83)

* Compute Cluster tiling: 4.0ms -> 1.0ms (300% speedup)
    * Coalesced UAV writes for structs, unaligned members were causing UAV readbacks
    * Coalesce per wave atomics into a groupshared atomic to reduce pressure on global/UAV atomics by 4x
* Fill clusters: 4.9ms to -> 1.0ms (390% speedup)
    * Fixed cases where the compiler was unable to unroll loops due to dynamic loop counts
    * Specialized shaders by subdivision surface type, with a special path for Pure BSpline surfaces. Prefetch all control points into shared memory to be used wave wide.

Minor Fixes

* Fix SpecularHitT guide buffer to DLSS-RR to improve coherence of specular rays.

## 0.9.1

Stability
* Fix crash on scenes with multiple subdivision mesh instances when the topology quality color mode was selected
* Fix a crash if scene with audio is loaded and no audio devices are present.

Topology Quality
* Add button in the Subdivision Evaluator tab to switch to topology quality view if issues are detected
* Fix Subdivision Evaluator UI not resetting upon scene load.

UI
* Application Window Size/Maximized/Fullscreen and Window state is now saved to and restored from imgui.ini. Delete imgui.ini to reset layout

Minor
* Make initial VRAM check non-fatal but add warnings about performance degradation, memory budgets.
* Made localToWorld transform use Matrix3x4 for consistency
* Style clean-up
* Update donut version

## 0.9.0

Initial beta release.

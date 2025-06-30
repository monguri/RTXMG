# RTX Mega Geometry SDK Change Log

## 1.0.0

Vulkan Support
* Requires Vulkan SDK 1.4.313 or greater which uses Cluster SPIRV intrinsics
* Convert all bindless arrays to use ResourceDescriptorHeap via Vulkan's mutable descriptor extension
* Fix validation warnings and Vulkan shutdown crashes

Minor Changes
* Expose isolation level in the UI
* Add smooth single crease sharpness, which prevents transition artifacts between single and multicrease edges. Only visible if the isolation level is lowered.
* Improve Profiler "Frame" Tab, add average times in tool tip, expose motion vector pass time
* Update to Streamline v2.8.0

Bug Fixes
* Fix thread-ordering to be compliant with SM6.6 1D quad lane ordering.
* Fix micro-triangle view toggle when in DLAA mode.
* Fix cases where a surface resulted in over U16_MAX clusters
* Fix tessellation for when there was per-material displacement scaling
* Fix crash for some malformed OBJ files with '0' values for some indices.

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

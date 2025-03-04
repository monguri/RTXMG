# RTXMG SDK Change Log

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

# Debugging Guide

How to get useful diagnostics out of `rtxmg_demo`, and which tool to reach for when.
See [README](../README.md) for build steps, [Quick Start](QuickStart.md) for the UI these
options mirror, and [ClusterLOD.md](ClusterLOD.md) for what the Cluster LOD ones act on.

## Running headless

`rtxmg_demo.exe` opens a window and never closes it on its own, so anything automated has
to bound the run:

```
rtxmg_demo.exe -mf cluster_lod/ABeautifulGame/glTF/ABeautifulGame.gltf -nf 120 -s out.png
```

- `-nf <n>` exits after n frames. It performs a full graceful shutdown, so shutdown-time
  diagnostics — debug-layer live-object reports, crash dumps — are actually written.
- `-s <file>` and `-sui <file>` save a screenshot without and with the UI, on exit.
- `--dump-stats <file.json>` writes the profiler timings, the streaming and memory blocks
  and the per-frame counter ladder, which is what to diff between two runs.

The executable is built as a GUI-subsystem binary, so its standard output does **not**
survive a pipe — `| tee` and `| grep` come back empty. Redirect to a file and search the
file instead:

```
rtxmg_demo.exe ... -nf 120 > run.log 2>&1
```

Capture once and search the log repeatedly. Re-running to grep a different pattern wastes
a full scene load, and the path tracer accumulates samples, so two runs are not
bit-identical anyway.

## Validation layers

Three independent switches, covering three different classes of failure.

| option | enables | catches |
| - | - | - |
| `-d` / `--debug` | graphics API debug layer + the NVRHI validation layer | resource state, barriers, binding sets, descriptor and API misuse |
| `-gd` / `--gpudebug` | GPU-based validation + ray tracing validation | out-of-bounds shader access, and per-CLAS cluster-builder argument violations |
| `--aftermath` | Aftermath GPU crash dumps | post-mortem state after a device removal or hang |

**Start with `--debug`.** Most corruption that shows up later as a wrong image or a device
removal begins as a resource-state or binding mistake, and the validation message names
the buffer, dispatch or shader involved.

**Escalate to `-gd`** when `--debug` is clean but geometry is still wrong or the device is
still being removed. Ray tracing validation is the only thing that inspects cluster and
CLAS build arguments, and it reports the offending argument index, build index and
primitive index rather than leaving you to infer them. It is the first tool to reach for
on a device removal in the Cluster LOD streaming path. `--debug` and `-gd` are independent
and combine cleanly; GPU-based validation is slow, so budget extra wall-clock for it.

**Aftermath is exclusive with the others.** Its crash-dump SDK cannot attach while the
graphics API validation layer is active, so `--aftermath` and `--debug` must not be
combined — the log will say the Aftermath initialize call failed. On Vulkan, attaching
Aftermath also stops the driver exposing ray tracing validation, so do not combine it
with `-gd` either. That one fails **silently**: the run still produces a dump, and the
log says ray tracing validation is inert this run, so an `-gd --aftermath` run reporting
no cluster-validation errors has proved nothing. Pick one per run. The crash-dump path is
printed to standard output.

Validation also perturbs timing enough to hide race-sensitive bugs. If something
reproduces bare but not under `--debug`, try `-gd` alone before concluding anything.

### Vulkan

`-vk` selects the Vulkan back-end. `--debug` enables the Vulkan validation layers, and
`-gd` additionally requests `VK_NV_ray_tracing_validation`; the log says either
`VK_NV_ray_tracing_validation: ENABLED` or a warning that the driver did not expose it —
check which, because an inert run reports no errors and proves nothing. Its messages
arrive only through the debug-utils messenger, so they are the debug-utils lines *without*
a `VUID-` prefix.

For GPU faults specifically, the Vulkan analogue of `-gd`'s GPU-based validation is the
validation layers' GPU-assisted validation, enabled by an environment variable:

```
set VK_LAYER_GPUAV_ENABLE=1
rtxmg_demo.exe -vk ... --debug
```

It instruments shaders with bounds and descriptor checks and clamps bad accesses, so a run
that would have removed the device instead keeps going and logs the offending HLSL file
and line.

## Cluster LOD diagnostics

### Per-frame logging

`--debug-clusterlod` turns on verbose per-frame readbacks. It forces extra synchronization
and is very slow, so use it on short runs. The streams worth knowing:

| line | reports |
| - | - |
| `ClusterLod traversal frame=…` | traversal counters: nodes and groups visited, clusters emitted, BLAS sharing providers and consumers |
| `ClusterLodStreaming request-stage frame=…` | what was requested, what was staged, and why a load was refused |
| `ClusterLodStreaming update-apply frame=…` | what the applied streaming task actually carried |
| `ClusterLodStreaming request-overflow frame=…` | traversal asked for more loads than the per-frame limit allows, and how many were dropped |

`renderedClusters` counts what traversal emitted **for BLAS building this frame**, not
what is on screen. With BLAS caching on it legitimately falls towards zero as a scene
converges, and with sharing on it is deduplicated across instances. It is only comparable
between two runs configured the same way. To judge residency instead, use `--dump-stats`
and compare `uniqueClusters` / `totalClusters`, which track what the TLAS references.

### Comparing against a non-streaming reference

`--preload` uploads the whole LOD hierarchy at load time, skipping request emission,
staging and streaming allocation. If a streamed frame looks wrong, render the same view
with `--preload`: if the preloaded frame is correct, the fault is in streaming rather
than in LOD selection, traversal or shading.

Two things make that comparison meaningful:

- **Turn the BLAS reuse strategies off on both sides** (`--no-blassharing`
  `--no-blascaching` `--no-blasmerging`). They resolve differently on the two paths, so
  leaving them on compares reuse policy rather than the geometry each path selected.
- **Turn occlusion culling off on both sides** (`--no-hiz-occlusion`). It is a temporal
  feedback loop, and the two paths reach it with different histories, so they settle on
  slightly different — both correct — sets of visible geometry.

### Seeing the state

`-cm lod` colours hits by LOD level, `-cm group` by cluster group, and `-cm blas` /
`-cm blascached` by which BLAS served the hit. Between them they answer "is this the
level of detail I expected, and where did its acceleration structure come from" without
any logging. `--show-occlusion-depth` displays the depth pyramid the occlusion test reads,
which is the fastest way to see whether something was wrongly culled.

The profiler's Streaming tab carries the same information numerically — pool occupancy,
resident groups and clusters, transfer rates, and how many BLAS were built this frame.
See [ClusterLOD.md — Visualizing Cluster LOD state](ClusterLOD.md#visualizing-cluster-lod-state).

### The cluster cache

Two cache behaviours cause confusing results rather than errors:

- Without `--cache-dir`, the cache is written next to the source asset. On a read-only or
  network asset tree that means re-baking on every load.
- Shard file names are derived from the source geometry only, so loading an existing cache
  with different bake settings silently **re-bakes over it** after logging a warning. A
  run configured differently from the one that filled the cache destroys it. Give each
  bake configuration its own `--cache-dir`.

## Shader debugging

`-dp <x> <y>` sets the shader-debug predicate pixel and fires a viewport pick at it. The
coordinates are display coordinates and are scaled to the render target internally; the
debug buffers are read back on the last frame of `-nf`. Pass `--dlssMode DLAA` for a 1:1
render-to-display ratio, so the pixel you name is exactly the pixel in the saved
screenshot.

The shader-debug print ring is compiled in by default and can be turned off with the
`RTXMG_SHADER_DEBUG` CMake option. `RTXMG_DEV_FEATURES` compiles in additional
development-only diagnostics and is off by default.

## Where to start

1. Reproduce headlessly with `-nf` and a redirected log, so the failure is repeatable and
   the evidence is on disk.
2. Run with `--debug`. Fix whatever it names before believing anything else.
3. If it is clean and geometry is still wrong or the device is removed, run with `-gd`.
4. If both are clean, isolate the path: does `--preload` render it correctly? If yes, the
   fault is in streaming; if no, it is in traversal, LOD selection or shading.
5. If the device is still being removed with no validation message, run with `--aftermath`
   alone and analyse the crash dump.

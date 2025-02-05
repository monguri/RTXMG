# Quick Start Guide

See [README](../README.md) for cloning & build steps.

## Path-tracer sample

This sample uses NVAPI cluster template extensions (RTXMG) to handle very large 
updates to the ray-tracing acceleration structure (BVH).

This enables real-time path-tracing of subdivision surfaces with displacement 
mapping for the first time.
(see [OpenSubdiv](https://graphics.pixar.com/opensubdiv/docs/intro.html))

### Getting started

On startup, the sample application should be displaying the user-interface in
default mode.
The interface is self-documenting : hover the mouse over widgets to access
detailed tool-tips.

![overview](./images/ui.jpg)


### Finding & loading content

On load, the application scans the `assets` folder located at the root of the
project and automatically populates the `scene` pull-down menu with compatible
assets. Additional asset file can be placed in this folder hierarchy and will
be detected on the next launch of the application.

![](./images/scene_loading.png)

> [!TIPS]
> The application will attempt to locate a media folder named ‘assets’; if
> unsuccessful, the path to the folder will be blank and the ‘Scene’ pull-down
> menu will remain empty.

### Camera controls

The camera follows the following conventions:

  | First-person mode  |                                                          |
  | ------------------ | -------------------------------------------------------- |
  | WASD               | forward / strafe left / backward / strafe right movement |   
  | LMB + drag         | orient the view                                          |   

  | Orbit mode         |                                                          |
  | ------------------ | -------------------------------------------------------- |
  | alt + LMB          | orbit around origin                                      |   
  | alt + MMP          | pan                                                      | 
  | mouse wheel        | dolly                                                    |

### Keyboard shortcuts

For the complete list, refer to the source code in `RTXMGDemoApp::KeyboardUpdate`

|            |                                                               |
| ---------- | ------------------------------------------------------------- |
| esc        | terminate application                                         |
| f          | reset camera                                                  |
| 1          | cycle shading modes                                           |
| 2          | cycle color modes forward                                     |
| /          | toggle dynamic tessellation ('freeze tess camera')            |
|            |                                                               |

### Profiler

The sample application tracks a number of run-time performance metrics that can
be accessed through the profiler window.

The profiler window can be toggled from the button next to the FPS counter: 

![profiler ctrl](./images/profiler_button.png)

Profiling data is split into the following tabs:
  * Frame : overall frame performance (ray-trace time, denoiser, ...)
  * AccelBuilder : granular breakdown of the BVH build
  * Evaluator : specifics of subdivision surface evaluation
  * Memory : tessellation & BVH memory usage


![profiler UI](./images/profiler_ui.png)

---

## Dynamic tessellation

![tessellation](./images/tessellation.png)

Dynamic tessellation generates new topology each frame, which forces a complete
rebuild of very large sections of the BVH. NVAPI's Mega-Geometry CLAS extensions
to the ray-tracing acceleration structure build system are crucial to meeting the
throughput requirements. However, partitioning large meshes into clusters is only
part of the solution. Tessellation algorithms generally produce large amounts of
repeating topology patterns: it is possible to take advantage of such a structured
output.

> [!IMPORTANT]
> All tessellation in this example is performed with generic compute : access
> to the fixed-function tessellation hardware is still accessible through the
> rasterizer's Hull and Domain shaders only.
> See: [DirectX Graphics Pipeline](https://learn.microsoft.com/en-us/windows/uwp/graphics-concepts/graphics-pipeline)

### Structured & Unstructured Clusters

Because Catmull-Clark surfaces are exclusively quad-based, we can define a
finite working set of rectangular tiles. Each tile in this set is a regular
grid of triangles, with the set covering all the combinations of M x N 
micro-triangle resolutions, up to a maximum of 8 x 8. 

Because the set can be defined with a finite number of re-usable cluster
topologies, we refer to these as 'structured clusters', as opposed to
'unstructured clusters', where no such finite set can be composed (ex. clusters
generated from photogrammetry data). The topology of the clusters itself can be
entirely arbitrary though, and any pattern can be used (ex: barycentric 
triangulation instead of quadrangulation). 

|  Quad cluster grids  |  Triangle cluster grids  |
| :---: | :---: |
| ![tessellation quads](./images/tessellation_quads.png) | ![tessellation triangles](./images/tessellation_triangles.png) |


### Cluster Template API

The Mega-Geometry NVAPI extensions expose a special BVH build path for structured
clusters, with substantial build performance advantages : the `CLAS` templates.

On application startup, each tiled grid configuration is passed to the `CLAS`
template builder (using NVAPI), which generates a uniquely identified
template specific to the triangle topology of that grid. The `CLAS` template
contains no vertex position data, only topology (the index buffer).

Each template represents a partially optimized BVH treelet that can be
instantiated at run-time to any `CLAS` of matching topology. This allows the
application to amortize enormous amounts of BVH build work, compounding each
time a given template is instantiated with new or updated vertex positions.

![bvh build](./images/clas_build.png)

This template `CLAS` workflow provides build performance comparable with 
traditional `BVH` refit, but is more flexible because it permits topology
changes. It also has lower memory usage and more stable traversal performance
versus traditional refit, and only clusters that are animated need to be
rebuilt. It does, however, rely on a good clustering of the mesh by the
application, where triangles in a cluster are near each other in space!

---

## Frame Pipeline

![pipeline](./images/pipeline.png)

1. Subdivision mesh animation:   
   Each frame starts with the interpolation of the subdivision control mesh (if animated).
   The computation of the limit surface samples is integrated within the tessellation
   algorithm, but follows the methods described in:
   [Efficient GPU Rendering of Subdivision Surfaces using Adaptive Quadtrees](https://dl.acm.org/doi/10.1145/2897824.2925874)
   (ACM TOG, Vol 35, Issue 4).

2. Cluster tiling:
   Applies heuristics to determine the tessellation edge-rates for each quadrangulated
   face of the control mesh:
   - Evaluate the limit surface at the locations of the 4 corners and 4 edge mid-points
   - Apply displacement, if any
   - Measure the lengths of the 8 projected segments in screen-space
   - Check against visibility oracles (frustum, Hierarchical Z-Buffer, ...)
   - Compute the final tessellation rates for the 4 edges
   - If necessary, split into cluster tiles (max tile size is 8x8)
   - Export clusters lists

3. Fill clusters:
   Computes the vertex positions for each cluster exported from step 2:
   - Evaluate the limit surface at all the micro-vertex locations in the cluster grid
   - Use polynomial basis derivatives to compute analytically surface tangents
     & normals
   - Apply displacement, if any
   - Export vertex & texcoord buffers

4. `CLAS` instantiation:
   Compresses each cluster's data in `CLAS` templates:
   - Select the cluster's template based on the cluster MxN edge-rate
   - Fill the CLAS descriptor in GPU memory with the template ID and
     pointers to the vertex data
   - Call NVAPI CLAS template 'indirect' build function to launch the
     build   

5. `BLAS` build:
    - Collect the pointers of the `CLAS`es instantiated in stage 4
    - Fill the `BLAS` build descriptor (also in GPU memory)
    - Call the NVAPI 'indirect' build function to launch the `BLAS`
      build

6. `TLAS` build:
    Build the `TLAS` using the same NVAPI 'indirect' pattern.

---

## Appendix

References to the file format extensions supported by the sample application.

### OBJ extensions

We are using an extended version of Autodesk's `OBJ` file format with a “tag” system
to express all information specific to subdivision surfaces. A Maya export plugin
could be made available upon request.

> [!NOTE]
> This is the same tagging system that is used in
> [OpenSubdiv](https://graphics.pixar.com/opensubdiv/docs/intro.html)

The tag syntax is as follows:

```
 t <tag name> <num int args>/<num float args>/<num string args> <args> 
```

Some examples:
```
# select vertex boundary interpolation mode VTX_BOUNDARY_EDGE_AND_CORNER
# 0 : VTX_BOUNDARY_NONE
# 1 : VTX_BOUNDARY_EDGE_AND_CORNER
# 2 : VTX_BOUNDARY_EDGE_ONLY
t interpolateboundary 1/0/0 1

# select face-varying boundary interpolation mode FVAR_LINEAR_ALL
# 0 : FVAR_LINEAR_NONE
# 1 : FVAR_LINEAR_CORNERS_ONLY
# 2 : FVAR_LINEAR_CORNERS_PLUS1
# 3 : FVAR_LINEAR_CORNERS_PLUS2
# 4 : FVAR_LINEAR_BOUNDARIES
# 5 : FVAR_LINEAR_ALL
t interpolateboundary 1/0/0 5

# tag the edge defined by vertices '1' and '3' with a sharpness of 2.0
t crease 2/1/0 1 3 2.0

# tag vertex '4' with a sharpness of 2.8
t corner 1/1/0 4 2.8

# set face '9' as a hole
t hole 1/0/0 9

# set the crease method to 'chaikin' (default is 'normal')
t creasemethod 0/0/1 chaikin 

```

### MTL extensions

The `OBJ` parser supports some of the physically based (PBR) materials
[extensions](https://en.wikipedia.org/wiki/Wavefront_.obj_file#Physically-based_rendering).
The path-tracer does not support transparent, transmissive or emissive materials
though, so while they are parsed, they will not be used.

Supported tags:
```
Kd: albedo
Ks: specular

Pr: roughness
Pm: metalness

map_Kd: albedo map
map_Ks: specular map
map_Pr: roughness map

# optional displacement scale & bias parameters
map_Bump -bm <scale> -bb <bias> : displacement map
```

### UDIM workflows

UDIM texture naming conventions are supported with the `<UDIM>` keyword.

> [!IMPORTANT]
> Our implementation has a limitation that does not allow UV islands to cross a
> UDIM tile boundary and will cause the application to throw a fatal error on load.

```
map_Kd textures/asset_name_D.<UDIM>.dds
map_Ks textures/asset_name_S.<UDIM>.dds
map_Bump -bm 0.002000 textures/asset_name_H.<UDIM>.dds
map_Pr textures/asset_name_R.<UDIM>.dds
```

#### JSON scene files

Multiple assets can be combined to create simple scenes with Donut's JSON schema.

Scene files should carry the extension ‘.scene.json’ and must be placed under root of
the ‘scenes’ folder (all asset paths are relative to the location of this file).

For historical reasons, the format used by this sample application differs slightly
from Donut's. Please consult the source code for details.

> [!NOTE]
> The `glTF` file format only supports triangle-based meshes and cannot represent
> subdivision surfaces without extensions. Because of these limitations, it is
> not supported in this sample application.

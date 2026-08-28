add_library(rtxmg)
target_include_directories(rtxmg PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

add_subdirectory(cluster_lod)
add_subdirectory(cluster_tess)
add_subdirectory(hiz)
add_subdirectory(profiler)
add_subdirectory(scene)
add_subdirectory(subdivision)
add_subdirectory(utils)

target_link_libraries(rtxmg cluster_lod cluster_tess profiler hiz subdivision rtxmg_utils scene envmap)
set_target_properties(rtxmg PROPERTIES FOLDER "RTXMG")

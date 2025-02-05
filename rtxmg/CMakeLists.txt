add_library(rtxmg)
target_include_directories(rtxmg PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

add_subdirectory(cluster_builder)
add_subdirectory(hiz)
add_subdirectory(profiler)
add_subdirectory(scene)
add_subdirectory(subdivision)
add_subdirectory(utils)

target_link_libraries(rtxmg cluster_builder profiler hiz subdivision rtxmg_utils scene envmap)
set_target_properties(rtxmg PROPERTIES FOLDER "RTXMG")

set(lib cluster_lod)
set(folder RTXMG)

include ("${DONUT_DIR}/compileshaders.cmake")

file(GLOB shaders "shaders/*")
file(GLOB sources "*.cpp" "baking/*.cpp" "baking/*.h"
                  "../include/rtxmg/${lib}/*.h" "../include/rtxmg/${lib}/baking/*.h" *.cfg)

add_library(${lib} OBJECT ${sources})
target_include_directories(${lib} PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
)
target_link_libraries(${lib} donut_engine cgltf meshoptimizer mimalloc-static)
set_target_properties(${lib} PROPERTIES FOLDER ${folder})

donut_compile_shaders_all_platforms(
    TARGET ${lib}_shaders
    CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/shaders.cfg
    FOLDER ${folder}
    SOURCES ${shaders}
    OUTPUT_BASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/${lib}
    SHADERMAKE_OPTIONS ${RTXMG_SHADERMAKE_OPTIONS}
    SHADERMAKE_OPTIONS_SPIRV ${RTXMG_SHADERMAKE_OPTIONS_SPIRV}
    SHADER_MODEL ${RTXMG_SHADERS_SHADERMODEL}
    IGNORE_INCLUDES ${RTXMG_SHADERS_IGNORED_INCLUDES}
    INCLUDES ${RTXMG_SHADERS_INCLUDE_DIR}
)
add_dependencies(${lib} ${lib}_shaders)

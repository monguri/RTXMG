set(lib cluster_builder)
set(folder RTXMG)

include ("${DONUT_DIR}/compileshaders.cmake")

OPTION(ENABLE_ACCEL_BUILDER_DEBUG "enable printing of raw debug buffers during acceleration builds" OFF)
if (ENABLE_ACCEL_BUILDER_DEBUG)
    add_compile_definitions(ENABLE_ACCEL_BUILDER_DEBUG=${ENABLE_ACCEL_BUILDER_DEBUG})
endif()

file(GLOB shaders "shaders/*" "../include/rtxmg/${lib}/*.hlsli")
file(GLOB sources "*.cpp" "../include/rtxmg/${lib}/*.h" *.cfg)

add_library(${lib} OBJECT ${sources})
target_include_directories(${lib} PUBLIC 
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
)
target_link_libraries(${lib} donut_engine subdivision)
set_target_properties(${lib} PROPERTIES FOLDER ${folder})

donut_compile_shaders(
    TARGET ${lib}_shaders
    CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/shaders.cfg
    FOLDER ${folder}
    SOURCES ${shaders}
    DXIL ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/${lib}/dxil
    SHADERMAKE_OPTIONS_DXIL ${RTXMG_SHADERMAKE_OPTIONS_DXIL}
    SHADER_MODEL ${RTXMG_SHADERS_SHADERMODEL}
    IGNORE_INCLUDES ${RTXMG_SHADERS_IGNORED_INCLUDES}
    INCLUDES ${RTXMG_SHADERS_INCLUDE_DIR}
)
add_dependencies(${lib} ${lib}_shaders)

set(lib envmap)
set(folder "Demo")
include ("${DONUT_DIR}/compileshaders.cmake")

file(GLOB shaders "shaders/*")
file(GLOB sources "*.cpp" "*.h" *.cfg)

add_library(${lib} OBJECT ${sources})
target_include_directories(${lib} PUBLIC 
    "${PROJECT_SOURCE_DIR}/rtxmg/include"
)
target_link_libraries(${lib} donut_engine)
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

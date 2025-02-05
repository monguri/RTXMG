# include ("private.cmake" OPTIONAL)

include ("${DONUT_DIR}/compileshaders.cmake")

add_subdirectory(audio)


file(GLOB shaders "shaders/*")
file(GLOB sources "*.cpp" "*.h" "osd_port_sources/tmr/*.h" *.cfg)

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/CMake/FetchImplot.cmake")
    message(FATAL_ERROR "Missing FetchImplot.cmake file")
endif()
include("${PROJECT_SOURCE_DIR}/CMake/FetchImplot.cmake")
set_target_properties(implot PROPERTIES FOLDER "Third-Party Libraries")

set(app rtxmg_demo)
set(folder "Demo")

list(APPEND RTXMG_SHADERS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

donut_compile_shaders(
    TARGET ${app}_shaders
    CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/shaders.cfg
    FOLDER ${folder}
    SOURCES ${shaders}
    DXIL ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/${app}/dxil
    # SPIRV_DXC ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/${app}/spirv
    SHADERMAKE_OPTIONS_DXIL ${RTXMG_SHADERMAKE_OPTIONS_DXIL}
    SHADER_MODEL ${RTXMG_SHADERS_SHADERMODEL}
    IGNORE_INCLUDES ${RTXMG_SHADERS_IGNORED_INCLUDES}
    INCLUDES ${RTXMG_SHADERS_INCLUDE_DIR}
)

add_executable(${app} 
    ${sources} 
)
target_link_libraries(${app} PRIVATE envmap rtxmg donut_render donut_app donut_engine implot)

if(ENABLE_AUDIO_ENGINE)
    if(TARGET audio_engine)
        target_link_libraries(${app} PRIVATE audio_engine)
    else()
        message(WARNING "Audio engine option is enabled, but configuration failed")
    endif()
endif()

add_dependencies(${app} ${app}_shaders)
set_target_properties(${app} PROPERTIES FOLDER ${folder})
add_compile_definitions( "_SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING=1")

target_include_directories(${app} PUBLIC 
    ${PROJECT_SOURCE_DIR} 
    ${PROJECT_SOURCE_DIR}/extern
)

add_subdirectory(envmap)

if(DONUT_D3D_AGILITY_SDK_ENABLED)
    target_compile_definitions(${app} PUBLIC DONUT_D3D_AGILITY_SDK_ENABLED=1)
    target_compile_definitions(${app} PUBLIC DONUT_D3D_AGILITY_SDK_VERSION=${DONUT_D3D_AGILITY_SDK_VERSION})
    add_custom_command(TARGET ${app} POST_BUILD
        COMMAND 
            ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${app}>/D3D12/"
        COMMAND 
            ${CMAKE_COMMAND} -E copy_if_different ${DONUT_D3D_AGILITY_SDK_CORE_DLL} "$<TARGET_FILE_DIR:${app}>/D3D12/"
    )
endif()

if(WIN32)
    add_custom_command( TARGET ${app} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:${app}> $<TARGET_FILE_DIR:${app}>
        COMMAND_EXPAND_LISTS)
endif()

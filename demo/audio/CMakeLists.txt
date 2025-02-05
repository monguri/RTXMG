
option(ENABLE_AUDIO_ENGINE "Enable audio playback" ON)

if (ENABLE_AUDIO_ENGINE)

    set(_audio_src_files
        audio.cpp
        audio.h
        waveFile.cpp
        waveFile.h
        xaudiofaudio.h)

    if (UNIX)

        # Linux builds emulate XAudio with the FAudio library ; typical installation with
        # 'sudo apt install libfaudio-dev' (same on Windows Subsystem for Linux)

        find_path(_faudio_include_dir FAudio.h HINTS "/usr/include")
        find_library(_faudio_lib libFAudio.so HINTS "/usr/lib/" "/usr/lib/x86_64-linux-gnu/" )

        if (_faudio_include_dir AND _faudio_lib)

            add_library(FAudio INTERFACE)

            target_include_directories(FAudio INTERFACE ${_faudio_include_dir})
            target_link_libraries(FAudio INTERFACE ${_faudio_lib})
            target_compile_definitions(FAudio INTERFACE AUDIO_ENGINE_XAUDIO_ON_FAUDIO)

            message(STATUS "Found FAudio: ${_faudio_include_dir}")
        endif()

        if (NOT TARGET FAudio)
            message(WARNING 
                "Audio engine is enabled but missing libfaudio-dev (disabling audio). "
                "Set ENABLE_AUDIO_ENGINE=OFF to skip this warning.")
            return()
        endif()
        
    endif()
   
    add_library(audio_engine STATIC ${_audio_src_files})

    target_include_directories(audio_engine PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")
    target_compile_definitions(audio_engine PUBLIC AUDIO_ENGINE_ENABLED)

    if (TARGET FAudio)
        target_link_libraries(audio_engine FAudio)
    endif()

    if (MSVC)
        target_compile_definitions(audio_engine PUBLIC AUDIO_ENGINE_WITH_XAUDIO)
    endif()

    set_target_properties(audio_engine PROPERTIES FOLDER Demo)

endif()

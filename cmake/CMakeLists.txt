project(MultiProcessorProgramming_exercise)
cmake_minimum_required(VERSION 3.1)
aux_source_directory(. SRC_LIST)

if(UNIX)
    find_package (Threads REQUIRED)
    find_package (OpenCL REQUIRED)

    # Compare build type against empty string to enforce default behavior
    # to be release-build.
    string(COMPARE EQUAL "${CMAKE_BUILD_TYPE}" "" emptyString)
    if(emptyString)
        MESSAGE(STATUS "Build type not set, setting it to \"Release\".")
        set(CMAKE_BUILD_TYPE Release)
    endif(emptyString)

    # Testing for "x86_64" would probably not work correctly on OSX
    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        MESSAGE(STATUS "Enabling x86_64 assembly.")
        enable_language(ASM_NASM)
    else(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        MESSAGE(STATUS "Not a X86_64-machine. Assembly-code will not be compiled!")
    endif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")

    set(SRC_LIST
        ../main.c
        ../lodepng.c
        ../depthmap_c.c
        ../depthmap64.asm
        ../common_opencl.c
        ../depthmap_opencl.c
        ../depthmap_opencl_amd.c
        ../depthmap_amd.cl
        ../depthmap_basic.cl)   # To get qt-creator to view it as one of the project files
    set(HDR_LIST
        ../lodepng.h
        ../depthmap_c.h
        ../doubleTime.h
        ../common_opencl.h
        ../depthmap_opencl.h
        ../depthmap_opencl_amd.h)

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")

    add_executable(${PROJECT_NAME} ${SRC_LIST})

    target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT} ${OpenCL_LIBRARY} m)

    # Copy .cl-file to the same directory as project executable
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/../depthmap_basic.cl $<TARGET_FILE_DIR:${PROJECT_NAME}>)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/../depthmap_amd.cl $<TARGET_FILE_DIR:${PROJECT_NAME}>)

else(UNIX)
    MESSAGE(FATAL_ERROR "Program requires UNIX-like operating system!")
endif(UNIX)

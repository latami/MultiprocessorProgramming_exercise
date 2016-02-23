project(MultiProcessorProgramming_exercise)
cmake_minimum_required(VERSION 2.8)
aux_source_directory(. SRC_LIST)
set(SRC_LIST
    ../main.c
    ../lodepng.c
    ../depthmap_c.c)
set(HDR_LIST
    ../lodepng.h
    ../depthmap_c.h
    ../doubleTime.h)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")

add_executable(${PROJECT_NAME} ${SRC_LIST})

target_link_libraries(${PROJECT_NAME} m)

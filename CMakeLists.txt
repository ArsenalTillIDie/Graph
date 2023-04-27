cmake_minimum_required(VERSION 2.8.11)
add_definitions(-std=c++11)
set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
project(GraphParallel)

add_executable(GraphTopDownBrandes GraphParallel.cpp GraphClass.h GraphClass.cpp Cormenium.cpp)

find_package(OpenMP)
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

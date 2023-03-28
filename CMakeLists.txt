cmake_minimum_required(VERSION 3.17)
project(GraphParallel)

add_executable(Graph GraphParallel.cpp GraphClass.h GraphClass.cpp Cormenium.cpp)

find_package(OpenMP)
target_link_libraries(Graph PUBLIC OpenMP::OpenMP_CXX)

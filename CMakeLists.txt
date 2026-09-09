cmake_minimum_required(VERSION 3.20)
project(desktop_agent_or LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 REQUIRED)

pybind11_add_module(_cxx_engine
    src/bindings.cpp
    src/alns_operators.cpp
)

target_compile_options(_cxx_engine PRIVATE -O3 -march=native -Wall)
target_include_directories(_cxx_engine PRIVATE src)


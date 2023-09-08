cmake_minimum_required(VERSION 3.8)

project(checking_cuda_autodiff_contraction)
enable_language("CUDA")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

include_directories(/home/doleron/book-workspace/pbp-deep-learning-in-modern-cpp/third-part/eigen/src)

file(GLOB cpu_sources "${PROJECT_SOURCE_DIR}/src/*.cc")
file(GLOB cuda_sources "")

add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/src/main.cc ${PROJECT_SOURCE_DIR}/src/functions.cu)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
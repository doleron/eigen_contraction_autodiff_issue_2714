cmake_minimum_required(VERSION 3.8)

project(checking_cuda_autodiff_contraction)
enable_language("CUDA")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

include_directories(/home/doleron/book-workspace/eigen-doleron/)

file(GLOB cpu_sources "${PROJECT_SOURCE_DIR}/src/*.cc")
file(GLOB cuda_sources "")

add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/src/main.cc ${PROJECT_SOURCE_DIR}/src/functions.cu)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# export CUDA_HOME=/usr/local/cuda
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
# export PATH=$PATH:$CUDA_HOME/bin
# cmake -DCMAKE_CUDA_ARCHITECTURES=86 ..
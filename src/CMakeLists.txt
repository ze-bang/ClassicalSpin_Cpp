cmake_minimum_required(VERSION 3.18)
project(CudaTensorContractions LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA package
find_package(CUDA REQUIRED)

# Find cuBLAS
find_package(CUDAToolkit REQUIRED)

# Enable CUDA separable compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Set CUDA architecture (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 89)  # For Turing, Ampere, and Ada Lovelace

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# Add the executable
add_executable(test_cuda_contractions 
    test_cuda_contractions.cu
    cuda_contractions.cu
)

# Set CUDA properties
set_target_properties(test_cuda_contractions PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

# Link libraries
target_link_libraries(test_cuda_contractions 
    ${CUDA_LIBRARIES}
    CUDA::cublas
)

# Optional: Add install target
install(TARGETS test_cuda_contractions DESTINATION bin)

# Optional: Enable debug symbols for debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ClassicalSpin_Cpp LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable modern CMake policies for CUDA
cmake_policy(SET CMP0104 NEW)
cmake_policy(SET CMP0105 NEW)

# Set build type to Release if not specified (for optimization)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Find required packages using modern CMake
find_package(CUDAToolkit REQUIRED)
find_package(Thrust REQUIRED CONFIG)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Boost 1.65 REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C CXX)

# Set CUDA architectures for multiple GPU support
set(CMAKE_CUDA_ARCHITECTURES "60 75 80 86 89" CACHE STRING "CUDA architectures")

# Enable position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set include directories
include_directories(src)

# Set optimized compiler flags with modern CMake approach
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native")
set(CMAKE_CUDA_FLAGS "-D__CUDA_NO_HOST_COMPILER_CHECK")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math -lineinfo")

# Additional optimization flags for specific configurations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # C++ optimization flags with numerical stability
    set(CXX_OPTIMIZATION_FLAGS
        -O3                    # Maximum optimization
        -DNDEBUG              # Disable assertions
        -march=native         # Optimize for current CPU
        -mtune=native         # Tune for current CPU
        -flto                 # Link-time optimization
        -funroll-loops        # Unroll loops
        -fomit-frame-pointer  # Omit frame pointer
        -fno-stack-protector  # Disable stack protector for performance
        -ffp-contract=fast    # Contract floating point operations when safe (but not -ffast-math)
        -fno-math-errno       # Don't set errno for math functions (safe optimization)
        -fno-trapping-math    # Assume floating-point operations don't trap (safer than fast-math)
        -fno-rounding-math    # Assume default rounding behavior (safer than fast-math)
    )
    
    # CUDA optimization flags
    set(CUDA_OPTIMIZATION_FLAGS
        -O3                   # Maximum optimization
        -DNDEBUG             # Disable assertions
        --use_fast_math      # Use fast math library
        --ptxas-options=-v,-O3  # Verbose PTX assembler output and optimization
        -lineinfo            # Line info for profiling
        --maxrregcount=32    # Limit register usage for better occupancy
        --extra-device-vectorization  # Enable extra vectorization
    )
    
    # CUDA host compiler optimization flags (separate from device flags)
    # Note: Omitting -march flag to avoid AMX intrinsics incompatibility between nvcc and GCC 12
    set(CUDA_HOST_OPTIMIZATION_FLAGS
        -Xcompiler=-O3       # Host compiler optimization
        -Xcompiler=-ffast-math
        -Xcompiler=-funroll-loops
    )
endif()

# OpenMP optimization flags
if(OpenMP_CXX_FOUND)
    list(APPEND CXX_OPTIMIZATION_FLAGS -fopenmp-simd)
endif()

# Collect source files with better organization
file(GLOB HEADER_FILES 
    "src/*.h"
    "src/*.hpp"
    "src/**/*.h"
    "src/**/*.hpp"
)

# Collect all CUDA source files (not headers) - include subdirectories
# Exclude GPU MD file which needs separate compilation
file(GLOB_RECURSE CUDA_SOURCE_FILES "src/*.cu")

# Collect all CUDA header files - include subdirectories
file(GLOB_RECURSE CUDA_HEADER_FILES 
    "src/*.cuh"
)

# Collect all C++ source files in results directory
file(GLOB CPP_SOURCE_FILES 
    "run_scripts/*.cpp"
)

# Create a modern header-only interface library
add_library(headers INTERFACE)
target_include_directories(headers INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)
target_sources(headers INTERFACE ${HEADER_FILES})

# Apply modern compiler features and optimization flags
target_compile_features(headers INTERFACE cxx_std_20)

# Add OpenMP and MPI support to header library with proper modern linking
if(OpenMP_CXX_FOUND)
    target_link_libraries(headers INTERFACE OpenMP::OpenMP_CXX)
    target_compile_definitions(headers INTERFACE OPENMP_ENABLED)
endif()

if(MPI_CXX_FOUND)
    target_link_libraries(headers INTERFACE MPI::MPI_CXX)
    target_compile_definitions(headers INTERFACE MPI_ENABLED)
endif()

if(Eigen3_FOUND)
    target_link_libraries(headers INTERFACE Eigen3::Eigen)
    target_compile_definitions(headers INTERFACE EIGEN3_ENABLED)
endif()

if(HDF5_FOUND)
    target_include_directories(headers INTERFACE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(headers INTERFACE ${HDF5_CXX_LIBRARIES})
    target_compile_definitions(headers INTERFACE HDF5_ENABLED)
endif()

# Create optimized CUDA library
if(CUDA_SOURCE_FILES)
    add_library(cuda_lib STATIC ${CUDA_SOURCE_FILES})
    
    # Set modern CUDA properties
    set_target_properties(cuda_lib PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        POSITION_INDEPENDENT_CODE ON
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    
    # Include directories for CUDA library
    target_include_directories(cuda_lib PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/dynamics>
        $<INSTALL_INTERFACE:include>
    )
    
    # Link with Eigen3 for GPU code
    if(Eigen3_FOUND)
        target_link_libraries(cuda_lib PUBLIC Eigen3::Eigen)
    endif()
    
    # Link with CUDA libraries using modern CMake
    target_link_libraries(cuda_lib PUBLIC CUDA::cudart CUDA::cublas CUDA::curand)
    
    # Add Thrust support (header-only library)
    if(Thrust_FOUND)
        # Thrust is header-only - just add compile definitions, no linking needed
        target_compile_definitions(cuda_lib PUBLIC THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA)
    endif()
    
    # Link with Boost for Odeint
    if(Boost_FOUND)
        target_include_directories(cuda_lib PUBLIC ${Boost_INCLUDE_DIRS})
    endif()
    
    # Add MPI dependency to CUDA library
    if(MPI_CXX_FOUND)
        target_link_libraries(cuda_lib PUBLIC MPI::MPI_CXX)
        target_compile_definitions(cuda_lib PUBLIC MPI_ENABLED)
    endif()
    
    # Apply CUDA optimization flags
    target_compile_options(cuda_lib PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_OPTIMIZATION_FLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_HOST_OPTIMIZATION_FLAGS}>
        # Workaround for nvcc/GCC 12 incompatibility: prevent inclusion of AMX headers
        $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-D_AMXTILEINTRIN_H_INCLUDED>
        $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-D_AMX_INT8INTRIN_H_INCLUDED>
        $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-D_AMX_BF16INTRIN_H_INCLUDED>
    )
    
    # Create separate library for GPU molecular dynamics (needs Boost.Odeint thrust)
    # DISABLED due to Boost.Odeint thrust integration compilation issues with NVCC
    # The Boost template specializations are being instantiated multiple times causing
    # "already defined" errors. Re-enable when compatibility issue is resolved.
    # add_library(cuda_md_lib STATIC src/dynamics/lattice_refactored_gpu_md.cu)
    # 
    # set_target_properties(cuda_md_lib PROPERTIES
    #     CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    #     POSITION_INDEPENDENT_CODE ON
    #     CUDA_STANDARD 20
    #     CUDA_STANDARD_REQUIRED ON
    #     CUDA_SEPARABLE_COMPILATION OFF
    #     CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    # )
    # 
    # target_include_directories(cuda_md_lib PUBLIC
    #     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    #     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/dynamics>
    #     $<INSTALL_INTERFACE:include>
    # )
    # 
    # target_link_libraries(cuda_md_lib PUBLIC 
    #     CUDA::cudart
    # )
    # 
    # if(Eigen3_FOUND)
    #     target_link_libraries(cuda_md_lib PUBLIC Eigen3::Eigen)
    # endif()
    # 
    # if(Boost_FOUND)
    #     target_include_directories(cuda_md_lib PUBLIC ${Boost_INCLUDE_DIRS})
    # endif()
    # 
    # if(Thrust_FOUND)
    #     target_compile_definitions(cuda_md_lib PUBLIC THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA)
    # endif()
    # 
    # target_compile_options(cuda_md_lib PRIVATE
    #     $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_OPTIMIZATION_FLAGS}>
    #     $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_HOST_OPTIMIZATION_FLAGS}>
    #     $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-D_AMXTILEINTRIN_H_INCLUDED>
    #     $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-D_AMX_INT8INTRIN_H_INCLUDED>
    #     $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-D_AMX_BF16INTRIN_H_INCLUDED>
    # )
    
    # Create interface library for CUDA headers
    add_library(cuda_headers INTERFACE)
    target_include_directories(cuda_headers INTERFACE src src/dynamics)
    target_sources(cuda_headers INTERFACE ${CUDA_HEADER_FILES})
    target_link_libraries(cuda_headers INTERFACE cuda_lib)
    # Note: cuda_md_lib is disabled due to compilation issues
endif()

# Create optimized executables for each C++ file in run_scripts
foreach(cpp_file ${CPP_SOURCE_FILES})
    # Get the filename without extension
    get_filename_component(exec_name ${cpp_file} NAME_WE)
    
    # Create executable
    add_executable(${exec_name} ${cpp_file})
    
    # Set modern C++ features
    target_compile_features(${exec_name} PRIVATE cxx_std_20)
    
    # Link with header libraries
    target_link_libraries(${exec_name} PRIVATE headers)
    
    # Link with CUDA libraries if CUDA source files exist
    if(CUDA_SOURCE_FILES)
        target_link_libraries(${exec_name} PRIVATE cuda_headers)
        set_target_properties(${exec_name} PROPERTIES
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        )
    endif()
    
    # Set include directories
    target_include_directories(${exec_name} PRIVATE src)
    
    # Apply optimized compiler flags using modern CMake
    target_compile_options(${exec_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${CXX_OPTIMIZATION_FLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_OPTIMIZATION_FLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_HOST_OPTIMIZATION_FLAGS}>
    )
    
    # Link-time optimization for Release builds
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        set_target_properties(${exec_name} PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION TRUE
        )
    endif()
    
    # Add debug information for profiling even in release mode
    target_compile_options(${exec_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-g1>  # Minimal debug info for profiling
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )
endforeach()

# Enable parallel builds
if(NOT DEFINED CMAKE_BUILD_PARALLEL_LEVEL)
    include(ProcessorCount)
    ProcessorCount(N)
    if(NOT N EQUAL 0)
        set(CMAKE_BUILD_PARALLEL_LEVEL ${N} CACHE STRING "Number of parallel build jobs")
        message(STATUS "Setting parallel build jobs to ${N}")
    endif()
endif()

# Enhanced build summary with optimization information
message(STATUS "================== Build Configuration ==================")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Parallel Build Jobs: ${CMAKE_BUILD_PARALLEL_LEVEL}")
message(STATUS "Found ${CMAKE_CURRENT_LIST_DIR}")
message(STATUS "Header files: ${HEADER_FILES}")
message(STATUS "CUDA source files: ${CUDA_SOURCE_FILES}")
message(STATUS "CUDA header files: ${CUDA_HEADER_FILES}")
message(STATUS "C++ source files: ${CPP_SOURCE_FILES}")

# Print dependency status with versions
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found and enabled (version: ${OpenMP_CXX_VERSION})")
else()
    message(WARNING "OpenMP not found")
endif()

if(MPI_CXX_FOUND)
    message(STATUS "MPI found and enabled (version: ${MPI_CXX_VERSION})")
else()
    message(WARNING "MPI not found")
endif()

if(CUDAToolkit_FOUND)
    message(STATUS "CUDA found and enabled (version: ${CUDAToolkit_VERSION})")
else()
    message(WARNING "CUDA not found")
endif()

if(Eigen3_FOUND)
    message(STATUS "Eigen3 found and enabled (version: ${Eigen3_VERSION})")
else()
    message(WARNING "Eigen3 not found")
endif()

if(Thrust_FOUND)
    message(STATUS "Thrust found and enabled")
else()
    message(WARNING "Thrust not found")
endif()

if(Boost_FOUND)
    message(STATUS "Boost found and enabled (version: ${Boost_VERSION})")
    message(STATUS "Boost include dirs: ${Boost_INCLUDE_DIRS}")
else()
    message(WARNING "Boost not found")
endif()

if(HDF5_FOUND)
    message(STATUS "HDF5 found and enabled (version: ${HDF5_VERSION})")
    message(STATUS "HDF5 include dirs: ${HDF5_INCLUDE_DIRS}")
    message(STATUS "HDF5 libraries: ${HDF5_CXX_LIBRARIES}")
else()
    message(WARNING "HDF5 not found")
endif()

message(STATUS "=====================================================")  # Print optimization flags being used
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "C++ Optimization Flags: ${CXX_OPTIMIZATION_FLAGS}")
    message(STATUS "CUDA Optimization Flags: ${CUDA_OPTIMIZATION_FLAGS}")
endif()

# Create a target to build all executables with optimized parallel build
add_custom_target(all_executables)
foreach(cpp_file ${CPP_SOURCE_FILES})
    get_filename_component(exec_name ${cpp_file} NAME_WE)
    add_dependencies(all_executables ${exec_name})
endforeach()

# Add a custom target for ninja builds to show build progress
if(CMAKE_GENERATOR STREQUAL "Ninja")
    message(STATUS "Using Ninja generator for optimized parallel builds")
    set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)
endif()

# Performance testing target
add_custom_target(performance_test
    COMMAND echo "Run performance tests here"
    DEPENDS all_executables
    COMMENT "Running performance tests on optimized executables"
)
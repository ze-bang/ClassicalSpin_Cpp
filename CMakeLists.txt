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
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

# Set CUDA architectures for multiple GPU support
set(CMAKE_CUDA_ARCHITECTURES "60 75 80 86 89" CACHE STRING "CUDA architectures")

# Enable position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set include directories
include_directories(src)

# Set optimized compiler flags with modern CMake approach
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native")
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
    set(CUDA_HOST_OPTIMIZATION_FLAGS
        -Xcompiler=-O3       # Host compiler optimization
        -Xcompiler=-ffast-math
        -Xcompiler=-funroll-loops
        -Xcompiler=-march=native
        -Xcompiler=-mtune=native
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
)

# Collect all CUDA source files (not headers)
file(GLOB CUDA_SOURCE_FILES "src/*.cu")

# Collect all CUDA header files
file(GLOB CUDA_HEADER_FILES 
    "src/*.cuh"
)

# Option to build legacy individual executables
option(BUILD_LEGACY_EXECUTABLES "Build individual executables for each simulation script" OFF)

# Collect all C++ source files in run_scripts directory
file(GLOB CPP_SOURCE_FILES 
    "run_scripts/*.cpp"
)

# Separate unified simulation from legacy scripts
list(FILTER CPP_SOURCE_FILES EXCLUDE REGEX ".*unified_simulation\\.cpp$")
set(UNIFIED_SIM_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/run_scripts/unified_simulation.cpp")

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

# Create optimized CUDA library
if(CUDA_SOURCE_FILES)
    add_library(cuda_lib STATIC ${CUDA_SOURCE_FILES})
    
    # Set modern CUDA properties
    set_target_properties(cuda_lib PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        POSITION_INDEPENDENT_CODE ON
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )
    
    # Include directories for CUDA library
    target_include_directories(cuda_lib PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    )
    
    # Link with CUDA libraries using modern CMake
    target_link_libraries(cuda_lib PUBLIC CUDA::cudart CUDA::cublas CUDA::curand)
    
    # Add MPI dependency to CUDA library
    if(MPI_CXX_FOUND)
        target_link_libraries(cuda_lib PUBLIC MPI::MPI_CXX)
        target_compile_definitions(cuda_lib PUBLIC MPI_ENABLED)
    endif()
    
    # Apply CUDA optimization flags
    target_compile_options(cuda_lib PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_OPTIMIZATION_FLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_HOST_OPTIMIZATION_FLAGS}>
    )
    
    # Create interface library for CUDA headers
    add_library(cuda_headers INTERFACE)
    target_include_directories(cuda_headers INTERFACE src)
    target_sources(cuda_headers INTERFACE ${CUDA_HEADER_FILES})
    target_link_libraries(cuda_headers INTERFACE cuda_lib)
endif()

# ============================================================================
# Build Unified Simulation (NEW APPROACH - RECOMMENDED)
# ============================================================================
message(STATUS "Building unified simulation executable")

add_executable(unified_simulation ${UNIFIED_SIM_SOURCE})

# Set modern C++ features
target_compile_features(unified_simulation PRIVATE cxx_std_20)

# Link with header libraries
target_link_libraries(unified_simulation PRIVATE headers)

# Link with CUDA libraries if CUDA source files exist
if(CUDA_SOURCE_FILES)
    target_link_libraries(unified_simulation PRIVATE cuda_headers)
    set_target_properties(unified_simulation PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
endif()

# Set include directories
target_include_directories(unified_simulation PRIVATE src run_scripts)

# Apply optimized compiler flags
target_compile_options(unified_simulation PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_OPTIMIZATION_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_OPTIMIZATION_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_HOST_OPTIMIZATION_FLAGS}>
)

# Link-time optimization for Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set_target_properties(unified_simulation PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION TRUE
    )
endif()

# Add debug information for profiling
target_compile_options(unified_simulation PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-g1>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

message(STATUS "Unified simulation will be built (recommended)")

# ============================================================================
# Build Legacy Individual Executables (OPTIONAL - FOR BACKWARDS COMPATIBILITY)
# ============================================================================
if(BUILD_LEGACY_EXECUTABLES)
    message(STATUS "Building legacy individual executables (BUILD_LEGACY_EXECUTABLES=ON)")
    
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
    
    # Create a target to build all legacy executables
    add_custom_target(all_legacy_executables)
    foreach(cpp_file ${CPP_SOURCE_FILES})
        get_filename_component(exec_name ${cpp_file} NAME_WE)
        add_dependencies(all_legacy_executables ${exec_name})
    endforeach()
else()
    message(STATUS "Skipping legacy executables (use -DBUILD_LEGACY_EXECUTABLES=ON to build them)")
endif()

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
message(STATUS "Build legacy executables: ${BUILD_LEGACY_EXECUTABLES}")
message(STATUS "Found ${CMAKE_CURRENT_LIST_DIR}")
message(STATUS "Header files: ${HEADER_FILES}")
message(STATUS "CUDA source files: ${CUDA_SOURCE_FILES}")
message(STATUS "CUDA header files: ${CUDA_HEADER_FILES}")
if(BUILD_LEGACY_EXECUTABLES)
    message(STATUS "Legacy C++ source files: ${CPP_SOURCE_FILES}")
endif()

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

message(STATUS "=====================================================")

# Print optimization flags being used
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "C++ Optimization Flags: ${CXX_OPTIMIZATION_FLAGS}")
    message(STATUS "CUDA Optimization Flags: ${CUDA_OPTIMIZATION_FLAGS}")
endif()

# Usage instructions
message(STATUS "")
message(STATUS "==================== Usage Instructions ====================")
message(STATUS "Build the unified simulation:")
message(STATUS "  cd build && cmake .. && make unified_simulation")
message(STATUS "")
message(STATUS "Run a simulation:")
message(STATUS "  mpirun -np 4 ./unified_simulation ../configs/bcao_sa.conf")
message(STATUS "  ./unified_simulation ../configs/pyrochlore_sa.conf")
message(STATUS "")
message(STATUS "Create your own config file based on examples in configs/")
message(STATUS "See configs/README.md for documentation")
message(STATUS "")
if(NOT BUILD_LEGACY_EXECUTABLES)
    message(STATUS "To build legacy executables: cmake -DBUILD_LEGACY_EXECUTABLES=ON ..")
endif()
message(STATUS "===========================================================")

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

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ClassicalSpin_Cpp LANGUAGES CXX CUDA)

option(CLASSICAL_SPIN_ENABLE_WARNINGS "Enable recommended warning flags" ON)
option(CLASSICAL_SPIN_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Modern CMake policies
cmake_policy(SET CMP0104 NEW)
cmake_policy(SET CMP0105 NEW)

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Enable position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CUDA architectures
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(_classical_spin_default_arch "native")
    if(DEFINED ENV{CLASSICAL_SPIN_CUDA_ARCHS} AND NOT "$ENV{CLASSICAL_SPIN_CUDA_ARCHS}" STREQUAL "")
        set(_classical_spin_default_arch "$ENV{CLASSICAL_SPIN_CUDA_ARCHS}")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${_classical_spin_default_arch}" CACHE STRING "CUDA architectures" FORCE)
endif()
unset(_classical_spin_default_arch)

# ============================================================================
# DEPENDENCIES
# ============================================================================

find_package(CUDAToolkit REQUIRED)
find_package(Thrust REQUIRED CONFIG)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Boost 1.65 REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C CXX)

# ============================================================================
# COMPILER FLAGS
# ============================================================================

# Compiler warnings configuration
set(CLASSICAL_SPIN_CXX_WARNING_FLAGS "")
set(CLASSICAL_SPIN_CUDA_WARNING_FLAGS "")

if(CLASSICAL_SPIN_ENABLE_WARNINGS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        list(APPEND CLASSICAL_SPIN_CXX_WARNING_FLAGS
            -Wall
            -Wextra
            -Wformat=2
            -Wnon-virtual-dtor
            # Suppress noisy warnings common in scientific computing code:
            -Wno-sign-compare          # size_t vs int comparisons (very common with Eigen)
            -Wno-sign-conversion       # signed/unsigned conversions
            -Wno-conversion            # narrowing conversions (common with Eigen Index types)
            -Wno-shadow                # constructor parameters shadowing members (common pattern)
            -Wno-old-style-cast        # C-style casts (from external headers like HDF5)
            -Wno-deprecated-copy       # deprecated implicit copy operators (HDF5 library issue)
            -Wno-float-conversion      # float/double conversions
            -Wno-unused-parameter      # unused parameters in callbacks/interfaces
            -Wno-pedantic              # strict ISO compliance (e.g., __int128)
            -Wno-reorder               # member initialization order
        )
    elseif(MSVC)
        list(APPEND CLASSICAL_SPIN_CXX_WARNING_FLAGS /W3 /permissive-)
    endif()

    if(CMAKE_CUDA_COMPILER)
        list(APPEND CLASSICAL_SPIN_CUDA_WARNING_FLAGS 
            -Xcompiler=-Wall 
            -Xcompiler=-Wextra
            -Xcompiler=-Wno-sign-compare
            -Xcompiler=-Wno-sign-conversion
            -Xcompiler=-Wno-conversion
            -Xcompiler=-Wno-shadow
            -Xcompiler=-Wno-old-style-cast
            -Xcompiler=-Wno-deprecated-copy
            -Xcompiler=-Wno-float-conversion
            -Xcompiler=-Wno-unused-parameter
            -Xcompiler=-Wno-pedantic
            -Xcompiler=-Wno-reorder
        )
    endif()

    if(CLASSICAL_SPIN_WARNINGS_AS_ERRORS)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
            list(APPEND CLASSICAL_SPIN_CXX_WARNING_FLAGS -Werror)
        elseif(MSVC)
            list(APPEND CLASSICAL_SPIN_CXX_WARNING_FLAGS /WX)
        endif()

        if(CMAKE_CUDA_COMPILER)
            list(APPEND CLASSICAL_SPIN_CUDA_WARNING_FLAGS -Xcompiler=-Werror)
        endif()
    endif()
endif()

function(classical_spin_apply_warnings target)
    if(CLASSICAL_SPIN_ENABLE_WARNINGS)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:${CLASSICAL_SPIN_CXX_WARNING_FLAGS}>
            $<$<COMPILE_LANGUAGE:CUDA>:${CLASSICAL_SPIN_CUDA_WARNING_FLAGS}>
        )
    endif()
endfunction()

# CUDA compiler flags
# --extended-lambda allows __device__ annotation on lambdas (required for Thrust)
set(CMAKE_CUDA_FLAGS "-D__CUDA_NO_HOST_COMPILER_CHECK --extended-lambda --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math -lineinfo")

# Optimization flags for Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CXX_OPTIMIZATION_FLAGS
        -O3 -DNDEBUG -march=native -mtune=native -flto
        -funroll-loops -fomit-frame-pointer -fno-stack-protector
        -ffp-contract=fast -fno-math-errno -fno-trapping-math -fno-rounding-math
    )
    
    set(CUDA_OPTIMIZATION_FLAGS
        -O3 -DNDEBUG --use_fast_math --ptxas-options=-v,-O3
        -lineinfo --maxrregcount=32 --extra-device-vectorization
    )
    
    set(CUDA_HOST_OPTIMIZATION_FLAGS
        -Xcompiler=-O3 -Xcompiler=-ffast-math -Xcompiler=-funroll-loops
    )
    
    # AMX intrinsics workaround for nvcc/GCC 12
    set(CUDA_AMX_WORKAROUND
        --compiler-options=-D_AMXTILEINTRIN_H_INCLUDED
        --compiler-options=-D_AMX_INT8INTRIN_H_INCLUDED
        --compiler-options=-D_AMX_BF16INTRIN_H_INCLUDED
    )
    
    if(OpenMP_CXX_FOUND)
        list(APPEND CXX_OPTIMIZATION_FLAGS -fopenmp-simd)
    endif()
endif()

# ============================================================================
# SOURCE FILES
# ============================================================================

set(CLASSICAL_SPIN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CLASSICAL_SPIN_HEADERS
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/core/simple_linear_alg.h
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/core/simulation_config.h
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/core/spin_config.h
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/core/unitcell.h
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/core/unitcell_builders.h
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/lattice/lattice.h
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/lattice/mixed_lattice.h
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/lattice/phonon_lattice.h
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/gpu/gpu_common_helpers.cuh
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/gpu/lattice_gpu.cuh
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/gpu/lattice_gpu_api.h
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/gpu/mixed_lattice_gpu.cuh
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/gpu/mixed_lattice_gpu_api.h
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/io/hdf5_io.h
)

set(CORE_SOURCES
    src/core/simple_linear_alg.cpp
    src/core/unitcell_builders.cpp
    src/core/spin_config.cpp
    src/core/phonon_lattice.cpp
)

set(GPU_SOURCES
    src/gpu/gpu_common_helpers.cu
    src/gpu/lattice_gpu.cu
    src/gpu/mixed_lattice_gpu.cu
)

# ============================================================================
# LIBRARIES
# ============================================================================

add_library(classical_spin_headers INTERFACE)
target_sources(classical_spin_headers INTERFACE ${CLASSICAL_SPIN_HEADERS})
target_include_directories(classical_spin_headers INTERFACE 
    $<BUILD_INTERFACE:${CLASSICAL_SPIN_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin>
    $<BUILD_INTERFACE:${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/core>
    $<BUILD_INTERFACE:${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/lattice>
    $<BUILD_INTERFACE:${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/gpu>
    $<BUILD_INTERFACE:${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/io>
    $<INSTALL_INTERFACE:include>
    $<INSTALL_INTERFACE:include/classical_spin>
    $<INSTALL_INTERFACE:include/classical_spin/core>
    $<INSTALL_INTERFACE:include/classical_spin/lattice>
    $<INSTALL_INTERFACE:include/classical_spin/gpu>
    $<INSTALL_INTERFACE:include/classical_spin/io>
)
target_compile_features(classical_spin_headers INTERFACE cxx_std_20)
target_link_libraries(classical_spin_headers INTERFACE 
    OpenMP::OpenMP_CXX 
    MPI::MPI_CXX 
    Eigen3::Eigen
    ${HDF5_CXX_LIBRARIES}
    CUDA::cudart
)
target_include_directories(classical_spin_headers INTERFACE ${HDF5_INCLUDE_DIRS})
target_compile_definitions(classical_spin_headers INTERFACE 
    OPENMP_ENABLED MPI_ENABLED EIGEN3_ENABLED HDF5_ENABLED CUDA_ENABLED
)

add_library(classical_spin_core STATIC ${CORE_SOURCES})
target_compile_features(classical_spin_core PUBLIC cxx_std_20)
target_link_libraries(classical_spin_core PUBLIC 
    classical_spin_headers
    Eigen3::Eigen
    ${HDF5_CXX_LIBRARIES}
)
target_include_directories(classical_spin_core PUBLIC ${HDF5_INCLUDE_DIRS})
target_include_directories(classical_spin_core PUBLIC 
    $<BUILD_INTERFACE:${CLASSICAL_SPIN_INCLUDE_DIR}>
    $<INSTALL_INTERFACE:include>
)
target_compile_options(classical_spin_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_OPTIMIZATION_FLAGS}>
)
classical_spin_apply_warnings(classical_spin_core)

# CUDA library
if(GPU_SOURCES)
    add_library(classical_spin_gpu STATIC ${GPU_SOURCES})
    
    set_target_properties(classical_spin_gpu PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        POSITION_INDEPENDENT_CODE ON
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    
    target_include_directories(classical_spin_gpu PUBLIC 
        $<BUILD_INTERFACE:${CLASSICAL_SPIN_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include>
        ${Boost_INCLUDE_DIRS}
        ${HDF5_INCLUDE_DIRS}
    )
    
    target_link_libraries(classical_spin_gpu PUBLIC 
        classical_spin_headers
        Eigen3::Eigen
        CUDA::cudart CUDA::cublas CUDA::curand
        MPI::MPI_CXX
        ${HDF5_CXX_LIBRARIES}
    )
    
    target_compile_definitions(classical_spin_gpu PUBLIC 
        THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
        MPI_ENABLED
        HDF5_ENABLED
    )
    
    target_compile_options(classical_spin_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_OPTIMIZATION_FLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_HOST_OPTIMIZATION_FLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_AMX_WORKAROUND}>
    )
    classical_spin_apply_warnings(classical_spin_gpu)
endif()

# ============================================================================
# HELPER FUNCTION FOR EXECUTABLES
# ============================================================================

function(add_optimized_executable exec_name source_file)
    add_executable(${exec_name} ${source_file})
    target_compile_features(${exec_name} PRIVATE cxx_std_20)
    target_link_libraries(${exec_name} PRIVATE classical_spin_headers classical_spin_core)
    target_include_directories(${exec_name} PRIVATE ${CLASSICAL_SPIN_INCLUDE_DIR})
    
    if(TARGET classical_spin_gpu)
        target_link_libraries(${exec_name} PRIVATE classical_spin_gpu)
        set_target_properties(${exec_name} PROPERTIES
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
            CUDA_SEPARABLE_COMPILATION OFF
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
    endif()
    
    target_compile_options(${exec_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${CXX_OPTIMIZATION_FLAGS}>
        $<$<COMPILE_LANGUAGE:CXX>:-g1>
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_OPTIMIZATION_FLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_HOST_OPTIMIZATION_FLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )
    classical_spin_apply_warnings(${exec_name})
    
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        set_target_properties(${exec_name} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endfunction()

# ============================================================================
# EXECUTABLES
# ============================================================================

# Main spin solver executable
add_optimized_executable(spin_solver src/apps/spin_solver.cpp)


# ============================================================================
# BUILD CONFIGURATION
# ============================================================================

# Enable parallel builds
if(NOT DEFINED CMAKE_BUILD_PARALLEL_LEVEL)
    include(ProcessorCount)
    ProcessorCount(N)
    if(NOT N EQUAL 0)
        set(CMAKE_BUILD_PARALLEL_LEVEL ${N} CACHE STRING "Number of parallel build jobs")
    endif()
endif()

# Ninja generator optimization
if(CMAKE_GENERATOR STREQUAL "Ninja")
    set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)
endif()

# ============================================================================
# BUILD SUMMARY
# ============================================================================

message(STATUS "================ ClassicalSpin_Cpp Build =================")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Parallel Build Jobs: ${CMAKE_BUILD_PARALLEL_LEVEL}")
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  OpenMP: ${OpenMP_CXX_VERSION}")
message(STATUS "  MPI: ${MPI_CXX_VERSION}")
message(STATUS "  CUDA: ${CUDAToolkit_VERSION}")
message(STATUS "  Eigen3: ${Eigen3_VERSION}")
message(STATUS "  Boost: ${Boost_VERSION}")
message(STATUS "  HDF5: ${HDF5_VERSION}")
message(STATUS "")
message(STATUS "Source Files:")
message(STATUS "  Core sources: ${CORE_SOURCES}")
if(TARGET classical_spin_gpu)
    message(STATUS "  CUDA sources: ${GPU_SOURCES}")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "")
    message(STATUS "Optimization Flags:")
    message(STATUS "  C++: -O3 -march=native -flto ...")
    message(STATUS "  CUDA: -O3 --use_fast_math ...")
endif()

message(STATUS "===========================================================")
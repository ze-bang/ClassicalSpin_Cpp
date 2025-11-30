cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ClassicalSpin_Cpp LANGUAGES CXX CUDA)

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
set(CMAKE_CUDA_ARCHITECTURES "60;75;80;86;89" CACHE STRING "CUDA architectures")

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

file(GLOB HEADER_FILES "src/*.h" "src/*.hpp" "src/**/*.h" "src/**/*.hpp")
file(GLOB_RECURSE CUDA_SOURCE_FILES "src/*.cu")
file(GLOB_RECURSE CUDA_HEADER_FILES "src/*.cuh")

# Exclude main executable source from library
list(FILTER CUDA_SOURCE_FILES EXCLUDE REGEX ".*unified_simulation\\.cu$")

# C++ source files (non-CUDA) - compiled separately to reduce compile times
set(CPP_LIB_SOURCES
    src/unified_config.cpp
    src/simple_linear_alg.cpp
    src/unitcell_builders.cpp
)

# ============================================================================
# LIBRARIES
# ============================================================================

# C++ library with implementations (reduces compile time by separating from headers)
add_library(cpp_lib STATIC ${CPP_LIB_SOURCES})
target_include_directories(cpp_lib PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(cpp_lib PUBLIC cxx_std_20)
target_link_libraries(cpp_lib PUBLIC 
    Eigen3::Eigen
    ${HDF5_CXX_LIBRARIES}
)
target_include_directories(cpp_lib PUBLIC ${HDF5_INCLUDE_DIRS})
target_compile_options(cpp_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CXX_OPTIMIZATION_FLAGS}>
)

# Header-only interface library
add_library(headers INTERFACE)
target_include_directories(headers INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)
target_sources(headers INTERFACE ${HEADER_FILES})
target_compile_features(headers INTERFACE cxx_std_20)

# Link dependencies
target_link_libraries(headers INTERFACE 
    OpenMP::OpenMP_CXX 
    MPI::MPI_CXX 
    Eigen3::Eigen
    ${HDF5_CXX_LIBRARIES}
    CUDA::cudart
)

target_include_directories(headers INTERFACE ${HDF5_INCLUDE_DIRS})

target_compile_definitions(headers INTERFACE 
    OPENMP_ENABLED MPI_ENABLED EIGEN3_ENABLED HDF5_ENABLED CUDA_ENABLED
)

# CUDA library
if(CUDA_SOURCE_FILES)
    add_library(cuda_lib STATIC ${CUDA_SOURCE_FILES})
    
    set_target_properties(cuda_lib PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        POSITION_INDEPENDENT_CODE ON
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    
    target_include_directories(cuda_lib PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/dynamics>
        $<INSTALL_INTERFACE:include>
        ${Boost_INCLUDE_DIRS}
        ${HDF5_INCLUDE_DIRS}
    )
    
    target_link_libraries(cuda_lib PUBLIC 
        Eigen3::Eigen
        CUDA::cudart CUDA::cublas CUDA::curand
        MPI::MPI_CXX
        ${HDF5_CXX_LIBRARIES}
    )
    
    target_compile_definitions(cuda_lib PUBLIC 
        THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA
        MPI_ENABLED
        HDF5_ENABLED
    )
    
    target_compile_options(cuda_lib PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_OPTIMIZATION_FLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_HOST_OPTIMIZATION_FLAGS}>
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_AMX_WORKAROUND}>
    )
    
    # CUDA headers interface library
    add_library(cuda_headers INTERFACE)
    target_include_directories(cuda_headers INTERFACE src src/dynamics)
    target_sources(cuda_headers INTERFACE ${CUDA_HEADER_FILES})
    target_link_libraries(cuda_headers INTERFACE cuda_lib)
endif()

# ============================================================================
# HELPER FUNCTION FOR EXECUTABLES
# ============================================================================

function(add_optimized_executable exec_name source_file)
    add_executable(${exec_name} ${source_file})
    target_compile_features(${exec_name} PRIVATE cxx_std_20)
    target_link_libraries(${exec_name} PRIVATE headers cpp_lib)
    target_include_directories(${exec_name} PRIVATE src)
    
    if(CUDA_SOURCE_FILES)
        target_link_libraries(${exec_name} PRIVATE cuda_headers)
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
    
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        set_target_properties(${exec_name} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endfunction()

# ============================================================================
# EXECUTABLES
# ============================================================================

# Main unified simulation executable
add_optimized_executable(unified_simulation src/unified_simulation.cu)


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
message(STATUS "  C++ executables: ${CPP_SOURCE_FILES}")
message(STATUS "  CUDA executables: ${CUDA_RUN_SCRIPTS}")

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "")
    message(STATUS "Optimization Flags:")
    message(STATUS "  C++: -O3 -march=native -flto ...")
    message(STATUS "  CUDA: -O3 --use_fast_math ...")
endif()

message(STATUS "===========================================================")
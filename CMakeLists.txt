cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

option(ENABLE_CUDA "Enable CUDA support" OFF)

if(ENABLE_CUDA)
    project(ClassicalSpin_Cpp LANGUAGES CXX CUDA)
else()
    project(ClassicalSpin_Cpp LANGUAGES CXX)
endif()

option(CLASSICAL_SPIN_ENABLE_WARNINGS "Enable recommended warning flags" ON)
option(CLASSICAL_SPIN_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)

# ============================================================================
# PROJECT CONFIGURATION
# ============================================================================

# Enable using *_ROOT environment variables for find_package
cmake_policy(SET CMP0144 NEW)

# Use Config mode for Boost (avoids deprecated FindBoost module)
# CMP0167 is only available in CMake 3.30+
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # Modern CMake policies for CUDA
    cmake_policy(SET CMP0104 NEW)
    cmake_policy(SET CMP0105 NEW)
endif()

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Enable position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CTest)

# CUDA architectures
if(ENABLE_CUDA)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(_classical_spin_default_arch "native")
        if(DEFINED ENV{CLASSICAL_SPIN_CUDA_ARCHS} AND NOT "$ENV{CLASSICAL_SPIN_CUDA_ARCHS}" STREQUAL "")
            set(_classical_spin_default_arch "$ENV{CLASSICAL_SPIN_CUDA_ARCHS}")
        endif()
        set(CMAKE_CUDA_ARCHITECTURES "${_classical_spin_default_arch}" CACHE STRING "CUDA architectures" FORCE)
    endif()
    unset(_classical_spin_default_arch)
endif()

# ============================================================================
# DEPENDENCIES
# ============================================================================

if(ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    find_package(Thrust REQUIRED CONFIG)
endif()
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Boost 1.65 REQUIRED CONFIG)

# Use HDF5_DIR environment variable if available (set by module system)
if(DEFINED ENV{HDF5_DIR} AND NOT HDF5_ROOT)
    set(HDF5_ROOT "$ENV{HDF5_DIR}")
endif()
find_package(HDF5 QUIET COMPONENTS C CXX)

if(NOT HDF5_FOUND)
    find_package(PkgConfig QUIET)

    set(CLASSICAL_SPIN_HDF5_INCLUDE_HINTS "")
    set(CLASSICAL_SPIN_HDF5_LIBRARY_HINTS "")

    if(PkgConfig_FOUND)
        pkg_check_modules(CLASSICAL_SPIN_HDF5_PKG QUIET hdf5-serial)
        if(NOT CLASSICAL_SPIN_HDF5_PKG_FOUND)
            pkg_check_modules(CLASSICAL_SPIN_HDF5_PKG QUIET hdf5)
        endif()

        if(CLASSICAL_SPIN_HDF5_PKG_FOUND)
            list(APPEND CLASSICAL_SPIN_HDF5_INCLUDE_HINTS ${CLASSICAL_SPIN_HDF5_PKG_INCLUDE_DIRS})
            list(APPEND CLASSICAL_SPIN_HDF5_LIBRARY_HINTS ${CLASSICAL_SPIN_HDF5_PKG_LIBRARY_DIRS})
            if(NOT HDF5_VERSION)
                set(HDF5_VERSION "${CLASSICAL_SPIN_HDF5_PKG_VERSION}")
            endif()
        endif()
    endif()

    list(APPEND CLASSICAL_SPIN_HDF5_INCLUDE_HINTS
        /usr/include/hdf5/serial
        /usr/include/hdf5/openmpi
        /usr/local/include/hdf5/serial
        /usr/local/include/hdf5/openmpi
    )
    list(APPEND CLASSICAL_SPIN_HDF5_LIBRARY_HINTS
        /usr/lib/x86_64-linux-gnu/hdf5/serial
        /usr/lib/x86_64-linux-gnu/hdf5/openmpi
        /usr/lib64/hdf5/serial
        /usr/lib64/hdf5/openmpi
        /usr/lib/hdf5/serial
        /usr/lib/hdf5/openmpi
        /usr/local/lib/hdf5/serial
        /usr/local/lib/hdf5/openmpi
    )

    find_path(CLASSICAL_SPIN_HDF5_INCLUDE_DIR
        NAMES hdf5.h H5Cpp.h
        HINTS ${CLASSICAL_SPIN_HDF5_INCLUDE_HINTS}
    )
    find_library(CLASSICAL_SPIN_HDF5_C_LIBRARY
        NAMES hdf5
        HINTS ${CLASSICAL_SPIN_HDF5_LIBRARY_HINTS}
    )
    find_library(CLASSICAL_SPIN_HDF5_CXX_LIBRARY
        NAMES hdf5_cpp
        HINTS ${CLASSICAL_SPIN_HDF5_LIBRARY_HINTS}
    )
    find_library(CLASSICAL_SPIN_HDF5_HL_LIBRARY
        NAMES hdf5_hl
        HINTS ${CLASSICAL_SPIN_HDF5_LIBRARY_HINTS}
    )
    find_library(CLASSICAL_SPIN_HDF5_HL_CXX_LIBRARY
        NAMES hdf5_hl_cpp
        HINTS ${CLASSICAL_SPIN_HDF5_LIBRARY_HINTS}
    )

    if(CLASSICAL_SPIN_HDF5_INCLUDE_DIR AND CLASSICAL_SPIN_HDF5_C_LIBRARY AND CLASSICAL_SPIN_HDF5_CXX_LIBRARY)
        set(HDF5_FOUND TRUE)
        set(HDF5_INCLUDE_DIRS ${CLASSICAL_SPIN_HDF5_INCLUDE_DIR})
        set(HDF5_C_LIBRARIES ${CLASSICAL_SPIN_HDF5_C_LIBRARY})
        set(HDF5_CXX_LIBRARIES ${CLASSICAL_SPIN_HDF5_CXX_LIBRARY} ${CLASSICAL_SPIN_HDF5_C_LIBRARY})

        if(CLASSICAL_SPIN_HDF5_HL_LIBRARY)
            list(APPEND HDF5_C_LIBRARIES ${CLASSICAL_SPIN_HDF5_HL_LIBRARY})
            list(APPEND HDF5_CXX_LIBRARIES ${CLASSICAL_SPIN_HDF5_HL_LIBRARY})
        endif()
        if(CLASSICAL_SPIN_HDF5_HL_CXX_LIBRARY)
            list(APPEND HDF5_CXX_LIBRARIES ${CLASSICAL_SPIN_HDF5_HL_CXX_LIBRARY})
        endif()

        set(HDF5_LIBRARIES ${HDF5_CXX_LIBRARIES})
        list(REMOVE_DUPLICATES HDF5_C_LIBRARIES)
        list(REMOVE_DUPLICATES HDF5_CXX_LIBRARIES)
        list(REMOVE_DUPLICATES HDF5_LIBRARIES)

        message(STATUS "HDF5 found via pkg-config/manual fallback")
    endif()

    unset(CLASSICAL_SPIN_HDF5_INCLUDE_HINTS)
    unset(CLASSICAL_SPIN_HDF5_LIBRARY_HINTS)
endif()

if(NOT HDF5_FOUND)
    message(FATAL_ERROR
        "Unable to find HDF5 C and C++ development files. "
        "Set HDF5_ROOT/HDF5_DIR or install the HDF5 development package."
    )
endif()

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
            -Wno-unused-variable       # unused variables (reserved for future use)
            -Wno-stringop-overread     # false positive in AVX intrinsics (GCC bug)
            -Wno-stringop-overflow     # false positive in AVX intrinsics (GCC bug)
        )
    elseif(MSVC)
        list(APPEND CLASSICAL_SPIN_CXX_WARNING_FLAGS /W3 /permissive-)
    endif()

    if(ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
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
            -Xcompiler=-Wno-unused-variable
            -Xcompiler=-Wno-stringop-overread
            -Xcompiler=-Wno-stringop-overflow
        )
    endif()

    if(CLASSICAL_SPIN_WARNINGS_AS_ERRORS)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
            list(APPEND CLASSICAL_SPIN_CXX_WARNING_FLAGS -Werror)
        elseif(MSVC)
            list(APPEND CLASSICAL_SPIN_CXX_WARNING_FLAGS /WX)
        endif()

        if(ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
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
if(ENABLE_CUDA)
    # --extended-lambda allows __device__ annotation on lambdas (required for Thrust)
    set(CMAKE_CUDA_FLAGS "-D__CUDA_NO_HOST_COMPILER_CHECK --extended-lambda --expt-relaxed-constexpr -Wno-deprecated-gpu-targets")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math -lineinfo")
    
    # Suppress nvlink warnings about incompatible libraries (32-bit libs on 64-bit system)
    set(CMAKE_CUDA_LINK_FLAGS "${CMAKE_CUDA_LINK_FLAGS} -Xnvlink=-w")
endif()

# Optimization flags for Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CXX_OPTIMIZATION_FLAGS
        -O3 -DNDEBUG -march=native -mtune=native -flto
        -funroll-loops -fomit-frame-pointer -fno-stack-protector
        -ffp-contract=fast -fno-math-errno -fno-trapping-math -fno-rounding-math
        -Wno-stringop-overread -Wno-stringop-overflow  # suppress false positives in AVX intrinsics during LTO
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
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/core/spin_config.h
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/core/unitcell.h
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/core/unitcell_builders.h
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/core/gneb.h
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/core/gneb_math.h
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/lattice/lattice.h
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/lattice/mixed_lattice.h
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/lattice/phonon_lattice.h
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/lattice/strain_phonon_lattice.h
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/lattice/kitaev_bonds.h
    ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/io/hdf5_io.h
)

if(ENABLE_CUDA)
    list(APPEND CLASSICAL_SPIN_HEADERS
        ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/gpu/gpu_common_helpers.cuh
        ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/gpu/lattice_gpu.cuh
        ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/gpu/lattice_gpu_api.h
        ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/gpu/mixed_lattice_gpu.cuh
        ${CLASSICAL_SPIN_INCLUDE_DIR}/classical_spin/gpu/mixed_lattice_gpu_api.h
    )
endif()

set(CORE_SOURCES
    src/core/simple_linear_alg.cpp
    src/core/unitcell_builders.cpp
    src/core/spin_config.cpp
    src/core/phonon_lattice.cpp
    src/core/strain_phonon_lattice.cpp
    src/core/gneb.cpp
    src/core/gneb_strain.cpp
    # Per-concern TUs split out of the `lattice.h` / `mixed_lattice.h`
    # god-object headers. Each one hosts a topical slice of the class so
    # the header stays slim and the slices build in parallel.
    src/core/lattice_accumulator.cpp
    src/core/lattice_mc.cpp
    src/core/lattice_pt.cpp
    src/core/lattice_observables.cpp
    src/core/lattice_md.cpp
    src/core/mixed_lattice_mc.cpp
    src/core/mixed_lattice_pt.cpp
    src/core/mixed_lattice_observables.cpp
    src/core/mixed_lattice_md.cpp
)

if(ENABLE_CUDA)
    set(GPU_SOURCES
        src/gpu/gpu_common_helpers.cu
        src/gpu/lattice_gpu.cu
        src/gpu/mixed_lattice_gpu.cu
    )
endif()

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
)
if(ENABLE_CUDA)
    target_link_libraries(classical_spin_headers INTERFACE CUDA::cudart)
endif()
target_include_directories(classical_spin_headers INTERFACE ${HDF5_INCLUDE_DIRS})
target_compile_definitions(classical_spin_headers INTERFACE 
    OPENMP_ENABLED MPI_ENABLED EIGEN3_ENABLED HDF5_ENABLED
)
if(ENABLE_CUDA)
    target_compile_definitions(classical_spin_headers INTERFACE CUDA_ENABLED)
endif()

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
if(ENABLE_CUDA AND GPU_SOURCES)
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
        CUDA::cudart_static CUDA::cuda_driver CUDA::cublas CUDA::curand
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
    # Accept any additional source files via ARGN so multi-TU executables
    # (e.g. spin_solver, which was split into runners_*.cpp) can use the
    # same wrapper as single-file targets.
    add_executable(${exec_name} ${source_file} ${ARGN})
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
    # Suppress LTO warnings from AVX intrinsics (GCC bug with -flto and avx512)
    target_link_options(${exec_name} PRIVATE
        -Wno-stringop-overread -Wno-stringop-overflow
    )
    classical_spin_apply_warnings(${exec_name})
    
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        set_target_properties(${exec_name} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endfunction()

# ============================================================================
# EXECUTABLES
# ============================================================================

# Main spin solver executable — main() plus the per-lattice runner TUs
# (split out of spin_solver.cpp for parallel compilation and readability).
add_optimized_executable(spin_solver
    src/apps/spin_solver.cpp
    src/apps/runners_lattice.cpp
    src/apps/runners_phonon.cpp
    src/apps/runners_strain.cpp
    src/apps/runners_mixed.cpp
    src/apps/runners_parameter_sweep.cpp
)

# Test: zigzag (C3-breaking) → finite strain
add_optimized_executable(test_zigzag_strain src/apps/test_zigzag_strain.cpp)

# Test: parallel tempering on StrainPhononLattice (NCTO_STRAIN)
add_optimized_executable(test_pt_strain src/apps/test_pt_strain.cpp)

# Regression tests for strain-phonon Hamiltonian consistency
add_optimized_executable(test_strain_phonon_regressions src/apps/test_strain_phonon_regressions.cpp)

# Regression tests for mixed SU2-SU3 trilinear bookkeeping
add_optimized_executable(test_mixed_trilinear_regressions src/apps/test_mixed_trilinear_regressions.cpp)

# Regression test for TmFeO3 use_global_frame=0 ↔ use_global_frame=1 equivalence
# (canonical local-frame patch in unitcell_builders.cpp).
add_optimized_executable(test_global_frame_equivalence src/apps/test_global_frame_equivalence.cpp)

# Regression test for Ingredient XVIII: pulse-drive trajectories at
# the new 1e-8 default must agree with the legacy 1e-10 reference to
# physics tolerance (~ 5e-6 on the magnetisation observable).
add_optimized_executable(test_pulse_drive_tolerance src/apps/test_pulse_drive_tolerance.cpp)

# Regression test for Ingredient XX (audit B1): pulse_chunking helper
# must snap every segment endpoint to the T_step grid so Boost's
# `integrate_const` never drops a sub-T_step remainder at a seam.
add_optimized_executable(test_pulse_chunking_alignment src/apps/test_pulse_chunking_alignment.cpp)

# HDF5 parallel-tempering writer smoke test (no MPI / GPU needed)
add_optimized_executable(test_hdf5_pt src/test_hdf5_pt.cpp)

# Monte-Carlo kernel micro-benchmark (sweeps/sec for each MC algorithm on
# a few canonical models). Intentionally single-rank — meant for measuring
# the local-update kernels in isolation.
add_optimized_executable(bench_mc src/apps/bench_mc.cpp)

# Molecular-dynamics RHS micro-benchmark (RHS-calls/sec for each LLG /
# spin-strain ODE kernel on a few canonical models). Same intent as
# bench_mc but for the time-evolution side of the library.
add_optimized_executable(bench_md src/apps/bench_md.cpp)

# Integrator-level micro-benchmark for Ingredient XVIII (pump-probe
# default tolerance relaxation + rk54 dispatch fix). Times
# single_pulse_drive() on a small TmFeO3 mixed lattice across method
# and tolerance combinations and prints speedup ratios.
add_optimized_executable(bench_integrators src/apps/bench_integrators.cpp)

if(BUILD_TESTING)
    add_test(NAME strain_phonon_regressions COMMAND test_strain_phonon_regressions)
    add_test(NAME mixed_trilinear_regressions COMMAND test_mixed_trilinear_regressions)
    add_test(NAME global_frame_equivalence COMMAND test_global_frame_equivalence)
    add_test(NAME pulse_drive_tolerance COMMAND test_pulse_drive_tolerance)
    set_tests_properties(pulse_drive_tolerance PROPERTIES TIMEOUT 120)
    add_test(NAME pulse_chunking_alignment COMMAND test_pulse_chunking_alignment)
    set_tests_properties(pulse_chunking_alignment PROPERTIES TIMEOUT 30)
    add_test(NAME hdf5_pt_writer_smoke COMMAND test_hdf5_pt)
    set_tests_properties(hdf5_pt_writer_smoke PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    # MPI-based tests: both use MPI_Init internally. Register the single-rank
    # invocation of zigzag_strain (Part 3's SA sweep is gated behind
    # TEST_ZIGZAG_FULL so the default run stays fast) and a 2-rank run of
    # test_pt_strain. Skip these gracefully if mpirun is not available.
    find_program(MPIEXEC_EXEC NAMES mpirun mpiexec)
    if(MPIEXEC_EXEC)
        add_test(NAME zigzag_strain
                 COMMAND ${MPIEXEC_EXEC} -n 1 $<TARGET_FILE:test_zigzag_strain>)
        set_tests_properties(zigzag_strain PROPERTIES TIMEOUT 60)

        add_test(NAME pt_strain
                 COMMAND ${MPIEXEC_EXEC} -n 2 --oversubscribe
                         $<TARGET_FILE:test_pt_strain>)
        set_tests_properties(pt_strain PROPERTIES TIMEOUT 120)
    endif()
endif()


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
if(ENABLE_CUDA)
    message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "CUDA: Disabled")
endif()
message(STATUS "Parallel Build Jobs: ${CMAKE_BUILD_PARALLEL_LEVEL}")
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  OpenMP: ${OpenMP_CXX_VERSION}")
message(STATUS "  MPI: ${MPI_CXX_VERSION}")
if(ENABLE_CUDA)
    message(STATUS "  CUDA: ${CUDAToolkit_VERSION}")
endif()
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
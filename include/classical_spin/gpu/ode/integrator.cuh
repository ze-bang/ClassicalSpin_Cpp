#ifndef CLASSICAL_SPIN_GPU_ODE_INTEGRATOR_CUH
#define CLASSICAL_SPIN_GPU_ODE_INTEGRATOR_CUH

// =============================================================================
// gpu::ode -- in-house, header-only GPU ODE integrator module.
//
// Goal: a single, physics-agnostic implementation of the explicit Runge-Kutta
// family (and SSPRK53 / Bulirsch-Stoer) that is shared by every GPU time
// propagator in the project (single lattice, mixed lattice, batched 2DCS),
// eliminating the previous byte-for-byte duplication of the stepper code.
//
// Design
// ------
//   * State          : thrust::device_vector<double>  (flat GPU array).
//   * System concept : any callable with
//                           void operator()(const State& in, State& out, double t)
//                       that evaluates dS/dt = f(t, S) on the device. The system
//                       owns its own kernel launch + synchronisation, so a single
//                       trajectory (LLG over lattice_size*spin_dim) and a batched
//                       ensemble (LLG over B*N*spin_dim) plug into the SAME
//                       steppers -- only the RHS functor differs.
//   * Workspace      : owns the reusable RK stage buffers (k1..k13, tmp). Lives
//                       with the caller so buffers persist across steps (no
//                       per-step reallocation).
//
// The element-wise update kernels (update_arrays_kernel, rk_stage_update_*,
// rk4_final_update_kernel, dopri5_final_update_kernel, ...) are declared in
// gpu_common_helpers.cuh and operate on raw pointers over `state.size()`
// elements, so they are identical for single and batched layouts.
//
// This header is templated and therefore compiled by nvcc inside every .cu TU
// that includes it. The fused-kernel sync counts are preserved exactly, so
// there is no performance regression relative to the hand-rolled steppers.
// =============================================================================

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

#include <string>
#include <vector>
#include <utility>
#include <iostream>

#include "classical_spin/gpu/gpu_common_helpers.cuh"

namespace gpu {
namespace ode {

using State = thrust::device_vector<double>;

// -----------------------------------------------------------------------------
// Workspace: reusable RK stage buffers. `ensure` grows (never shrinks) the
// buffers needed for the requested number of stages.
// -----------------------------------------------------------------------------
struct Workspace {
    State k1, k2, k3, k4, k5, k6, k7;
    State k8, k9, k10, k11, k12, k13;
    State tmp;

    void ensure(size_t n, int stages) {
        auto grow = [&](State& a) { if (a.size() < n) a.resize(n, 0.0); };
        grow(k1); grow(k2); grow(tmp);
        if (stages >= 4)  { grow(k3); grow(k4); }
        if (stages >= 6)  { grow(k5); grow(k6); }
        if (stages >= 7)  { grow(k7); }
        if (stages >= 11) { grow(k8); grow(k9); grow(k10); grow(k11); }
        if (stages >= 13) { grow(k12); grow(k13); }
    }
};

// -----------------------------------------------------------------------------
// step: advance `state` by one step of size `dt` using `method`.
//
// `system` is the RHS functor (const State&, State&, double). `ws` provides the
// stage buffers. The available methods mirror the historical step_gpu():
//   euler, rk2/midpoint, rk4, rk5/rkck54, dopri5, rk78/rkf78,
//   bulirsch_stoer/bs, ssprk53, rk54/rkf54, and a fallback to ssprk53.
// -----------------------------------------------------------------------------
template <class System>
void step(System& system, State& state, double t, double dt,
          const std::string& method, Workspace& ws) {
    const size_t array_size = state.size();

    const int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid(static_cast<unsigned>((array_size + BLOCK_SIZE - 1) / BLOCK_SIZE));

    double* d_state = thrust::raw_pointer_cast(state.data());

    if (method == "euler") {
        // y_{n+1} = y_n + h * f(t_n, y_n)  (1st order, 1 eval, 1 sync)
        ws.ensure(array_size, 2);
        double* d_k1 = thrust::raw_pointer_cast(ws.k1.data());

        system(state, ws.k1, t);
        ::update_arrays_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k1, dt, array_size);
        cudaDeviceSynchronize();

    } else if (method == "rk2" || method == "midpoint") {
        // Modified midpoint (2nd order, 2 evals, 2 syncs)
        ws.ensure(array_size, 2);
        double* d_k1 = thrust::raw_pointer_cast(ws.k1.data());
        double* d_k2 = thrust::raw_pointer_cast(ws.k2.data());
        double* d_tmp = thrust::raw_pointer_cast(ws.tmp.data());

        system(state, ws.k1, t);
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, 0.5 * dt, array_size);
        cudaDeviceSynchronize();

        system(ws.tmp, ws.k2, t + 0.5 * dt);
        ::update_arrays_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k2, dt, array_size);
        cudaDeviceSynchronize();

    } else if (method == "rk4") {
        // Classic RK4 (4th order, 4 evals, 4 syncs via fused final update)
        ws.ensure(array_size, 4);
        double* d_k1 = thrust::raw_pointer_cast(ws.k1.data());
        double* d_k2 = thrust::raw_pointer_cast(ws.k2.data());
        double* d_k3 = thrust::raw_pointer_cast(ws.k3.data());
        double* d_k4 = thrust::raw_pointer_cast(ws.k4.data());
        double* d_tmp = thrust::raw_pointer_cast(ws.tmp.data());

        system(state, ws.k1, t);

        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, 0.5 * dt, array_size);
        cudaDeviceSynchronize();
        system(ws.tmp, ws.k2, t + 0.5 * dt);

        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k2, 0.5 * dt, array_size);
        cudaDeviceSynchronize();
        system(ws.tmp, ws.k3, t + 0.5 * dt);

        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k3, dt, array_size);
        cudaDeviceSynchronize();
        system(ws.tmp, ws.k4, t + dt);

        ::rk4_final_update_kernel<<<grid, block>>>(d_state, d_k1, d_k2, d_k3, d_k4, dt / 6.0, array_size);
        cudaDeviceSynchronize();

    } else if (method == "rk5" || method == "rkck54") {
        // Cash-Karp RK5(4) (5th order, 6 evals)
        ws.ensure(array_size, 6);
        double* d_k1 = thrust::raw_pointer_cast(ws.k1.data());
        double* d_k2 = thrust::raw_pointer_cast(ws.k2.data());
        double* d_k3 = thrust::raw_pointer_cast(ws.k3.data());
        double* d_k4 = thrust::raw_pointer_cast(ws.k4.data());
        double* d_k5 = thrust::raw_pointer_cast(ws.k5.data());
        double* d_k6 = thrust::raw_pointer_cast(ws.k6.data());
        double* d_tmp = thrust::raw_pointer_cast(ws.tmp.data());

        constexpr double a21 = 1.0/5.0;
        constexpr double a31 = 3.0/40.0, a32 = 9.0/40.0;
        constexpr double a41 = 3.0/10.0, a42 = -9.0/10.0, a43 = 6.0/5.0;
        constexpr double a51 = -11.0/54.0, a52 = 5.0/2.0, a53 = -70.0/27.0, a54 = 35.0/27.0;
        constexpr double a61 = 1631.0/55296.0, a62 = 175.0/512.0, a63 = 575.0/13824.0;
        constexpr double a64 = 44275.0/110592.0, a65 = 253.0/4096.0;
        constexpr double c2 = 1.0/5.0, c3 = 3.0/10.0, c4 = 3.0/5.0, c5 = 1.0, c6 = 7.0/8.0;
        constexpr double b1 = 37.0/378.0, b3 = 250.0/621.0, b4 = 125.0/594.0, b6 = 512.0/1771.0;

        system(state, ws.k1, t);

        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a21 * dt, array_size);
        cudaDeviceSynchronize();
        system(ws.tmp, ws.k2, t + c2 * dt);

        ::rk_stage_update_2_kernel<<<grid, block>>>(d_tmp, d_state, d_k1, a31, d_k2, a32, dt, array_size);
        cudaDeviceSynchronize();
        system(ws.tmp, ws.k3, t + c3 * dt);

        ::rk_stage_update_3_kernel<<<grid, block>>>(d_tmp, d_state, d_k1, a41, d_k2, a42, d_k3, a43, dt, array_size);
        cudaDeviceSynchronize();
        system(ws.tmp, ws.k4, t + c4 * dt);

        ::rk_stage_update_4_kernel<<<grid, block>>>(d_tmp, d_state, d_k1, a51, d_k2, a52, d_k3, a53, d_k4, a54, dt, array_size);
        cudaDeviceSynchronize();
        system(ws.tmp, ws.k5, t + c5 * dt);

        ::rk_stage_update_5_kernel<<<grid, block>>>(d_tmp, d_state, d_k1, a61, d_k2, a62, d_k3, a63, d_k4, a64, d_k5, a65, dt, array_size);
        cudaDeviceSynchronize();
        system(ws.tmp, ws.k6, t + c6 * dt);

        ::dopri5_final_update_kernel<<<grid, block>>>(d_state, d_k1, d_k3, d_k4, d_k6, d_k6, dt, b1, b3, b4, 0.0, b6, array_size);
        cudaDeviceSynchronize();

    } else if (method == "dopri5") {
        // Dormand-Prince 5(4) (5th order, 6 syncs via fused kernels)
        ws.ensure(array_size, 6);
        double* d_k1 = thrust::raw_pointer_cast(ws.k1.data());
        double* d_k2 = thrust::raw_pointer_cast(ws.k2.data());
        double* d_k3 = thrust::raw_pointer_cast(ws.k3.data());
        double* d_k4 = thrust::raw_pointer_cast(ws.k4.data());
        double* d_k5 = thrust::raw_pointer_cast(ws.k5.data());
        double* d_k6 = thrust::raw_pointer_cast(ws.k6.data());
        double* d_tmp = thrust::raw_pointer_cast(ws.tmp.data());

        constexpr double a21 = 1.0/5.0;
        constexpr double a31 = 3.0/40.0, a32 = 9.0/40.0;
        constexpr double a41 = 44.0/45.0, a42 = -56.0/15.0, a43 = 32.0/9.0;
        constexpr double a51 = 19372.0/6561.0, a52 = -25360.0/2187.0, a53 = 64448.0/6561.0, a54 = -212.0/729.0;
        constexpr double a61 = 9017.0/3168.0, a62 = -355.0/33.0, a63 = 46732.0/5247.0, a64 = 49.0/176.0, a65 = -5103.0/18656.0;
        constexpr double c2 = 1.0/5.0, c3 = 3.0/10.0, c4 = 4.0/5.0, c5 = 8.0/9.0, c6 = 1.0;
        constexpr double b1 = 35.0/384.0, b3 = 500.0/1113.0, b4 = 125.0/192.0, b5 = -2187.0/6784.0, b6 = 11.0/84.0;

        system(state, ws.k1, t);

        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a21 * dt, array_size);
        cudaDeviceSynchronize();
        system(ws.tmp, ws.k2, t + c2 * dt);

        ::rk_stage_update_2_kernel<<<grid, block>>>(d_tmp, d_state, d_k1, a31, d_k2, a32, dt, array_size);
        cudaDeviceSynchronize();
        system(ws.tmp, ws.k3, t + c3 * dt);

        ::rk_stage_update_3_kernel<<<grid, block>>>(d_tmp, d_state, d_k1, a41, d_k2, a42, d_k3, a43, dt, array_size);
        cudaDeviceSynchronize();
        system(ws.tmp, ws.k4, t + c4 * dt);

        ::rk_stage_update_4_kernel<<<grid, block>>>(d_tmp, d_state, d_k1, a51, d_k2, a52, d_k3, a53, d_k4, a54, dt, array_size);
        cudaDeviceSynchronize();
        system(ws.tmp, ws.k5, t + c5 * dt);

        ::rk_stage_update_5_kernel<<<grid, block>>>(d_tmp, d_state, d_k1, a61, d_k2, a62, d_k3, a63, d_k4, a64, d_k5, a65, dt, array_size);
        cudaDeviceSynchronize();
        system(ws.tmp, ws.k6, t + c6 * dt);

        ::dopri5_final_update_kernel<<<grid, block>>>(d_state, d_k1, d_k3, d_k4, d_k5, d_k6, dt, b1, b3, b4, b5, b6, array_size);
        cudaDeviceSynchronize();

    } else if (method == "rk78" || method == "rkf78") {
        // Runge-Kutta-Fehlberg 7(8) (8th order, 13 evals; simplified weights)
        ws.ensure(array_size, 11);
        double* d_k1  = thrust::raw_pointer_cast(ws.k1.data());
        double* d_k2  = thrust::raw_pointer_cast(ws.k2.data());
        double* d_k3  = thrust::raw_pointer_cast(ws.k3.data());
        double* d_k4  = thrust::raw_pointer_cast(ws.k4.data());
        double* d_k5  = thrust::raw_pointer_cast(ws.k5.data());
        double* d_k6  = thrust::raw_pointer_cast(ws.k6.data());
        double* d_k7  = thrust::raw_pointer_cast(ws.k7.data());
        double* d_k8  = thrust::raw_pointer_cast(ws.k8.data());
        double* d_k9  = thrust::raw_pointer_cast(ws.k9.data());
        double* d_k10 = thrust::raw_pointer_cast(ws.k10.data());
        double* d_k11 = thrust::raw_pointer_cast(ws.k11.data());
        double* d_tmp = thrust::raw_pointer_cast(ws.tmp.data());

        constexpr double c2 = 2.0/27.0, c3 = 1.0/9.0, c4 = 1.0/6.0, c5 = 5.0/12.0;
        constexpr double c6 = 1.0/2.0, c7 = 5.0/6.0, c8 = 1.0/6.0, c9 = 2.0/3.0;
        constexpr double c10 = 1.0/3.0, c11 = 1.0;

        constexpr double b1 = 41.0/840.0, b6 = 34.0/105.0, b7 = 9.0/35.0, b8 = 9.0/35.0;
        constexpr double b9 = 9.0/280.0, b10 = 9.0/280.0, b11 = 41.0/840.0;

        system(state, ws.k1, t);

        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, c2 * dt, array_size);
        cudaDeviceSynchronize();
        system(ws.tmp, ws.k2, t + c2 * dt);

        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, (1.0/36.0) * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k2, (1.0/12.0) * dt, array_size);
        cudaDeviceSynchronize();
        system(ws.tmp, ws.k3, t + c3 * dt);

        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, (1.0/24.0) * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k3, (1.0/8.0) * dt, array_size);
        cudaDeviceSynchronize();
        system(ws.tmp, ws.k4, t + c4 * dt);

        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, (5.0/12.0) * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k4, (-25.0/16.0) * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k3, (25.0/16.0) * dt, array_size);
        cudaDeviceSynchronize();
        system(ws.tmp, ws.k5, t + c5 * dt);

        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, (1.0/20.0) * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k4, (1.0/4.0) * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k5, (1.0/5.0) * dt, array_size);
        cudaDeviceSynchronize();
        system(ws.tmp, ws.k6, t + c6 * dt);

        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, (-25.0/108.0) * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k4, (125.0/108.0) * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k5, (-65.0/27.0) * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k6, (125.0/54.0) * dt, array_size);
        cudaDeviceSynchronize();
        system(ws.tmp, ws.k7, t + c7 * dt);

        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, (31.0/300.0) * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k5, (61.0/225.0) * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k6, (-2.0/9.0) * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k7, (13.0/900.0) * dt, array_size);
        cudaDeviceSynchronize();
        system(ws.tmp, ws.k8, t + c8 * dt);

        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, 2.0 * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k4, (-53.0/6.0) * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k5, (704.0/45.0) * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k6, (-107.0/9.0) * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k7, (67.0/90.0) * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k8, 3.0 * dt, array_size);
        cudaDeviceSynchronize();
        system(ws.tmp, ws.k9, t + c9 * dt);

        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, (-91.0/108.0) * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k4, (23.0/108.0) * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k5, (-976.0/135.0) * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k6, (311.0/54.0) * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k7, (-19.0/60.0) * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k8, (17.0/6.0) * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k9, (-1.0/12.0) * dt, array_size);
        cudaDeviceSynchronize();
        system(ws.tmp, ws.k10, t + c10 * dt);

        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, (2383.0/4100.0) * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k4, (-341.0/164.0) * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k5, (4496.0/1025.0) * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k6, (-301.0/82.0) * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k7, (2133.0/4100.0) * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k8, (45.0/82.0) * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k9, (45.0/164.0) * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k10, (18.0/41.0) * dt, array_size);
        cudaDeviceSynchronize();
        system(ws.tmp, ws.k11, t + c11 * dt);

        // 8th order solution
        ::update_arrays_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k1, b1 * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k6, b6 * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k7, b7 * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k8, b8 * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k9, b9 * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k10, b10 * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k11, b11 * dt, array_size);
        cudaDeviceSynchronize();

    } else if (method == "bulirsch_stoer" || method == "bs") {
        // Modified midpoint with 4 substeps (Bulirsch-Stoer, simplified)
        ws.ensure(array_size, 4);
        double* d_k      = thrust::raw_pointer_cast(ws.k1.data());   // k
        double* d_y_prev = thrust::raw_pointer_cast(ws.k2.data());   // y_prev
        double* d_y_mid  = thrust::raw_pointer_cast(ws.k3.data());   // y_mid
        double* d_y_next = thrust::raw_pointer_cast(ws.k4.data());   // y_next

        const int n_substeps = 4;
        const double h_sub = dt / n_substeps;

        // y_prev = y
        thrust::copy(state.begin(), state.end(), ws.k2.begin());

        // First Euler step: y_mid = y + h_sub * f(t, y)
        system(state, ws.k1, t);
        ::update_arrays_kernel<<<grid, block>>>(d_y_mid, d_state, 1.0, d_k, h_sub, array_size);
        cudaDeviceSynchronize();

        for (int i = 1; i < n_substeps; ++i) {
            double t_curr = t + i * h_sub;
            system(ws.k3, ws.k1, t_curr);
            ::update_arrays_kernel<<<grid, block>>>(d_y_next, d_y_prev, 1.0, d_k, 2.0 * h_sub, array_size);
            cudaDeviceSynchronize();
            thrust::copy(ws.k3.begin(), ws.k3.end(), ws.k2.begin());   // y_prev = y_mid
            thrust::copy(ws.k4.begin(), ws.k4.end(), ws.k3.begin());   // y_mid  = y_next
        }

        // Final correction step
        system(ws.k3, ws.k1, t + dt);
        ::update_arrays_kernel<<<grid, block>>>(d_y_next, d_y_mid, 0.5, d_y_prev, 0.5, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_state, d_y_next, 1.0, d_k, 0.5 * h_sub, array_size);
        cudaDeviceSynchronize();

    } else if (method == "ssprk53") {
        // Strong-Stability-Preserving RK, 5 stages, 3rd order
        constexpr double a30 = 0.355909775063327;
        constexpr double a32 = 0.644090224936674;
        constexpr double a40 = 0.367933791638137;
        constexpr double a43 = 0.632066208361863;
        constexpr double a52 = 0.237593836598569;
        constexpr double a54 = 0.762406163401431;
        constexpr double b10 = 0.377268915331368;
        constexpr double b21 = 0.377268915331368;
        constexpr double b32 = 0.242995220537396;
        constexpr double b43 = 0.238458932846290;
        constexpr double b54 = 0.287632146308408;
        constexpr double c1 = 0.377268915331368;
        constexpr double c2 = 0.754537830662736;
        constexpr double c3 = 0.728985661612188;
        constexpr double c4 = 0.699226135931670;

        ws.ensure(array_size, 2);
        double* d_k   = thrust::raw_pointer_cast(ws.k1.data());        // k
        double* d_tmp = thrust::raw_pointer_cast(ws.tmp.data());       // tmp
        double* d_u   = thrust::raw_pointer_cast(ws.k2.data());        // u

        system(state, ws.k1, t);
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k, b10 * dt, array_size);
        cudaDeviceSynchronize();

        system(ws.tmp, ws.k1, t + c1 * dt);
        ::update_arrays_kernel<<<grid, block>>>(d_u, d_tmp, 1.0, d_k, b21 * dt, array_size);
        cudaDeviceSynchronize();

        system(ws.k2, ws.k1, t + c2 * dt);
        ::update_arrays_three_kernel<<<grid, block>>>(d_tmp, d_state, a30, d_u, a32, d_k, b32 * dt, array_size);
        cudaDeviceSynchronize();

        system(ws.tmp, ws.k1, t + c3 * dt);
        ::update_arrays_three_kernel<<<grid, block>>>(d_tmp, d_state, a40, d_tmp, a43, d_k, b43 * dt, array_size);
        cudaDeviceSynchronize();

        system(ws.tmp, ws.k1, t + c4 * dt);
        ::update_arrays_three_kernel<<<grid, block>>>(d_state, d_u, a52, d_tmp, a54, d_k, b54 * dt, array_size);
        cudaDeviceSynchronize();
        cudaDeviceSynchronize();

    } else if (method == "rk54" || method == "rkf54") {
        // 5(4) embedded -> reuse dopri5
        step(system, state, t, dt, "dopri5", ws);

    } else {
        std::cerr << "Warning: Unknown GPU integration method '" << method
                  << "', using ssprk53" << std::endl;
        step(system, state, t, dt, "ssprk53", ws);
    }
}

// -----------------------------------------------------------------------------
// integrate: fixed-step driver.
//
// Loops `while (t < T_end)`, invoking `observe(t, state)` every `save_interval`
// steps and once more on the final state. `observe` is any callable
// `void(double t, const State& state)`. This reproduces the historical
// integrate_gpu loop exactly; callers that need a different stopping rule
// (e.g. the batched 2DCS half-open grid) can drive `step` directly.
// -----------------------------------------------------------------------------
template <class System, class Observer>
void integrate(System& system, State& state, double T_start, double T_end,
               double dt, size_t save_interval, Observer&& observe,
               const std::string& method, Workspace& ws) {
    double t = T_start;
    size_t step_index = 0;

    while (t < T_end) {
        if (step_index % save_interval == 0) {
            observe(t, state);
        }
        step(system, state, t, dt, method, ws);
        t += dt;
        ++step_index;
    }

    GPU_CHECK_KERNEL();
    observe(t, state);
}

} // namespace ode
} // namespace gpu

#endif // CLASSICAL_SPIN_GPU_ODE_INTEGRATOR_CUH

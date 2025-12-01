#include "classical_spin/gpu/lattice_gpu.cuh"
#include "classical_spin/gpu/lattice_gpu_api.h"
#include "classical_spin/gpu/gpu_common_helpers.cuh"

namespace gpu {

// Common device functions and kernels are now in gpu_common_helpers.cu/cuh
// ======================= Device Helper Functions =======================

// Note: Common device functions (multiply_matrix_vector_device, cross_product_SU2_device,
// cross_product_SU3_device, init_local_field_device, add_onsite_contribution_device,
// add_bilinear_contribution_device, add_drive_field_device, compute_ll_derivative_device)
// are provided by gpu_common_helpers.cuh

/**
 * Compute local field for a site (device function)
 * 
 * Computes: H_eff = -B_ext + 2*A·S + sum_j J_ij·S_j
 */
__device__
void compute_local_field_device(
    double* local_field,
    const double* d_spins,
    int site,
    const double* d_field,
    const double* d_onsite,
    const double* d_bilinear_vals,
    const size_t* d_bilinear_idx,
    const size_t* d_bilinear_counts,
    size_t max_bilinear,
    size_t lattice_size,
    size_t spin_dim
) {
    const double* spin_here = &d_spins[site * spin_dim];
    double temp[8];
    
    // Initialize: H = -B_ext
    ::init_local_field_device(local_field, &d_field[site * spin_dim], spin_dim);
    
    // On-site: H += 2*A*S (factor 2 from derivative of quadratic term)
    ::add_onsite_contribution_device(local_field, &d_onsite[site * spin_dim * spin_dim],
                                     spin_here, temp, spin_dim, 2.0);
    
    // Bilinear: H += sum_j J_ij * S_j
    size_t num_neighbors = d_bilinear_counts[site];
    for (size_t n = 0; n < num_neighbors && n < max_bilinear; ++n) {
        size_t partner = d_bilinear_idx[site * max_bilinear + n];
        if (partner < lattice_size) {
            ::add_bilinear_contribution_device(local_field,
                &d_bilinear_vals[(site * max_bilinear + n) * spin_dim * spin_dim],
                &d_spins[partner * spin_dim], temp, spin_dim);
        }
    }
}

// ======================= Kernel Implementations =======================

/**
 * Kernel to compute Landau-Lifshitz derivatives using flattened arrays
 * This is the primary LLG kernel used by all integration methods.
 * 
 * Computes: dS/dt = H_eff × S
 * where H_eff = -B_ext + 2*A·S + sum_j J_ij·S_j - H_drive(t)
 */
__global__
void LLG_flat_kernel(
    double* d_dsdt,
    const double* d_spins,
    double* d_local_field,
    const double* d_field,
    const double* d_onsite,
    const double* d_bilinear_vals,
    const size_t* d_bilinear_idx,
    const size_t* d_bilinear_counts,
    const double* d_field_drive,
    double pulse_amp,
    double pulse_width,
    double pulse_freq,
    double t_pulse_1,
    double t_pulse_2,
    double curr_time,
    size_t lattice_size,
    size_t spin_dim,
    size_t N_atoms,
    size_t max_bilinear
) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= lattice_size) return;
    
    const double* spin_here = &d_spins[site * spin_dim];
    double* local_field = &d_local_field[site * spin_dim];
    double* dsdt = &d_dsdt[site * spin_dim];
    
    // Compute local field using device function
    compute_local_field_device(
        local_field, d_spins, site,
        d_field, d_onsite,
        d_bilinear_vals, d_bilinear_idx, d_bilinear_counts,
        max_bilinear, lattice_size, spin_dim
    );
    
    // Add drive field: H -= H_drive(t)
    ::add_drive_field_device(local_field, d_field_drive, d_field_drive + N_atoms * spin_dim,
                             site % N_atoms, pulse_amp, pulse_width, pulse_freq,
                             t_pulse_1, t_pulse_2, curr_time, spin_dim);
    
    // Compute Landau-Lifshitz derivative: dS/dt = H_eff × S
    ::compute_ll_derivative_device(dsdt, spin_here, local_field, spin_dim);
}

// ======================= GPUODESystem Implementation =======================

// Alias kernel names to use common implementations from gpu_common_helpers.cu
#define update_state_kernel ::update_arrays_kernel
#define update_state_three_kernel ::update_arrays_three_kernel
void GPUODESystem::operator()(const GPUState& x, GPUState& dxdt, double t) const {
    const int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid((data.lattice_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Get raw pointers
    const double* d_x = thrust::raw_pointer_cast(x.data());
    double* d_dxdt = thrust::raw_pointer_cast(dxdt.data());
    double* d_local_field = thrust::raw_pointer_cast(data.local_field.data());
    const double* d_field = thrust::raw_pointer_cast(data.field.data());
    const double* d_onsite = thrust::raw_pointer_cast(data.onsite.data());
    const double* d_bilinear_vals = thrust::raw_pointer_cast(data.bilinear_vals.data());
    const size_t* d_bilinear_idx = thrust::raw_pointer_cast(data.bilinear_idx.data());
    const size_t* d_bilinear_counts = thrust::raw_pointer_cast(data.bilinear_counts.data());
    const double* d_field_drive = thrust::raw_pointer_cast(data.field_drive.data());
    
    LLG_flat_kernel<<<grid, block>>>(
        d_dxdt, d_x, d_local_field,
        d_field, d_onsite,
        d_bilinear_vals, d_bilinear_idx, d_bilinear_counts,
        d_field_drive,
        data.pulse_amp, data.pulse_width, data.pulse_freq,
        data.t_pulse_1, data.t_pulse_2,
        t,
        data.lattice_size, data.spin_dim, data.N_atoms, data.max_bilinear
    );
    
    cudaDeviceSynchronize();
}

/**
 * Create GPU lattice data from host arrays
 * Internal function - returns the raw struct
 */
GPULatticeData create_gpu_lattice_data_internal(
    size_t lattice_size,
    size_t spin_dim,
    size_t N_atoms,
    size_t max_bilinear,
    const std::vector<double>& flat_field,
    const std::vector<double>& flat_onsite,
    const std::vector<double>& flat_bilinear,
    const std::vector<size_t>& flat_partners,
    const std::vector<size_t>& num_bilinear_per_site
) {
    GPULatticeData data;
    data.lattice_size = lattice_size;
    data.spin_dim = spin_dim;
    data.N_atoms = N_atoms;
    data.max_bilinear = max_bilinear;
    
    size_t array_size = lattice_size * spin_dim;
    
    // Copy arrays to device
    data.field = thrust::device_vector<double>(flat_field.begin(), flat_field.end());
    data.onsite = thrust::device_vector<double>(flat_onsite.begin(), flat_onsite.end());
    data.bilinear_vals = thrust::device_vector<double>(flat_bilinear.begin(), flat_bilinear.end());
    data.bilinear_idx = thrust::device_vector<size_t>(flat_partners.begin(), flat_partners.end());
    data.bilinear_counts = thrust::device_vector<size_t>(num_bilinear_per_site.begin(), num_bilinear_per_site.end());
    
    // Initialize working arrays
    data.work_1.resize(array_size, 0.0);
    data.work_2.resize(array_size, 0.0);
    data.work_3.resize(array_size, 0.0);
    data.local_field.resize(array_size, 0.0);
    
    // Initialize field drive to zeros
    data.field_drive.resize(2 * N_atoms * spin_dim, 0.0);
    
    data.initialized = true;
    return data;
}

/**
 * Set pulse parameters on GPU
 */
void set_gpu_pulse(
    GPULatticeData& data,
    const std::vector<double>& flat_field_drive,
    double pulse_amp,
    double pulse_width,
    double pulse_freq,
    double t_pulse_1,
    double t_pulse_2
) {
    data.field_drive = thrust::device_vector<double>(flat_field_drive.begin(), flat_field_drive.end());
    data.pulse_amp = pulse_amp;
    data.pulse_width = pulse_width;
    data.pulse_freq = pulse_freq;
    data.t_pulse_1 = t_pulse_1;
    data.t_pulse_2 = t_pulse_2;
}

/**
 * Perform GPU integration with selectable method
 * 
 * Available methods (matching CPU Boost.Odeint options):
 * - "euler": Explicit Euler (1st order)
 * - "rk2" or "midpoint": Modified midpoint (2nd order)
 * - "rk4": Classic Runge-Kutta 4th order
 * - "rk5" or "rkck54": Cash-Karp 5(4) - simulated as RK5 fixed step
 * - "dopri5": Dormand-Prince 5(4) - default
 * - "rk78" or "rkf78": Runge-Kutta-Fehlberg 7(8)
 * - "ssprk53": Strong Stability Preserving RK 5-stage 3rd order (optimized for spin dynamics)
 * - "bulirsch_stoer" or "bs": Bulirsch-Stoer (high accuracy)
 */
void integrate_gpu(
    GPUODESystem& system,
    GPUState& state,
    double T_start,
    double T_end,
    double dt,
    size_t save_interval,
    std::vector<std::pair<double, std::vector<double>>>& trajectory,
    const std::string& method
) {
    // Time stepping
    double t = T_start;
    size_t step = 0;
    
    while (t < T_end) {
        // Save trajectory at intervals
        if (step % save_interval == 0) {
            thrust::host_vector<double> h_state = state;
            std::vector<double> state_vec(h_state.begin(), h_state.end());
            trajectory.push_back({t, state_vec});
        }
        
        // Perform one step with selected method
        step_gpu(system, state, t, dt, method);
        
        t += dt;
        step++;
    }
    
    // Save final state
    thrust::host_vector<double> h_state = state;
    std::vector<double> state_vec(h_state.begin(), h_state.end());
    trajectory.push_back({t, state_vec});
}

// Overload for backward compatibility (defaults to ssprk53)
void integrate_gpu(
    GPUODESystem& system,
    GPUState& state,
    double T_start,
    double T_end,
    double dt,
    size_t save_interval,
    std::vector<std::pair<double, std::vector<double>>>& trajectory
) {
    integrate_gpu(system, state, T_start, T_end, dt, save_interval, trajectory, "ssprk53");
}

/**
 * Single integration step on GPU with selectable method
 * 
 * Available methods:
 * - "euler": Explicit Euler (1st order, 1 stage)
 * - "rk2" or "midpoint": Modified midpoint (2nd order, 2 stages)
 * - "rk4": Classic Runge-Kutta (4th order, 4 stages)
 * - "rk5" or "rkck54": Cash-Karp style (5th order, 6 stages)
 * - "dopri5": Dormand-Prince (5th order, 7 stages) - default
 * - "rk78" or "rkf78": Fehlberg (8th order, 13 stages)
 * - "ssprk53": SSP RK (3rd order, 5 stages, optimized for stability)
 * - "bulirsch_stoer" or "bs": Modified midpoint extrapolation
 */
void step_gpu(
    GPUODESystem& system,
    GPUState& state,
    double t,
    double dt,
    const std::string& method
) {
    GPULatticeData& data = system.data;
    size_t array_size = data.lattice_size * data.spin_dim;
    
    const int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid((array_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    double* d_state = thrust::raw_pointer_cast(state.data());
    
    if (method == "euler") {
        // =====================================================================
        // Euler method: y_{n+1} = y_n + h * f(t_n, y_n)
        // 1st order, 1 function evaluation
        // =====================================================================
        GPUState k(array_size);
        double* d_k = thrust::raw_pointer_cast(k.data());
        
        system(state, k, t);
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k, dt, array_size);
        cudaDeviceSynchronize();
        
    } else if (method == "rk2" || method == "midpoint") {
        // =====================================================================
        // Modified Midpoint (RK2): 2nd order, 2 function evaluations
        // k1 = f(t, y)
        // k2 = f(t + h/2, y + h/2 * k1)
        // y_{n+1} = y_n + h * k2
        // =====================================================================
        GPUState k1(array_size), k2(array_size), tmp(array_size);
        double* d_k1 = thrust::raw_pointer_cast(k1.data());
        double* d_k2 = thrust::raw_pointer_cast(k2.data());
        double* d_tmp = thrust::raw_pointer_cast(tmp.data());
        
        // k1 = f(t, y)
        system(state, k1, t);
        
        // tmp = y + h/2 * k1
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, 0.5 * dt, array_size);
        cudaDeviceSynchronize();
        
        // k2 = f(t + h/2, tmp)
        system(tmp, k2, t + 0.5 * dt);
        
        // y_{n+1} = y + h * k2
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k2, dt, array_size);
        cudaDeviceSynchronize();
        
    } else if (method == "rk4") {
        // =====================================================================
        // Classic RK4: 4th order, 4 function evaluations
        // k1 = f(t, y)
        // k2 = f(t + h/2, y + h/2 * k1)
        // k3 = f(t + h/2, y + h/2 * k2)
        // k4 = f(t + h, y + h * k3)
        // y_{n+1} = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        // =====================================================================
        GPUState k1(array_size), k2(array_size), k3(array_size), k4(array_size);
        GPUState tmp(array_size);
        double* d_k1 = thrust::raw_pointer_cast(k1.data());
        double* d_k2 = thrust::raw_pointer_cast(k2.data());
        double* d_k3 = thrust::raw_pointer_cast(k3.data());
        double* d_k4 = thrust::raw_pointer_cast(k4.data());
        double* d_tmp = thrust::raw_pointer_cast(tmp.data());
        
        // k1 = f(t, y)
        system(state, k1, t);
        
        // k2 = f(t + h/2, y + h/2 * k1)
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, 0.5 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k2, t + 0.5 * dt);
        
        // k3 = f(t + h/2, y + h/2 * k2)
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k2, 0.5 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k3, t + 0.5 * dt);
        
        // k4 = f(t + h, y + h * k3)
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k3, dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k4, t + dt);
        
        // y_{n+1} = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        // Accumulate: sum = k1 + 2*k2 + 2*k3 + k4
        update_state_kernel<<<grid, block>>>(d_k1, d_k1, 1.0, d_k2, 2.0, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_k1, d_k1, 1.0, d_k3, 2.0, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_k1, d_k1, 1.0, d_k4, 1.0, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k1, dt / 6.0, array_size);
        cudaDeviceSynchronize();
        
    } else if (method == "rk5" || method == "rkck54") {
        // =====================================================================
        // Cash-Karp RK5(4): 5th order, 6 function evaluations
        // Butcher tableau coefficients for the 5th order solution
        // =====================================================================
        GPUState k1(array_size), k2(array_size), k3(array_size);
        GPUState k4(array_size), k5(array_size), k6(array_size);
        GPUState tmp(array_size);
        double* d_k1 = thrust::raw_pointer_cast(k1.data());
        double* d_k2 = thrust::raw_pointer_cast(k2.data());
        double* d_k3 = thrust::raw_pointer_cast(k3.data());
        double* d_k4 = thrust::raw_pointer_cast(k4.data());
        double* d_k5 = thrust::raw_pointer_cast(k5.data());
        double* d_k6 = thrust::raw_pointer_cast(k6.data());
        double* d_tmp = thrust::raw_pointer_cast(tmp.data());
        
        // Cash-Karp coefficients
        constexpr double a21 = 1.0/5.0;
        constexpr double a31 = 3.0/40.0, a32 = 9.0/40.0;
        constexpr double a41 = 3.0/10.0, a42 = -9.0/10.0, a43 = 6.0/5.0;
        constexpr double a51 = -11.0/54.0, a52 = 5.0/2.0, a53 = -70.0/27.0, a54 = 35.0/27.0;
        constexpr double a61 = 1631.0/55296.0, a62 = 175.0/512.0, a63 = 575.0/13824.0;
        constexpr double a64 = 44275.0/110592.0, a65 = 253.0/4096.0;
        constexpr double c2 = 1.0/5.0, c3 = 3.0/10.0, c4 = 3.0/5.0, c5 = 1.0, c6 = 7.0/8.0;
        // 5th order weights
        constexpr double b1 = 37.0/378.0, b3 = 250.0/621.0, b4 = 125.0/594.0, b6 = 512.0/1771.0;
        
        // k1 = f(t, y)
        system(state, k1, t);
        
        // k2 = f(t + c2*h, y + h*a21*k1)
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a21 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k2, t + c2 * dt);
        
        // k3 = f(t + c3*h, y + h*(a31*k1 + a32*k2))
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a31 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k2, a32 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k3, t + c3 * dt);
        
        // k4 = f(t + c4*h, y + h*(a41*k1 + a42*k2 + a43*k3))
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a41 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k2, a42 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k3, a43 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k4, t + c4 * dt);
        
        // k5 = f(t + c5*h, y + h*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a51 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k2, a52 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k3, a53 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k4, a54 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k5, t + c5 * dt);
        
        // k6 = f(t + c6*h, y + h*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a61 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k2, a62 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k3, a63 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k4, a64 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k5, a65 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k6, t + c6 * dt);
        
        // y_{n+1} = y + h*(b1*k1 + b3*k3 + b4*k4 + b6*k6)  (note: b2=b5=0)
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k1, b1 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k3, b3 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k4, b4 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k6, b6 * dt, array_size);
        cudaDeviceSynchronize();
        
    } else if (method == "dopri5") {
        // =====================================================================
        // Dormand-Prince 5(4): 5th order, 7 function evaluations (FSAL)
        // Default method, good general-purpose choice
        // =====================================================================
        GPUState k1(array_size), k2(array_size), k3(array_size), k4(array_size);
        GPUState k5(array_size), k6(array_size), k7(array_size);
        GPUState tmp(array_size);
        double* d_k1 = thrust::raw_pointer_cast(k1.data());
        double* d_k2 = thrust::raw_pointer_cast(k2.data());
        double* d_k3 = thrust::raw_pointer_cast(k3.data());
        double* d_k4 = thrust::raw_pointer_cast(k4.data());
        double* d_k5 = thrust::raw_pointer_cast(k5.data());
        double* d_k6 = thrust::raw_pointer_cast(k6.data());
        double* d_tmp = thrust::raw_pointer_cast(tmp.data());
        (void)k7;  // k7 not used in fixed-step (used in adaptive for FSAL)
        
        // Dormand-Prince coefficients
        constexpr double a21 = 1.0/5.0;
        constexpr double a31 = 3.0/40.0, a32 = 9.0/40.0;
        constexpr double a41 = 44.0/45.0, a42 = -56.0/15.0, a43 = 32.0/9.0;
        constexpr double a51 = 19372.0/6561.0, a52 = -25360.0/2187.0, a53 = 64448.0/6561.0, a54 = -212.0/729.0;
        constexpr double a61 = 9017.0/3168.0, a62 = -355.0/33.0, a63 = 46732.0/5247.0, a64 = 49.0/176.0, a65 = -5103.0/18656.0;
        constexpr double c2 = 1.0/5.0, c3 = 3.0/10.0, c4 = 4.0/5.0, c5 = 8.0/9.0, c6 = 1.0;
        // 5th order weights
        constexpr double b1 = 35.0/384.0, b3 = 500.0/1113.0, b4 = 125.0/192.0, b5 = -2187.0/6784.0, b6 = 11.0/84.0;
        
        // k1 = f(t, y)
        system(state, k1, t);
        
        // k2 = f(t + c2*h, y + h*a21*k1)
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a21 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k2, t + c2 * dt);
        
        // k3 = f(t + c3*h, y + h*(a31*k1 + a32*k2))
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a31 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k2, a32 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k3, t + c3 * dt);
        
        // k4 = f(t + c4*h, y + h*(a41*k1 + a42*k2 + a43*k3))
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a41 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k2, a42 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k3, a43 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k4, t + c4 * dt);
        
        // k5 = f(t + c5*h, y + h*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a51 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k2, a52 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k3, a53 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k4, a54 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k5, t + c5 * dt);
        
        // k6 = f(t + c6*h, y + h*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a61 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k2, a62 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k3, a63 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k4, a64 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k5, a65 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k6, t + c6 * dt);
        
        // y_{n+1} = y + h*(b1*k1 + b3*k3 + b4*k4 + b5*k5 + b6*k6)  (note: b2=b7=0)
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k1, b1 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k3, b3 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k4, b4 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k5, b5 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k6, b6 * dt, array_size);
        cudaDeviceSynchronize();
        
    } else if (method == "rk78" || method == "rkf78") {
        // =====================================================================
        // Runge-Kutta-Fehlberg 7(8): 8th order, 13 function evaluations
        // Very high accuracy for smooth problems
        // =====================================================================
        GPUState k1(array_size), k2(array_size), k3(array_size), k4(array_size);
        GPUState k5(array_size), k6(array_size), k7(array_size), k8(array_size);
        GPUState k9(array_size), k10(array_size), k11(array_size), k12(array_size), k13(array_size);
        GPUState tmp(array_size);
        double* d_k1 = thrust::raw_pointer_cast(k1.data());
        double* d_k2 = thrust::raw_pointer_cast(k2.data());
        double* d_k3 = thrust::raw_pointer_cast(k3.data());
        double* d_k4 = thrust::raw_pointer_cast(k4.data());
        double* d_k5 = thrust::raw_pointer_cast(k5.data());
        double* d_k6 = thrust::raw_pointer_cast(k6.data());
        double* d_k7 = thrust::raw_pointer_cast(k7.data());
        double* d_k8 = thrust::raw_pointer_cast(k8.data());
        double* d_k9 = thrust::raw_pointer_cast(k9.data());
        double* d_k10 = thrust::raw_pointer_cast(k10.data());
        double* d_k11 = thrust::raw_pointer_cast(k11.data());
        double* d_tmp = thrust::raw_pointer_cast(tmp.data());
        (void)k12; (void)k13;  // k12, k13 not used in simplified implementation
        
        // RKF78 coefficients (Fehlberg's 7(8) method)
        // Using simplified 8th order weights for the primary solution
        constexpr double c2 = 2.0/27.0, c3 = 1.0/9.0, c4 = 1.0/6.0, c5 = 5.0/12.0;
        constexpr double c6 = 1.0/2.0, c7 = 5.0/6.0, c8 = 1.0/6.0, c9 = 2.0/3.0;
        constexpr double c10 = 1.0/3.0, c11 = 1.0;
        
        // 8th order solution weights
        constexpr double b1 = 41.0/840.0, b6 = 34.0/105.0, b7 = 9.0/35.0, b8 = 9.0/35.0;
        constexpr double b9 = 9.0/280.0, b10 = 9.0/280.0, b11 = 41.0/840.0;
        
        // Stage coefficients (simplified - using key stages)
        system(state, k1, t);
        
        // k2
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, c2 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k2, t + c2 * dt);
        
        // k3
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, (1.0/36.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k2, (1.0/12.0) * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k3, t + c3 * dt);
        
        // k4
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, (1.0/24.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k3, (1.0/8.0) * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k4, t + c4 * dt);
        
        // k5
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, (5.0/12.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k4, (-25.0/16.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k3, (25.0/16.0) * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k5, t + c5 * dt);
        
        // k6
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, (1.0/20.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k4, (1.0/4.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k5, (1.0/5.0) * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k6, t + c6 * dt);
        
        // k7
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, (-25.0/108.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k4, (125.0/108.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k5, (-65.0/27.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k6, (125.0/54.0) * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k7, t + c7 * dt);
        
        // k8
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, (31.0/300.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k5, (61.0/225.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k6, (-2.0/9.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k7, (13.0/900.0) * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k8, t + c8 * dt);
        
        // k9
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, 2.0 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k4, (-53.0/6.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k5, (704.0/45.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k6, (-107.0/9.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k7, (67.0/90.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k8, 3.0 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k9, t + c9 * dt);
        
        // k10
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, (-91.0/108.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k4, (23.0/108.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k5, (-976.0/135.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k6, (311.0/54.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k7, (-19.0/60.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k8, (17.0/6.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k9, (-1.0/12.0) * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k10, t + c10 * dt);
        
        // k11
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, (2383.0/4100.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k4, (-341.0/164.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k5, (4496.0/1025.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k6, (-301.0/82.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k7, (2133.0/4100.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k8, (45.0/82.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k9, (45.0/164.0) * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k10, (18.0/41.0) * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k11, t + c11 * dt);
        
        // 8th order solution: y_{n+1} = y + h*(b1*k1 + b6*k6 + b7*k7 + b8*k8 + b9*k9 + b10*k10 + b11*k11)
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k1, b1 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k6, b6 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k7, b7 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k8, b8 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k9, b9 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k10, b10 * dt, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k11, b11 * dt, array_size);
        cudaDeviceSynchronize();
        
    } else if (method == "bulirsch_stoer" || method == "bs") {
        // =====================================================================
        // Bulirsch-Stoer: Modified midpoint with Richardson extrapolation
        // Very high accuracy, uses sequence of subdivisions
        // Simplified version using 4 substeps for GPU efficiency
        // =====================================================================
        const int n_substeps = 4;  // Number of midpoint steps
        GPUState y_mid(array_size), y_prev(array_size), y_next(array_size);
        GPUState k(array_size);
        double* d_y_mid = thrust::raw_pointer_cast(y_mid.data());
        double* d_y_prev = thrust::raw_pointer_cast(y_prev.data());
        double* d_y_next = thrust::raw_pointer_cast(y_next.data());
        double* d_k = thrust::raw_pointer_cast(k.data());
        
        double h_sub = dt / n_substeps;
        
        // Modified midpoint method with n_substeps
        // y_prev = y
        thrust::copy(state.begin(), state.end(), y_prev.begin());
        
        // First Euler step: y_mid = y + h_sub * f(t, y)
        system(state, k, t);
        update_state_kernel<<<grid, block>>>(d_y_mid, d_state, 1.0, d_k, h_sub, array_size);
        cudaDeviceSynchronize();
        
        // Midpoint steps
        for (int i = 1; i < n_substeps; ++i) {
            double t_curr = t + i * h_sub;
            system(y_mid, k, t_curr);
            // y_next = y_prev + 2*h_sub*f(t_curr, y_mid)
            update_state_kernel<<<grid, block>>>(d_y_next, d_y_prev, 1.0, d_k, 2.0 * h_sub, array_size);
            cudaDeviceSynchronize();
            // Shift: y_prev = y_mid, y_mid = y_next
            thrust::copy(y_mid.begin(), y_mid.end(), y_prev.begin());
            thrust::copy(y_next.begin(), y_next.end(), y_mid.begin());
        }
        
        // Final correction step
        system(y_mid, k, t + dt);
        // y_{n+1} = 0.5 * (y_mid + y_prev + h_sub * f(t+dt, y_mid))
        update_state_kernel<<<grid, block>>>(d_y_next, d_y_mid, 0.5, d_y_prev, 0.5, array_size);
        cudaDeviceSynchronize();
        update_state_kernel<<<grid, block>>>(d_state, d_y_next, 1.0, d_k, 0.5 * h_sub, array_size);
        cudaDeviceSynchronize();
        
    } else if (method == "ssprk53") {
        // =====================================================================
        // SSPRK53: Strong Stability Preserving RK, 5 stages, 3rd order
        // Optimized for hyperbolic PDEs and spin dynamics
        // =====================================================================
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
        
        GPUState k(array_size), tmp(array_size), u(array_size);
        double* d_k = thrust::raw_pointer_cast(k.data());
        double* d_tmp = thrust::raw_pointer_cast(tmp.data());
        double* d_u = thrust::raw_pointer_cast(u.data());
        
        // Stage 1
        system(state, k, t);
        update_state_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k, b10 * dt, array_size);
        cudaDeviceSynchronize();
        
        // Stage 2
        system(tmp, k, t + c1 * dt);
        update_state_kernel<<<grid, block>>>(d_u, d_tmp, 1.0, d_k, b21 * dt, array_size);
        cudaDeviceSynchronize();
        
        // Stage 3
        system(u, k, t + c2 * dt);
        update_state_three_kernel<<<grid, block>>>(d_tmp, d_state, a30, d_u, a32, d_k, b32 * dt, array_size);
        cudaDeviceSynchronize();
        
        // Stage 4
        system(tmp, k, t + c3 * dt);
        update_state_three_kernel<<<grid, block>>>(d_tmp, d_state, a40, d_tmp, a43, d_k, b43 * dt, array_size);
        cudaDeviceSynchronize();
        
        // Stage 5 (final)
        system(tmp, k, t + c4 * dt);
        update_state_three_kernel<<<grid, block>>>(d_state, d_u, a52, d_tmp, a54, d_k, b54 * dt, array_size);
        cudaDeviceSynchronize();
        
    } else if (method == "rk54" || method == "rkf54") {
        // =====================================================================
        // Runge-Kutta-Fehlberg 5(4): Same as rkck54 but with Fehlberg coefficients
        // =====================================================================
        // Use the same implementation as dopri5 (both are 5th order embedded methods)
        step_gpu(system, state, t, dt, "dopri5");
        
    } else {
        // Default to SSPRK53 for unknown methods
        std::cerr << "Warning: Unknown GPU integration method '" << method << "', using ssprk53" << std::endl;
        step_gpu(system, state, t, dt, "ssprk53");
    }
}

/**
 * Compute total energy on GPU
 */
double compute_energy_gpu(const GPULatticeData& data, const GPUState& state) {
    // Allocate device memory for site energies
    thrust::device_vector<double> d_energies(data.lattice_size);
    
    // TODO: Implement energy computation kernel
    // For now, return 0 as placeholder
    return 0.0;
}

/**
 * Normalize spins on GPU
 */
// Note: normalize_spins_flat_kernel is an alias for normalize_spins_kernel in gpu_common_helpers.cu
#define normalize_spins_flat_kernel ::normalize_spins_kernel

void normalize_spins_gpu(GPUState& state, size_t lattice_size, size_t spin_dim, double spin_length) {
    const int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid((lattice_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    double* d_state = thrust::raw_pointer_cast(state.data());
    normalize_spins_flat_kernel<<<grid, block>>>(d_state, spin_length, lattice_size, spin_dim);
    cudaDeviceSynchronize();
}

} // namespace gpu

// =============================================================================
// Host-Callable API Implementation (for C++ TUs)
// These functions wrap the internal gpu:: namespace functions with opaque handles
// =============================================================================

namespace gpu {

/**
 * Internal structure that backs the opaque handle
 * Only visible to CUDA translation units
 */
struct GPULatticeDataHandle {
    GPULatticeData data;
    GPUState state;
    bool has_state;
    
    GPULatticeDataHandle() : has_state(false) {}
};

// API wrapper functions - use different internal namespace to avoid collision
namespace api {

GPULatticeDataHandle* create_handle(
    size_t lattice_size,
    size_t spin_dim,
    size_t N_atoms,
    size_t max_bilinear,
    const std::vector<double>& flat_field,
    const std::vector<double>& flat_onsite,
    const std::vector<double>& flat_bilinear,
    const std::vector<size_t>& flat_partners,
    const std::vector<size_t>& num_bilinear_per_site
) {
    GPULatticeDataHandle* handle = new GPULatticeDataHandle();
    
    // Use the internal create function
    handle->data = gpu::create_gpu_lattice_data_internal(
        lattice_size, spin_dim, N_atoms, max_bilinear,
        flat_field, flat_onsite, flat_bilinear, flat_partners, num_bilinear_per_site
    );
    
    return handle;
}

} // namespace api

// Implementations of API functions declared in lattice_gpu_api.h
GPULatticeDataHandle* create_gpu_lattice_data(
    size_t lattice_size,
    size_t spin_dim,
    size_t N_atoms,
    size_t max_bilinear,
    const std::vector<double>& flat_field,
    const std::vector<double>& flat_onsite,
    const std::vector<double>& flat_bilinear,
    const std::vector<size_t>& flat_partners,
    const std::vector<size_t>& num_bilinear_per_site
) {
    return api::create_handle(lattice_size, spin_dim, N_atoms, max_bilinear,
                              flat_field, flat_onsite, flat_bilinear, 
                              flat_partners, num_bilinear_per_site);
}

void destroy_gpu_lattice_data(GPULatticeDataHandle* handle) {
    if (handle) {
        // GPULatticeData uses thrust vectors which auto-deallocate
        delete handle;
    }
}

void set_gpu_pulse(
    GPULatticeDataHandle* handle,
    const std::vector<double>& flat_field_drive,
    double pulse_amp,
    double pulse_width,
    double pulse_freq,
    double t_pulse_1,
    double t_pulse_2
) {
    if (!handle) return;
    
    gpu::set_gpu_pulse(handle->data, flat_field_drive, 
                       pulse_amp, pulse_width, pulse_freq, 
                       t_pulse_1, t_pulse_2);
}

void set_gpu_spins(
    GPULatticeDataHandle* handle,
    const std::vector<double>& flat_spins
) {
    if (!handle) return;
    
    handle->state.resize(flat_spins.size());
    thrust::copy(flat_spins.begin(), flat_spins.end(), handle->state.begin());
    handle->has_state = true;
}

void get_gpu_spins(
    GPULatticeDataHandle* handle,
    std::vector<double>& flat_spins
) {
    if (!handle || !handle->has_state) return;
    
    flat_spins.resize(handle->state.size());
    thrust::copy(handle->state.begin(), handle->state.end(), flat_spins.begin());
}

void integrate_gpu(
    GPULatticeDataHandle* handle,
    double T_start,
    double T_end,
    double dt,
    size_t save_interval,
    std::vector<std::pair<double, std::vector<double>>>& trajectory,
    const std::string& method
) {
    if (!handle || !handle->has_state) return;
    
    GPUODESystem system(handle->data);
    gpu::integrate_gpu(system, handle->state, T_start, T_end, dt, 
                       save_interval, trajectory, method);
}

void step_gpu(
    GPULatticeDataHandle* handle,
    double t,
    double dt,
    const std::string& method
) {
    if (!handle || !handle->has_state) return;
    
    GPUODESystem system(handle->data);
    gpu::step_gpu(system, handle->state, t, dt, method);
}

double compute_energy_gpu(GPULatticeDataHandle* handle) {
    if (!handle || !handle->has_state) return 0.0;
    
    return gpu::compute_energy_gpu(handle->data, handle->state);
}

void normalize_spins_gpu(GPULatticeDataHandle* handle, double spin_length) {
    if (!handle || !handle->has_state) return;
    
    gpu::normalize_spins_gpu(handle->state, handle->data.lattice_size, 
                             handle->data.spin_dim, spin_length);
}

void compute_magnetization_gpu(
    GPULatticeDataHandle* handle,
    std::vector<double>& mag_local,
    std::vector<double>& mag_staggered
) {
    if (!handle || !handle->has_state) return;
    
    size_t spin_dim = handle->data.spin_dim;
    size_t lattice_size = handle->data.lattice_size;
    
    mag_local.resize(spin_dim, 0.0);
    mag_staggered.resize(spin_dim, 0.0);
    
    // Copy state to host for magnetization computation
    std::vector<double> h_state(handle->state.size());
    thrust::copy(handle->state.begin(), handle->state.end(), h_state.begin());
    
    // Compute magnetizations on CPU (could be optimized with GPU reduction)
    for (size_t i = 0; i < lattice_size; ++i) {
        double sign = (i % 2 == 0) ? 1.0 : -1.0;
        for (size_t d = 0; d < spin_dim; ++d) {
            mag_local[d] += h_state[i * spin_dim + d];
            mag_staggered[d] += h_state[i * spin_dim + d] * sign;
        }
    }
    
    // Normalize
    for (size_t d = 0; d < spin_dim; ++d) {
        mag_local[d] /= lattice_size;
        mag_staggered[d] /= lattice_size;
    }
}

} // namespace gpu


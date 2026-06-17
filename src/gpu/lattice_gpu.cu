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
 *
 * NOTE: trilinear couplings are intentionally absent here. The CPU
 *       `Lattice` supports trilinear terms, but they are not yet
 *       uploaded to or evaluated on the GPU. `Lattice::ensure_gpu_data_initialized`
 *       throws if the lattice has any trilinear partners so we never
 *       silently drop a term of the Hamiltonian. Implementing trilinear
 *       on the GPU also requires extending `create_gpu_lattice_data_internal`
 *       to upload `trilinear_*` arrays (see audit item T3).
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
    
    // NOTE: No sync here - caller is responsible for synchronization
    // This allows kernel pipelining when multiple GPU operations follow
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
    // Time stepping (delegated to the shared gpu::ode fixed-step driver).
    // The observer snapshots the device state to host at each save point,
    // reproducing the historical sampling (every save_interval steps plus the
    // final state).
    auto observe = [&](double t_obs, const gpu::ode::State& st) {
        thrust::host_vector<double> h_state = st;
        trajectory.push_back({t_obs, std::vector<double>(h_state.begin(), h_state.end())});
    };

    gpu::ode::integrate(system, state, T_start, T_end, dt, save_interval,
                        observe, method, system.data.rk_ws);
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
    // Delegate to the shared in-house gpu::ode stepper module. All Runge-Kutta
    // coefficient logic now lives in gpu/ode/integrator.cuh; this lattice path
    // only supplies the Landau-Lifshitz RHS (GPUODESystem) and the persistent
    // per-system stage workspace.
    gpu::ode::step(system, state, t, dt, method, system.data.rk_ws);
}

/**
 * Compute total energy on GPU
 *
 * Not yet implemented on the device. Rather than silently returning a
 * bogus 0.0 (which previously could be mistaken for a real energy), we
 * throw so callers fall back to the CPU energy path (total_energy_flat).
 */
double compute_energy_gpu(const GPULatticeData& /*data*/, const GPUState& /*state*/) {
    throw std::runtime_error(
        "gpu::compute_energy_gpu: GPU energy reduction is not implemented. "
        "Use the CPU total_energy_flat() helper on the downloaded state "
        "instead of relying on a GPU energy value.");
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

// =============================================================================
// Batched τ-scan kernels for 2DCS GPU parallelization
// =============================================================================

/**
 * Batched LLG kernel for τ-parallelized 2DCS.
 *
 * Runs B replicas of the same lattice simultaneously.  Each thread handles
 * one (replica, site) pair.  Lattice topology and shared pulse parameters
 * (t_pulse_1, amplitude, width, frequency, field directions) are broadcast
 * across all replicas.  Only t_pulse_2[b] differs per replica.
 *
 * State layout: d_spins[b * N * spin_dim + i * spin_dim + d]
 *   replica b, site i, component d.
 */
__global__
void LLG_flat_kernel_batched(
    double* __restrict__ d_dsdt,              // [B * N * spin_dim]
    const double* __restrict__ d_spins,       // [B * N * spin_dim]
    double* __restrict__ d_local_field,       // [B * N * spin_dim]
    // Lattice data — shared across all replicas
    const double* __restrict__ d_field,
    const double* __restrict__ d_onsite,
    const double* __restrict__ d_bilinear_vals,
    const size_t* __restrict__ d_bilinear_idx,
    const size_t* __restrict__ d_bilinear_counts,
    // Drive field — direction is shared; only t_pulse_2 varies
    const double* __restrict__ d_field_drive,
    double pulse_amp,
    double pulse_width,
    double pulse_freq,
    double t_pulse_1,
    const double* __restrict__ d_t_pulse2,    // [B] per-replica second-pulse times
    double curr_time,
    // Dimensions
    size_t N,           // sites per replica = lattice_size
    size_t spin_dim,
    size_t N_atoms,
    size_t max_bilinear,
    size_t B            // batch size = number of τ replicas
) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= B * N) return;

    size_t b    = tid / N;
    size_t site = tid % N;

    // Per-replica spin/field/dsdt base pointers
    size_t replica_offset = b * N * spin_dim;
    const double* d_spins_b   = d_spins      + replica_offset;
    double*       d_field_b   = d_local_field + replica_offset;
    double*       d_dsdt_b    = d_dsdt        + replica_offset;

    const double* spin_here = &d_spins_b[site * spin_dim];
    double*       lf        = &d_field_b[site * spin_dim];
    double*       dsdt      = &d_dsdt_b[site * spin_dim];

    // Compute H_eff = -B_ext + 2A·S + Σ J·S_j
    // Partner indices refer to sites within [0, N) — same for all replicas.
    compute_local_field_device(
        lf, d_spins_b, (int)site,
        d_field, d_onsite,
        d_bilinear_vals, d_bilinear_idx, d_bilinear_counts,
        max_bilinear, N, spin_dim
    );

    // Drive field: pulse1 is shared, pulse2 is per-replica
    ::add_drive_field_device(lf,
        d_field_drive, d_field_drive + N_atoms * spin_dim,
        site % N_atoms,
        pulse_amp, pulse_width, pulse_freq,
        t_pulse_1, d_t_pulse2[b],
        curr_time, spin_dim);

    // dS/dt = H_eff × S
    ::compute_ll_derivative_device(dsdt, spin_here, lf, spin_dim);
}

/**
 * Per-replica magnetization reduction for batched 2DCS.
 *
 * Computes M_antiferro, M_local, M_global for each replica using atomic adds.
 * Caller must zero d_mags before launching.
 *
 * d_mags layout: [b * 3 * spin_dim + m * spin_dim + d]
 *   m = 0 → M_antiferro (frame-rotated + sublattice sign)
 *   m = 1 → M_local     (raw sum, no frame)
 *   m = 2 → M_global    (frame-rotated, no sign)
 * Values are RAW sums (not divided by N); host normalizes afterward.
 */
__global__
void compute_batched_mag_kernel(
    const double* __restrict__ d_state,        // [B * N * spin_dim]
    double* __restrict__ d_mags,               // [B * 3 * spin_dim] (zeroed by caller)
    const double* __restrict__ d_afm_signs,    // [N_atoms]
    const double* __restrict__ d_frames,       // [N_atoms * spin_dim * spin_dim] row-major
    size_t B, size_t N, size_t N_atoms, size_t spin_dim
) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= B * N) return;

    size_t b    = tid / N;
    size_t site = tid % N;
    size_t atom = site % N_atoms;

    const double* s       = d_state + b * N * spin_dim + site * spin_dim;
    double*       m_anti  = d_mags  + b * 3 * spin_dim + 0 * spin_dim;
    double*       m_local = d_mags  + b * 3 * spin_dim + 1 * spin_dim;
    double*       m_glob  = d_mags  + b * 3 * spin_dim + 2 * spin_dim;

    const double* frame   = d_frames + atom * spin_dim * spin_dim;
    double        sign    = d_afm_signs[atom];

    for (size_t d = 0; d < spin_dim; ++d) {
        // M_local: plain sum
        ATOMIC_ADD_DOUBLE(&m_local[d], s[d]);

        // M_antiferro + M_global via frame transform
        double contrib = 0.0;
        for (size_t nu = 0; nu < spin_dim; ++nu) {
            contrib += frame[d * spin_dim + nu] * s[nu];
        }
        ATOMIC_ADD_DOUBLE(&m_glob[d],  contrib);
        ATOMIC_ADD_DOUBLE(&m_anti[d],  sign * contrib);
    }
}

/**
 * Internal batched integration: RK4 over B τ-replicas with on-device
 * magnetization extraction.
 *
 * Returns a BatchedMagResult (declared in lattice_gpu_api.h) using only the
 * gpu:: namespace types internally.
 */
static gpu::BatchedMagResult integrate_gpu_batched_internal(
    GPULatticeData& data,
    const std::vector<double>& flat_initial_state,
    const std::vector<double>& tau2_values,
    const std::vector<double>& afm_signs,
    const std::vector<double>& sublattice_frames,
    double T_start, double T_end, double dt,
    size_t save_interval
) {
    const size_t N        = data.lattice_size;
    const size_t spin_dim = data.spin_dim;
    const size_t N_atoms  = data.N_atoms;
    const size_t max_bi   = data.max_bilinear;
    const size_t B        = tau2_values.size();

    const size_t array_size = B * N * spin_dim;
    const size_t mag_stride = B * 3 * spin_dim;

    // ---- Allocate batched device arrays ----
    thrust::device_vector<double> d_state(array_size);
    thrust::device_vector<double> d_local_field(array_size);
    thrust::device_vector<double> d_mags(mag_stride, 0.0);  // zeroed per step
    thrust::device_vector<double> d_t_pulse2(tau2_values);
    thrust::device_vector<double> d_afm(afm_signs);
    thrust::device_vector<double> d_frames(sublattice_frames);

    // ---- Replicate initial state into all B replicas ----
    for (size_t b = 0; b < B; ++b) {
        thrust::copy(flat_initial_state.begin(), flat_initial_state.end(),
                     d_state.begin() + b * N * spin_dim);
    }

    // ---- Raw device pointers (shared lattice data) ----
    const double* d_field_ptr       = thrust::raw_pointer_cast(data.field.data());
    const double* d_onsite_ptr      = thrust::raw_pointer_cast(data.onsite.data());
    const double* d_bilinear_ptr    = thrust::raw_pointer_cast(data.bilinear_vals.data());
    const size_t* d_bidx_ptr        = thrust::raw_pointer_cast(data.bilinear_idx.data());
    const size_t* d_bicnt_ptr       = thrust::raw_pointer_cast(data.bilinear_counts.data());
    const double* d_fdrive_ptr      = thrust::raw_pointer_cast(data.field_drive.data());
    const double* d_t2_ptr          = thrust::raw_pointer_cast(d_t_pulse2.data());
    const double* d_afm_ptr         = thrust::raw_pointer_cast(d_afm.data());
    const double* d_frames_ptr      = thrust::raw_pointer_cast(d_frames.data());

    double* d_state_ptr      = thrust::raw_pointer_cast(d_state.data());
    double* d_lf_ptr         = thrust::raw_pointer_cast(d_local_field.data());
    double* d_mags_ptr       = thrust::raw_pointer_cast(d_mags.data());

    const int BLOCK = 256;
    dim3 block_llg(BLOCK);
    dim3 grid_llg((B * N + BLOCK - 1) / BLOCK);

    // Helper: launch batched LLG kernel to fill d_dsdt at time t
    auto batched_rhs = [&](const double* d_in, double* d_dsdt, double t) {
        LLG_flat_kernel_batched<<<grid_llg, block_llg>>>(
            d_dsdt, d_in, d_lf_ptr,
            d_field_ptr, d_onsite_ptr,
            d_bilinear_ptr, d_bidx_ptr, d_bicnt_ptr,
            d_fdrive_ptr,
            data.pulse_amp, data.pulse_width, data.pulse_freq,
            data.t_pulse_1, d_t2_ptr,
            t, N, spin_dim, N_atoms, max_bi, B);
        cudaDeviceSynchronize();
    };

    // Adapt the batched RHS to the gpu::ode System concept so it can drive the
    // shared stepper. The flat state array spans all B replicas (B*N*spin_dim);
    // the elementwise RK update kernels inside the stepper are oblivious to the
    // batching, so the single-trajectory stepper applies unchanged.
    auto batched_system = [&](const gpu::ode::State& in, gpu::ode::State& out, double tt) {
        batched_rhs(thrust::raw_pointer_cast(in.data()),
                    thrust::raw_pointer_cast(out.data()), tt);
    };
    gpu::ode::Workspace ws;
    ws.ensure(array_size, 4);

    // Helper: extract magnetizations, append to result
    gpu::BatchedMagResult result;
    result.B             = B;
    result.spin_dim      = spin_dim;
    result.n_time_points = 0;

    auto snapshot = [&](double t) {
        thrust::fill(d_mags.begin(), d_mags.end(), 0.0);
        compute_batched_mag_kernel<<<grid_llg, block_llg>>>(
            d_state_ptr, d_mags_ptr,
            d_afm_ptr, d_frames_ptr,
            B, N, N_atoms, spin_dim);
        cudaDeviceSynchronize();

        // Download B * 3 * spin_dim doubles
        thrust::host_vector<double> h_mags = d_mags;

        result.times.push_back(t);
        result.mag_data.insert(result.mag_data.end(),
                               h_mags.begin(), h_mags.end());
        ++result.n_time_points;
    };

    // ---- Time loop ----
    double t    = T_start;
    size_t step = 0;

    while (t < T_end - 1e-12) {
        if (step % save_interval == 0) snapshot(t);

        // One RK4 step over all B replicas via the shared gpu::ode stepper.
        gpu::ode::step(batched_system, d_state, t, dt, "rk4", ws);

        t += dt;
        ++step;
    }

    // Final snapshot
    snapshot(t);

    // Normalize all magnetizations by N
    const double inv_N = 1.0 / static_cast<double>(N);
    for (auto& v : result.mag_data) v *= inv_N;

    GPU_CHECK_KERNEL();

    return result;
}

} // namespace gpu  (internal)

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

/**
 * Batched τ-scan API wrapper: runs B replicas simultaneously on GPU.
 * See lattice_gpu_api.h for full documentation.
 */
BatchedMagResult integrate_gpu_batched(
    GPULatticeDataHandle* handle,
    const std::vector<double>& flat_initial_state,
    const std::vector<double>& tau2_values,
    const std::vector<double>& afm_signs,
    const std::vector<double>& sublattice_frames,
    double T_start, double T_end, double dt,
    size_t save_interval,
    const std::string& method
) {
    if (!handle) return BatchedMagResult{};
    if (tau2_values.empty()) return BatchedMagResult{};

    // Only RK4 is supported for batched mode (validated to be bit-identical with CPU).
    // For other methods, warn and fall back to rk4.
    if (method != "rk4") {
        std::cerr << "[integrate_gpu_batched] method='" << method
                  << "' is not supported in batched mode; using rk4." << std::endl;
    }

    return integrate_gpu_batched_internal(
        handle->data,
        flat_initial_state, tau2_values,
        afm_signs, sublattice_frames,
        T_start, T_end, dt, save_interval);
}

} // namespace gpu


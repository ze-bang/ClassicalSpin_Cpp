#include "classical_spin/gpu/mixed_lattice_gpu.cuh"
#include "classical_spin/gpu/mixed_lattice_gpu_api.h"
#include "classical_spin/gpu/gpu_common_helpers.cuh"

namespace mixed_gpu {

// Common device functions and kernels are now in gpu_common_helpers.cu/cuh
// ======================= Device Helper Functions (Mixed Lattice Specific) =======================

// Note: Common device functions (dot_device, contract_device, multiply_matrix_vector_device,
// cross_product_SU2_device, cross_product_SU3_device, init_local_field_device, 
// add_onsite_contribution_device, add_bilinear_contribution_device, add_drive_field_device,
// compute_ll_derivative_device) are now provided by lattice_gpu.cuh

/**
 * Mixed bilinear contraction for SU(2)-SU(3) coupling
 * result_i = sum_j J_{ij} * spin3_j  (result has spin_dim_SU2 components)
 */
__device__
void multiply_mixed_matrix_vector_device(double* result, const double* matrix, 
                                          const double* vector, 
                                          size_t n_out, size_t n_in) {
    // matrix is n_out x n_in
    for (size_t i = 0; i < n_out; ++i) {
        result[i] = 0.0;
        for (size_t j = 0; j < n_in; ++j) {
            result[i] += matrix[i * n_in + j] * vector[j];
        }
    }
}

/**
 * Mixed bilinear contraction with transpose for SU(3)-SU(2) coupling
 * result_j = sum_i J_{ij}^T * spin2_i = sum_i J_{ji} * spin2_i
 * J is stored as n_rows x n_cols, so transpose gives n_cols x n_rows
 */
__device__
void multiply_mixed_matrix_transpose_vector_device(double* result, const double* matrix, 
                                                    const double* vector, 
                                                    size_t n_rows, size_t n_cols) {
    // matrix is n_rows x n_cols (stored row-major)
    // result has n_cols components
    for (size_t j = 0; j < n_cols; ++j) {
        result[j] = 0.0;
        for (size_t i = 0; i < n_rows; ++i) {
            result[j] += matrix[i * n_cols + j] * vector[i];
        }
    }
}



/**
 * Compute local field for SU(2) site in mixed lattice (unified function)
 * Includes SU(2)-SU(2) bilinear and SU(2)-SU(3) mixed bilinear interactions
 */
__device__
void compute_local_field_SU2_device(
    double* local_field,
    const double* d_spins_SU2,
    const double* d_spins_SU3,
    int site,
    const double* field,
    const double* onsite_interaction,
    const double* bilinear_interaction,
    const size_t* bilinear_partners,
    const size_t* bilinear_counts,
    const double* mixed_bilinear_interaction,
    const size_t* mixed_bilinear_partners_SU3,
    const size_t* mixed_bilinear_counts_SU2,
    size_t max_bilinear,
    size_t max_mixed_bilinear,
    size_t lattice_size_SU2,
    size_t lattice_size_SU3,
    size_t spin_dim_SU2,
    size_t spin_dim_SU3
) {
    const double* spin_here = &d_spins_SU2[site * spin_dim_SU2];
    double temp[8];
    
    // Initialize: H = -B
    ::init_local_field_device(local_field, &field[site * spin_dim_SU2], spin_dim_SU2);
    
    // On-site: H += 2*A*S (factor 2 from derivative of quadratic term)
    ::add_onsite_contribution_device(local_field, &onsite_interaction[site * spin_dim_SU2 * spin_dim_SU2],
                                     spin_here, temp, spin_dim_SU2, 2.0);
    
    // Bilinear SU(2)-SU(2): H += sum_j J_ij * S_j
    size_t num_neighbors = bilinear_counts[site];
    for (size_t n = 0; n < num_neighbors && n < max_bilinear; ++n) {
        size_t partner = bilinear_partners[site * max_bilinear + n];
        if (partner < lattice_size_SU2) {
            ::add_bilinear_contribution_device(local_field,
                &bilinear_interaction[(site * max_bilinear + n) * spin_dim_SU2 * spin_dim_SU2],
                &d_spins_SU2[partner * spin_dim_SU2], temp, spin_dim_SU2);
        }
    }
    
    // Mixed SU(2)-SU(3): H += sum_j J_mixed * S3_j
    size_t num_mixed = mixed_bilinear_counts_SU2[site];
    for (size_t n = 0; n < num_mixed && n < max_mixed_bilinear; ++n) {
        size_t partner_SU3 = mixed_bilinear_partners_SU3[site * max_mixed_bilinear + n];
        if (partner_SU3 < lattice_size_SU3) {
            const double* partner_spin = &d_spins_SU3[partner_SU3 * spin_dim_SU3];
            const double* J_mixed = &mixed_bilinear_interaction[
                (site * max_mixed_bilinear + n) * spin_dim_SU2 * spin_dim_SU3];
            
            multiply_mixed_matrix_vector_device(temp, J_mixed, partner_spin, spin_dim_SU2, spin_dim_SU3);
            for (size_t i = 0; i < spin_dim_SU2; ++i) {
                local_field[i] += temp[i];
            }
        }
    }
}

/**
 * Compute local field for SU(3) site in mixed lattice (unified function)
 * Includes SU(3)-SU(3) bilinear and SU(3)-SU(2) mixed bilinear interactions
 */
__device__
void compute_local_field_SU3_device(
    double* local_field,
    const double* d_spins_SU2,
    const double* d_spins_SU3,
    int site,
    const double* field,
    const double* onsite_interaction,
    const double* bilinear_interaction,
    const size_t* bilinear_partners,
    const size_t* bilinear_counts,
    const double* mixed_bilinear_interaction,
    const size_t* mixed_bilinear_partners_SU2,
    const size_t* mixed_bilinear_counts_SU3,
    size_t max_bilinear,
    size_t max_mixed_bilinear,
    size_t lattice_size_SU2,
    size_t lattice_size_SU3,
    size_t spin_dim_SU2,
    size_t spin_dim_SU3
) {
    const double* spin_here = &d_spins_SU3[site * spin_dim_SU3];
    double temp[8];
    
    // Initialize: H = -B
    ::init_local_field_device(local_field, &field[site * spin_dim_SU3], spin_dim_SU3);
    
    // On-site: H += 2*A*S (factor 2 from derivative of quadratic term)
    ::add_onsite_contribution_device(local_field, &onsite_interaction[site * spin_dim_SU3 * spin_dim_SU3],
                                     spin_here, temp, spin_dim_SU3, 2.0);
    
    // Bilinear SU(3)-SU(3): H += sum_j J_ij * S_j
    size_t num_neighbors = bilinear_counts[site];
    for (size_t n = 0; n < num_neighbors && n < max_bilinear; ++n) {
        size_t partner = bilinear_partners[site * max_bilinear + n];
        if (partner < lattice_size_SU3) {
            ::add_bilinear_contribution_device(local_field,
                &bilinear_interaction[(site * max_bilinear + n) * spin_dim_SU3 * spin_dim_SU3],
                &d_spins_SU3[partner * spin_dim_SU3], temp, spin_dim_SU3);
        }
    }
    
    // Mixed SU(3)-SU(2): H += sum_j J_mixed^T * S2_j
    size_t num_mixed = mixed_bilinear_counts_SU3[site];
    for (size_t n = 0; n < num_mixed && n < max_mixed_bilinear; ++n) {
        size_t partner_SU2 = mixed_bilinear_partners_SU2[site * max_mixed_bilinear + n];
        if (partner_SU2 < lattice_size_SU2) {
            const double* partner_spin = &d_spins_SU2[partner_SU2 * spin_dim_SU2];
            const double* J_mixed = &mixed_bilinear_interaction[
                (site * max_mixed_bilinear + n) * spin_dim_SU2 * spin_dim_SU3];
            
            multiply_mixed_matrix_transpose_vector_device(temp, J_mixed, partner_spin, spin_dim_SU2, spin_dim_SU3);
            for (size_t j = 0; j < spin_dim_SU3; ++j) {
                local_field[j] += temp[j];
            }
        }
    }
}

// ======================= Kernel Implementations =======================

__global__
void LLG_SU2_kernel(
    double* d_dsdt,
    const double* d_spins_SU2,
    const double* d_spins_SU3,
    double* d_local_field,
    InteractionDataSU2GPU interactions_SU2,
    MixedInteractionDataGPU mixed_interactions,
    FieldDriveParamsSU2GPU field_drive,
    TimeStepParamsMixedGPU time_params,
    LatticeDimsMixedGPU dims,
    NeighborCountsMixedGPU neighbors
) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= dims.lattice_size_SU2) return;
    
    const double* spin_here = &d_spins_SU2[site * dims.spin_dim_SU2];
    double* local_field = &d_local_field[site * dims.spin_dim_SU2];
    double* dsdt = &d_dsdt[site * dims.spin_dim_SU2];
    
    // Compute local field using unified device function
    compute_local_field_SU2_device(
        local_field, d_spins_SU2, d_spins_SU3, site,
        interactions_SU2.field,
        interactions_SU2.onsite_interaction,
        interactions_SU2.bilinear_interaction,
        interactions_SU2.bilinear_partners,
        interactions_SU2.bilinear_counts,
        mixed_interactions.bilinear_interaction,
        mixed_interactions.bilinear_partners_SU3,
        mixed_interactions.bilinear_counts_SU2,
        neighbors.max_bilinear_SU2, neighbors.max_mixed_bilinear,
        dims.lattice_size_SU2, dims.lattice_size_SU3,
        dims.spin_dim_SU2, dims.spin_dim_SU3
    );
    
    // Add drive field
    ::add_drive_field_device(local_field, field_drive.field_drive_1, field_drive.field_drive_2,
                           site % dims.N_atoms_SU2, field_drive.amplitude, field_drive.width,
                           field_drive.frequency, field_drive.t_pulse_1, field_drive.t_pulse_2,
                           time_params.curr_time, dims.spin_dim_SU2);
    
    // Compute Landau-Lifshitz derivative
    ::compute_ll_derivative_device(dsdt, spin_here, local_field, dims.spin_dim_SU2);
}

__global__
void LLG_SU3_kernel(
    double* d_dsdt,
    const double* d_spins_SU2,
    const double* d_spins_SU3,
    double* d_local_field,
    InteractionDataSU3GPU interactions_SU3,
    MixedInteractionDataGPU mixed_interactions,
    FieldDriveParamsSU3GPU field_drive,
    TimeStepParamsMixedGPU time_params,
    LatticeDimsMixedGPU dims,
    NeighborCountsMixedGPU neighbors
) {
    int site = blockIdx.x * blockDim.x + threadIdx.x;
    if (site >= dims.lattice_size_SU3) return;
    
    const double* spin_here = &d_spins_SU3[site * dims.spin_dim_SU3];
    double* local_field = &d_local_field[site * dims.spin_dim_SU3];
    double* dsdt = &d_dsdt[dims.SU3_offset + site * dims.spin_dim_SU3];
    
    // Compute local field using unified device function
    compute_local_field_SU3_device(
        local_field, d_spins_SU2, d_spins_SU3, site,
        interactions_SU3.field,
        interactions_SU3.onsite_interaction,
        interactions_SU3.bilinear_interaction,
        interactions_SU3.bilinear_partners,
        interactions_SU3.bilinear_counts,
        mixed_interactions.bilinear_interaction,
        mixed_interactions.bilinear_partners_SU2,
        mixed_interactions.bilinear_counts_SU3,
        neighbors.max_bilinear_SU3, neighbors.max_mixed_bilinear,
        dims.lattice_size_SU2, dims.lattice_size_SU3,
        dims.spin_dim_SU2, dims.spin_dim_SU3
    );
    
    // Add drive field
    ::add_drive_field_device(local_field, field_drive.field_drive_1, field_drive.field_drive_2,
                           site % dims.N_atoms_SU3, field_drive.amplitude, field_drive.width,
                           field_drive.frequency, field_drive.t_pulse_1, field_drive.t_pulse_2,
                           time_params.curr_time, dims.spin_dim_SU3);
    
    // Compute Landau-Lifshitz derivative
    ::compute_ll_derivative_device(dsdt, spin_here, local_field, dims.spin_dim_SU3);
}

// ======================= GPUMixedODESystem Implementation =======================

void GPUMixedODESystem::operator()(const GPUState& x, GPUState& dxdt, double t) const {
    const int BLOCK_SIZE = 256;
    
    // Get pointers
    const double* d_x = thrust::raw_pointer_cast(x.data());
    double* d_dxdt = thrust::raw_pointer_cast(dxdt.data());
    
    // SU(2) spins are at the beginning
    const double* d_spins_SU2 = d_x;
    // SU(3) spins start at offset
    const double* d_spins_SU3 = d_x + data.SU3_offset();
    
    // Working arrays for local fields (separate for SU2 and SU3)
    double* d_local_field_SU2 = thrust::raw_pointer_cast(data.local_field.data());
    double* d_local_field_SU3 = d_local_field_SU2 + data.lattice_size_SU2 * data.spin_dim_SU2;
    
    // Build shared structs using helper methods
    LatticeDimsMixedGPU dims = data.get_dims();
    NeighborCountsMixedGPU neighbors = data.get_neighbor_counts();
    TimeStepParamsMixedGPU time_params = data.get_time_params(t);
    MixedInteractionDataGPU mixed_interactions = data.get_mixed_interactions();
    
    // Launch SU(2) kernel
    {
        dim3 block(BLOCK_SIZE);
        dim3 grid((data.lattice_size_SU2 + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        InteractionDataSU2GPU interactions_SU2 = data.get_interactions_SU2();
        FieldDriveParamsSU2GPU field_drive_SU2 = data.get_field_drive_SU2();
        
        LLG_SU2_kernel<<<grid, block>>>(
            d_dxdt,
            d_spins_SU2,
            d_spins_SU3,
            d_local_field_SU2,
            interactions_SU2,
            mixed_interactions,
            field_drive_SU2,
            time_params,
            dims,
            neighbors
        );
    }
    
    // Launch SU(3) kernel
    {
        dim3 block(BLOCK_SIZE);
        dim3 grid((data.lattice_size_SU3 + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        InteractionDataSU3GPU interactions_SU3 = data.get_interactions_SU3();
        FieldDriveParamsSU3GPU field_drive_SU3 = data.get_field_drive_SU3();
        
        LLG_SU3_kernel<<<grid, block>>>(
            d_dxdt,
            d_spins_SU2,
            d_spins_SU3,
            d_local_field_SU3,
            interactions_SU3,
            mixed_interactions,
            field_drive_SU3,
            time_params,
            dims,
            neighbors
        );
    }
    
    cudaDeviceSynchronize();
}

// ======================= Host Integration Functions =======================

GPUMixedLatticeData create_gpu_mixed_lattice_data_internal(
    size_t lattice_size_SU2, size_t spin_dim_SU2, size_t N_atoms_SU2,
    size_t lattice_size_SU3, size_t spin_dim_SU3, size_t N_atoms_SU3,
    size_t max_bilinear_SU2, size_t max_bilinear_SU3, size_t max_mixed_bilinear,
    const std::vector<double>& flat_field_SU2,
    const std::vector<double>& flat_onsite_SU2,
    const std::vector<double>& flat_bilinear_SU2,
    const std::vector<size_t>& flat_partners_SU2,
    const std::vector<size_t>& num_bilinear_per_site_SU2,
    const std::vector<double>& flat_field_SU3,
    const std::vector<double>& flat_onsite_SU3,
    const std::vector<double>& flat_bilinear_SU3,
    const std::vector<size_t>& flat_partners_SU3,
    const std::vector<size_t>& num_bilinear_per_site_SU3,
    const std::vector<double>& flat_mixed_bilinear,
    const std::vector<size_t>& flat_mixed_partners_SU2,
    const std::vector<size_t>& flat_mixed_partners_SU3,
    const std::vector<size_t>& num_mixed_per_site_SU2
) {
    GPUMixedLatticeData data;
    
    // Set dimensions
    data.lattice_size_SU2 = lattice_size_SU2;
    data.lattice_size_SU3 = lattice_size_SU3;
    data.spin_dim_SU2 = spin_dim_SU2;
    data.spin_dim_SU3 = spin_dim_SU3;
    data.N_atoms_SU2 = N_atoms_SU2;
    data.N_atoms_SU3 = N_atoms_SU3;
    data.max_bilinear_SU2 = max_bilinear_SU2;
    data.max_bilinear_SU3 = max_bilinear_SU3;
    data.max_mixed_bilinear = max_mixed_bilinear;
    
    size_t total_size = data.state_size();
    
    // Copy SU(2) data to device
    data.field_SU2 = thrust::device_vector<double>(flat_field_SU2.begin(), flat_field_SU2.end());
    data.onsite_SU2 = thrust::device_vector<double>(flat_onsite_SU2.begin(), flat_onsite_SU2.end());
    data.bilinear_vals_SU2 = thrust::device_vector<double>(flat_bilinear_SU2.begin(), flat_bilinear_SU2.end());
    data.bilinear_idx_SU2 = thrust::device_vector<size_t>(flat_partners_SU2.begin(), flat_partners_SU2.end());
    data.bilinear_counts_SU2 = thrust::device_vector<size_t>(num_bilinear_per_site_SU2.begin(), num_bilinear_per_site_SU2.end());
    
    // Copy SU(3) data to device
    data.field_SU3 = thrust::device_vector<double>(flat_field_SU3.begin(), flat_field_SU3.end());
    data.onsite_SU3 = thrust::device_vector<double>(flat_onsite_SU3.begin(), flat_onsite_SU3.end());
    data.bilinear_vals_SU3 = thrust::device_vector<double>(flat_bilinear_SU3.begin(), flat_bilinear_SU3.end());
    data.bilinear_idx_SU3 = thrust::device_vector<size_t>(flat_partners_SU3.begin(), flat_partners_SU3.end());
    data.bilinear_counts_SU3 = thrust::device_vector<size_t>(num_bilinear_per_site_SU3.begin(), num_bilinear_per_site_SU3.end());
    
    // Copy mixed interaction data
    data.mixed_bilinear_vals = thrust::device_vector<double>(flat_mixed_bilinear.begin(), flat_mixed_bilinear.end());
    data.mixed_bilinear_idx_SU2 = thrust::device_vector<size_t>(flat_mixed_partners_SU2.begin(), flat_mixed_partners_SU2.end());
    data.mixed_bilinear_idx_SU3 = thrust::device_vector<size_t>(flat_mixed_partners_SU3.begin(), flat_mixed_partners_SU3.end());
    data.mixed_bilinear_counts_SU2 = thrust::device_vector<size_t>(num_mixed_per_site_SU2.begin(), num_mixed_per_site_SU2.end());
    
    // Initialize working arrays
    data.work_1.resize(total_size, 0.0);
    data.work_2.resize(total_size, 0.0);
    data.work_3.resize(total_size, 0.0);
    data.local_field.resize(total_size, 0.0);
    
    // Initialize field drive to zeros
    data.field_drive_SU2.resize(2 * N_atoms_SU2 * spin_dim_SU2, 0.0);
    data.field_drive_SU3.resize(2 * N_atoms_SU3 * spin_dim_SU3, 0.0);
    
    data.initialized = true;
    return data;
}

void set_gpu_pulse_SU2(
    GPUMixedLatticeData& data,
    const std::vector<double>& flat_field_drive,
    double pulse_amp, double pulse_width, double pulse_freq,
    double t_pulse_1, double t_pulse_2
) {
    data.field_drive_SU2 = thrust::device_vector<double>(flat_field_drive.begin(), flat_field_drive.end());
    data.pulse_amp_SU2 = pulse_amp;
    data.pulse_width_SU2 = pulse_width;
    data.pulse_freq_SU2 = pulse_freq;
    data.t_pulse_1_SU2 = t_pulse_1;
    data.t_pulse_2_SU2 = t_pulse_2;
}

void set_gpu_pulse_SU3(
    GPUMixedLatticeData& data,
    const std::vector<double>& flat_field_drive,
    double pulse_amp, double pulse_width, double pulse_freq,
    double t_pulse_1, double t_pulse_2
) {
    data.field_drive_SU3 = thrust::device_vector<double>(flat_field_drive.begin(), flat_field_drive.end());
    data.pulse_amp_SU3 = pulse_amp;
    data.pulse_width_SU3 = pulse_width;
    data.pulse_freq_SU3 = pulse_freq;
    data.t_pulse_1_SU3 = t_pulse_1;
    data.t_pulse_2_SU3 = t_pulse_2;
}

void step_mixed_gpu(
    GPUMixedODESystem& system,
    GPUState& state,
    double t,
    double dt,
    const std::string& method
) {
    GPUMixedLatticeData& data = system.data;
    size_t array_size = data.state_size();
    
    const int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid((array_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    double* d_state = thrust::raw_pointer_cast(state.data());
    
    if (method == "euler") {
        // Euler: y_{n+1} = y_n + h * f(t_n, y_n)
        GPUState k(array_size);
        double* d_k = thrust::raw_pointer_cast(k.data());
        
        system(state, k, t);
        ::update_arrays_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k, dt, array_size);
        cudaDeviceSynchronize();
        
    } else if (method == "rk2" || method == "midpoint") {
        // RK2: midpoint method
        GPUState k1(array_size), k2(array_size), tmp(array_size);
        double* d_k1 = thrust::raw_pointer_cast(k1.data());
        double* d_k2 = thrust::raw_pointer_cast(k2.data());
        double* d_tmp = thrust::raw_pointer_cast(tmp.data());
        
        system(state, k1, t);
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, 0.5 * dt, array_size);
        cudaDeviceSynchronize();
        
        system(tmp, k2, t + 0.5 * dt);
        ::update_arrays_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k2, dt, array_size);
        cudaDeviceSynchronize();
        
    } else if (method == "rk4") {
        // Classic RK4
        GPUState k1(array_size), k2(array_size), k3(array_size), k4(array_size);
        GPUState tmp(array_size);
        double* d_k1 = thrust::raw_pointer_cast(k1.data());
        double* d_k2 = thrust::raw_pointer_cast(k2.data());
        double* d_k3 = thrust::raw_pointer_cast(k3.data());
        double* d_k4 = thrust::raw_pointer_cast(k4.data());
        double* d_tmp = thrust::raw_pointer_cast(tmp.data());
        
        system(state, k1, t);
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, 0.5 * dt, array_size);
        cudaDeviceSynchronize();
        
        system(tmp, k2, t + 0.5 * dt);
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k2, 0.5 * dt, array_size);
        cudaDeviceSynchronize();
        
        system(tmp, k3, t + 0.5 * dt);
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k3, dt, array_size);
        cudaDeviceSynchronize();
        
        system(tmp, k4, t + dt);
        
        // Combine: y_{n+1} = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        ::update_arrays_kernel<<<grid, block>>>(d_k1, d_k1, 1.0, d_k2, 2.0, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_k1, d_k1, 1.0, d_k3, 2.0, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_k1, d_k1, 1.0, d_k4, 1.0, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k1, dt / 6.0, array_size);
        cudaDeviceSynchronize();
        
    } else if (method == "dopri5") {
        // Dormand-Prince 5(4)
        GPUState k1(array_size), k2(array_size), k3(array_size), k4(array_size);
        GPUState k5(array_size), k6(array_size);
        GPUState tmp(array_size);
        double* d_k1 = thrust::raw_pointer_cast(k1.data());
        double* d_k2 = thrust::raw_pointer_cast(k2.data());
        double* d_k3 = thrust::raw_pointer_cast(k3.data());
        double* d_k4 = thrust::raw_pointer_cast(k4.data());
        double* d_k5 = thrust::raw_pointer_cast(k5.data());
        double* d_k6 = thrust::raw_pointer_cast(k6.data());
        double* d_tmp = thrust::raw_pointer_cast(tmp.data());
        
        // Dormand-Prince coefficients
        constexpr double a21 = 1.0/5.0;
        constexpr double a31 = 3.0/40.0, a32 = 9.0/40.0;
        constexpr double a41 = 44.0/45.0, a42 = -56.0/15.0, a43 = 32.0/9.0;
        constexpr double a51 = 19372.0/6561.0, a52 = -25360.0/2187.0, a53 = 64448.0/6561.0, a54 = -212.0/729.0;
        constexpr double a61 = 9017.0/3168.0, a62 = -355.0/33.0, a63 = 46732.0/5247.0, a64 = 49.0/176.0, a65 = -5103.0/18656.0;
        constexpr double c2 = 1.0/5.0, c3 = 3.0/10.0, c4 = 4.0/5.0, c5 = 8.0/9.0, c6 = 1.0;
        constexpr double b1 = 35.0/384.0, b3 = 500.0/1113.0, b4 = 125.0/192.0, b5 = -2187.0/6784.0, b6 = 11.0/84.0;
        
        system(state, k1, t);
        
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a21 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k2, t + c2 * dt);
        
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a31 * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k2, a32 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k3, t + c3 * dt);
        
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a41 * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k2, a42 * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k3, a43 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k4, t + c4 * dt);
        
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a51 * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k2, a52 * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k3, a53 * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k4, a54 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k5, t + c5 * dt);
        
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k1, a61 * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k2, a62 * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k3, a63 * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k4, a64 * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_tmp, 1.0, d_k5, a65 * dt, array_size);
        cudaDeviceSynchronize();
        system(tmp, k6, t + c6 * dt);
        
        // Final combination
        ::update_arrays_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k1, b1 * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k3, b3 * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k4, b4 * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k5, b5 * dt, array_size);
        cudaDeviceSynchronize();
        ::update_arrays_kernel<<<grid, block>>>(d_state, d_state, 1.0, d_k6, b6 * dt, array_size);
        cudaDeviceSynchronize();
        
    } else if (method == "ssprk53") {
        // SSPRK53: Strong Stability Preserving RK, 5 stages, 3rd order
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
        ::update_arrays_kernel<<<grid, block>>>(d_tmp, d_state, 1.0, d_k, b10 * dt, array_size);
        cudaDeviceSynchronize();
        
        // Stage 2
        system(tmp, k, t + c1 * dt);
        ::update_arrays_kernel<<<grid, block>>>(d_u, d_tmp, 1.0, d_k, b21 * dt, array_size);
        cudaDeviceSynchronize();
        
        // Stage 3
        system(u, k, t + c2 * dt);
        ::update_arrays_three_kernel<<<grid, block>>>(d_tmp, d_state, a30, d_u, a32, d_k, b32 * dt, array_size);
        cudaDeviceSynchronize();
        
        // Stage 4
        system(tmp, k, t + c3 * dt);
        ::update_arrays_three_kernel<<<grid, block>>>(d_tmp, d_state, a40, d_tmp, a43, d_k, b43 * dt, array_size);
        cudaDeviceSynchronize();
        
        // Stage 5 (final)
        system(tmp, k, t + c4 * dt);
        ::update_arrays_three_kernel<<<grid, block>>>(d_state, d_u, a52, d_tmp, a54, d_k, b54 * dt, array_size);
        cudaDeviceSynchronize();
        
    } else {
        // Default to SSPRK53
        std::cerr << "Warning: Unknown GPU integration method '" << method << "', using ssprk53" << std::endl;
        step_mixed_gpu(system, state, t, dt, "ssprk53");
    }
}

void integrate_mixed_gpu(
    GPUMixedODESystem& system,
    GPUState& state,
    double T_start,
    double T_end,
    double dt,
    size_t save_interval,
    std::vector<std::pair<double, std::vector<double>>>& trajectory,
    const std::string& method
) {
    double t = T_start;
    size_t step = 0;
    
    while (t < T_end) {
        // Save trajectory at intervals
        if (step % save_interval == 0) {
            thrust::host_vector<double> h_state = state;
            std::vector<double> state_vec(h_state.begin(), h_state.end());
            trajectory.push_back({t, state_vec});
        }
        
        // Perform one step
        step_mixed_gpu(system, state, t, dt, method);
        
        t += dt;
        step++;
    }
    
    // Save final state
    thrust::host_vector<double> h_state = state;
    std::vector<double> state_vec(h_state.begin(), h_state.end());
    trajectory.push_back({t, state_vec});
}

void compute_magnetization_mixed_gpu(
    const GPUMixedLatticeData& data,
    const GPUState& state,
    double* h_mag_SU2,
    double* h_mag_staggered_SU2,
    double* h_mag_SU3,
    double* h_mag_staggered_SU3
) {
    thrust::device_vector<double> d_mag_local_SU2(data.spin_dim_SU2);
    thrust::device_vector<double> d_mag_staggered_SU2(data.spin_dim_SU2);
    thrust::device_vector<double> d_mag_local_SU3(data.spin_dim_SU3);
    thrust::device_vector<double> d_mag_staggered_SU3(data.spin_dim_SU3);
    
    const double* d_state = thrust::raw_pointer_cast(state.data());
    
    // Compute SU(2) magnetization
    const int threads_per_block = 256;
    ::compute_magnetization_kernel<<<data.spin_dim_SU2, threads_per_block>>>(
        d_state,
        thrust::raw_pointer_cast(d_mag_local_SU2.data()),
        thrust::raw_pointer_cast(d_mag_staggered_SU2.data()),
        data.lattice_size_SU2,
        data.spin_dim_SU2,
        data.N_atoms_SU2
    );
    
    // Compute SU(3) magnetization
    ::compute_magnetization_kernel<<<data.spin_dim_SU3, threads_per_block>>>(
        d_state + data.SU3_offset(),
        thrust::raw_pointer_cast(d_mag_local_SU3.data()),
        thrust::raw_pointer_cast(d_mag_staggered_SU3.data()),
        data.lattice_size_SU3,
        data.spin_dim_SU3,
        data.N_atoms_SU3
    );
    
    cudaDeviceSynchronize();
    
    // Copy results back to host
    thrust::copy(d_mag_local_SU2.begin(), d_mag_local_SU2.end(), h_mag_SU2);
    thrust::copy(d_mag_staggered_SU2.begin(), d_mag_staggered_SU2.end(), h_mag_staggered_SU2);
    thrust::copy(d_mag_local_SU3.begin(), d_mag_local_SU3.end(), h_mag_SU3);
    thrust::copy(d_mag_staggered_SU3.begin(), d_mag_staggered_SU3.end(), h_mag_staggered_SU3);
}

void normalize_spins_mixed_gpu(
    GPUState& state,
    size_t lattice_size_SU2, size_t spin_dim_SU2, double spin_length_SU2,
    size_t lattice_size_SU3, size_t spin_dim_SU3, double spin_length_SU3
) {
    const int BLOCK_SIZE = 256;
    double* d_state = thrust::raw_pointer_cast(state.data());
    
    // Normalize SU(2) spins
    {
        dim3 block(BLOCK_SIZE);
        dim3 grid((lattice_size_SU2 + BLOCK_SIZE - 1) / BLOCK_SIZE);
        ::normalize_spins_kernel<<<grid, block>>>(d_state, spin_length_SU2, lattice_size_SU2, spin_dim_SU2);
    }
    
    // Normalize SU(3) spins
    {
        dim3 block(BLOCK_SIZE);
        dim3 grid((lattice_size_SU3 + BLOCK_SIZE - 1) / BLOCK_SIZE);
        size_t SU3_offset = lattice_size_SU2 * spin_dim_SU2;
        ::normalize_spins_kernel<<<grid, block>>>(d_state + SU3_offset, spin_length_SU3, lattice_size_SU3, spin_dim_SU3);
    }
    
    cudaDeviceSynchronize();
}

} // namespace mixed_gpu

// =============================================================================
// Host-Callable API Implementation (for C++ TUs)
// These functions wrap the internal mixed_gpu:: namespace functions with opaque handles
// =============================================================================

namespace mixed_gpu {

/**
 * Internal structure that backs the opaque handle
 * Only visible to CUDA translation units
 */
struct GPUMixedLatticeDataHandle {
    GPUMixedLatticeData data;
    GPUState state;
    bool has_state;
    
    GPUMixedLatticeDataHandle() : has_state(false) {}
};

GPUMixedLatticeDataHandle* create_gpu_mixed_lattice_data(
    size_t lattice_size_SU2, size_t spin_dim_SU2, size_t N_atoms_SU2,
    size_t lattice_size_SU3, size_t spin_dim_SU3, size_t N_atoms_SU3,
    size_t max_bilinear_SU2, size_t max_bilinear_SU3, size_t max_mixed_bilinear,
    const std::vector<double>& flat_field_SU2,
    const std::vector<double>& flat_onsite_SU2,
    const std::vector<double>& flat_bilinear_SU2,
    const std::vector<size_t>& flat_partners_SU2,
    const std::vector<size_t>& num_bilinear_per_site_SU2,
    const std::vector<double>& flat_field_SU3,
    const std::vector<double>& flat_onsite_SU3,
    const std::vector<double>& flat_bilinear_SU3,
    const std::vector<size_t>& flat_partners_SU3,
    const std::vector<size_t>& num_bilinear_per_site_SU3,
    const std::vector<double>& flat_mixed_bilinear,
    const std::vector<size_t>& flat_mixed_partners_SU2,
    const std::vector<size_t>& flat_mixed_partners_SU3,
    const std::vector<size_t>& num_mixed_per_site_SU2
) {
    GPUMixedLatticeDataHandle* handle = new GPUMixedLatticeDataHandle();
    
    // Use the internal create function
    handle->data = mixed_gpu::create_gpu_mixed_lattice_data_internal(
        lattice_size_SU2, spin_dim_SU2, N_atoms_SU2,
        lattice_size_SU3, spin_dim_SU3, N_atoms_SU3,
        max_bilinear_SU2, max_bilinear_SU3, max_mixed_bilinear,
        flat_field_SU2, flat_onsite_SU2, flat_bilinear_SU2, 
        flat_partners_SU2, num_bilinear_per_site_SU2,
        flat_field_SU3, flat_onsite_SU3, flat_bilinear_SU3, 
        flat_partners_SU3, num_bilinear_per_site_SU3,
        flat_mixed_bilinear, flat_mixed_partners_SU2, 
        flat_mixed_partners_SU3, num_mixed_per_site_SU2
    );
    
    return handle;
}

void destroy_gpu_mixed_lattice_data(GPUMixedLatticeDataHandle* handle) {
    if (handle) {
        // GPUMixedLatticeData uses thrust vectors which auto-deallocate
        delete handle;
    }
}

void set_gpu_pulse_SU2(
    GPUMixedLatticeDataHandle* handle,
    const std::vector<double>& flat_field_drive,
    double pulse_amp, double pulse_width, double pulse_freq,
    double t_pulse_1, double t_pulse_2
) {
    if (!handle) return;
    
    mixed_gpu::set_gpu_pulse_SU2(handle->data, flat_field_drive, 
                                  pulse_amp, pulse_width, pulse_freq, 
                                  t_pulse_1, t_pulse_2);
}

void set_gpu_pulse_SU3(
    GPUMixedLatticeDataHandle* handle,
    const std::vector<double>& flat_field_drive,
    double pulse_amp, double pulse_width, double pulse_freq,
    double t_pulse_1, double t_pulse_2
) {
    if (!handle) return;
    
    mixed_gpu::set_gpu_pulse_SU3(handle->data, flat_field_drive, 
                                  pulse_amp, pulse_width, pulse_freq, 
                                  t_pulse_1, t_pulse_2);
}

void set_gpu_mixed_spins(
    GPUMixedLatticeDataHandle* handle,
    const std::vector<double>& flat_spins
) {
    if (!handle) return;
    
    handle->state.resize(flat_spins.size());
    thrust::copy(flat_spins.begin(), flat_spins.end(), handle->state.begin());
    handle->has_state = true;
}

void get_gpu_mixed_spins(
    GPUMixedLatticeDataHandle* handle,
    std::vector<double>& flat_spins
) {
    if (!handle || !handle->has_state) return;
    
    flat_spins.resize(handle->state.size());
    thrust::copy(handle->state.begin(), handle->state.end(), flat_spins.begin());
}

void integrate_mixed_gpu(
    GPUMixedLatticeDataHandle* handle,
    double T_start,
    double T_end,
    double dt,
    size_t save_interval,
    std::vector<std::pair<double, std::vector<double>>>& trajectory,
    const std::string& method
) {
    if (!handle || !handle->has_state) return;
    
    GPUMixedODESystem system(handle->data);
    mixed_gpu::integrate_mixed_gpu(system, handle->state, T_start, T_end, dt, 
                                    save_interval, trajectory, method);
}

void step_mixed_gpu(
    GPUMixedLatticeDataHandle* handle,
    double t,
    double dt,
    const std::string& method
) {
    if (!handle || !handle->has_state) return;
    
    GPUMixedODESystem system(handle->data);
    mixed_gpu::step_mixed_gpu(system, handle->state, t, dt, method);
}

void compute_magnetization_mixed_gpu(
    GPUMixedLatticeDataHandle* handle,
    std::vector<double>& mag_SU2,
    std::vector<double>& mag_staggered_SU2,
    std::vector<double>& mag_SU3,
    std::vector<double>& mag_staggered_SU3
) {
    if (!handle || !handle->has_state) return;
    
    size_t spin_dim_SU2 = handle->data.spin_dim_SU2;
    size_t spin_dim_SU3 = handle->data.spin_dim_SU3;
    
    mag_SU2.resize(spin_dim_SU2);
    mag_staggered_SU2.resize(spin_dim_SU2);
    mag_SU3.resize(spin_dim_SU3);
    mag_staggered_SU3.resize(spin_dim_SU3);
    
    mixed_gpu::compute_magnetization_mixed_gpu(
        handle->data, handle->state,
        mag_SU2.data(), mag_staggered_SU2.data(),
        mag_SU3.data(), mag_staggered_SU3.data()
    );
}

void normalize_spins_mixed_gpu(
    GPUMixedLatticeDataHandle* handle,
    double spin_length_SU2,
    double spin_length_SU3
) {
    if (!handle || !handle->has_state) return;
    
    mixed_gpu::normalize_spins_mixed_gpu(
        handle->state,
        handle->data.lattice_size_SU2, handle->data.spin_dim_SU2, spin_length_SU2,
        handle->data.lattice_size_SU3, handle->data.spin_dim_SU3, spin_length_SU3
    );
}

void get_mixed_lattice_dims(
    GPUMixedLatticeDataHandle* handle,
    size_t& lattice_size_SU2, size_t& spin_dim_SU2, size_t& N_atoms_SU2,
    size_t& lattice_size_SU3, size_t& spin_dim_SU3, size_t& N_atoms_SU3
) {
    if (!handle) {
        lattice_size_SU2 = spin_dim_SU2 = N_atoms_SU2 = 0;
        lattice_size_SU3 = spin_dim_SU3 = N_atoms_SU3 = 0;
        return;
    }
    
    lattice_size_SU2 = handle->data.lattice_size_SU2;
    spin_dim_SU2 = handle->data.spin_dim_SU2;
    N_atoms_SU2 = handle->data.N_atoms_SU2;
    lattice_size_SU3 = handle->data.lattice_size_SU3;
    spin_dim_SU3 = handle->data.spin_dim_SU3;
    N_atoms_SU3 = handle->data.N_atoms_SU3;
}

} // namespace mixed_gpu

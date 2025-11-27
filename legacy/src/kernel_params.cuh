#ifndef KERNEL_PARAMS_CUH
#define KERNEL_PARAMS_CUH

#include <cuda_runtime.h>

// Structure to hold all SU2-related device pointers
template<size_t N_SU2, size_t lattice_size_SU2>
struct SU2_DeviceParams {
    double* spins;
    double* local_field;
    double* field;
    double* onsite_interaction;
    double* bilinear_interaction;
    size_t* bilinear_partners;
    double* trilinear_interaction;
    size_t* trilinear_partners;
    double* field_drive_1;
    double* field_drive_2;
    
    size_t num_bi;
    size_t num_tri;
    size_t max_bi_neighbors;
    size_t max_tri_neighbors;
};

// Structure to hold all SU3-related device pointers
template<size_t N_SU3, size_t lattice_size_SU3>
struct SU3_DeviceParams {
    double* spins;
    double* local_field;
    double* field;
    double* onsite_interaction;
    double* bilinear_interaction;
    size_t* bilinear_partners;
    double* trilinear_interaction;
    size_t* trilinear_partners;
    double* field_drive_1;
    double* field_drive_2;
    
    size_t num_bi;
    size_t num_tri;
    size_t max_bi_neighbors;
    size_t max_tri_neighbors;
};

// Structure to hold mixed interaction parameters
template<size_t N_SU2, size_t N_SU3>
struct MixedInteractionParams {
    double* mixed_bilinear_interaction_SU2;
    double* mixed_bilinear_interaction_SU3;
    size_t* mixed_bilinear_partners_SU2;
    size_t* mixed_bilinear_partners_SU3;
    
    double* mixed_trilinear_interaction_SU2;
    double* mixed_trilinear_interaction_SU3;
    size_t* mixed_trilinear_partners_SU2;
    size_t* mixed_trilinear_partners_SU3;
    
    size_t num_bi_SU2_SU3;
    size_t num_tri_SU2_SU3;
    size_t max_mixed_bi_neighbors_SU2;
    size_t max_mixed_bi_neighbors_SU3;
    size_t max_mixed_tri_neighbors_SU2;
    size_t max_mixed_tri_neighbors_SU3;
};

// Structure to hold time-dependent field drive parameters
struct DriveFieldParams {
    double amp;
    double width;
    double freq;
    double t_B_1;
    double t_B_2;
    double curr_time;
    double dt;
};

// Structure to hold working arrays for RK methods
template<size_t N_SU2, size_t N_SU3>
struct RKWorkArrays {
    double* k_SU2;
    double* k_SU3;
    double* work_SU2_1;
    double* work_SU2_2;
    double* work_SU2_3;
    double* work_SU3_1;
    double* work_SU3_2;
    double* work_SU3_3;
};

#endif // KERNEL_PARAMS_CUH

/**
 * runners_parameter_sweep.cpp — multi-lattice parameter sweep driver.
 *
 * The parameter sweep is the one runner that spans every lattice family,
 * so it pulls in all four lattice headers and dispatches via the other
 * `run_*` functions declared in `spin_solver_runners.h`.
 *
 * Split out of `spin_solver.cpp` to keep the main TU small.
 */

#include "spin_solver_runners.h"

#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/unitcell.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/lattice/lattice.h"
#include "classical_spin/lattice/mixed_lattice.h"
#include "classical_spin/lattice/phonon_lattice.h"
#include "classical_spin/lattice/strain_phonon_lattice.h"

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <cmath>

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif

using namespace std;

/**
 * Run the multi-lattice parameter sweep dispatcher.
 */
void run_parameter_sweep(const SpinConfig& base_config, int rank, int size) {
    // Determine which sweep mode to use
    vector<string> params;
    vector<vector<double>> param_grids;
    
    if (!base_config.sweep_parameters.empty() && base_config.sweep_parameters.size() == base_config.sweep_starts.size() &&
        base_config.sweep_parameters.size() == base_config.sweep_ends.size() && 
        base_config.sweep_parameters.size() == base_config.sweep_steps.size()) {
        // N-dimensional sweep mode
        params = base_config.sweep_parameters;
        for (size_t p = 0; p < params.size(); ++p) {
            vector<double> grid;
            for (double val = base_config.sweep_starts[p];
                 (base_config.sweep_steps[p] > 0 ? val <= base_config.sweep_ends[p] : val >= base_config.sweep_ends[p]);
                 val += base_config.sweep_steps[p]) {
                grid.push_back(val);
            }
            param_grids.push_back(grid);
        }
    } else if (!base_config.sweep_parameter.empty()) {
        // Legacy 1D sweep mode
        params.push_back(base_config.sweep_parameter);
        vector<double> grid;
        for (double val = base_config.sweep_start;
             (base_config.sweep_step > 0 ? val <= base_config.sweep_end : val >= base_config.sweep_end);
             val += base_config.sweep_step) {
            grid.push_back(val);
        }
        param_grids.push_back(grid);
    } else {
        if (rank == 0) {
            cerr << "Error: No sweep parameters specified!" << endl;
        }
        return;
    }
    
    // Check if GPU is needed for the base simulation
    bool needs_gpu = base_config.use_gpu && (
        base_config.sweep_base_simulation == SimulationType::MOLECULAR_DYNAMICS ||
        base_config.sweep_base_simulation == SimulationType::PUMP_PROBE ||
        base_config.sweep_base_simulation == SimulationType::TWOD_COHERENT_SPECTROSCOPY
    );
    
    if (rank == 0) {
        cout << "Running " << params.size() << "D parameter sweep..." << endl;
        for (size_t p = 0; p < params.size(); ++p) {
            cout << "  Parameter " << (p+1) << ": " << params[p] 
                 << " (" << param_grids[p].size() << " points)" << endl;
        }
        cout << "Base simulation: ";
        switch (base_config.sweep_base_simulation) {
            case SimulationType::SIMULATED_ANNEALING: cout << "Simulated Annealing"; break;
            case SimulationType::PARALLEL_TEMPERING: cout << "Parallel Tempering"; break;
            case SimulationType::MOLECULAR_DYNAMICS: cout << "Molecular Dynamics"; break;
            case SimulationType::PUMP_PROBE: cout << "Pump-Probe"; break;
            case SimulationType::TWOD_COHERENT_SPECTROSCOPY: cout << "2DCS Spectroscopy"; break;
            default: cout << "Unknown"; break;
        }
        cout << endl;
        cout << "MPI ranks: " << size << endl;
        if (needs_gpu) {
#ifdef CUDA_ENABLED
            cout << "GPU acceleration: ENABLED" << endl;
#else
            cout << "GPU acceleration: REQUESTED but not available (compiled without CUDA)" << endl;
            cout << "Falling back to CPU implementation" << endl;
#endif
        } else if (base_config.use_gpu) {
            cout << "GPU acceleration: Not used by base simulation type" << endl;
        } else {
            cout << "GPU acceleration: DISABLED (using CPU)" << endl;
        }
    }
    
#ifdef CUDA_ENABLED
    // Set GPU device based on local rank (for multi-GPU nodes)
    // Do this ONCE before the sweep loop to avoid repeated setup
    if (needs_gpu) {
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count > 0) {
            int device_id = rank % device_count;
            cudaSetDevice(device_id);
            // Log GPU assignment for all ranks (synchronized output)
            for (int r = 0; r < size; ++r) {
                if (rank == r) {
                    cout << "[Rank " << rank << "] Assigned to GPU " << device_id 
                         << " (parameter sweep, " << device_count << " GPU(s) available)" << endl;
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
        } else {
            if (rank == 0) {
                cout << "Warning: No GPUs detected, falling back to CPU" << endl;
            }
        }
    }
#endif
    
    // Generate all combinations of parameter values (Cartesian product)
    vector<vector<double>> all_sweep_points;
    function<void(size_t, vector<double>&)> generate_combinations;
    generate_combinations = [&](size_t depth, vector<double>& current) {
        if (depth == params.size()) {
            all_sweep_points.push_back(current);
            return;
        }
        for (double val : param_grids[depth]) {
            current.push_back(val);
            generate_combinations(depth + 1, current);
            current.pop_back();
        }
    };
    vector<double> current;
    generate_combinations(0, current);
    
    if (rank == 0) {
        cout << "Total sweep points: " << all_sweep_points.size() << endl;
    }
    
    // ========================================================================
    // PARALLEL TEMPERING SPECIAL HANDLING
    // For parallel tempering, we need to split MPI_COMM_WORLD into sub-communicators
    // so each sweep point can run proper PT with multiple temperature replicas
    // ========================================================================
    bool is_parallel_tempering = (base_config.sweep_base_simulation == SimulationType::PARALLEL_TEMPERING);
    
    MPI_Comm sweep_comm = MPI_COMM_WORLD;  // Communicator for this rank's sweep point
    int sweep_rank = rank;                  // Rank within sweep_comm
    int sweep_size = size;                  // Size of sweep_comm
    int sweep_point_idx = -1;               // Which sweep point this rank is assigned to
    
    if (is_parallel_tempering) {
        // Determine ranks per sweep point
        int ranks_per_point = base_config.pt_ranks_per_point;
        if (ranks_per_point <= 0) {
            // Auto-detect: divide total ranks among sweep points, minimum 2 for PT
            ranks_per_point = max(2, size / static_cast<int>(all_sweep_points.size()));
        }
        
        // Calculate number of concurrent sweep points we can run
        int num_concurrent_points = size / ranks_per_point;
        if (num_concurrent_points < 1) {
            num_concurrent_points = 1;
            ranks_per_point = size;
        }
        
        // Assign this rank to a sweep point group (color) and compute local rank
        int color = rank / ranks_per_point;  // Which group (sweep point) this rank belongs to
        if (color >= num_concurrent_points) {
            // Extra ranks that don't fit into groups - assign to last group
            color = num_concurrent_points - 1;
        }
        
        // Create sub-communicator for this group
        MPI_Comm_split(MPI_COMM_WORLD, color, rank, &sweep_comm);
        MPI_Comm_rank(sweep_comm, &sweep_rank);
        MPI_Comm_size(sweep_comm, &sweep_size);
        
        if (rank == 0) {
            cout << "\nParallel Tempering in Parameter Sweep Mode:" << endl;
            cout << "  Ranks per sweep point: " << ranks_per_point << endl;
            cout << "  Concurrent sweep points: " << num_concurrent_points << endl;
            cout << "  Total sweep points: " << all_sweep_points.size() << endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        
        // Distribute sweep points across groups (not individual ranks)
        // Each group processes sweep points in round-robin fashion
        for (size_t i = color; i < all_sweep_points.size(); i += num_concurrent_points) {
            const auto& param_values = all_sweep_points[i];
            
            // Print progress (only rank 0 of each group)
            if (sweep_rank == 0) {
                stringstream progress;
                progress << "[Group " << color << "] Processing point " << i+1 << "/" << all_sweep_points.size() << ": ";
                for (size_t p = 0; p < params.size(); ++p) {
                    if (p > 0) progress << ", ";
                    progress << params[p] << "=" << param_values[p];
                }
                cout << progress.str() << endl;
            }
            
            // Create modified config for this sweep point
            SpinConfig sweep_config = base_config;
            sweep_config.simulation = base_config.sweep_base_simulation;
            
            // Apply all parameter values
            for (size_t p = 0; p < params.size(); ++p) {
                const string& param_name = params[p];
                double param_value = param_values[p];
                
                sweep_config.set_param(param_name, param_value);
                
                if (param_name == "field_strength" || param_name == "h") {
                    sweep_config.field_strength = param_value;
                }
            }
            
            // Create output directory for this sweep point
            stringstream ss;
            ss << base_config.output_dir;
            for (size_t p = 0; p < params.size(); ++p) {
                ss << "/" << params[p] << "_" << scientific << param_values[p];
            }
            sweep_config.output_dir = ss.str();
            if (sweep_rank == 0) {
                filesystem::create_directories(sweep_config.output_dir);
            }
            MPI_Barrier(sweep_comm);  // Ensure directory is created before others proceed
            
            // Run parallel tempering with the sub-communicator's rank/size
            if (sweep_config.system == SystemType::TMFEO3) {
                MixedUnitCell mixed_uc = build_tmfeo3(sweep_config);
                MixedLattice mixed_lattice(mixed_uc, sweep_config.lattice_size[0], 
                                          sweep_config.lattice_size[1], 
                                          sweep_config.lattice_size[2],
                                          sweep_config.spin_length,
                                          sweep_config.spin_length_su3);
                
                // Initialize spins
                if (sweep_config.use_ferromagnetic_init) {
                    SpinVector dir_su2(mixed_lattice.spin_dim_SU2);
                    for (size_t d = 0; d < mixed_lattice.spin_dim_SU2; ++d) {
                        dir_su2(d) = (d < sweep_config.ferromagnetic_direction.size()) ? 
                                     sweep_config.ferromagnetic_direction[d] : 0.0;
                    }
                    SpinVector dir_su3 = SpinVector::Zero(mixed_lattice.spin_dim_SU3);
                    const int su3_init_component = static_cast<int>(sweep_config.get_param("su3_init_component", 2.0));
                    if (su3_init_component >= 0 && su3_init_component < static_cast<int>(mixed_lattice.spin_dim_SU3)) {
                        dir_su3(su3_init_component) = 1.0;
                    } else {
                        dir_su3(2) = 1.0;
                    }
                    mixed_lattice.init_ferromagnetic(dir_su2, dir_su3);
                } else if (!sweep_config.initial_spin_config.empty()) {
                    mixed_lattice.load_spin_config(sweep_config.initial_spin_config);
                }
                
                // Run PT with proper sub-communicator rank/size
                run_parallel_tempering_mixed(mixed_lattice, sweep_config, sweep_rank, sweep_size, sweep_comm);
            } else if (sweep_config.system == SystemType::NCTO_STRAIN) {
                // StrainPhononLattice magnetoelastic system
                UnitCell strain_uc = build_strain_honeycomb(sweep_config);
                StrainPhononLattice strain_lattice(strain_uc,
                                                   sweep_config.lattice_size[0],
                                                   sweep_config.lattice_size[1],
                                                   sweep_config.lattice_size[2],
                                                   sweep_config.spin_length);
                
                // Build parameters from config
                MagnetoelasticParams me_params;
                ElasticParams el_params;
                StrainDriveParams dr_params;
                build_strain_params(sweep_config, me_params, el_params, dr_params);
                
                // Set parameters
                strain_lattice.set_parameters(me_params, el_params, dr_params);
                strain_lattice.alpha_gilbert = sweep_config.get_param("alpha_gilbert", 0.0);
                
                // Set magnetic field
                Eigen::Vector3d B;
                B << sweep_config.field_strength * sweep_config.field_direction[0],
                     sweep_config.field_strength * sweep_config.field_direction[1],
                     sweep_config.field_strength * sweep_config.field_direction[2];
                strain_lattice.set_uniform_field(B);
                
                // Initialize spins
                if (!sweep_config.initial_spin_config.empty()) {
                    strain_lattice.load_spin_config(sweep_config.initial_spin_config);
                } else {
                    strain_lattice.init_random();
                }
                
                // Run PT with proper sub-communicator rank/size
                run_parallel_tempering_strain(strain_lattice, sweep_config, sweep_rank, sweep_size, sweep_comm);
            } else {
                // Standard lattice systems
                UnitCell* uc_ptr = nullptr;
                switch (sweep_config.system) {
                    case SystemType::HONEYCOMB_BCAO:
                        uc_ptr = new UnitCell(build_bcao_honeycomb(sweep_config));
                        break;
                    case SystemType::HONEYCOMB_KITAEV:
                        uc_ptr = new UnitCell(build_kitaev_honeycomb(sweep_config));
                        break;
                    case SystemType::PYROCHLORE:
                        uc_ptr = new UnitCell(build_pyrochlore(sweep_config));
                        break;
                    case SystemType::PYROCHLORE_NON_KRAMER:
                        uc_ptr = new UnitCell(build_pyrochlore_non_kramer(sweep_config));
                        break;
                    case SystemType::TMFEO3_FE:
                        uc_ptr = new UnitCell(build_tmfeo3_fe(sweep_config));
                        break;
                    case SystemType::TMFEO3_TM:
                        uc_ptr = new UnitCell(build_tmfeo3_tm(sweep_config));
                        break;
                    default:
                        if (sweep_rank == 0) {
                            cerr << "Error: Unknown system type for parameter sweep with parallel tempering" << endl;
                        }
                        MPI_Abort(MPI_COMM_WORLD, 1);
                        return;
                }
                
                Lattice lattice(*uc_ptr, sweep_config.lattice_size[0], 
                              sweep_config.lattice_size[1], 
                              sweep_config.lattice_size[2],
                              sweep_config.spin_length);
                lattice.lattice_type = system_type_to_string(sweep_config.system);
                
                // Initialize spins
                if (sweep_config.use_ferromagnetic_init) {
                    SpinVector dir(lattice.spin_dim);
                    for (size_t d = 0; d < lattice.spin_dim; ++d) {
                        dir(d) = (d < sweep_config.ferromagnetic_direction.size()) ? 
                                 sweep_config.ferromagnetic_direction[d] : 0.0;
                    }
                    lattice.init_ferromagnetic(dir);
                } else if (!sweep_config.initial_spin_config.empty()) {
                    lattice.load_spin_config(sweep_config.initial_spin_config);
                }
                
                // Run PT with proper sub-communicator rank/size
                run_parallel_tempering(lattice, sweep_config, sweep_rank, sweep_size, sweep_comm);
                
                delete uc_ptr;
            }
            
            // Print completion message (only rank 0 of each group)
            if (sweep_rank == 0) {
                stringstream completion;
                completion << "[Group " << color << "] Completed point " << i+1 << ": ";
                for (size_t p = 0; p < params.size(); ++p) {
                    if (p > 0) completion << ", ";
                    completion << params[p] << "=" << param_values[p];
                }
                cout << completion.str() << endl;
            }
            
            // Synchronize within the group before moving to next sweep point
            MPI_Barrier(sweep_comm);
        }
        
        // Free the sub-communicator
        MPI_Comm_free(&sweep_comm);
        
    } else {
        // ====================================================================
        // NON-PARALLEL-TEMPERING: Original behavior
        // Distribute sweep points across MPI ranks (one rank per point)
        // ====================================================================
        for (size_t i = rank; i < all_sweep_points.size(); i += size) {
        const auto& param_values = all_sweep_points[i];
        
        // Print progress
        stringstream progress;
        progress << "[Rank " << rank << "] Processing point " << i+1 << "/" << all_sweep_points.size() << ": ";
        for (size_t p = 0; p < params.size(); ++p) {
            if (p > 0) progress << ", ";
            progress << params[p] << "=" << param_values[p];
        }
        if (rank == 0 || all_sweep_points.size() > 1) {
            cout << progress.str() << endl;
        }
        
        // Create modified config for this sweep point
        SpinConfig sweep_config = base_config;
        sweep_config.simulation = base_config.sweep_base_simulation;
        
        // Apply all parameter values
        for (size_t p = 0; p < params.size(); ++p) {
            const string& param_name = params[p];
            double param_value = param_values[p];
            
            sweep_config.set_param(param_name, param_value);
            
            // Also check if it's a special field parameter
            if (param_name == "field_strength" || param_name == "h") {
                sweep_config.field_strength = param_value;
            }
        }
        
        // Create output directory for this sweep point
        stringstream ss;
        ss << base_config.output_dir;
        for (size_t p = 0; p < params.size(); ++p) {
            ss << "/" << params[p] << "_" << scientific << param_values[p];
        }
        sweep_config.output_dir = ss.str();
        filesystem::create_directories(sweep_config.output_dir);
        
        // Build unit cell with updated parameters
        if (sweep_config.system == SystemType::NCTO) {
            // PhononLattice spin-phonon coupled system (honeycomb)
            UnitCell phonon_uc = build_phonon_honeycomb(sweep_config);
            PhononLattice phonon_lattice(phonon_uc,
                                         sweep_config.lattice_size[0],
                                         sweep_config.lattice_size[1],
                                         sweep_config.lattice_size[2],
                                         sweep_config.spin_length);
            
            // Build parameters from config
            SpinPhononCouplingParams sp_params;
            PhononParams ph_params;
            DriveParams dr_params;
            TimeDependentSpinPhononParams td_sp_params;
            build_phonon_params(sweep_config, sp_params, ph_params, dr_params, td_sp_params);
            
            // Set parameters
            phonon_lattice.set_parameters(sp_params, ph_params, dr_params);
            phonon_lattice.set_time_dependent_spin_phonon(td_sp_params);
            phonon_lattice.alpha_gilbert = sweep_config.get_param("alpha_gilbert", 0.0);
            
            // Set magnetic field
            Eigen::Vector3d B;
            B << sweep_config.field_strength * sweep_config.field_direction[0],
                 sweep_config.field_strength * sweep_config.field_direction[1],
                 sweep_config.field_strength * sweep_config.field_direction[2];
            phonon_lattice.set_field(B);
            
            // Initialize spins
            if (sweep_config.use_ferromagnetic_init) {
                Eigen::Vector3d dir;
                dir << sweep_config.ferromagnetic_direction[0],
                       sweep_config.ferromagnetic_direction[1],
                       sweep_config.ferromagnetic_direction[2];
                phonon_lattice.init_ferromagnetic(dir);
            } else if (!sweep_config.initial_spin_config.empty()) {
                phonon_lattice.load_spin_config(sweep_config.initial_spin_config);
            } else {
                phonon_lattice.init_random();
            }
            
            // Run appropriate simulation
            switch (sweep_config.simulation) {
                case SimulationType::SIMULATED_ANNEALING:
                    run_simulated_annealing_phonon(phonon_lattice, sweep_config, 0, 1);
                    break;
                case SimulationType::MOLECULAR_DYNAMICS:
                    run_molecular_dynamics_phonon(phonon_lattice, sweep_config, 0, 1);
                    break;
                case SimulationType::PUMP_PROBE:
                    run_pump_probe_phonon(phonon_lattice, sweep_config, 0, 1);
                    break;
                case SimulationType::TWOD_COHERENT_SPECTROSCOPY:
                    run_2dcs_phonon(phonon_lattice, sweep_config, 0, 1);
                    break;
                default:
                    cerr << "[Rank " << rank << "] Error: Unsupported base simulation for parameter sweep with PhononLattice" << endl;
                    break;
            }
        } else if (sweep_config.system == SystemType::NCTO_STRAIN) {
            // StrainPhononLattice magnetoelastic (spin-strain) coupled system (honeycomb)
            UnitCell strain_uc = build_strain_honeycomb(sweep_config);
            StrainPhononLattice strain_lattice(strain_uc,
                                               sweep_config.lattice_size[0],
                                               sweep_config.lattice_size[1],
                                               sweep_config.lattice_size[2],
                                               sweep_config.spin_length);
            
            // Build parameters from config
            MagnetoelasticParams me_params;
            ElasticParams el_params;
            StrainDriveParams dr_params;
            build_strain_params(sweep_config, me_params, el_params, dr_params);
            
            // Set parameters
            strain_lattice.set_parameters(me_params, el_params, dr_params);
            strain_lattice.alpha_gilbert = sweep_config.get_param("alpha_gilbert", 0.0);
            
            // Set magnetic field
            Eigen::Vector3d B;
            B << sweep_config.field_strength * sweep_config.field_direction[0],
                 sweep_config.field_strength * sweep_config.field_direction[1],
                 sweep_config.field_strength * sweep_config.field_direction[2];
            strain_lattice.set_uniform_field(B);
            
            // Initialize spins
            if (!sweep_config.initial_spin_config.empty()) {
                strain_lattice.load_spin_config(sweep_config.initial_spin_config);
            } else {
                strain_lattice.init_random();
            }
            
            // Run appropriate simulation
            switch (sweep_config.simulation) {
                case SimulationType::SIMULATED_ANNEALING:
                    run_simulated_annealing_strain(strain_lattice, sweep_config, 0, 1);
                    break;
                case SimulationType::MOLECULAR_DYNAMICS:
                    run_molecular_dynamics_strain(strain_lattice, sweep_config, 0, 1);
                    break;
                case SimulationType::PUMP_PROBE:
                    run_pump_probe_strain(strain_lattice, sweep_config, 0, 1);
                    break;
                case SimulationType::KINETIC_BARRIER_ANALYSIS:
                    run_kinetic_barrier_analysis_strain(strain_lattice, sweep_config, 0, 1);
                    break;
                default:
                    cerr << "[Rank " << rank << "] Error: Unsupported base simulation for parameter sweep with StrainPhononLattice" << endl;
                    break;
            }
        } else if (sweep_config.system == SystemType::TMFEO3) {
            MixedUnitCell mixed_uc = build_tmfeo3(sweep_config);
            MixedLattice mixed_lattice(mixed_uc, sweep_config.lattice_size[0], 
                                      sweep_config.lattice_size[1], 
                                      sweep_config.lattice_size[2],
                                      sweep_config.spin_length,
                                      sweep_config.spin_length_su3);
            
            // Initialize spins
            if (sweep_config.use_ferromagnetic_init) {
                // Create SU2 direction from config (use general dimension)
                SpinVector dir_su2(mixed_lattice.spin_dim_SU2);
                for (size_t d = 0; d < mixed_lattice.spin_dim_SU2; ++d) {
                    dir_su2(d) = (d < sweep_config.ferromagnetic_direction.size()) ? 
                                 sweep_config.ferromagnetic_direction[d] : 0.0;
                }
                // Create SU3 direction
                SpinVector dir_su3 = SpinVector::Zero(mixed_lattice.spin_dim_SU3);
                const int su3_init_component = static_cast<int>(sweep_config.get_param("su3_init_component", 2.0));
                if (su3_init_component >= 0 && su3_init_component < static_cast<int>(mixed_lattice.spin_dim_SU3)) {
                    dir_su3(su3_init_component) = 1.0;
                } else {
                    dir_su3(2) = 1.0;  // Default to λ3
                }
                mixed_lattice.init_ferromagnetic(dir_su2, dir_su3);
            } else if (!sweep_config.initial_spin_config.empty()) {
                mixed_lattice.load_spin_config(sweep_config.initial_spin_config);
            }
            // else: spins already initialized randomly in constructor
            
            // Run appropriate simulation
            // Pass 0, 1 for rank/size since each MPI rank works independently on different sweep points
            // GPU is already set up at the beginning of run_parameter_sweep
            switch (sweep_config.simulation) {
                case SimulationType::SIMULATED_ANNEALING:
                    run_simulated_annealing_mixed(mixed_lattice, sweep_config, 0, 1);
                    break;
                case SimulationType::PARALLEL_TEMPERING:
                    run_parallel_tempering_mixed(mixed_lattice, sweep_config, 0, 1);
                    break;
                case SimulationType::MOLECULAR_DYNAMICS:
                    run_molecular_dynamics_mixed(mixed_lattice, sweep_config, 0, 1);
                    break;
                case SimulationType::PUMP_PROBE:
                    run_pump_probe_mixed(mixed_lattice, sweep_config, 0, 1);
                    break;
                case SimulationType::TWOD_COHERENT_SPECTROSCOPY:
                    run_2dcs_spectroscopy_mixed(mixed_lattice, sweep_config, 0, 1);
                    break;
                default:
                    cerr << "[Rank " << rank << "] Error: Unsupported base simulation for parameter sweep with mixed lattice" << endl;
                    break;
            }
        } else {
            // Standard lattice systems - build unit cell based on system type
            UnitCell* uc_ptr = nullptr;
            switch (sweep_config.system) {
                case SystemType::HONEYCOMB_BCAO:
                    uc_ptr = new UnitCell(build_bcao_honeycomb(sweep_config));
                    break;
                case SystemType::HONEYCOMB_KITAEV:
                    uc_ptr = new UnitCell(build_kitaev_honeycomb(sweep_config));
                    break;
                case SystemType::PYROCHLORE:
                    uc_ptr = new UnitCell(build_pyrochlore(sweep_config));
                    break;
                case SystemType::PYROCHLORE_NON_KRAMER:
                    uc_ptr = new UnitCell(build_pyrochlore_non_kramer(sweep_config));
                    break;
                case SystemType::TMFEO3_FE:
                    uc_ptr = new UnitCell(build_tmfeo3_fe(sweep_config));
                    break;
                case SystemType::TMFEO3_TM:
                    uc_ptr = new UnitCell(build_tmfeo3_tm(sweep_config));
                    break;
                default:
                    if (rank == 0) {
                        cerr << "Error: Unknown system type for parameter sweep" << endl;
                    }
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    return;
            }
            
            Lattice lattice(*uc_ptr, sweep_config.lattice_size[0], 
                          sweep_config.lattice_size[1], 
                          sweep_config.lattice_size[2],
                          sweep_config.spin_length);
            lattice.lattice_type = system_type_to_string(sweep_config.system);
            
            // Initialize spins
            if (sweep_config.use_ferromagnetic_init) {
                // Initialize all spins in same direction (use general spin_dim)
                SpinVector dir(lattice.spin_dim);
                for (size_t d = 0; d < lattice.spin_dim; ++d) {
                    dir(d) = (d < sweep_config.ferromagnetic_direction.size()) ? 
                             sweep_config.ferromagnetic_direction[d] : 0.0;
                }
                lattice.init_ferromagnetic(dir);
            } else if (!sweep_config.initial_spin_config.empty()) {
                lattice.load_spin_config(sweep_config.initial_spin_config);
            }
            // else: spins already initialized randomly in constructor
            
            // Run appropriate simulation
            // Pass 0, 1 for rank/size since each MPI rank works independently on different sweep points
            // GPU is already set up at the beginning of run_parameter_sweep
            switch (sweep_config.simulation) {
                case SimulationType::SIMULATED_ANNEALING:
                    run_simulated_annealing(lattice, sweep_config, 0, 1);
                    break;
                case SimulationType::PARALLEL_TEMPERING:
                    run_parallel_tempering(lattice, sweep_config, 0, 1);
                    break;
                case SimulationType::MOLECULAR_DYNAMICS:
                    run_molecular_dynamics(lattice, sweep_config, 0, 1);
                    break;
                case SimulationType::PUMP_PROBE:
                    run_pump_probe(lattice, sweep_config, 0, 1);
                    break;
                case SimulationType::TWOD_COHERENT_SPECTROSCOPY:
                    run_2dcs_spectroscopy(lattice, sweep_config, 0, 1);
                    break;
                default:
                    cerr << "[Rank " << rank << "] Error: Unsupported base simulation for parameter sweep" << endl;
                    break;
            }
            
            // Clean up unit cell pointer
            delete uc_ptr;
        }
        
        // Print completion message
        stringstream completion;
        completion << "[Rank " << rank << "] Completed point " << i+1 << ": ";
        for (size_t p = 0; p < params.size(); ++p) {
            if (p > 0) completion << ", ";
            completion << params[p] << "=" << param_values[p];
        }
        cout << completion.str() << endl;
        }  // End of for loop over sweep points
    }  // End of else (non-PT)
    
    // Synchronize all ranks
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "Parameter sweep completed (" << all_sweep_points.size() << " points)." << endl;
    }
}

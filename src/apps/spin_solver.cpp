/**
 * spin_solver.cpp — simulation executable entry point.
 *
 * The heavy lifting (every `run_<simulation>_<lattice>()` function, plus
 * the parameter-sweep driver and the strain/phonon parameter builders)
 * lives in sibling TUs: `runners_lattice.cpp`, `runners_phonon.cpp`,
 * `runners_strain.cpp`, `runners_mixed.cpp`, `runners_parameter_sweep.cpp`.
 * All of them are declared in `spin_solver_runners.h`. This file is now
 * just CLI parsing, MPI lifetime management, and the lattice-family
 * dispatch.
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
#include <memory>
#include <cmath>
#include <filesystem>
#include <fstream>

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#endif

using namespace std;


int main(int argc, char** argv) {
    // Initialize MPI
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(&argc, &argv);
    }
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Parse command line arguments
    string config_file = "simulation.param";
    if (argc > 1) {
        config_file = argv[1];
    }
    
    // Load configuration
    SpinConfig config;
    try {
        config = SpinConfig::from_file(config_file);
    } catch (const exception& e) {
        if (rank == 0) {
            cerr << "Error loading configuration: " << e.what() << endl;
            cerr << "Usage: " << argv[0] << " [config_file]\n";
        }
        MPI_Finalize();
        return 1;
    }
    
    // Validate configuration
    if (!config.validate()) {
        if (rank == 0) {
            cerr << "Configuration validation failed\n";
        }
        MPI_Finalize();
        return 1;
    }
    
    // Print configuration on rank 0
    if (rank == 0) {
        config.print();
    }
    
    // Build system and run simulation
    try {
        if (config.system == SystemType::NCTO) {
            // PhononLattice spin-phonon coupled system (honeycomb)
            if (rank == 0) {
                cout << "\nBuilding PhononLattice spin-phonon lattice..." << endl;
            }
            
            UnitCell phonon_uc = build_phonon_honeycomb(config);
            PhononLattice phonon_lattice(phonon_uc,
                                         config.lattice_size[0],
                                         config.lattice_size[1],
                                         config.lattice_size[2],
                                         config.spin_length);
            
            // Build parameters from config
            SpinPhononCouplingParams sp_params;
            PhononParams ph_params;
            DriveParams dr_params;
            TimeDependentSpinPhononParams td_sp_params;
            build_phonon_params(config, sp_params, ph_params, dr_params, td_sp_params);
            
            // Set parameters (this builds the interaction matrices)
            phonon_lattice.set_parameters(sp_params, ph_params, dr_params);
            
            // Set time-dependent spin-phonon coupling parameters
            phonon_lattice.set_time_dependent_spin_phonon(td_sp_params);
            
            // Set Gilbert damping if specified
            phonon_lattice.alpha_gilbert = config.get_param("alpha_gilbert", 0.0);
            
            // Set magnetic field
            Eigen::Vector3d B;
            B << config.field_strength * config.field_direction[0],
                 config.field_strength * config.field_direction[1],
                 config.field_strength * config.field_direction[2];
            phonon_lattice.set_field(B);
            
            // Initialize spins
            if (config.use_ferromagnetic_init) {
                Eigen::Vector3d dir;
                dir << config.ferromagnetic_direction[0],
                       config.ferromagnetic_direction[1],
                       config.ferromagnetic_direction[2];
                phonon_lattice.init_ferromagnetic(dir);
            } else if (!config.initial_spin_config.empty()) {
                phonon_lattice.load_spin_config(config.initial_spin_config);
            } else {
                phonon_lattice.init_random();
            }
            
            // Run simulation
            switch (config.simulation) {
                case SimulationType::SIMULATED_ANNEALING:
                    run_simulated_annealing_phonon(phonon_lattice, config, rank, size);
                    break;
                case SimulationType::MOLECULAR_DYNAMICS:
                    run_molecular_dynamics_phonon(phonon_lattice, config, rank, size);
                    break;
                case SimulationType::PUMP_PROBE:
                    run_pump_probe_phonon(phonon_lattice, config, rank, size);
                    break;
                case SimulationType::TWOD_COHERENT_SPECTROSCOPY:
                    run_2dcs_phonon(phonon_lattice, config, rank, size);
                    break;
                case SimulationType::PARAMETER_SWEEP:
                    run_parameter_sweep(config, rank, size);
                    break;
                default:
                    if (rank == 0) {
                        cerr << "Simulation type not supported for PhononLattice. "
                             << "Supported: SA, MD, pump_probe, 2dcs, parameter_sweep" << endl;
                    }
                    break;
            }
        } else if (config.system == SystemType::NCTO_STRAIN) {
            // StrainPhononLattice magnetoelastic (spin-strain) coupled system (honeycomb)
            if (rank == 0) {
                cout << "\nBuilding StrainPhononLattice magnetoelastic lattice..." << endl;
            }
            
            UnitCell strain_uc = build_strain_honeycomb(config);
            StrainPhononLattice strain_lattice(strain_uc,
                                               config.lattice_size[0],
                                               config.lattice_size[1],
                                               config.lattice_size[2],
                                               config.spin_length);
            
            // Build parameters from config
            MagnetoelasticParams me_params;
            ElasticParams el_params;
            StrainDriveParams dr_params;
            build_strain_params(config, me_params, el_params, dr_params);
            
            // Set parameters (this builds the interaction matrices)
            strain_lattice.set_parameters(me_params, el_params, dr_params);
            
            // Set static drive force on Eg phonon (for GNEB barrier)
            strain_lattice.drive_F_Eg1_ = config.drive_F_Eg1;
            strain_lattice.drive_F_Eg2_ = config.drive_F_Eg2;
            if (rank == 0 && (std::abs(config.drive_F_Eg1) > 1e-15 || std::abs(config.drive_F_Eg2) > 1e-15)) {
                cout << "Static drive force: F_Eg1 = " << config.drive_F_Eg1
                     << ", F_Eg2 = " << config.drive_F_Eg2 << endl;
            }
            
            // Set Gilbert damping if specified
            strain_lattice.alpha_gilbert = config.get_param("alpha_gilbert", 0.0);
            
            // Set magnetic field
            Eigen::Vector3d B;
            B << config.field_strength * config.field_direction[0],
                 config.field_strength * config.field_direction[1],
                 config.field_strength * config.field_direction[2];
            strain_lattice.set_uniform_field(B);
            
            // Initialize spins
            if (!config.initial_spin_config.empty()) {
                strain_lattice.load_spin_config(config.initial_spin_config);
            } else {
                strain_lattice.init_random();
            }
            
            // Run simulation
            switch (config.simulation) {
                case SimulationType::SIMULATED_ANNEALING:
                    run_simulated_annealing_strain(strain_lattice, config, rank, size);
                    break;
                case SimulationType::PARALLEL_TEMPERING:
                    run_parallel_tempering_strain(strain_lattice, config, rank, size, MPI_COMM_WORLD);
                    break;
                case SimulationType::MOLECULAR_DYNAMICS:
                    run_molecular_dynamics_strain(strain_lattice, config, rank, size);
                    break;
                case SimulationType::PUMP_PROBE:
                    run_pump_probe_strain(strain_lattice, config, rank, size);
                    break;
                case SimulationType::KINETIC_BARRIER_ANALYSIS:
                    run_kinetic_barrier_analysis_strain(strain_lattice, config, rank, size);
                    break;
                case SimulationType::PARAMETER_SWEEP:
                    run_parameter_sweep(config, rank, size);
                    break;
                default:
                    if (rank == 0) {
                        cerr << "Simulation type not supported for StrainPhononLattice. "
                             << "Supported: SA, PT, MD, pump_probe, kinetic_barrier, parameter_sweep" << endl;
                    }
                    break;
            }
        } else if (config.system == SystemType::TMFEO3) {
            // Mixed SU(2)+SU(3) system
            if (rank == 0) {
                cout << "\nBuilding TmFeO3 system..." << endl;
            }
            
            auto mixed_uc = build_tmfeo3(config);
            MixedLattice mixed_lattice(mixed_uc, 
                                       config.lattice_size[0],
                                       config.lattice_size[1],
                                       config.lattice_size[2],
                                       config.spin_length,
                                       config.spin_length_su3);
            
            // Initialize spins
            if (config.use_ferromagnetic_init) {
                // Create SU2 direction from config (use general dimension)
                SpinVector dir_su2(mixed_lattice.spin_dim_SU2);
                for (size_t d = 0; d < mixed_lattice.spin_dim_SU2; ++d) {
                    dir_su2(d) = (d < config.ferromagnetic_direction.size()) ? 
                                 config.ferromagnetic_direction[d] : 0.0;
                }
                // Create SU3 direction
                SpinVector dir_su3 = SpinVector::Zero(mixed_lattice.spin_dim_SU3);
                const int su3_init_component = static_cast<int>(config.get_param("su3_init_component", 2.0));
                if (su3_init_component >= 0 && su3_init_component < static_cast<int>(mixed_lattice.spin_dim_SU3)) {
                    dir_su3(su3_init_component) = 1.0;
                } else {
                    dir_su3(2) = 1.0;  // Default to λ3
                }
                mixed_lattice.init_ferromagnetic(dir_su2, dir_su3);
            } else if (!config.initial_spin_config.empty()) {
                mixed_lattice.load_spin_config(config.initial_spin_config);
            } else {
                mixed_lattice.init_random();
            }

            // Gilbert damping (optional; defaults to 0 = undamped)
            mixed_lattice.alpha_gilbert = config.get_param("alpha_gilbert", 0.0);
            if (mixed_lattice.alpha_gilbert > 0.0 && rank == 0) {
                cout << "Gilbert damping: α = " << mixed_lattice.alpha_gilbert << endl;
            }
            
            // Whether to save full spin state trajectories (large, for diagnosis)
            const bool save_spin_trajectories = (config.get_param("save_spin_trajectories", 0.0) > 0.5);

            // Run simulation
            switch (config.simulation) {
                case SimulationType::SIMULATED_ANNEALING:
                    run_simulated_annealing_mixed(mixed_lattice, config, rank, size);
                    break;
                case SimulationType::PARALLEL_TEMPERING:
                    run_parallel_tempering_mixed(mixed_lattice, config, rank, size);
                    break;
                case SimulationType::MOLECULAR_DYNAMICS:
                    run_molecular_dynamics_mixed(mixed_lattice, config, rank, size);
                    break;
                case SimulationType::PUMP_PROBE:
                    run_pump_probe_mixed(mixed_lattice, config, rank, size);
                    break;
                case SimulationType::TWOD_COHERENT_SPECTROSCOPY:
                    run_2dcs_spectroscopy_mixed(mixed_lattice, config, rank, size);
                    break;
                case SimulationType::PARAMETER_SWEEP:
                    run_parameter_sweep(config, rank, size);
                    break;
                default:
                    if (rank == 0) {
                        cerr << "Simulation type not supported for mixed lattice\n";
                    }
                    break;
            }
        } else {
            // Regular SU(2) system
            if (rank == 0) {
                cout << "\nBuilding unit cell..." << endl;
            }
            
            UnitCell* uc_ptr = nullptr;
            switch (config.system) {
                case SystemType::HONEYCOMB_BCAO:
                    uc_ptr = new UnitCell(build_bcao_honeycomb(config));
                    break;
                case SystemType::HONEYCOMB_KITAEV:
                    uc_ptr = new UnitCell(build_kitaev_honeycomb(config));
                    break;
                case SystemType::PYROCHLORE:
                    uc_ptr = new UnitCell(build_pyrochlore(config));
                    break;
                case SystemType::PYROCHLORE_NON_KRAMER:
                    uc_ptr = new UnitCell(build_pyrochlore_non_kramer(config));
                    break;
                case SystemType::TMFEO3_FE:
                    uc_ptr = new UnitCell(build_tmfeo3_fe(config));
                    break;
                case SystemType::TMFEO3_TM:
                    uc_ptr = new UnitCell(build_tmfeo3_tm(config));
                    break;
                default:
                    if (rank == 0) {
                        cerr << "Error: Unknown system type for parameter sweep" << endl;
                    }
                    MPI_Abort(MPI_COMM_WORLD, 1);
                    return 1;
                }
            
            Lattice lattice(*uc_ptr, 
                          config.lattice_size[0],
                          config.lattice_size[1],
                          config.lattice_size[2],
                          config.spin_length);
            lattice.lattice_type = system_type_to_string(config.system);
            
            // Initialize spins
            if (config.use_ferromagnetic_init) {
                // Initialize all spins in same direction (use general spin_dim)
                SpinVector dir(lattice.spin_dim);
                for (size_t d = 0; d < lattice.spin_dim; ++d) {
                    dir(d) = (d < config.ferromagnetic_direction.size()) ? 
                             config.ferromagnetic_direction[d] : 0.0;
                }
                dir.normalize();
                dir *= config.spin_length;
                for (size_t i = 0; i < lattice.lattice_size; ++i) {
                    lattice.spins[i] = dir;
                }
            } else if (!config.initial_spin_config.empty()) {
                lattice.load_spin_config(config.initial_spin_config);
            }
            // else: spins already initialized randomly in constructor
            
            // Run simulation
            switch (config.simulation) {
                case SimulationType::SIMULATED_ANNEALING:
                    run_simulated_annealing(lattice, config, rank, size);
                    break;
                case SimulationType::PARALLEL_TEMPERING:
                    run_parallel_tempering(lattice, config, rank, size);
                    break;
                case SimulationType::MOLECULAR_DYNAMICS:
                    run_molecular_dynamics(lattice, config, rank, size);
                    break;
                case SimulationType::PUMP_PROBE:
                    run_pump_probe(lattice, config, rank, size);
                    break;
                case SimulationType::TWOD_COHERENT_SPECTROSCOPY:
                    run_2dcs_spectroscopy(lattice, config, rank, size);
                    break;
                case SimulationType::PARAMETER_SWEEP:
                    run_parameter_sweep(config, rank, size);
                    break;
                default:
                    if (rank == 0) {
                        cerr << "Simulation type not implemented\n";
                    }
                    break;
            }
        }
    } catch (const exception& e) {
        if (rank == 0) {
            cerr << "Error during simulation: " << e.what() << endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    // Finalize MPI
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized) {
        MPI_Finalize();
    }
    
    if (rank == 0) {
        cout << "\n=== Simulation completed successfully ===" << endl;
    }
    
    return 0;
}

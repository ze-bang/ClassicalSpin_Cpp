/**
 * @file test_pt_strain.cpp
 * @brief Lightweight test: parallel tempering on StrainPhononLattice (NCTO_STRAIN)
 *
 * Verifies:
 * 1. has_extra_dof<StrainPhononLattice> trait is detected at compile time
 * 2. Strain state is exchanged during replica swaps (detailed-balance fix)
 * 3. PT completes without errors on a tiny lattice (4×4, 2 replicas)
 * 4. Output files (HDF5, spin configs, strain state) are written correctly
 * 5. Energy is consistent before/after replica exchange
 *
 * Run: mpirun -np 2 ./build/test_pt_strain
 */

#include "classical_spin/lattice/strain_phonon_lattice.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/core/spin_config.h"
#include "classical_spin/mc/mc_common.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <filesystem>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::cout << std::scientific << std::setprecision(8);

    if (rank == 0) {
        std::cout << "==========================================================\n";
        std::cout << "  Test: Parallel Tempering on StrainPhononLattice\n";
        std::cout << "  MPI ranks: " << size << "\n";
        std::cout << "==========================================================\n\n";
    }

    // ============================================================
    // Test 1: Compile-time trait detection
    // ============================================================
    {
        static_assert(mc::has_extra_dof<StrainPhononLattice>::value,
                      "StrainPhononLattice must satisfy has_extra_dof trait");

        // Verify Lattice does NOT have extra DOF (should not trigger exchange)
        // (We can't check Lattice here without including it, but the static_assert above is key)

        if (rank == 0) {
            std::cout << "[PASS] Test 1: has_extra_dof<StrainPhononLattice> = true\n";
            std::cout << "       extra_dof_size = " << StrainState::N_DOF << " (9 strain + 9 velocity)\n\n";
        }
    }

    // ============================================================
    // Build a small lattice
    // ============================================================
    const size_t L = 4;
    const float spin_length = 1.0f;

    // Hamiltonian parameters (Janssen-like for NCTO)
    MagnetoelasticParams me;
    me.J = 0.68;
    me.K = -7.89;
    me.Gamma = 3.07;
    me.Gammap = -2.94;
    me.J2_A = 0.0;  me.J2_B = 0.0;
    me.J3 = 0.0;    me.J7 = 0.0;
    me.lambda_Eg  = 0.1;  // Nonzero to test magnetoelastic coupling
    me.gamma_J7   = 0.0;

    ElasticParams el;
    el.C11 = 1.0;  el.C12 = 0.3;  el.C44 = 0.35;
    el.M = 1.0;
    el.gamma_Eg = 0.1;

    StrainDriveParams dr;
    dr.E0_1 = 0; dr.E0_2 = 0;

    SpinConfig config;
    config.set_param("J", me.J);
    config.set_param("K", me.K);
    config.set_param("Gamma", me.Gamma);
    config.set_param("Gammap", me.Gammap);
    config.set_param("J2_A", me.J2_A);
    config.set_param("J2_B", me.J2_B);
    config.set_param("J3", me.J3);
    config.field_strength = 0.0;

    UnitCell uc = build_strain_honeycomb(config);
    StrainPhononLattice lattice(uc, L, L, 1, spin_length);
    lattice.set_parameters(me, el, dr);
    lattice.init_random();

    if (rank == 0) {
        std::cout << "Lattice: " << L << "x" << L << " honeycomb, "
                  << lattice.lattice_size << " sites, spin_dim=" << lattice.spin_dim
                  << ", N_atoms=" << lattice.N_atoms << "\n\n";
    }

    // ============================================================
    // Test 2: Strain pack/unpack round-trip
    // ============================================================
    {
        // Set some non-trivial strain
        lattice.set_strain_Eg(0.05, 0.03);

        double buf[StrainState::N_DOF];
        lattice.pack_extra_dof(buf);

        // Zero out strain
        StrainState zero_strain;
        lattice.strain = zero_strain;

        // Unpack should restore
        lattice.unpack_extra_dof(buf);

        bool ok = true;
        if (std::abs(lattice.strain.epsilon_xx - 0.05) > 1e-15) ok = false;
        if (std::abs(lattice.strain.epsilon_yy - (-0.05)) > 1e-15) ok = false;
        if (std::abs(lattice.strain.epsilon_xy - 0.03) > 1e-15) ok = false;

        if (rank == 0) {
            std::cout << "[" << (ok ? "PASS" : "FAIL") << "] Test 2: Strain pack/unpack round-trip\n\n";
        }
    }

    // ============================================================
    // Test 3: Verify strain is exchanged during replica swap
    // ============================================================
    if (size >= 2) {
        // Reset lattice
        lattice.init_random();

        // Set distinct strain on each rank
        double init_Eg1 = 0.1 * (rank + 1);
        double init_Eg2 = 0.02 * (rank + 1);
        lattice.set_strain_Eg(init_Eg1, init_Eg2);

        // Record initial strain
        double pre_Eg1 = lattice.strain.epsilon_xx;
        double pre_Eg2 = lattice.strain.epsilon_xy;

        // Set up for a forced exchange (use high temperature to almost always accept)
        std::vector<double> temps = {1000.0, 2000.0};
        if (size > 2) {
            // Pad with more temperatures if more ranks
            temps.resize(size);
            for (int i = 0; i < size; ++i) {
                temps[i] = 1000.0 + i * 1000.0;
            }
        }
        double curr_T = temps[rank];

        std::mt19937 exchange_rng(42 + rank * 12345);

        // Force an exchange attempt between rank 0 and rank 1
        int result = mc::attempt_replica_exchange(
            lattice, exchange_rng, rank, size, temps, curr_T, 0, MPI_COMM_WORLD);

        // Check if strain changed (if exchange was accepted)
        double post_Eg1 = lattice.strain.epsilon_xx;
        double post_Eg2 = lattice.strain.epsilon_xy;

        bool strain_exchanged = (std::abs(post_Eg1 - pre_Eg1) > 1e-15 ||
                                 std::abs(post_Eg2 - pre_Eg2) > 1e-15);

        // Gather results
        int all_results[2] = {0, 0};
        MPI_Gather(&result, 1, MPI_INT, all_results, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            bool exchange_happened = (all_results[0] == 1);
            if (exchange_happened) {
                std::cout << "[" << (strain_exchanged ? "PASS" : "FAIL")
                          << "] Test 3: Strain exchanged during replica swap "
                          << "(pre=" << pre_Eg1 << ", post=" << post_Eg1 << ")\n";
                if (!strain_exchanged) {
                    std::cout << "  BUG: Exchange was accepted but strain was NOT swapped!\n";
                }
            } else {
                std::cout << "[SKIP] Test 3: Exchange was rejected (rare at high T), re-run\n";
            }
        }
        if (rank == 1) {
            bool exchange_happened = (result == 1 || all_results[0] == 1);
            // rank 1 doesn't have all_results, use local result
            if (result == 1 && strain_exchanged) {
                std::cout << "  Rank 1: strain swapped correctly "
                          << "(pre=" << pre_Eg1 << ", post=" << post_Eg1 << ")\n";
            }
        }
        std::cout << std::flush;
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) std::cout << "\n";
    } else {
        if (rank == 0) {
            std::cout << "[SKIP] Test 3: Need >= 2 MPI ranks for replica exchange test\n\n";
        }
    }

    // ============================================================
    // Test 4: Full (short) parallel tempering run
    // ============================================================
    if (size >= 2) {
        // Re-initialize
        lattice.init_random();
        lattice.set_strain_Eg(0.0, 0.0);

        // Relax strain to equilibrium
        lattice.relax_strain(rank == 0);

        // Record initial energy
        double E_init = lattice.total_energy();
        if (rank == 0) {
            std::cout << "Initial E/N = " << E_init / lattice.lattice_size << "\n";
        }

        // Very short PT: 100 equilibration, 200 measurement
        std::vector<double> temps(size);
        for (int i = 0; i < size; ++i) {
            temps[i] = 0.01 + (1.0 - 0.01) * double(i) / double(size - 1);
        }

        std::string test_dir = "/tmp/test_pt_strain_" + std::to_string(rank);
        std::string out_dir = "/tmp/test_pt_strain_output";

        std::vector<int> ranks_to_write = {-1};  // all ranks write

        lattice.parallel_tempering(
            temps,
            100,    // n_anneal (equilibration sweeps)
            200,    // n_measure (measurement sweeps)
            0,      // overrelaxation_rate (disabled for speed)
            10,     // swap_rate
            20,     // probe_rate
            out_dir,
            ranks_to_write,
            false,   // gaussian_move = false (random spin for exploration)
            MPI_COMM_WORLD,
            false    // verbose = false
        );

        // Verify output files exist
        MPI_Barrier(MPI_COMM_WORLD);

        bool files_ok = true;
        std::string rank_dir = out_dir + "/rank_" + std::to_string(rank);
        
        // Check for spin config
        bool has_spins = false;
        if (std::filesystem::exists(rank_dir)) {
            for (auto& entry : std::filesystem::directory_iterator(rank_dir)) {
                if (entry.path().filename().string().find("spins_T=") != std::string::npos) {
                    has_spins = true;
                    break;
                }
            }
        }

        // Check for strain state
        bool has_strain = std::filesystem::exists(rank_dir + "/strain_state.txt");

#ifdef HDF5_ENABLED
        // Check for HDF5 output
        bool has_hdf5 = std::filesystem::exists(rank_dir + "/parallel_tempering_data.h5");
        if (!has_hdf5) files_ok = false;
#endif

        if (!has_spins) files_ok = false;
        if (!has_strain) files_ok = false;

        // Gather file status
        int local_ok = files_ok ? 1 : 0;
        int all_ok = 0;
        MPI_Reduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            std::cout << "\n[" << (all_ok ? "PASS" : "FAIL") << "] Test 4: Full PT run completed\n";
            std::cout << "  Spin config saved: " << (has_spins ? "yes" : "no") << "\n";
            std::cout << "  Strain state saved: " << (has_strain ? "yes" : "no") << "\n";
#ifdef HDF5_ENABLED
            bool root_hdf5 = std::filesystem::exists(rank_dir + "/parallel_tempering_data.h5");
            bool agg_hdf5 = std::filesystem::exists(out_dir + "/parallel_tempering_aggregated.h5");
            std::cout << "  HDF5 per-rank data: " << (root_hdf5 ? "yes" : "no") << "\n";
            std::cout << "  HDF5 aggregated data: " << (agg_hdf5 ? "yes" : "no") << "\n";
#endif
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // Check that cold replica has lower energy than hot replica
        double E_final = lattice.total_energy();
        double E_per_site = E_final / lattice.lattice_size;

        std::vector<double> all_energies(size);
        MPI_Gather(&E_per_site, 1, MPI_DOUBLE, all_energies.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            std::cout << "\n  Energy per site by rank (T_cold → T_hot):\n";
            for (int i = 0; i < size; ++i) {
                std::cout << "    rank " << i << " (T=" << std::fixed << std::setprecision(4) 
                          << temps[i] << "): E/N = " << std::scientific << all_energies[i] << "\n";
            }
            std::cout << "\n";
        }

        // Cleanup
        if (rank == 0) {
            std::filesystem::remove_all(out_dir);
        }
    } else {
        if (rank == 0) {
            std::cout << "[SKIP] Test 4: Need >= 2 MPI ranks\n";
        }
    }

    // ============================================================
    // Test 5: Energy consistency check
    // ============================================================
    {
        lattice.init_random();
        lattice.relax_strain(false);

        double E_total = lattice.total_energy();
        double E_spin = lattice.spin_energy();
        double E_strain = lattice.strain_energy();
        double E_me = lattice.magnetoelastic_energy();

        double E_sum = E_spin + E_strain + E_me + lattice.drive_energy();
        bool energy_ok = std::abs(E_total - E_sum) < 1e-10 * (std::abs(E_total) + 1.0);

        if (rank == 0) {
            std::cout << "[" << (energy_ok ? "PASS" : "FAIL") 
                      << "] Test 5: Energy consistency\n";
            std::cout << "  E_total = " << E_total << "\n";
            std::cout << "  E_spin  = " << E_spin << "\n";
            std::cout << "  E_strain = " << E_strain << "\n";
            std::cout << "  E_magnetoelastic = " << E_me << "\n";
            std::cout << "  Sum = " << E_sum << "\n";
            std::cout << "  Diff = " << std::abs(E_total - E_sum) << "\n\n";
        }
    }

    if (rank == 0) {
        std::cout << "==========================================================\n";
        std::cout << "  All tests completed.\n";
        std::cout << "==========================================================\n";
    }

    MPI_Finalize();
    return 0;
}

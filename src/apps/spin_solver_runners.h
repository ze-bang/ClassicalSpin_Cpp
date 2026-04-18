#pragma once
/**
 * spin_solver_runners.h — private forward declarations for spin_solver.
 *
 * Before this split, every `run_<simulation>_<lattice>()` function lived in
 * the same 4.9 kLOC translation unit as `main()`. The file is now broken up
 * into one .cpp per lattice family (regular Lattice, PhononLattice,
 * StrainPhononLattice, MixedLattice) plus a dedicated TU for the parameter
 * sweep driver. This header is the one place every runner is declared so
 * the builder (`main`, `run_parameter_sweep`) can call across TUs without
 * re-including all of the big lattice headers.
 *
 * Notes:
 *   - This is an executable-private header. It lives next to spin_solver.cpp
 *     and is not installed, not in include/classical_spin/.
 *   - Only forward declarations live here, so including it stays cheap.
 *     Each runner's .cpp includes the corresponding lattice header.
 */

#include <mpi.h>

// --- Forward declarations -------------------------------------------------

struct SpinConfig;

class Lattice;
class PhononLattice;
class StrainPhononLattice;
class MixedLattice;

struct SpinPhononCouplingParams;
struct PhononParams;
struct DriveParams;
struct TimeDependentSpinPhononParams;

struct MagnetoelasticParams;
struct ElasticParams;
struct StrainDriveParams;

// --- Regular Lattice (SU(2), Heisenberg / Kitaev / BCAO / pyrochlore) -----

void run_simulated_annealing   (Lattice& lattice, const SpinConfig& config,
                                int rank, int size);
void run_parallel_tempering    (Lattice& lattice, const SpinConfig& config,
                                int rank, int size,
                                MPI_Comm comm = MPI_COMM_WORLD);
void run_molecular_dynamics    (Lattice& lattice, const SpinConfig& config,
                                int rank, int size);
void run_pump_probe            (Lattice& lattice, const SpinConfig& config,
                                int rank, int size);
void run_2dcs_spectroscopy     (Lattice& lattice, const SpinConfig& config,
                                int rank, int size);

// --- PhononLattice (spin + discrete phonon modes) -------------------------

void build_phonon_params(const SpinConfig& config,
                         SpinPhononCouplingParams& sp_params,
                         PhononParams& ph_params,
                         DriveParams& dr_params,
                         TimeDependentSpinPhononParams& td_sp_params);

void run_simulated_annealing_phonon (PhononLattice& lattice, const SpinConfig& config,
                                     int rank, int size);
void run_molecular_dynamics_phonon  (PhononLattice& lattice, const SpinConfig& config,
                                     int rank, int size);
void run_pump_probe_phonon          (PhononLattice& lattice, const SpinConfig& config,
                                     int rank, int size);
void run_2dcs_phonon                (PhononLattice& lattice, const SpinConfig& config,
                                     int rank, int size);

// --- StrainPhononLattice (spin + magnetoelastic strain field) -------------

void build_strain_params(const SpinConfig& config,
                         MagnetoelasticParams& me_params,
                         ElasticParams& el_params,
                         StrainDriveParams& dr_params);

void run_simulated_annealing_strain      (StrainPhononLattice& lattice, const SpinConfig& config,
                                          int rank, int size);
void run_molecular_dynamics_strain       (StrainPhononLattice& lattice, const SpinConfig& config,
                                          int rank, int size);
void run_pump_probe_strain               (StrainPhononLattice& lattice, const SpinConfig& config,
                                          int rank, int size);
void run_kinetic_barrier_analysis_strain (StrainPhononLattice& lattice, const SpinConfig& config,
                                          int rank, int size);
void run_parallel_tempering_strain       (StrainPhononLattice& lattice, const SpinConfig& config,
                                          int rank, int size,
                                          MPI_Comm comm = MPI_COMM_WORLD);

// --- MixedLattice (SU(2) + SU(3), TmFeO3) ---------------------------------

void run_simulated_annealing_mixed  (MixedLattice& lattice, const SpinConfig& config,
                                     int rank, int size);
void run_parallel_tempering_mixed   (MixedLattice& lattice, const SpinConfig& config,
                                     int rank, int size,
                                     MPI_Comm comm = MPI_COMM_WORLD);
void run_molecular_dynamics_mixed   (MixedLattice& lattice, const SpinConfig& config,
                                     int rank, int size);
void run_pump_probe_mixed           (MixedLattice& lattice, const SpinConfig& config,
                                     int rank, int size);
void run_2dcs_spectroscopy_mixed    (MixedLattice& lattice, const SpinConfig& config,
                                     int rank, int size);

// --- Parameter sweep driver (multi-lattice dispatcher) --------------------

void run_parameter_sweep(const SpinConfig& base_config, int rank, int size);

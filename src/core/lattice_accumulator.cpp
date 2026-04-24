/**
 * @file lattice_accumulator.cpp
 * @brief Implementation of RealSpaceCorrelationAccumulator.
 *
 * Method bodies for the real-space correlation accumulator were
 * originally defined inline in `lattice.h`, which bloated that header
 * by ~1000 lines and forced every TU including `lattice.h` to also
 * pull in HDF5, MPI reduction code, and large looped spin- and
 * dimer-correlation kernels. They now live here; the struct
 * declaration (data members + trivial inline getters) remains in
 * `lattice.h`.
 */

#include "classical_spin/lattice/lattice.h"

#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#ifdef HDF5_ENABLED
#include <H5Cpp.h>
#endif

// ---- RealSpaceCorrelationAccumulator::initialize ----
    void RealSpaceCorrelationAccumulator::initialize(size_t d1, size_t d2, size_t d3, 
                   size_t n_sub, size_t /* n_bonds */, size_t sdim,
                   const array<Eigen::Vector3d, 3>& lattice_vectors,
                   const vector<Eigen::Vector3d>& sublattice_positions) {
        dim1 = d1;
        dim2 = d2;
        dim3 = d3;
        n_sublattices = n_sub;
        spin_dim = sdim;
        n_sites = d1 * d2 * d3 * n_sub;
        
        // Cell displacements (no sublattice info)
        n_cell_displacements = d1 * d2 * d3;
        
        // Sublattice pairs with symmetry: (i,j) with i ≤ j
        n_sublattice_pairs = n_sub * (n_sub + 1) / 2;
        
        // Bond types = sublattice pairs (bond classified by which sublattices it connects)
        n_bond_types = n_sublattice_pairs;
        
        // Total spin correlation storage: [cell_disp][sub_pair][spin_comp]
        size_t spin_corr_size = n_cell_displacements * n_sublattice_pairs * n_spin_components;
        spin_corr_sum.resize(spin_corr_size, 0.0);
        spin_corr_sq_sum.resize(spin_corr_size, 0.0);
        spin_mean_sum.resize(n_sublattices, Eigen::Vector3d::Zero());
        spin_mean_sq_sum.resize(n_sublattices, Eigen::Vector3d::Zero());
        
        // Dimer correlation storage: [cell_disp][bond_type_mu][bond_type_nu][dimer_comp]
        // = [n_cell_displacements * n_bond_types * n_bond_types * 3]
        size_t dimer_corr_size = n_cell_displacements * n_bond_types * n_bond_types * n_dimer_components;
        dimer_corr_sum.resize(dimer_corr_size, 0.0);
        dimer_corr_sq_sum.resize(dimer_corr_size, 0.0);
        
        // Dimer means: [n_bond_types * n_dimer_components]
        dimer_mean_sum.resize(n_bond_types * n_dimer_components, 0.0);
        dimer_mean_sq_sum.resize(n_bond_types * n_dimer_components, 0.0);
        
        // Build cell displacement vectors and indices
        cell_displacement_vectors.resize(n_cell_displacements);
        cell_displacement_indices.resize(n_cell_displacements);

        for (size_t dn1 = 0; dn1 < d1; ++dn1) {
            for (size_t dn2 = 0; dn2 < d2; ++dn2) {
                for (size_t dn3 = 0; dn3 < d3; ++dn3) {
                    size_t idx = cell_displacement_index(dn1, dn2, dn3);
                    cell_displacement_vectors[idx] =
                        double(dn1) * lattice_vectors[0] +
                        double(dn2) * lattice_vectors[1] +
                        double(dn3) * lattice_vectors[2];
                    cell_displacement_indices[idx] = {int(dn1), int(dn2), int(dn3)};
                }
            }
        }

        // Store sublattice positions for computing full displacement vectors
        sublattice_positions_ = sublattice_positions;

        // Pre-compute displaced_cell_idx[disp_idx * n_cells + cell_i] = cell_j
        // (PBC-shifted cell index). This lookup was previously rebuilt on every
        // accumulate_*_correlations() call (~24 MB allocation per measurement on
        // 12³ unit cells). Caching it here removes the allocation entirely from
        // the measurement path; the table is purely geometric so it's correct
        // to share across samples.
        const size_t n_cells = n_cell_displacements;
        displaced_cell_idx_cache.assign(n_cells * n_cells, 0);
        for (size_t dn1 = 0; dn1 < d1; ++dn1) {
            for (size_t dn2 = 0; dn2 < d2; ++dn2) {
                for (size_t dn3 = 0; dn3 < d3; ++dn3) {
                    const size_t disp_idx = cell_displacement_index(dn1, dn2, dn3);
                    const size_t row_base = disp_idx * n_cells;
                    for (size_t n1 = 0; n1 < d1; ++n1) {
                        const size_t m1 = (n1 + dn1) % d1;
                        for (size_t n2 = 0; n2 < d2; ++n2) {
                            const size_t m2 = (n2 + dn2) % d2;
                            for (size_t n3 = 0; n3 < d3; ++n3) {
                                const size_t m3 = (n3 + dn3) % d3;
                                const size_t cell_i = (n1 * d2 + n2) * d3 + n3;
                                const size_t cell_j = (m1 * d2 + m2) * d3 + m3;
                                displaced_cell_idx_cache[row_base + cell_i] = cell_j;
                            }
                        }
                    }
                }
            }
        }

        // Pre-allocate sublattice-resolved spin scratch buffer (filled per call
        // by accumulate_spin_correlations).
        spins_by_sub_buf.assign(n_sublattices * n_cells, Eigen::Vector3d::Zero());

        n_samples = 0;
        initialized = true;
    }

// ---- RealSpaceCorrelationAccumulator::bond_center ----
    Eigen::Vector3d RealSpaceCorrelationAccumulator::bond_center(size_t bond_type) const {
        auto [sub_i, sub_j] = bond_type_to_sublattices(bond_type);
        return 0.5 * (sublattice_positions_[sub_i] + sublattice_positions_[sub_j]);
    }

// ---- RealSpaceCorrelationAccumulator::accumulate_spin_correlations ----
    void RealSpaceCorrelationAccumulator::accumulate_spin_correlations(
        const vector<Eigen::VectorXd>& spins,
        const function<size_t(size_t)>& /* site_to_sublattice */,
        const function<array<size_t, 3>(size_t)>& /* site_to_cell */)
    {
        if (!initialized) {
            throw std::runtime_error("RealSpaceCorrelationAccumulator not initialized");
        }

        const size_t n_cells = dim1 * dim2 * dim3;
        const double inv_n_cells = 1.0 / double(n_cells);

        // ========== PHASE 1: Pre-compute spin arrays organized by sublattice ==========
        // spins_by_sub_buf[sub * n_cells + cell_idx] holds the 3-vec spin for
        // (cell, sublattice). Reuses the persistent member buffer (was a fresh
        // vector<vector<Eigen::Vector3d>> per call → ~MB of malloc traffic per
        // measurement on a 12³ pyrochlore lattice).
        if (spins_by_sub_buf.size() != n_sublattices * n_cells) {
            spins_by_sub_buf.assign(n_sublattices * n_cells, Eigen::Vector3d::Zero());
        }
        Eigen::Vector3d* const S_buf = spins_by_sub_buf.data();

        std::vector<Eigen::Vector3d> M_sub(n_sublattices, Eigen::Vector3d::Zero());

        for (size_t cell_idx = 0; cell_idx < n_cells; ++cell_idx) {
            for (size_t sub = 0; sub < n_sublattices; ++sub) {
                const size_t site = cell_idx * n_sublattices + sub;
                Eigen::Vector3d S = spins[site].head<3>();
                S_buf[sub * n_cells + cell_idx] = S;
                M_sub[sub] += S;
            }
        }

        // Finalize sublattice magnetizations
        for (size_t sub = 0; sub < n_sublattices; ++sub) {
            M_sub[sub] *= inv_n_cells;
            spin_mean_sum[sub] += M_sub[sub];
            spin_mean_sq_sum[sub] += M_sub[sub].cwiseProduct(M_sub[sub]);
        }

        // ========== PHASE 2: Use cached displacement lookup ==========
        // displaced_cell_idx_cache[disp_idx * n_cells + cell_i] = cell_j
        // (built once in initialize(); see lattice.h struct documentation).
        const size_t* const disp_table = displaced_cell_idx_cache.data();

        // ========== PHASE 3: Accumulate correlations ==========
        for (size_t sub_i = 0; sub_i < n_sublattices; ++sub_i) {
            const Eigen::Vector3d* const S_sub_i = S_buf + sub_i * n_cells;

            for (size_t sub_j = sub_i; sub_j < n_sublattices; ++sub_j) {
                const Eigen::Vector3d* const S_sub_j = S_buf + sub_j * n_cells;
                const size_t sub_pair_idx = sublattice_pair_index(sub_i, sub_j);

                for (size_t cell_disp_idx = 0; cell_disp_idx < n_cells; ++cell_disp_idx) {
                    const size_t* const disp_row = disp_table + cell_disp_idx * n_cells;

                    double corr_xx = 0.0, corr_xy = 0.0, corr_xz = 0.0;
                    double corr_yy = 0.0, corr_yz = 0.0, corr_zz = 0.0;

                    for (size_t cell_i = 0; cell_i < n_cells; ++cell_i) {
                        const size_t cell_j = disp_row[cell_i];
                        const Eigen::Vector3d& Si = S_sub_i[cell_i];
                        const Eigen::Vector3d& Sj = S_sub_j[cell_j];

                        corr_xx += Si[0] * Sj[0];
                        corr_xy += 0.5 * (Si[0] * Sj[1] + Si[1] * Sj[0]);
                        corr_xz += 0.5 * (Si[0] * Sj[2] + Si[2] * Sj[0]);
                        corr_yy += Si[1] * Sj[1];
                        corr_yz += 0.5 * (Si[1] * Sj[2] + Si[2] * Sj[1]);
                        corr_zz += Si[2] * Sj[2];
                    }

                    const size_t base_idx = spin_corr_index(cell_disp_idx, sub_pair_idx, 0);
                    const array<double, 6> corr = {
                        corr_xx * inv_n_cells,
                        corr_xy * inv_n_cells,
                        corr_xz * inv_n_cells,
                        corr_yy * inv_n_cells,
                        corr_yz * inv_n_cells,
                        corr_zz * inv_n_cells
                    };
                    for (size_t c = 0; c < 6; ++c) {
                        spin_corr_sum[base_idx + c]    += corr[c];
                        spin_corr_sq_sum[base_idx + c] += corr[c] * corr[c];
                    }
                }
            }
        }

        n_samples++;
    }

// ---- RealSpaceCorrelationAccumulator::accumulate_dimer_correlations ----
    void RealSpaceCorrelationAccumulator::accumulate_dimer_correlations(
        const vector<Eigen::VectorXd>& spins,
        const vector<array<size_t, 2>>& bonds,
        const vector<size_t>& bond_types,
        const vector<array<size_t, 3>>& bond_cells)
    {
        if (!initialized) {
            throw std::runtime_error("RealSpaceCorrelationAccumulator not initialized");
        }
        
        size_t n_bonds = bonds.size();
        size_t n_cells = dim1 * dim2 * dim3;
        
        // ========== PHASE 1: Compute all dimer operators ==========
        // D^α_b = S_i^α S_j^α for each component
        vector<array<double, 3>> D(n_bonds);
        for (size_t b = 0; b < n_bonds; ++b) {
            size_t i = bonds[b][0];
            size_t j = bonds[b][1];
            const Eigen::Vector3d Si = spins[i].head<3>();
            const Eigen::Vector3d Sj = spins[j].head<3>();
            D[b][0] = Si[0] * Sj[0];  // D^x
            D[b][1] = Si[1] * Sj[1];  // D^y
            D[b][2] = Si[2] * Sj[2];  // D^z
        }
        
        // ========== PHASE 2: Compute mean dimer per bond type ==========
        vector<array<double, 3>> D_mean(n_bond_types, {0.0, 0.0, 0.0});
        vector<size_t> D_count(n_bond_types, 0);
        for (size_t b = 0; b < n_bonds; ++b) {
            size_t type = bond_types[b];
            for (size_t c = 0; c < 3; ++c) {
                D_mean[type][c] += D[b][c];
            }
            D_count[type]++;
        }
        for (size_t t = 0; t < n_bond_types; ++t) {
            if (D_count[t] > 0) {
                double inv_count = 1.0 / double(D_count[t]);
                for (size_t c = 0; c < 3; ++c) {
                    D_mean[t][c] *= inv_count;
                    size_t idx = dimer_mean_index(t, c);
                    dimer_mean_sum[idx] += D_mean[t][c];
                    dimer_mean_sq_sum[idx] += D_mean[t][c] * D_mean[t][c];
                }
            }
        }
        
        // ========== PHASE 3: Group bonds by cell and type ==========
        // bonds_by_cell_type[cell_idx][type] = list of bond indices
        vector<vector<vector<size_t>>> bonds_by_cell_type(
            n_cells, vector<vector<size_t>>(n_bond_types));
        
        for (size_t b = 0; b < n_bonds; ++b) {
            auto& [n1, n2, n3] = bond_cells[b];
            size_t cell_idx = (n1 * dim2 + n2) * dim3 + n3;
            size_t type = bond_types[b];
            bonds_by_cell_type[cell_idx][type].push_back(b);
        }
        
        // ========== PHASE 4: Use cached cell displacement lookup ==========
        // displaced_cell_idx_cache built once in initialize() — was previously
        // a fresh vector<vector<size_t>>(N,N) per call.
        const size_t* const disp_table = displaced_cell_idx_cache.data();

        // ========== PHASE 5: Accumulate dimer-dimer correlations ==========
        // Restructure: loop over bond type pairs FIRST (fewer iterations)
        for (size_t type_mu = 0; type_mu < n_bond_types; ++type_mu) {
            for (size_t type_nu = 0; type_nu < n_bond_types; ++type_nu) {

                // Loop over cell displacements
                for (size_t cell_disp_idx = 0; cell_disp_idx < n_cells; ++cell_disp_idx) {
                    const size_t* const disp_map = disp_table + cell_disp_idx * n_cells;
                    
                    // Correlations for each dimer component
                    array<double, 3> corr = {0.0, 0.0, 0.0};
                    size_t count = 0;
                    
                    // Sum over all cell pairs with this offset
                    for (size_t cell_i = 0; cell_i < n_cells; ++cell_i) {
                        size_t cell_j = disp_map[cell_i];
                        
                        const auto& bonds_mu = bonds_by_cell_type[cell_i][type_mu];
                        const auto& bonds_nu = bonds_by_cell_type[cell_j][type_nu];
                        
                        // All bond pairs of these types
                        for (size_t bi : bonds_mu) {
                            for (size_t bj : bonds_nu) {
                                for (size_t c = 0; c < 3; ++c) {
                                    corr[c] += D[bi][c] * D[bj][c];
                                }
                                count++;
                            }
                        }
                    }
                    
                    if (count > 0) {
                        double inv_count = 1.0 / double(count);
                        for (size_t c = 0; c < 3; ++c) {
                            corr[c] *= inv_count;
                            size_t idx = dimer_corr_index(cell_disp_idx, type_mu, type_nu, c);
                            dimer_corr_sum[idx] += corr[c];
                            dimer_corr_sq_sum[idx] += corr[c] * corr[c];
                        }
                    }
                }
            }
        }
    }

// ---- RealSpaceCorrelationAccumulator::compute_Sq ----
    Eigen::Matrix3d RealSpaceCorrelationAccumulator::compute_Sq(const Eigen::Vector3d& q, bool connected) const {
        if (n_samples == 0) {
            return Eigen::Matrix3d::Zero();
        }
        
        // Use 6 symmetric components, then expand to 3x3
        array<double, 6> Sq_sym = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        
        for (size_t cell_disp_idx = 0; cell_disp_idx < n_cell_displacements; ++cell_disp_idx) {
            for (size_t sub_i = 0; sub_i < n_sublattices; ++sub_i) {
                for (size_t sub_j = sub_i; sub_j < n_sublattices; ++sub_j) {
                    size_t sub_pair_idx = sublattice_pair_index(sub_i, sub_j);
                    
                    // Compute full displacement including sublattice positions
                    Eigen::Vector3d dr = cell_displacement_vectors[cell_disp_idx] 
                                        + sublattice_positions_[sub_j] - sublattice_positions_[sub_i];
                    double phase = q.dot(dr);
                    double cos_phase = std::cos(phase);
                    
                    // Multiplicity: 1 for diagonal (i=j), 2 for off-diagonal (symmetry)
                    double mult = (sub_i == sub_j) ? 1.0 : 2.0;
                    
                    for (size_t c = 0; c < 6; ++c) {
                        size_t idx = spin_corr_index(cell_disp_idx, sub_pair_idx, c);
                        double C = spin_corr_sum[idx] / double(n_samples);
                        
                        // Subtract disconnected part if requested
                        if (connected) {
                            // For symmetric components, need to match symmetrization
                            // Component c corresponds to (α,β) pair
                            // 0=xx, 1=xy, 2=xz, 3=yy, 4=yz, 5=zz
                            Eigen::Vector3d mi = spin_mean_sum[sub_i] / double(n_samples);
                            Eigen::Vector3d mj = spin_mean_sum[sub_j] / double(n_samples);
                            
                            double disc = 0.0;
                            switch(c) {
                                case 0: disc = mi[0] * mj[0]; break;  // xx
                                case 1: disc = 0.5 * (mi[0] * mj[1] + mi[1] * mj[0]); break;  // xy
                                case 2: disc = 0.5 * (mi[0] * mj[2] + mi[2] * mj[0]); break;  // xz
                                case 3: disc = mi[1] * mj[1]; break;  // yy
                                case 4: disc = 0.5 * (mi[1] * mj[2] + mi[2] * mj[1]); break;  // yz
                                case 5: disc = mi[2] * mj[2]; break;  // zz
                            }
                            C -= disc;
                        }
                        
                        Sq_sym[c] += mult * C * cos_phase;
                    }
                }
            }
        }
        
        // Expand to 3x3 symmetric matrix
        Eigen::Matrix3d Sq;
        Sq(0,0) = Sq_sym[0];  // xx
        Sq(0,1) = Sq_sym[1];  Sq(1,0) = Sq_sym[1];  // xy
        Sq(0,2) = Sq_sym[2];  Sq(2,0) = Sq_sym[2];  // xz
        Sq(1,1) = Sq_sym[3];  // yy
        Sq(1,2) = Sq_sym[4];  Sq(2,1) = Sq_sym[4];  // yz
        Sq(2,2) = Sq_sym[5];  // zz
        
        return Sq;
    }

// ---- RealSpaceCorrelationAccumulator::compute_Sq_dimer ----
    array<Eigen::MatrixXd, 3> RealSpaceCorrelationAccumulator::compute_Sq_dimer(const Eigen::Vector3d& q, bool connected) const {
        array<Eigen::MatrixXd, 3> Sq_D;
        for (size_t c = 0; c < 3; ++c) {
            Sq_D[c] = Eigen::MatrixXd::Zero(n_bond_types, n_bond_types);
        }
        
        if (n_samples == 0) {
            return Sq_D;
        }
        
        // Precompute bond centers for each bond type
        vector<Eigen::Vector3d> bond_centers(n_bond_types);
        for (size_t mu = 0; mu < n_bond_types; ++mu) {
            bond_centers[mu] = bond_center(mu);
        }
        
        for (size_t cell_disp_idx = 0; cell_disp_idx < n_cell_displacements; ++cell_disp_idx) {
            for (size_t mu = 0; mu < n_bond_types; ++mu) {
                for (size_t nu = 0; nu < n_bond_types; ++nu) {
                    // Displacement between bond centers: ΔR + center(ν) - center(μ)
                    Eigen::Vector3d dr = cell_displacement_vectors[cell_disp_idx] 
                                        + bond_centers[nu] - bond_centers[mu];
                    double phase = q.dot(dr);
                    double cos_phase = std::cos(phase);
                    
                    for (size_t c = 0; c < 3; ++c) {
                        size_t idx = dimer_corr_index(cell_disp_idx, mu, nu, c);
                        double chi = dimer_corr_sum[idx] / double(n_samples);
                        
                        if (connected) {
                            double D_mu = dimer_mean_sum[dimer_mean_index(mu, c)] / double(n_samples);
                            double D_nu = dimer_mean_sum[dimer_mean_index(nu, c)] / double(n_samples);
                            chi -= D_mu * D_nu;
                        }
                        
                        Sq_D[c](mu, nu) += chi * cos_phase;
                    }
                }
            }
        }
        
        return Sq_D;
    }

// ---- RealSpaceCorrelationAccumulator::compute_Sq_dimer_total ----
    Eigen::MatrixXd RealSpaceCorrelationAccumulator::compute_Sq_dimer_total(const Eigen::Vector3d& q, bool connected) const {
        auto Sq_components = compute_Sq_dimer(q, connected);
        Eigen::MatrixXd Sq_total = Eigen::MatrixXd::Zero(n_bond_types, n_bond_types);
        for (size_t c = 0; c < 3; ++c) {
            Sq_total += Sq_components[c];
        }
        return Sq_total;
    }

// ---- RealSpaceCorrelationAccumulator::merge ----
    void RealSpaceCorrelationAccumulator::merge(const RealSpaceCorrelationAccumulator& other) {
        if (!initialized || !other.initialized) return;
        if (n_cell_displacements != other.n_cell_displacements) {
            throw std::runtime_error("Cannot merge accumulators with different geometry");
        }
        
        for (size_t i = 0; i < spin_corr_sum.size(); ++i) {
            spin_corr_sum[i] += other.spin_corr_sum[i];
            spin_corr_sq_sum[i] += other.spin_corr_sq_sum[i];
        }
        for (size_t i = 0; i < n_sublattices; ++i) {
            spin_mean_sum[i] += other.spin_mean_sum[i];
            spin_mean_sq_sum[i] += other.spin_mean_sq_sum[i];
        }
        for (size_t i = 0; i < dimer_corr_sum.size(); ++i) {
            dimer_corr_sum[i] += other.dimer_corr_sum[i];
            dimer_corr_sq_sum[i] += other.dimer_corr_sq_sum[i];
        }
        for (size_t i = 0; i < n_bond_types; ++i) {
            dimer_mean_sum[i] += other.dimer_mean_sum[i];
            dimer_mean_sq_sum[i] += other.dimer_mean_sq_sum[i];
        }
        n_samples += other.n_samples;
    }

// ---- RealSpaceCorrelationAccumulator::save_hdf5 ----
    void RealSpaceCorrelationAccumulator::save_hdf5(const string& filename, const string& group_name) const {
        try {
            H5::H5File file(filename, H5F_ACC_TRUNC);
            H5::Group group = file.createGroup(group_name);
            
            // Helper lambda for writing scalar attributes
            auto write_attr = [](H5::Group& grp, const string& name, hsize_t val) {
                H5::DataSpace scalar_space(H5S_SCALAR);
                H5::Attribute attr = grp.createAttribute(name, H5::PredType::NATIVE_HSIZE, scalar_space);
                attr.write(H5::PredType::NATIVE_HSIZE, &val);
            };
            
            // Helper lambda for writing 1D double vectors
            auto write_vec = [](H5::Group& grp, const string& name, const vector<double>& vec) {
                hsize_t dims[1] = {vec.size()};
                H5::DataSpace dataspace(1, dims);
                H5::DataSet dataset = grp.createDataSet(name, H5::PredType::NATIVE_DOUBLE, dataspace);
                dataset.write(vec.data(), H5::PredType::NATIVE_DOUBLE);
            };
            
            // Helper lambda for writing 1D int vectors
            auto write_int_vec = [](H5::Group& grp, const string& name, const vector<int>& vec) {
                hsize_t dims[1] = {vec.size()};
                H5::DataSpace dataspace(1, dims);
                H5::DataSet dataset = grp.createDataSet(name, H5::PredType::NATIVE_INT, dataspace);
                dataset.write(vec.data(), H5::PredType::NATIVE_INT);
            };
            
            // Save metadata
            write_attr(group, "n_samples", n_samples);
            write_attr(group, "dim1", dim1);
            write_attr(group, "dim2", dim2);
            write_attr(group, "dim3", dim3);
            write_attr(group, "n_sublattices", n_sublattices);
            write_attr(group, "n_bond_types", n_bond_types);
            write_attr(group, "n_cell_displacements", n_cell_displacements);
            write_attr(group, "n_sublattice_pairs", n_sublattice_pairs);
            write_attr(group, "n_spin_components", n_spin_components);
            write_attr(group, "n_dimer_components", n_dimer_components);
            
            // ========== CELL DISPLACEMENT METADATA ==========
            // Save cell displacement indices: (dn1, dn2, dn3) for each cell offset
            vector<int> cell_disp_indices_flat(n_cell_displacements * 3);
            for (size_t i = 0; i < n_cell_displacements; ++i) {
                cell_disp_indices_flat[i * 3 + 0] = cell_displacement_indices[i][0];
                cell_disp_indices_flat[i * 3 + 1] = cell_displacement_indices[i][1];
                cell_disp_indices_flat[i * 3 + 2] = cell_displacement_indices[i][2];
            }
            write_int_vec(group, "cell_displacement_indices", cell_disp_indices_flat);
            
            // Save cell displacement vectors
            vector<double> cell_disp_flat(n_cell_displacements * 3);
            for (size_t i = 0; i < n_cell_displacements; ++i) {
                cell_disp_flat[i * 3 + 0] = cell_displacement_vectors[i](0);
                cell_disp_flat[i * 3 + 1] = cell_displacement_vectors[i](1);
                cell_disp_flat[i * 3 + 2] = cell_displacement_vectors[i](2);
            }
            write_vec(group, "cell_displacement_vectors", cell_disp_flat);
            
            // ========== SUBLATTICE PAIR MAPPING ==========
            // For each sublattice pair index (also = bond type index), store (sub_i, sub_j)
            // This maps bond type → which sublattices the bond connects
            vector<int> sublattice_pairs(n_sublattice_pairs * 2);
            size_t pair_idx = 0;
            for (size_t sub_i = 0; sub_i < n_sublattices; ++sub_i) {
                for (size_t sub_j = sub_i; sub_j < n_sublattices; ++sub_j) {
                    sublattice_pairs[pair_idx * 2 + 0] = static_cast<int>(sub_i);
                    sublattice_pairs[pair_idx * 2 + 1] = static_cast<int>(sub_j);
                    pair_idx++;
                }
            }
            write_int_vec(group, "sublattice_pairs", sublattice_pairs);
            // Note: bond_types array is same as sublattice_pairs since bond type = sublattice pair
            write_int_vec(group, "bond_types", sublattice_pairs);
            
            // Save sublattice positions
            vector<double> sub_pos_flat(n_sublattices * 3);
            for (size_t i = 0; i < n_sublattices; ++i) {
                sub_pos_flat[i * 3 + 0] = sublattice_positions_[i](0);
                sub_pos_flat[i * 3 + 1] = sublattice_positions_[i](1);
                sub_pos_flat[i * 3 + 2] = sublattice_positions_[i](2);
            }
            write_vec(group, "sublattice_positions", sub_pos_flat);
            
            // Save bond centers: center(μ) = (r_{sub_i} + r_{sub_j})/2
            // Shape: [n_bond_types × 3]
            vector<double> bond_centers_flat(n_bond_types * 3);
            for (size_t mu = 0; mu < n_bond_types; ++mu) {
                Eigen::Vector3d center = bond_center(mu);
                bond_centers_flat[mu * 3 + 0] = center(0);
                bond_centers_flat[mu * 3 + 1] = center(1);
                bond_centers_flat[mu * 3 + 2] = center(2);
            }
            write_vec(group, "bond_centers", bond_centers_flat);
            
            // ========== SPIN CORRELATIONS ==========
            // Shape: [n_cell_displacements][n_sublattice_pairs][6]
            // Component order: xx=0, xy=1, xz=2, yy=3, yz=4, zz=5
            write_vec(group, "spin_corr_sum", spin_corr_sum);
            
            // Save spin means per sublattice
            vector<double> spin_mean_flat(n_sublattices * 3);
            for (size_t i = 0; i < n_sublattices; ++i) {
                spin_mean_flat[i * 3 + 0] = spin_mean_sum[i](0);
                spin_mean_flat[i * 3 + 1] = spin_mean_sum[i](1);
                spin_mean_flat[i * 3 + 2] = spin_mean_sum[i](2);
            }
            write_vec(group, "spin_mean_sum", spin_mean_flat);
            
            // ========== DIMER CORRELATIONS ==========
            // Shape: [n_cell_displacements][n_bond_types][n_bond_types][3]
            // Component order: x=0, y=1, z=2
            write_vec(group, "dimer_corr_sum", dimer_corr_sum);
            
            // Dimer means: [n_bond_types][3]
            write_vec(group, "dimer_mean_sum", dimer_mean_sum);
            
            file.close();
        } catch (const H5::Exception& e) {
            std::cerr << "HDF5 error saving correlations: " << e.getDetailMsg() << std::endl;
        }
    }

// ---- RealSpaceCorrelationAccumulator::save_structure_factor_grid ----
    void RealSpaceCorrelationAccumulator::save_structure_factor_grid(
        const string& filename,
        pair<double, double> q1_range,
        pair<double, double> q2_range,
        pair<double, double> q3_range,
        size_t n_q1, size_t n_q2, size_t n_q3,
        const Eigen::Vector3d& b1,
        const Eigen::Vector3d& b2,
        const Eigen::Vector3d& b3,
        bool connected) const 
    {
        ofstream file(filename);
        if (!file) {
            std::cerr << "Error: Cannot open file " << filename << endl;
            return;
        }
        
        file << "# Spin structure factor S(q) from real-space correlation accumulator\n";
        file << "# n_samples = " << n_samples << "\n";
        file << "# Lattice: " << dim1 << " x " << dim2 << " x " << dim3 
             << " x " << n_sublattices << " sublattices\n";
        file << "# Connected correlator: " << (connected ? "yes" : "no") << "\n";
        file << "# Columns: q1(rlu) q2(rlu) q3(rlu) qx qy qz S_total S_xx S_yy S_zz S_xy S_xz S_yz\n";
        file << std::scientific << std::setprecision(8);
        
        for (size_t i1 = 0; i1 < n_q1; ++i1) {
            double q1 = (n_q1 > 1) ? q1_range.first + (q1_range.second - q1_range.first) * i1 / (n_q1 - 1)
                                   : 0.5 * (q1_range.first + q1_range.second);
            for (size_t i2 = 0; i2 < n_q2; ++i2) {
                double q2 = (n_q2 > 1) ? q2_range.first + (q2_range.second - q2_range.first) * i2 / (n_q2 - 1)
                                       : 0.5 * (q2_range.first + q2_range.second);
                for (size_t i3 = 0; i3 < n_q3; ++i3) {
                    double q3 = (n_q3 > 1) ? q3_range.first + (q3_range.second - q3_range.first) * i3 / (n_q3 - 1)
                                           : 0.5 * (q3_range.first + q3_range.second);
                    
                    // Convert to Cartesian q-vector
                    Eigen::Vector3d q = q1 * b1 + q2 * b2 + q3 * b3;
                    
                    // Compute S(q)
                    Eigen::Matrix3d Sq = compute_Sq(q, connected);
                    double S_total = Sq.trace();
                    
                    file << q1 << " " << q2 << " " << q3 << " "
                         << q(0) << " " << q(1) << " " << q(2) << " "
                         << S_total << " "
                         << Sq(0,0) << " " << Sq(1,1) << " " << Sq(2,2) << " "
                         << Sq(0,1) << " " << Sq(0,2) << " " << Sq(1,2) << "\n";
                }
            }
        }
        
        file.close();
    }

// ---- RealSpaceCorrelationAccumulator::mpi_reduce ----
    void RealSpaceCorrelationAccumulator::mpi_reduce(MPI_Comm comm) {
        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
        
        if (size == 1) return;  // Nothing to reduce
        
        // Flatten spin means for MPI
        vector<double> spin_mean_flat(n_sublattices * 3);
        vector<double> spin_mean_sq_flat(n_sublattices * 3);
        for (size_t i = 0; i < n_sublattices; ++i) {
            for (size_t d = 0; d < 3; ++d) {
                spin_mean_flat[i * 3 + d] = spin_mean_sum[i](d);
                spin_mean_sq_flat[i * 3 + d] = spin_mean_sq_sum[i](d);
            }
        }
        
        // Reduce to rank 0
        if (rank == 0) {
            vector<double> recv_spin(spin_corr_sum.size());
            vector<double> recv_spin_sq(spin_corr_sq_sum.size());
            vector<double> recv_mean(n_sublattices * 3);
            vector<double> recv_mean_sq(n_sublattices * 3);
            vector<double> recv_dimer(dimer_corr_sum.size());
            vector<double> recv_dimer_sq(dimer_corr_sq_sum.size());
            vector<double> recv_dimer_mean(dimer_mean_sum.size());
            vector<double> recv_dimer_mean_sq(dimer_mean_sq_sum.size());
            size_t recv_samples;
            
            for (int src = 1; src < size; ++src) {
                MPI_Recv(&recv_samples, 1, MPI_UNSIGNED_LONG, src, 0, comm, MPI_STATUS_IGNORE);
                MPI_Recv(recv_spin.data(), spin_corr_sum.size(), MPI_DOUBLE, src, 1, comm, MPI_STATUS_IGNORE);
                MPI_Recv(recv_spin_sq.data(), spin_corr_sq_sum.size(), MPI_DOUBLE, src, 2, comm, MPI_STATUS_IGNORE);
                MPI_Recv(recv_mean.data(), n_sublattices * 3, MPI_DOUBLE, src, 3, comm, MPI_STATUS_IGNORE);
                MPI_Recv(recv_mean_sq.data(), n_sublattices * 3, MPI_DOUBLE, src, 4, comm, MPI_STATUS_IGNORE);
                MPI_Recv(recv_dimer.data(), dimer_corr_sum.size(), MPI_DOUBLE, src, 5, comm, MPI_STATUS_IGNORE);
                MPI_Recv(recv_dimer_sq.data(), dimer_corr_sq_sum.size(), MPI_DOUBLE, src, 6, comm, MPI_STATUS_IGNORE);
                MPI_Recv(recv_dimer_mean.data(), dimer_mean_sum.size(), MPI_DOUBLE, src, 7, comm, MPI_STATUS_IGNORE);
                MPI_Recv(recv_dimer_mean_sq.data(), dimer_mean_sq_sum.size(), MPI_DOUBLE, src, 8, comm, MPI_STATUS_IGNORE);
                
                // Add to local
                n_samples += recv_samples;
                for (size_t i = 0; i < spin_corr_sum.size(); ++i) {
                    spin_corr_sum[i] += recv_spin[i];
                    spin_corr_sq_sum[i] += recv_spin_sq[i];
                }
                for (size_t i = 0; i < n_sublattices * 3; ++i) {
                    spin_mean_flat[i] += recv_mean[i];
                    spin_mean_sq_flat[i] += recv_mean_sq[i];
                }
                for (size_t i = 0; i < dimer_corr_sum.size(); ++i) {
                    dimer_corr_sum[i] += recv_dimer[i];
                    dimer_corr_sq_sum[i] += recv_dimer_sq[i];
                }
                for (size_t i = 0; i < dimer_mean_sum.size(); ++i) {
                    dimer_mean_sum[i] += recv_dimer_mean[i];
                    dimer_mean_sq_sum[i] += recv_dimer_mean_sq[i];
                }
            }
            
            // Unflatten spin means back
            for (size_t i = 0; i < n_sublattices; ++i) {
                for (size_t d = 0; d < 3; ++d) {
                    spin_mean_sum[i](d) = spin_mean_flat[i * 3 + d];
                    spin_mean_sq_sum[i](d) = spin_mean_sq_flat[i * 3 + d];
                }
            }
        } else {
            // Send to rank 0
            MPI_Send(&n_samples, 1, MPI_UNSIGNED_LONG, 0, 0, comm);
            MPI_Send(spin_corr_sum.data(), spin_corr_sum.size(), MPI_DOUBLE, 0, 1, comm);
            MPI_Send(spin_corr_sq_sum.data(), spin_corr_sq_sum.size(), MPI_DOUBLE, 0, 2, comm);
            MPI_Send(spin_mean_flat.data(), n_sublattices * 3, MPI_DOUBLE, 0, 3, comm);
            MPI_Send(spin_mean_sq_flat.data(), n_sublattices * 3, MPI_DOUBLE, 0, 4, comm);
            MPI_Send(dimer_corr_sum.data(), dimer_corr_sum.size(), MPI_DOUBLE, 0, 5, comm);
            MPI_Send(dimer_corr_sq_sum.data(), dimer_corr_sq_sum.size(), MPI_DOUBLE, 0, 6, comm);
            MPI_Send(dimer_mean_sum.data(), dimer_mean_sum.size(), MPI_DOUBLE, 0, 7, comm);
            MPI_Send(dimer_mean_sq_sum.data(), dimer_mean_sq_sum.size(), MPI_DOUBLE, 0, 8, comm);
        }
    }


#pragma once
/**
 * mc_common.h — Common Monte Carlo types, analysis, and template algorithms
 *
 * Provides unified structs, statistical analysis functions, and template
 * MC algorithms shared by Lattice, PhononLattice, and StrainPhononLattice.
 *
 * Template algorithms require the lattice type L to provide:
 *   Members:
 *     vector<SpinVector> spins;
 *     size_t lattice_size, spin_dim, N_atoms;
 *     float  spin_length;
 *   Methods:
 *     double total_energy();                    // total energy of current config
 *     double metropolis(double T, bool gaussian_move, double sigma);
 *     void   overrelaxation();
 *     void   deterministic_sweep(size_t num_sweeps);
 *     SpinVector magnetization_global()  const;
 *     vector<SpinVector> magnetization_sublattice() const;
 *     void save_spin_config(const string& filename) const;
 *     void save_positions(const string& filename) const;
 */

#include <vector>
#include <string>
#include <cmath>
#include <ctime>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <functional>
#include <filesystem>
#include <mpi.h>

#include "classical_spin/core/spin_config.h" // should_rank_write

#ifdef HDF5_ENABLED
#include "classical_spin/io/hdf5_io.h"
#include <H5Cpp.h>
#endif

namespace mc {

using std::vector;
using std::string;
using std::cout;
using std::endl;

// ============================================================
// COMMON STRUCTS
// ============================================================

struct BinningResult {
    double mean = 0.0;
    double error = 0.0;
    double tau_int = 1.0;
    size_t optimal_bin_level = 0;
    vector<double> errors_by_level;
};

struct Observable {
    double value = 0.0;
    double error = 0.0;
    Observable(double v = 0.0, double e = 0.0) : value(v), error(e) {}
};

struct VectorObservable {
    vector<double> values;
    vector<double> errors;
    VectorObservable() = default;
    VectorObservable(size_t dim) : values(dim, 0.0), errors(dim, 0.0) {}
};

struct ThermodynamicObservables {
    double temperature = 0.0;
    Observable energy;
    Observable specific_heat;
    VectorObservable magnetization;
    vector<VectorObservable> sublattice_magnetization;
    vector<VectorObservable> energy_sublattice_cross;
};

struct OptimizedTempGridResult {
    vector<double> temperatures;
    vector<double> acceptance_rates;
    vector<double> local_diffusivities;
    vector<double> autocorrelation_times;
    vector<size_t> sweeps_per_temp;
    double mean_acceptance_rate = 0.0;
    double round_trip_estimate  = 0.0;
    size_t feedback_iterations_used = 0;
    bool converged = false;
};

struct AutocorrelationResult {
    double tau_int = 1.0;
    size_t sampling_interval = 100;
    vector<double> correlation_function;
};

// ============================================================
// PURE ANALYSIS FUNCTIONS
// ============================================================

/**
 * Binning analysis (recursive blocking) for error estimation
 * with automatic integrated autocorrelation time detection.
 */
inline BinningResult binning_analysis(const vector<double>& data) {
    BinningResult result;
    if (data.empty()) return result;

    size_t n = data.size();
    result.mean = std::accumulate(data.begin(), data.end(), 0.0) / double(n);

    if (n < 4) {
        double var = 0.0;
        for (double x : data) var += (x - result.mean) * (x - result.mean);
        result.error = std::sqrt(var / (n * (n - 1)));
        return result;
    }

    // Recursive blocking
    vector<double> binned = data;
    size_t max_levels = static_cast<size_t>(std::log2(n)) - 1;
    result.errors_by_level.reserve(max_levels);

    while (binned.size() >= 4) {
        size_t m = binned.size();
        double s = 0.0, s2 = 0.0;
        for (double x : binned) { s += x; s2 += x * x; }
        double mean_l = s / m;
        double var_l  = s2 / m - mean_l * mean_l;
        result.errors_by_level.push_back(std::sqrt(var_l / (m - 1)));

        vector<double> next;
        next.reserve(m / 2);
        for (size_t i = 0; i + 1 < m; i += 2)
            next.push_back(0.5 * (binned[i] + binned[i + 1]));
        binned = std::move(next);
    }

    // Optimal level = level with maximum error (plateau)
    if (result.errors_by_level.size() > 2) {
        double mx = 0.0;
        for (size_t l = 0; l < result.errors_by_level.size(); ++l)
            if (result.errors_by_level[l] > mx) {
                mx = result.errors_by_level[l];
                result.optimal_bin_level = l;
            }
    }

    if (!result.errors_by_level.empty()) {
        size_t use = std::min(result.optimal_bin_level + 1,
                              result.errors_by_level.size() - 1);
        result.error = result.errors_by_level[use];
        if (result.errors_by_level[0] > 1e-20) {
            double ratio = result.error / result.errors_by_level[0];
            result.tau_int = 0.5 * ratio * ratio;
        }
    }
    return result;
}

/**
 * Estimate integrated autocorrelation time using Sokal's
 * self-consistent window method (C = 6).
 */
inline void estimate_autocorrelation_time(const vector<double>& energies,
                                          size_t base_interval,
                                          double& tau_int_out,
                                          size_t& sampling_interval_out) {
    size_t N = energies.size();
    if (N < 10) {
        tau_int_out = 1.0;
        sampling_interval_out = base_interval;
        return;
    }

    double mean = std::accumulate(energies.begin(), energies.end(), 0.0) / N;
    double variance = 0.0;
    for (double e : energies) variance += (e - mean) * (e - mean);
    variance /= N;

    if (variance < 1e-20) {
        tau_int_out = 1.0;
        sampling_interval_out = base_interval;
        return;
    }

    size_t max_lag = std::min(N / 4, size_t(1000));
    constexpr double sokal_C = 6.0;
    tau_int_out = 0.5;
    for (size_t lag = 1; lag < max_lag; ++lag) {
        double corr = 0.0;
        size_t count = N - lag;
        for (size_t i = 0; i < count; ++i)
            corr += (energies[i] - mean) * (energies[i + lag] - mean);
        double rho = corr / (count * variance);
        if (rho < 0.0) break;
        tau_int_out += rho;
        if (static_cast<double>(lag) >= sokal_C * tau_int_out) break;
    }

    if (2.0 * tau_int_out > 0.1 * N)
        cout << "[WARNING] τ_int=" << tau_int_out
             << " large relative to N=" << N << endl;

    size_t tau_sweeps = static_cast<size_t>(std::ceil(tau_int_out)) * base_interval;
    sampling_interval_out = std::max(size_t(2) * tau_sweeps, size_t(100));
}

/**
 * Full autocorrelation computation — richer output than
 * estimate_autocorrelation_time.
 */
inline AutocorrelationResult compute_autocorrelation(
    const vector<double>& energies, size_t base_interval = 10) {
    AutocorrelationResult result;
    size_t N = energies.size();
    if (N < 10) {
        result.sampling_interval = base_interval;
        return result;
    }

    double mean = std::accumulate(energies.begin(), energies.end(), 0.0) / N;
    double variance = 0.0;
    for (double e : energies) variance += (e - mean) * (e - mean);
    variance /= N;

    if (variance < 1e-20) {
        result.sampling_interval = base_interval;
        return result;
    }

    size_t max_lag = std::min(N / 4, size_t(1000));
    result.correlation_function.resize(max_lag, 0.0);

    constexpr double sokal_C = 6.0;
    result.tau_int = 0.5;
    for (size_t lag = 1; lag < max_lag; ++lag) {
        double corr = 0.0;
        size_t count = N - lag;
        for (size_t i = 0; i < count; ++i)
            corr += (energies[i] - mean) * (energies[i + lag] - mean);
        double rho = corr / (count * variance);
        result.correlation_function[lag] = rho;
        if (rho < 0.0) break;
        result.tau_int += rho;
        if (static_cast<double>(lag) >= sokal_C * result.tau_int) break;
    }

    size_t tau_sweeps = static_cast<size_t>(std::ceil(result.tau_int)) * base_interval;
    result.sampling_interval = std::max(size_t(2) * tau_sweeps, size_t(100));
    return result;
}

/**
 * Geometrically-spaced temperature ladder.
 */
inline vector<double> generate_geometric_temperature_ladder(
    double Tmin, double Tmax, size_t R) {
    vector<double> temps(R);
    if (R == 1) { temps[0] = Tmin; return temps; }
    for (size_t i = 0; i < R; ++i)
        temps[i] = Tmin * std::pow(Tmax / Tmin, double(i) / double(R - 1));
    return temps;
}

/**
 * Component-wise binning analysis for vector observables.
 */
template<typename SpinVec>
inline vector<BinningResult> binning_analysis_vector(const vector<SpinVec>& data) {
    if (data.empty()) return {};
    size_t dim = data[0].size();
    vector<BinningResult> results(dim);
    for (size_t d = 0; d < dim; ++d) {
        vector<double> comp(data.size());
        for (size_t i = 0; i < data.size(); ++i) comp[i] = data[i](d);
        results[d] = binning_analysis(comp);
    }
    return results;
}

// ============================================================
// TEMPLATE: THERMODYNAMIC OBSERVABLES
// ============================================================

/**
 * Compute thermodynamic observables with binning error estimates
 * and jackknife specific heat.
 */
template<typename SpinVec>
inline ThermodynamicObservables compute_thermodynamic_observables(
    const vector<double>& energies,
    const vector<vector<SpinVec>>& sublattice_mags,
    double temperature, size_t lattice_size) {

    ThermodynamicObservables obs;
    obs.temperature = temperature;
    double T = temperature;
    size_t n_samples = energies.size();
    if (n_samples == 0) return obs;

    // Energy per site
    vector<double> e_per_site(n_samples);
    for (size_t i = 0; i < n_samples; ++i)
        e_per_site[i] = energies[i] / double(lattice_size);
    BinningResult E_res = binning_analysis(e_per_site);
    obs.energy.value = E_res.mean;
    obs.energy.error = E_res.error;

    // Specific heat via jackknife
    {
        double N2 = double(lattice_size) * double(lattice_size);
        if (n_samples < 2) {
            obs.specific_heat = Observable(0.0, 0.0);
        } else {
            double E_mean = 0.0;
            for (size_t i = 0; i < n_samples; ++i) E_mean += energies[i];
            E_mean /= n_samples;
            double var_E = 0.0;
            for (size_t i = 0; i < n_samples; ++i) {
                double d = energies[i] - E_mean; var_E += d * d;
            }
            var_E /= n_samples;
            obs.specific_heat.value = std::max(0.0, var_E) / (T * T * N2);

            size_t n_jack = std::min(n_samples, size_t(100));
            n_jack = std::max(n_jack, size_t(2));
            size_t block_size = std::max(size_t(1), n_samples / n_jack);
            n_jack = (n_samples + block_size - 1) / block_size;

            vector<double> C_jack(n_jack);
            for (size_t j = 0; j < n_jack; ++j) {
                size_t bs = j * block_size;
                size_t be = std::min((j + 1) * block_size, n_samples);
                double Ej = 0.0; size_t cnt = 0;
                for (size_t i = 0; i < n_samples; ++i)
                    if (i < bs || i >= be) { Ej += energies[i]; ++cnt; }
                if (cnt < 2) { C_jack[j] = obs.specific_heat.value; continue; }
                Ej /= cnt;
                double vj = 0.0;
                for (size_t i = 0; i < n_samples; ++i)
                    if (i < bs || i >= be) { double d = energies[i] - Ej; vj += d * d; }
                vj /= cnt;
                C_jack[j] = std::max(0.0, vj) / (T * T * N2);
            }
            double C_mean = 0.0;
            for (double c : C_jack) C_mean += c;
            C_mean /= n_jack;
            double C_var = 0.0;
            for (double c : C_jack) C_var += (c - C_mean) * (c - C_mean);
            C_var *= double(n_jack - 1) / double(n_jack);
            obs.specific_heat.error = std::sqrt(std::max(0.0, C_var));
        }
    }

    // Sublattice magnetizations
    if (!sublattice_mags.empty() && !sublattice_mags[0].empty()) {
        size_t n_sub = sublattice_mags[0].size();
        size_t sdim  = sublattice_mags[0][0].size();

        obs.sublattice_magnetization.resize(n_sub);
        for (size_t alpha = 0; alpha < n_sub; ++alpha) {
            obs.sublattice_magnetization[alpha] = VectorObservable(sdim);
            for (size_t d = 0; d < sdim; ++d) {
                vector<double> Md(n_samples);
                for (size_t i = 0; i < n_samples; ++i)
                    Md[i] = sublattice_mags[i][alpha](d);
                BinningResult r = binning_analysis(Md);
                obs.sublattice_magnetization[alpha].values[d] = r.mean;
                obs.sublattice_magnetization[alpha].errors[d] = r.error;
            }
        }

        // Total magnetization
        obs.magnetization = VectorObservable(sdim);
        for (size_t d = 0; d < sdim; ++d) {
            vector<double> Mt(n_samples);
            for (size_t i = 0; i < n_samples; ++i) {
                double tot = 0.0;
                for (size_t a = 0; a < n_sub; ++a) tot += sublattice_mags[i][a](d);
                Mt[i] = tot / double(n_sub);
            }
            BinningResult r = binning_analysis(Mt);
            obs.magnetization.values[d] = r.mean;
            obs.magnetization.errors[d] = r.error;
        }

        // Cross term <E · S_α> − <E><S_α>
        double E_mean_total = 0.0;
        for (double E : energies) E_mean_total += E;
        E_mean_total /= n_samples;

        obs.energy_sublattice_cross.resize(n_sub);
        for (size_t alpha = 0; alpha < n_sub; ++alpha) {
            obs.energy_sublattice_cross[alpha] = VectorObservable(sdim);
            for (size_t d = 0; d < sdim; ++d) {
                vector<double> ESd(n_samples);
                for (size_t i = 0; i < n_samples; ++i)
                    ESd[i] = energies[i] * sublattice_mags[i][alpha](d);
                BinningResult ES_res = binning_analysis(ESd);
                double S_mean = obs.sublattice_magnetization[alpha].values[d];
                double cross  = ES_res.mean - E_mean_total * S_mean;

                // Jackknife for cross error
                size_t nj = std::min(n_samples, size_t(100));
                size_t bs = n_samples / nj;
                if (bs == 0) bs = 1;
                nj = n_samples / bs;
                vector<double> cj(nj);
                for (size_t j = 0; j < nj; ++j) {
                    double Es = 0, Ss = 0, ESs = 0; size_t cnt = 0;
                    for (size_t i = 0; i < n_samples; ++i) {
                        if (i / bs != j) {
                            Es  += energies[i];
                            Ss  += sublattice_mags[i][alpha](d);
                            ESs += energies[i] * sublattice_mags[i][alpha](d);
                            ++cnt;
                        }
                    }
                    if (cnt == 0) { cj[j] = cross; continue; }
                    cj[j] = ESs / cnt - (Es / cnt) * (Ss / cnt);
                }
                double cm = 0;
                for (double c : cj) cm += c;
                cm /= nj;
                double cv = 0;
                for (double c : cj) cv += (c - cm) * (c - cm);
                cv *= double(nj - 1) / nj;

                obs.energy_sublattice_cross[alpha].values[d] = cross;
                obs.energy_sublattice_cross[alpha].errors[d] =
                    std::sqrt(std::max(0.0, cv));
            }
        }
    }
    return obs;
}

// ============================================================
// TEMPLATE: MC ALGORITHMS
// ============================================================

/**
 * Greedy quench: iterate deterministic sweeps until energy converges.
 */
template<typename L>
inline void greedy_quench(L& lat, double rel_tol = 1e-12,
                          size_t max_sweeps = 10000) {
    double E_prev = lat.total_energy();
    for (size_t s = 0; s < max_sweeps; ++s) {
        lat.deterministic_sweep(1);
        double E = lat.total_energy();
        double dE = std::abs(E - E_prev);
        if (E_prev != 0.0 && dE / (std::abs(E_prev) + 1e-18) < rel_tol) {
            cout << "Greedy quench converged at sweep " << s + 1
                 << ", E/N = " << E / lat.lattice_size << endl;
            return;
        }
        E_prev = E;
    }
    cout << "Greedy quench reached max sweeps (" << max_sweeps
         << "), E/N = " << E_prev / lat.lattice_size << endl;
}

// ============================================================
// SFINAE trait: detect if lattice L has extra DOF to exchange
// (e.g., strain fields in StrainPhononLattice)
// Required methods: extra_dof_size(), pack_extra_dof(double*), unpack_extra_dof(const double*)
// ============================================================
template<typename T, typename = void>
struct has_extra_dof : std::false_type {};

template<typename T>
struct has_extra_dof<T, std::void_t<
    decltype(std::declval<const T&>().extra_dof_size()),
    decltype(std::declval<const T&>().pack_extra_dof(std::declval<double*>())),
    decltype(std::declval<T&>().unpack_extra_dof(std::declval<const double*>()))
>> : std::true_type {};

/**
 * Attempt replica exchange between neighbouring replicas.
 * Returns 1 if accepted, 0 otherwise.
 *
 * If L provides extra_dof_size() / pack_extra_dof() / unpack_extra_dof(),
 * the extra degrees of freedom (e.g. strain state) are exchanged together
 * with the spin configuration, preserving detailed balance.
 */
template<typename L>
inline int attempt_replica_exchange(
    L& lat, std::mt19937& exchange_rng,
    int rank, int size, const vector<double>& temp,
    double curr_Temp, size_t swap_parity, MPI_Comm comm) {

    int partner = (swap_parity % 2 == 0)
        ? (rank % 2 == 0 ? rank + 1 : rank - 1)
        : (rank % 2 == 1 ? rank + 1 : rank - 1);
    if (partner < 0 || partner >= size) return 0;

    double E = lat.total_energy();
    double E_partner, T_partner = temp[partner];
    MPI_Sendrecv(&E, 1, MPI_DOUBLE, partner, 0,
                 &E_partner, 1, MPI_DOUBLE, partner, 0, comm, MPI_STATUS_IGNORE);

    int accept_int = 0;
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    if (rank < partner) {
        double delta = (1.0 / curr_Temp - 1.0 / T_partner) * (E - E_partner);
        accept_int = (delta >= 0 || uni(exchange_rng) < std::exp(delta)) ? 1 : 0;
    }
    int recv_accept = 0;
    MPI_Sendrecv(&accept_int, 1, MPI_INT, partner, 2,
                 &recv_accept, 1, MPI_INT, partner, 2, comm, MPI_STATUS_IGNORE);
    bool accepted = (rank < partner) ? (accept_int == 1) : (recv_accept == 1);

    if (accepted) {
        // Exchange spin configurations
        size_t buf_len = lat.lattice_size * lat.spin_dim;
        vector<double> sendbuf(buf_len), recvbuf(buf_len);
        for (size_t i = 0; i < lat.lattice_size; ++i)
            for (size_t j = 0; j < lat.spin_dim; ++j)
                sendbuf[i * lat.spin_dim + j] = lat.spins[i](j);
        MPI_Sendrecv(sendbuf.data(), buf_len, MPI_DOUBLE, partner, 1,
                     recvbuf.data(), buf_len, MPI_DOUBLE, partner, 1,
                     comm, MPI_STATUS_IGNORE);
        for (size_t i = 0; i < lat.lattice_size; ++i)
            for (size_t j = 0; j < lat.spin_dim; ++j)
                lat.spins[i](j) = recvbuf[i * lat.spin_dim + j];

        // Exchange extra DOF (e.g. strain state) if the lattice provides them
        if constexpr (has_extra_dof<L>::value) {
            size_t extra_n = lat.extra_dof_size();
            if (extra_n > 0) {
                vector<double> extra_send(extra_n), extra_recv(extra_n);
                lat.pack_extra_dof(extra_send.data());
                MPI_Sendrecv(extra_send.data(), extra_n, MPI_DOUBLE, partner, 3,
                             extra_recv.data(), extra_n, MPI_DOUBLE, partner, 3,
                             comm, MPI_STATUS_IGNORE);
                lat.unpack_extra_dof(extra_recv.data());
            }
        }
    }
    return accepted ? 1 : 0;
}

// Forward declarations for HDF5 functions used by gather_and_save_statistics_comprehensive
#ifdef HDF5_ENABLED
template<typename SpinVec>
inline void save_thermodynamic_observables_hdf5(
    const string& out_dir,
    double temperature, size_t lattice_size, size_t spin_dim, size_t N_atoms,
    const ThermodynamicObservables& obs,
    const vector<double>& energies,
    const vector<SpinVec>& magnetizations,
    const vector<vector<SpinVec>>& sublattice_mags,
    size_t n_anneal, size_t n_measure,
    size_t probe_rate, size_t swap_rate,
    size_t overrelaxation_rate,
    double acceptance_rate, double swap_acceptance_rate);

inline void save_heat_capacity_hdf5(
    const string& out_dir,
    const vector<double>& temperatures,
    const vector<double>& heat_capacity,
    const vector<double>& dHeat);
#endif

/**
 * Gather thermodynamic statistics across replicas, save HDF5 + configs.
 * ExtraSaveFn(rank_dir, curr_Temp) is called for lattice-specific saves.
 */
template<typename L,
         typename ExtraSaveFn = std::function<void(const string&, double)>>
inline void gather_and_save_statistics_comprehensive(
    L& lat, int rank, int size, double curr_Temp,
    const vector<double>& energies,
    const vector<std::decay_t<decltype(lat.spins[0])>>& magnetizations,
    const vector<vector<std::decay_t<decltype(lat.spins[0])>>>& sublattice_mags,
    vector<double>& heat_capacity, vector<double>& dHeat,
    const vector<double>& temp, const string& dir_name,
    const vector<int>& rank_to_write,
    size_t n_anneal, size_t n_measure,
    double curr_accept, int swap_accept,
    size_t swap_rate, size_t overrelaxation_rate,
    size_t probe_rate, MPI_Comm comm,
    bool verbose = false,
    ExtraSaveFn extra_save = [](const string&, double){}) {

    using SpinVec = std::decay_t<decltype(lat.spins[0])>;
    auto obs = compute_thermodynamic_observables<SpinVec>(
        energies, sublattice_mags, curr_Temp, lat.lattice_size);

    double c_val = obs.specific_heat.value, c_err = obs.specific_heat.error;
    MPI_Gather(&c_val, 1, MPI_DOUBLE, heat_capacity.data(), 1, MPI_DOUBLE, 0, comm);
    MPI_Gather(&c_err, 1, MPI_DOUBLE, dHeat.data(),         1, MPI_DOUBLE, 0, comm);

    double total_steps = n_anneal + n_measure;
    double metro_steps = (overrelaxation_rate > 0) ? total_steps / overrelaxation_rate : total_steps;
    double acc_rate    = curr_accept / metro_steps;
    double swap_act    = (swap_rate > 0) ? double(swap_accept) / (total_steps / swap_rate) : 0.0;

    cout << "Rank " << rank << ": T=" << curr_Temp
         << ", acc=" << acc_rate
         << ", swap_acc=" << swap_act
         << ", <E>/N=" << obs.energy.value << "±" << obs.energy.error
         << ", C_V=" << obs.specific_heat.value << "±" << obs.specific_heat.error
         << endl;

    if (!dir_name.empty()) {
        if (rank == 0) std::filesystem::create_directories(dir_name);
        MPI_Barrier(comm);

        if (should_rank_write(rank, rank_to_write)) {
            string rank_dir = dir_name + "/rank_" + std::to_string(rank);
            std::filesystem::create_directories(rank_dir);

#ifdef HDF5_ENABLED
            save_thermodynamic_observables_hdf5<SpinVec>(
                rank_dir, curr_Temp, lat.lattice_size, lat.spin_dim, lat.N_atoms,
                obs, energies, magnetizations, sublattice_mags,
                n_anneal, n_measure, probe_rate, swap_rate,
                overrelaxation_rate, acc_rate, swap_act);
#endif
            lat.save_spin_config(rank_dir + "/spins_T=" +
                                 std::to_string(curr_Temp) + ".txt");
            lat.save_positions(rank_dir + "/positions.txt");
            extra_save(rank_dir, curr_Temp);
        }

        MPI_Barrier(comm);
        if (rank == 0) {
#ifdef HDF5_ENABLED
            save_heat_capacity_hdf5(dir_name, temp, heat_capacity, dHeat);
#endif
        }
    }
    MPI_Barrier(comm);
}

/**
 * Parallel tempering with Bittner-adaptive sweep scheduling.
 * ExtraSaveFn(rank_dir, T) is called per-rank for lattice-specific saves.
 */
template<typename L,
         typename ExtraSaveFn = std::function<void(const string&, double)>>
inline void parallel_tempering(
    L& lat, vector<double> temp, size_t n_anneal, size_t n_measure,
    size_t overrelaxation_rate, size_t swap_rate, size_t probe_rate,
    string dir_name, const vector<int>& rank_to_write,
    bool gaussian_move = true, MPI_Comm comm = MPI_COMM_WORLD,
    bool verbose = false,
    const vector<size_t>& sweeps_per_temp = {},
    ExtraSaveFn extra_save = [](const string&, double){}) {

    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) MPI_Init(nullptr, nullptr);

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Exchange-decision RNG (separate from lattice's MC RNG)
    std::mt19937 exchange_rng(
        static_cast<unsigned>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count()
            + rank * 12345));

    double curr_Temp = temp[rank];
    double sigma     = 1000.0;
    int    swap_accept = 0;
    double curr_accept = 0;

    bool   use_adaptive = !sweeps_per_temp.empty()
                          && sweeps_per_temp.size() >= static_cast<size_t>(size);
    size_t eff_swap = swap_rate;
    if (use_adaptive) {
        eff_swap = *std::max_element(sweeps_per_temp.begin(),
                                     sweeps_per_temp.end());
        if (rank == 0)
            cout << "Bittner adaptive sweep schedule: exchange freq = "
                 << eff_swap << endl;
    }

    vector<double> heat_cap, dHeat;
    if (rank == 0) { heat_cap.resize(size); dHeat.resize(size); }

    using SpinVec = std::decay_t<decltype(lat.spins[0])>;
    vector<double>           energies;
    vector<SpinVec>          magnetizations;
    vector<vector<SpinVec>>  sublattice_mags;
    size_t expected = n_measure / probe_rate + 100;
    energies.reserve(expected);
    magnetizations.reserve(expected);
    sublattice_mags.reserve(expected);

    cout << "Rank " << rank << ": T=" << curr_Temp << endl;

    // --- Equilibration ---
    cout << "Rank " << rank << ": Equilibrating..." << endl;
    for (size_t i = 0; i < n_anneal; ++i) {
        if (overrelaxation_rate > 0) {
            lat.overrelaxation();
            if (i % overrelaxation_rate == 0)
                curr_accept += lat.metropolis(curr_Temp, gaussian_move, sigma);
        } else {
            curr_accept += lat.metropolis(curr_Temp, gaussian_move, sigma);
        }
        if (eff_swap > 0 && i % eff_swap == 0)
            swap_accept += attempt_replica_exchange(lat, exchange_rng,
                rank, size, temp, curr_Temp, i / eff_swap, comm);
    }

    // --- Pilot autocorrelation ---
    {
        size_t pilot_n  = std::min(size_t(5000), n_measure / 5);
        size_t pilot_iv = std::max(size_t(1), std::min(probe_rate, size_t(10)));
        vector<double> pilot_e;
        pilot_e.reserve(pilot_n / pilot_iv + 1);
        for (size_t i = 0; i < pilot_n; ++i) {
            if (overrelaxation_rate > 0) {
                lat.overrelaxation();
                if (i % overrelaxation_rate == 0)
                    lat.metropolis(curr_Temp, gaussian_move, sigma);
            } else {
                lat.metropolis(curr_Temp, gaussian_move, sigma);
            }
            if (eff_swap > 0 && i % eff_swap == 0)
                attempt_replica_exchange(lat, exchange_rng,
                    rank, size, temp, curr_Temp, i / eff_swap, comm);
            if (i % pilot_iv == 0) pilot_e.push_back(lat.total_energy());
        }
        double ptau; size_t psamp;
        estimate_autocorrelation_time(pilot_e, pilot_iv, ptau, psamp);
        size_t tau_sw   = static_cast<size_t>(std::ceil(ptau)) * pilot_iv;
        size_t min_rate = 2 * tau_sw;

        cout << "Rank " << rank << ": τ_int=" << ptau
             << " (=" << tau_sw << " sweeps), recommended probe_rate ≥ "
             << min_rate << endl;
        if (probe_rate < min_rate)
            cout << "[WARNING] Rank " << rank << ": probe_rate=" << probe_rate
                 << " < 2·τ_int=" << min_rate
                 << ". Samples will be correlated!" << endl;
    }

    // --- Measurement ---
    cout << "Rank " << rank << ": Measuring..." << endl;
    for (size_t i = 0; i < n_measure; ++i) {
        if (overrelaxation_rate > 0) {
            lat.overrelaxation();
            if (i % overrelaxation_rate == 0)
                curr_accept += lat.metropolis(curr_Temp, gaussian_move, sigma);
        } else {
            curr_accept += lat.metropolis(curr_Temp, gaussian_move, sigma);
        }
        if (eff_swap > 0 && i % eff_swap == 0)
            swap_accept += attempt_replica_exchange(lat, exchange_rng,
                rank, size, temp, curr_Temp, i / eff_swap, comm);
        if (i % probe_rate == 0) {
            energies.push_back(lat.total_energy());
            magnetizations.push_back(lat.magnetization_global());
            sublattice_mags.push_back(lat.magnetization_sublattice());
        }
    }
    cout << "Rank " << rank << ": Collected " << energies.size()
         << " samples" << endl;

    gather_and_save_statistics_comprehensive(
        lat, rank, size, curr_Temp, energies, magnetizations, sublattice_mags,
        heat_cap, dHeat, temp, dir_name, rank_to_write,
        n_anneal, n_measure, curr_accept, swap_accept,
        swap_rate, overrelaxation_rate, probe_rate, comm, verbose, extra_save);
}

/**
 * Gradient-based temperature grid optimization (Miyata et al., 2024).
 * Minimizes variance of acceptance rates with reparameterization L_k = log(Δβ_k)
 * so that β ordering is strictly preserved. Uses score-function gradient estimation.
 * Ref: Miyata et al., "Refined Gradient-Based Temperature Optimization for the
 * Replica-Exchange Monte-Carlo Method" (arXiv:2601.13542).
 */
template<typename Lat>
inline OptimizedTempGridResult generate_optimized_temperature_grid_mpi_gradient(
    Lat& lat, double Tmin, double Tmax,
    size_t warmup_sweeps   = 500,
    size_t sweeps_per_iter = 500,
    size_t gradient_iters  = 50,
    bool   gaussian_move   = false,
    size_t overrelaxation_rate = 0,
    double learning_rate   = 0.1,
    double convergence_tol = 1e-4,
    MPI_Comm comm = MPI_COMM_WORLD) {

    int rank, R;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &R);

    OptimizedTempGridResult result;
    if (R < 2)  { result.temperatures = {Tmin}; result.converged = true; return result; }
    if (R == 2) { result.temperatures = {Tmin, Tmax};
                  result.acceptance_rates = {0.5}; result.converged = true; return result; }

    std::mt19937 exchange_rng(
        static_cast<unsigned>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count()
            + rank * 12345));

    if (rank == 0) {
        cout << "=== Gradient-based Temperature Grid (MPI) ===" << endl;
        cout << "Ref: Miyata et al., arXiv:2601.13542 (2024)" << endl;
        cout << "T_min = " << Tmin << ", T_max = " << Tmax << ", R = " << R << endl;
    }

    double beta_min = 1.0 / Tmax, beta_max = 1.0 / Tmin;
    double beta_range = beta_max - beta_min;

    // Softmax parameterization: log_delta[0..R-2], Δβ_k = beta_range * exp(log_delta_k) / sum(exp(log_delta))
    // Ensures monotonicity and positive intervals (Miyata et al. reparameterization).
    vector<double> log_delta(R - 1, 0.0);  // uniform spacing initially
    vector<double> beta(R);

    auto L_to_beta = [&]() {
        double sum_exp = 0;
        for (size_t k = 0; k < R - 1; ++k) sum_exp += std::exp(log_delta[k]);
        double scale = beta_range / (sum_exp + 1e-20);
        beta[0] = beta_min;
        for (size_t k = 0; k < R - 1; ++k) {
            beta[k + 1] = beta[k] + scale * std::exp(log_delta[k]);
        }
        beta[R - 1] = beta_max;
    };
    L_to_beta();

    double my_beta = beta[rank], my_T = 1.0 / my_beta;
    double sigma = 1000.0;

    if (rank == 0) cout << "Warming up replicas..." << endl;
    for (size_t i = 0; i < warmup_sweeps; ++i) {
        lat.metropolis(my_T, gaussian_move, sigma);
        if (overrelaxation_rate > 0 && i % overrelaxation_rate == 0) lat.overrelaxation();
    }
    MPI_Barrier(comm);

    vector<double> acceptance_rates(R - 1, 0.0);

    for (size_t iter = 0; iter < gradient_iters; ++iter) {
        double sum_energy = 0.0;
        size_t n_steps = 0;
        int attempts_low = 0, accepts_low = 0;
        double sum_E_accept_low = 0.0;
        int attempts_high = 0, accepts_high = 0;
        double sum_E_accept_high = 0.0;

        for (size_t sw = 0; sw < sweeps_per_iter; ++sw) {
            lat.metropolis(my_T, gaussian_move, sigma);
            if (overrelaxation_rate > 0 && sw % overrelaxation_rate == 0)
                lat.overrelaxation();
            double E = lat.total_energy();
            sum_energy += E;
            ++n_steps;

            for (int parity = 0; parity <= 1; ++parity) {
                int partner = (parity == 0)
                    ? (rank % 2 == 0 ? rank + 1 : rank - 1)
                    : (rank % 2 == 1 ? rank + 1 : rank - 1);
                if (partner < 0 || partner >= R) continue;

                double partner_E;
                MPI_Sendrecv(&E, 1, MPI_DOUBLE, partner, 0,
                             &partner_E, 1, MPI_DOUBLE, partner, 0,
                             comm, MPI_STATUS_IGNORE);

                int acc = 0;
                std::uniform_real_distribution<double> uni(0.0, 1.0);
                if (rank < partner) {
                    double delta = -(beta[partner] - my_beta) * (E - partner_E);
                    acc = (delta >= 0 || uni(exchange_rng) < std::exp(delta)) ? 1 : 0;
                    ++attempts_high;
                    if (acc) { accepts_high++; sum_E_accept_high += E; }
                } else {
                    double delta = -(my_beta - beta[partner]) * (partner_E - E);
                    acc = (delta >= 0 || uni(exchange_rng) < std::exp(delta)) ? 1 : 0;
                    ++attempts_low;
                    if (acc) { accepts_low++; sum_E_accept_low += E; }
                }
                int racc = 0;
                MPI_Sendrecv(&acc, 1, MPI_INT, partner, 1,
                             &racc, 1, MPI_INT, partner, 1, comm, MPI_STATUS_IGNORE);
                bool accepted = (rank < partner) ? (acc == 1) : (racc == 1);
                if (accepted) {
                    size_t bl = lat.lattice_size * lat.spin_dim;
                    vector<double> sb(bl), rb(bl);
                    for (size_t i = 0; i < lat.lattice_size; ++i)
                        for (size_t j = 0; j < lat.spin_dim; ++j)
                            sb[i * lat.spin_dim + j] = lat.spins[i](j);
                    MPI_Sendrecv(sb.data(), bl, MPI_DOUBLE, partner, 2,
                                 rb.data(), bl, MPI_DOUBLE, partner, 2,
                                 comm, MPI_STATUS_IGNORE);
                    for (size_t i = 0; i < lat.lattice_size; ++i)
                        for (size_t j = 0; j < lat.spin_dim; ++j)
                            lat.spins[i](j) = rb[i * lat.spin_dim + j];
                    if constexpr (has_extra_dof<Lat>::value) {
                        size_t extra_n = lat.extra_dof_size();
                        if (extra_n > 0) {
                            vector<double> es(extra_n), er(extra_n);
                            lat.pack_extra_dof(es.data());
                            MPI_Sendrecv(es.data(), extra_n, MPI_DOUBLE, partner, 3,
                                         er.data(), extra_n, MPI_DOUBLE, partner, 3,
                                         comm, MPI_STATUS_IGNORE);
                            lat.unpack_extra_dof(er.data());
                        }
                    }
                }
            }
        }

        double mean_energy_j = (n_steps > 0) ? (sum_energy / n_steps) : 0.0;
        vector<double> recv_mean_energy(R);
        vector<int> recv_attempts_low(R), recv_accepts_low(R);
        vector<int> recv_attempts_high(R), recv_accepts_high(R);
        vector<double> recv_sum_E_low(R), recv_sum_E_high(R);
        MPI_Gather(&mean_energy_j, 1, MPI_DOUBLE, recv_mean_energy.data(), 1, MPI_DOUBLE, 0, comm);
        MPI_Gather(&attempts_low, 1, MPI_INT, recv_attempts_low.data(), 1, MPI_INT, 0, comm);
        MPI_Gather(&accepts_low, 1, MPI_INT, recv_accepts_low.data(), 1, MPI_INT, 0, comm);
        MPI_Gather(&attempts_high, 1, MPI_INT, recv_attempts_high.data(), 1, MPI_INT, 0, comm);
        MPI_Gather(&accepts_high, 1, MPI_INT, recv_accepts_high.data(), 1, MPI_INT, 0, comm);
        MPI_Gather(&sum_E_accept_low, 1, MPI_DOUBLE, recv_sum_E_low.data(), 1, MPI_DOUBLE, 0, comm);
        MPI_Gather(&sum_E_accept_high, 1, MPI_DOUBLE, recv_sum_E_high.data(), 1, MPI_DOUBLE, 0, comm);

        bool converged = false;
        if (rank == 0) {
            vector<double> E_A(R - 1), E_HA_hot(R - 1), E_HA_cold(R - 1);
            vector<double> H_mean(R);
            for (int j = 0; j < R; ++j) H_mean[j] = recv_mean_energy[j];
            for (int e = 0; e < R - 1; ++e) {
                int att = recv_attempts_high[e];
                E_A[e] = (att > 0) ? (double(recv_accepts_high[e]) / att) : 0.5;
                E_HA_hot[e] = (att > 0 && recv_accepts_high[e] > 0)
                    ? (recv_sum_E_high[e] / recv_accepts_high[e]) : H_mean[e];
                int att_c = recv_attempts_low[e + 1];
                E_HA_cold[e] = (att_c > 0 && recv_accepts_low[e + 1] > 0)
                    ? (recv_sum_E_low[e + 1] / recv_accepts_low[e + 1]) : H_mean[e + 1];
            }
            double A_bar = 0;
            for (int e = 0; e < R - 1; ++e) A_bar += E_A[e];
            A_bar /= (R - 1);

            double loss = 0;
            for (int e = 0; e < R - 1; ++e) loss += (E_A[e] - A_bar) * (E_A[e] - A_bar);
            loss /= (R - 1);
            acceptance_rates.assign(E_A.begin(), E_A.end());

            vector<double> df_dbeta(R, 0.0);
            for (int j = 1; j < R - 1; ++j) {
                double t1 = (E_A[j - 1] - A_bar) * (H_mean[j] * E_A[j - 1] - E_HA_cold[j - 1]);
                double t2 = (E_A[j] - A_bar) * (H_mean[j] * E_A[j] - E_HA_hot[j]);
                df_dbeta[j] = (2.0 / (R - 1)) * (t1 + t2);
            }
            df_dbeta[0] = (2.0 / (R - 1)) * (E_A[0] - A_bar) * (H_mean[0] * E_A[0] - E_HA_hot[0]);
            df_dbeta[R - 1] = (2.0 / (R - 1)) * (E_A[R - 2] - A_bar) * (H_mean[R - 1] * E_A[R - 2] - E_HA_cold[R - 2]);

            double sum_exp = 0;
            for (size_t k = 0; k < R - 1; ++k) sum_exp += std::exp(log_delta[k]);
            double S = sum_exp + 1e-20;
            vector<double> dbeta_dL(R - 1);
            for (size_t k = 0; k < R - 1; ++k)
                dbeta_dL[k] = beta_range * std::exp(log_delta[k]) / S;

            vector<double> grad_L(R - 1, 0.0);
            for (size_t k = 0; k < R - 1; ++k) {
                for (int j = 1; j < R; ++j) {
                    double dbeta_j_dLk = 0;
                    if (k < j) {
                        double del = (beta[j] - beta_min) / (beta_range + 1e-20);
                        dbeta_j_dLk = dbeta_dL[k] * (1.0 - del);
                    } else {
                        double del = (beta[j] - beta_min) / (beta_range + 1e-20);
                        dbeta_j_dLk = -dbeta_dL[k] * del;
                    }
                    grad_L[k] += df_dbeta[j] * dbeta_j_dLk;
                }
            }

            double lr = learning_rate * (1.0 - 0.8 * double(iter) / gradient_iters);
            for (size_t k = 0; k < R - 1; ++k) {
                log_delta[k] -= lr * grad_L[k];
                log_delta[k] = std::max(-10.0, std::min(10.0, log_delta[k]));
            }

            cout << "Iter " << iter + 1 << "/" << gradient_iters
                 << ": loss = " << std::scientific << loss
                 << ", mean A = " << std::fixed << std::setprecision(3) << A_bar << endl;
            result.feedback_iterations_used = iter + 1;
            if (loss < convergence_tol) {
                converged = true;
                cout << "Converged at iteration " << iter + 1 << endl;
            }
        }

        int ci = converged ? 1 : 0;
        MPI_Bcast(&ci, 1, MPI_INT, 0, comm);
        MPI_Bcast(log_delta.data(), static_cast<int>(R - 1), MPI_DOUBLE, 0, comm);
        L_to_beta();  // all ranks compute beta from log_delta
        my_beta = beta[rank];
        my_T = 1.0 / my_beta;
        if (ci) { result.converged = true; break; }
    }

    MPI_Bcast(acceptance_rates.data(), R - 1, MPI_DOUBLE, 0, comm);

    if (rank == 0) cout << "\nMeasuring τ for adaptive sweep schedule..." << endl;
    size_t tau_n = std::max(size_t(500), sweeps_per_iter);
    vector<double> eseries;
    eseries.reserve(tau_n);
    for (size_t i = 0; i < tau_n; ++i) {
        lat.metropolis(my_T, gaussian_move, sigma);
        if (overrelaxation_rate > 0 && i % overrelaxation_rate == 0) lat.overrelaxation();
        eseries.push_back(lat.total_energy());
    }
    double my_tau = 1.0; size_t dummy;
    estimate_autocorrelation_time(eseries, 1, my_tau, dummy);
    my_tau = std::max(1.0, my_tau);

    vector<double> all_tau(R);
    MPI_Allgather(&my_tau, 1, MPI_DOUBLE, all_tau.data(), 1, MPI_DOUBLE, comm);
    double tau_min_v = *std::min_element(all_tau.begin(), all_tau.end());
    size_t n_base = 10;
    result.autocorrelation_times = all_tau;
    result.sweeps_per_temp.resize(R);
    for (int k = 0; k < R; ++k)
        result.sweeps_per_temp[k] = std::max(size_t(1),
            static_cast<size_t>(std::ceil(n_base * all_tau[k] / tau_min_v)));

    result.temperatures.resize(R);
    for (int i = 0; i < R; ++i) result.temperatures[i] = 1.0 / beta[i];
    std::sort(result.temperatures.begin(), result.temperatures.end());
    result.acceptance_rates = acceptance_rates;
    result.local_diffusivities.resize(R - 1);
    for (int e = 0; e < R - 1; ++e) {
        double A = acceptance_rates[e];
        result.local_diffusivities[e] = A * (1.0 - A);
    }
    result.mean_acceptance_rate = 0;
    for (double A : acceptance_rates) result.mean_acceptance_rate += A;
    result.mean_acceptance_rate /= (R - 1);
    double sum_inv_f = 0, total_current = 0;
    for (int e = 0; e < R - 1; ++e) {
        double db = std::abs(beta[e + 1] - beta[e]);
        total_current += std::max(acceptance_rates[e], 1e-6) * db;
    }
    for (int e = 0; e < R - 1; ++e) {
        double db = std::abs(beta[e + 1] - beta[e]);
        double A = std::max(acceptance_rates[e], 1e-6);
        double f_i = A * db / total_current;
        double n_avg = 0.5 * (result.sweeps_per_temp[e] + result.sweeps_per_temp[e + 1]);
        sum_inv_f += n_avg / f_i;
    }
    result.round_trip_estimate = sum_inv_f;

    if (rank == 0) {
        cout << "\n=== Gradient-Optimised Temperature Grid Summary ===" << endl;
        cout << "Mean acceptance: " << std::fixed << std::setprecision(3)
             << result.mean_acceptance_rate * 100 << "%" << endl;
        cout << "Est round-trip: " << std::scientific << result.round_trip_estimate << endl;
        cout << "Converged: " << (result.converged ? "YES" : "NO") << endl;
    }
    MPI_Barrier(comm);
    return result;
}

/**
 * Feedback-optimised temperature grid (Katzgraber + Bittner adaptive).
 * Set use_gradient=true (default) for gradient-based optimization (Miyata et al. 2024).
 */
template<typename L>
inline OptimizedTempGridResult generate_optimized_temperature_grid_mpi(
    L& lat, double Tmin, double Tmax,
    size_t warmup_sweeps   = 500,
    size_t sweeps_per_iter = 500,
    size_t feedback_iters  = 20,
    bool   gaussian_move   = false,
    size_t overrelaxation_rate = 0,
    double target_acceptance   = 0.5,
    double convergence_tol     = 0.05,
    MPI_Comm comm = MPI_COMM_WORLD,
    bool use_gradient = true) {

    if (use_gradient) {
        return generate_optimized_temperature_grid_mpi_gradient<L>(
            lat, Tmin, Tmax,
            warmup_sweeps, sweeps_per_iter,
            std::max(feedback_iters, size_t(50)),
            gaussian_move, overrelaxation_rate,
            0.15,   // learning_rate
            convergence_tol, comm);
    }

    int rank, R;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &R);

    OptimizedTempGridResult result;
    if (R < 2)  { result.temperatures = {Tmin}; result.converged = true; return result; }
    if (R == 2) { result.temperatures = {Tmin, Tmax};
                  result.acceptance_rates = {0.5}; result.converged = true; return result; }

    // Per-rank exchange RNG
    std::mt19937 exchange_rng(
        static_cast<unsigned>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count()
            + rank * 12345));

    if (rank == 0) {
        cout << "=== Feedback-Optimised Temperature Grid (MPI) ===" << endl;
        cout << "T_min = " << Tmin << ", T_max = " << Tmax << ", R = " << R << endl;
        cout << "Target acceptance rate: " << target_acceptance * 100 << "%" << endl;
    }

    double beta_min = 1.0 / Tmax, beta_max = 1.0 / Tmin;
    vector<double> beta(R);
    for (int i = 0; i < R; ++i)
        beta[i] = beta_min + (beta_max - beta_min) * double(i) / double(R - 1);

    double my_beta = beta[rank], my_T = 1.0 / my_beta;
    double sigma = 1000.0;

    // Warmup
    if (rank == 0) cout << "Warming up replicas..." << endl;
    for (size_t i = 0; i < warmup_sweeps; ++i) {
        lat.metropolis(my_T, gaussian_move, sigma);
        if (overrelaxation_rate > 0 && i % overrelaxation_rate == 0) lat.overrelaxation();
    }
    MPI_Barrier(comm);

    vector<double> acceptance_rates(R - 1, 0.0);
    double base_damping = 0.5;

    // Feedback iterations
    for (size_t iter = 0; iter < feedback_iters; ++iter) {
        double damping = base_damping + 0.3 * (double(iter) / double(feedback_iters));
        int local_attempts = 0, local_accepts = 0;

        for (size_t sw = 0; sw < sweeps_per_iter; ++sw) {
            lat.metropolis(my_T, gaussian_move, sigma);
            if (overrelaxation_rate > 0 && sw % overrelaxation_rate == 0)
                lat.overrelaxation();

            for (int parity = 0; parity <= 1; ++parity) {
                int partner = (parity == 0)
                    ? (rank % 2 == 0 ? rank + 1 : rank - 1)
                    : (rank % 2 == 1 ? rank + 1 : rank - 1);
                if (partner < 0 || partner >= R) continue;

                double my_E = lat.total_energy(), partner_E;
                MPI_Sendrecv(&my_E,      1, MPI_DOUBLE, partner, 0,
                             &partner_E, 1, MPI_DOUBLE, partner, 0,
                             comm, MPI_STATUS_IGNORE);

                int acc = 0;
                std::uniform_real_distribution<double> uni(0.0, 1.0);
                if (rank < partner) {
                    double delta = -(beta[partner] - my_beta) * (my_E - partner_E);
                    acc = (delta >= 0 || uni(exchange_rng) < std::exp(delta)) ? 1 : 0;
                    ++local_attempts;
                    if (acc) ++local_accepts;
                }
                int racc = 0;
                MPI_Sendrecv(&acc,  1, MPI_INT, partner, 1,
                             &racc, 1, MPI_INT, partner, 1, comm, MPI_STATUS_IGNORE);
                bool accepted = (rank < partner) ? (acc == 1) : (racc == 1);
                if (accepted) {
                    size_t bl = lat.lattice_size * lat.spin_dim;
                    vector<double> sb(bl), rb(bl);
                    for (size_t i = 0; i < lat.lattice_size; ++i)
                        for (size_t j = 0; j < lat.spin_dim; ++j)
                            sb[i * lat.spin_dim + j] = lat.spins[i](j);
                    MPI_Sendrecv(sb.data(), bl, MPI_DOUBLE, partner, 2,
                                 rb.data(), bl, MPI_DOUBLE, partner, 2,
                                 comm, MPI_STATUS_IGNORE);
                    for (size_t i = 0; i < lat.lattice_size; ++i)
                        for (size_t j = 0; j < lat.spin_dim; ++j)
                            lat.spins[i](j) = rb[i * lat.spin_dim + j];

                    // Exchange extra DOF (e.g. strain) if available
                    if constexpr (has_extra_dof<L>::value) {
                        size_t extra_n = lat.extra_dof_size();
                        if (extra_n > 0) {
                            vector<double> es(extra_n), er(extra_n);
                            lat.pack_extra_dof(es.data());
                            MPI_Sendrecv(es.data(), extra_n, MPI_DOUBLE, partner, 3,
                                         er.data(), extra_n, MPI_DOUBLE, partner, 3,
                                         comm, MPI_STATUS_IGNORE);
                            lat.unpack_extra_dof(er.data());
                        }
                    }
                }
            }
        }

        // Gather acceptance
        int ma = (rank < R - 1) ? local_attempts : 0;
        int mx = (rank < R - 1) ? local_accepts  : 0;
        vector<int> ra(R), rx(R);
        MPI_Gather(&ma, 1, MPI_INT, ra.data(), 1, MPI_INT, 0, comm);
        MPI_Gather(&mx, 1, MPI_INT, rx.data(), 1, MPI_INT, 0, comm);

        bool converged = false;
        if (rank == 0) {
            for (int e = 0; e < R - 1; ++e)
                if (ra[e] > 0) acceptance_rates[e] = double(rx[e]) / double(ra[e]);

            double mean_dev = 0.0, mean_rate = 0.0;
            double mn = 1.0, mx_r = 0.0;
            for (int e = 0; e < R - 1; ++e) {
                mean_dev  += std::abs(acceptance_rates[e] - target_acceptance);
                mean_rate += acceptance_rates[e];
                mn = std::min(mn, acceptance_rates[e]);
                mx_r = std::max(mx_r, acceptance_rates[e]);
            }
            mean_rate /= (R - 1); mean_dev /= (R - 1);

            cout << "Iter " << iter + 1 << "/" << feedback_iters
                 << ": mean A = " << std::fixed << std::setprecision(3) << mean_rate
                 << " [" << mn << ", " << mx_r << "]"
                 << ", mean dev = " << mean_dev << endl;

            result.feedback_iterations_used = iter + 1;
            if (mean_dev < convergence_tol) {
                converged = true;
                cout << "Converged at iteration " << iter + 1 << endl;
            }

            if (!converged) {
                vector<double> wt(R - 1); double tw = 0;
                for (int e = 0; e < R - 1; ++e) {
                    wt[e] = std::max(0.01, std::min(0.99, acceptance_rates[e]));
                    tw += wt[e];
                }
                for (auto& w : wt) w /= tw;

                vector<double> nb(R);
                nb[0] = beta_min;
                double cum = 0;
                for (int e = 0; e < R - 1; ++e) {
                    cum += wt[e];
                    nb[e + 1] = beta_min + cum * (beta_max - beta_min);
                }
                nb[R - 1] = beta_max;
                for (int k = 1; k < R - 1; ++k)
                    nb[k] = (1.0 - damping) * beta[k] + damping * nb[k];
                beta = nb;
            }
        }

        int ci = converged ? 1 : 0;
        MPI_Bcast(&ci, 1, MPI_INT, 0, comm);
        MPI_Bcast(beta.data(), R, MPI_DOUBLE, 0, comm);
        my_beta = beta[rank]; my_T = 1.0 / my_beta;
        if (ci) { result.converged = true; break; }
    }

    MPI_Bcast(acceptance_rates.data(), R - 1, MPI_DOUBLE, 0, comm);

    // Phase 2: measure tau for adaptive sweep schedule
    if (rank == 0) cout << "\nMeasuring τ for adaptive sweep schedule..." << endl;
    size_t tau_n = std::max(size_t(500), sweeps_per_iter);
    vector<double> eseries;
    eseries.reserve(tau_n);
    for (size_t i = 0; i < tau_n; ++i) {
        lat.metropolis(my_T, gaussian_move, sigma);
        if (overrelaxation_rate > 0 && i % overrelaxation_rate == 0) lat.overrelaxation();
        eseries.push_back(lat.total_energy());
    }
    double my_tau = 1.0; size_t dummy;
    estimate_autocorrelation_time(eseries, 1, my_tau, dummy);
    my_tau = std::max(1.0, my_tau);

    vector<double> all_tau(R);
    MPI_Allgather(&my_tau, 1, MPI_DOUBLE, all_tau.data(), 1, MPI_DOUBLE, comm);

    double tau_min_v = *std::min_element(all_tau.begin(), all_tau.end());
    size_t n_base = 10;
    result.autocorrelation_times = all_tau;
    result.sweeps_per_temp.resize(R);
    for (int k = 0; k < R; ++k)
        result.sweeps_per_temp[k] = std::max(size_t(1),
            static_cast<size_t>(std::ceil(n_base * all_tau[k] / tau_min_v)));

    if (rank == 0) {
        cout << "Autocorrelation times:" << endl;
        for (int k = 0; k < std::min(R, 15); ++k)
            cout << "  T[" << k << "] = " << std::scientific << std::setprecision(4)
                 << 1.0 / beta[k] << "  τ=" << std::fixed << std::setprecision(1)
                 << all_tau[k] << "  n_sw=" << result.sweeps_per_temp[k] << endl;
        if (R > 15) cout << "  ... (" << R - 15 << " more)" << endl;
    }

    result.temperatures.resize(R);
    for (int i = 0; i < R; ++i) result.temperatures[i] = 1.0 / beta[i];
    std::sort(result.temperatures.begin(), result.temperatures.end());
    result.acceptance_rates = acceptance_rates;

    result.local_diffusivities.resize(R - 1);
    for (int e = 0; e < R - 1; ++e) {
        double A = acceptance_rates[e];
        result.local_diffusivities[e] = A * (1.0 - A);
    }
    result.mean_acceptance_rate = 0;
    for (double A : acceptance_rates) result.mean_acceptance_rate += A;
    result.mean_acceptance_rate /= (R - 1);

    double sum_inv_f = 0, total_current = 0;
    for (int e = 0; e < R - 1; ++e) {
        double db = std::abs(beta[e + 1] - beta[e]);
        total_current += std::max(acceptance_rates[e], 1e-6) * db;
    }
    for (int e = 0; e < R - 1; ++e) {
        double db  = std::abs(beta[e + 1] - beta[e]);
        double A   = std::max(acceptance_rates[e], 1e-6);
        double f_i = A * db / total_current;
        double n_avg = 0.5 * (result.sweeps_per_temp[e] + result.sweeps_per_temp[e + 1]);
        sum_inv_f += n_avg / f_i;
    }
    result.round_trip_estimate = sum_inv_f;

    if (rank == 0) {
        cout << "\n=== Optimised Temperature Grid Summary ===" << endl;
        cout << "Mean acceptance: " << std::fixed << std::setprecision(3)
             << result.mean_acceptance_rate * 100 << "%" << endl;
        cout << "Est round-trip: " << std::scientific
             << result.round_trip_estimate << endl;
        cout << "Converged: " << (result.converged ? "YES" : "NO") << endl;
    }
    MPI_Barrier(comm);
    return result;
}

// ============================================================
// HDF5 OUTPUT
// ============================================================

#ifdef HDF5_ENABLED

/**
 * Save per-replica thermodynamic data to HDF5.
 */
template<typename SpinVec>
inline void save_thermodynamic_observables_hdf5(
    const string& out_dir,
    double temperature, size_t lattice_size, size_t spin_dim, size_t N_atoms,
    const ThermodynamicObservables& obs,
    const vector<double>& energies,
    const vector<SpinVec>& magnetizations,
    const vector<vector<SpinVec>>& sublattice_mags,
    size_t n_anneal, size_t n_measure,
    size_t probe_rate, size_t swap_rate,
    size_t overrelaxation_rate,
    double acceptance_rate, double swap_acceptance_rate) {

    std::filesystem::create_directories(out_dir);
    string filename = out_dir + "/parallel_tempering_data.h5";
    size_t n_samples = energies.size();

    HDF5PTWriter writer(filename, temperature, lattice_size, spin_dim, N_atoms,
                        n_samples, n_anneal, n_measure, probe_rate, swap_rate,
                        overrelaxation_rate, acceptance_rate, swap_acceptance_rate);

    writer.write_timeseries(energies, magnetizations, sublattice_mags);

    vector<vector<double>> sub_means(N_atoms), sub_errs(N_atoms);
    vector<vector<double>> cross_means(N_atoms), cross_errs(N_atoms);
    for (size_t a = 0; a < N_atoms; ++a) {
        sub_means[a]   = obs.sublattice_magnetization[a].values;
        sub_errs[a]    = obs.sublattice_magnetization[a].errors;
        cross_means[a] = obs.energy_sublattice_cross[a].values;
        cross_errs[a]  = obs.energy_sublattice_cross[a].errors;
    }
    writer.write_observables(obs.energy.value, obs.energy.error,
                             obs.specific_heat.value, obs.specific_heat.error,
                             obs.magnetization.values, obs.magnetization.errors,
                             sub_means, sub_errs, cross_means, cross_errs);
    writer.close();
}

/**
 * Save aggregated heat capacity data (root rank only).
 */
inline void save_heat_capacity_hdf5(
    const string& out_dir,
    const vector<double>& temperatures,
    const vector<double>& heat_capacity,
    const vector<double>& dHeat) {

    std::filesystem::create_directories(out_dir);
    string filename = out_dir + "/parallel_tempering_aggregated.h5";
    size_t n = temperatures.size();

    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group data_grp = file.createGroup("/temperature_scan");
    H5::Group meta_grp = file.createGroup("/metadata");

    std::time_t now = std::time(nullptr);
    char ts[100];
    std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));

    H5::DataSpace scalar(H5S_SCALAR);
    hsize_t nv = n;
    H5::Attribute a1 = meta_grp.createAttribute("n_temperatures",
                            H5::PredType::NATIVE_HSIZE, scalar);
    a1.write(H5::PredType::NATIVE_HSIZE, &nv);
    H5::StrType st(H5::PredType::C_S1, strlen(ts) + 1);
    H5::Attribute a2 = meta_grp.createAttribute("creation_time", st, scalar);
    a2.write(st, ts);

    hsize_t dims[1] = {n};
    H5::DataSpace ds(1, dims);
    data_grp.createDataSet("temperature",        H5::PredType::NATIVE_DOUBLE, ds)
            .write(temperatures.data(),          H5::PredType::NATIVE_DOUBLE);
    data_grp.createDataSet("specific_heat",      H5::PredType::NATIVE_DOUBLE, ds)
            .write(heat_capacity.data(),         H5::PredType::NATIVE_DOUBLE);
    data_grp.createDataSet("specific_heat_error",H5::PredType::NATIVE_DOUBLE, ds)
            .write(dHeat.data(),                 H5::PredType::NATIVE_DOUBLE);
    file.close();
}

#endif // HDF5_ENABLED

} // namespace mc

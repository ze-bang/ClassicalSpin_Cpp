/**
 * bench_mc.cpp — micro-benchmark harness for the classical-spin Monte Carlo
 *                kernels.
 *
 * Reports the throughput of the local-update kernels (Metropolis,
 * overrelaxation, Wolff, Swendsen-Wang) in spin updates per second per core
 * and in wall time per sweep, on a few canonical models of increasing
 * difficulty across three lattice families:
 *
 *   Lattice (single-species, the historical workhorse):
 *     - heisenberg-honeycomb : isotropic NN Heisenberg, 3 NN, SU(2)
 *     - kitaev-honeycomb     : Kitaev-Γ-Γ', 3 NN, anisotropic exchange
 *     - pyrochlore           : isotropic NN Heisenberg, 6 NN, frustrated 3D
 *
 *   MixedLattice (two species, SU(2) + SU(3), with mixed couplings):
 *     - tmfeo3-bilinear      : TmFeO3, only bilinear + mixed-bilinear (chi*=0)
 *     - tmfeo3-trilinear     : TmFeO3 with the SU(2)-SU(2)-SU(3) mixed
 *                              trilinear couplings on (chi2, chi5, chi7 != 0).
 *                              This is the kernel where O(d^3) trilinear
 *                              contraction dominates and is the primary target
 *                              of the trilinear-pre-contraction optimization.
 *
 *   StrainPhononLattice (spins coupled to elastic + magnetoelastic strain):
 *     - strain-honeycomb-bilin : bilinear-only Kitaev-honeycomb on the strain
 *                                lattice, with magnetoelastic OFF (lambda=0).
 *                                Isolates the bilinear hot path.
 *     - strain-honeycomb-me    : same plus magnetoelastic Eg coupling ON,
 *                                exposing the 8-call ME-field hot loop.
 *
 * The output is intentionally machine-parseable (one row per measurement)
 * so the numbers can be fed straight into tables or CSV.
 *
 * Usage:
 *
 *   bench_mc                                       # full default suite
 *   bench_mc --family=mixed --L=4 --sweeps=2000
 *   bench_mc --model=tmfeo3-trilinear --algo=metropolis
 *
 * Notes on methodology:
 *   * Every measurement is preceded by a warmup that is at least as long as
 *     the measurement itself, so JIT-y effects (page faults, allocator
 *     warm-up, branch-predictor training) are amortized out.
 *   * `sweeps/sec` is reported as
 *         sweeps_done / wall_time
 *     and `updates/sec/site` as
 *         (sweeps_done * lattice_size) / wall_time.
 *     For MixedLattice, lattice_size = lattice_size_SU2 + lattice_size_SU3.
 *   * Peak resident set size is read from /proc/self/status (VmPeak/VmHWM).
 */

#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/unitcell.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/lattice/lattice.h"
#include "classical_spin/lattice/mixed_lattice.h"
#include "classical_spin/lattice/strain_phonon_lattice.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

namespace {

// Read VmPeak / VmHWM from /proc/self/status (Linux). Returns kB.
size_t read_proc_status_kb(const char* key) {
    std::ifstream f("/proc/self/status");
    if (!f) return 0;
    std::string line;
    const std::string needle = std::string(key) + ":";
    while (std::getline(f, line)) {
        if (line.compare(0, needle.size(), needle) == 0) {
            std::istringstream iss(line.substr(needle.size()));
            size_t kb;
            std::string unit;
            iss >> kb >> unit;
            return kb;
        }
    }
    return 0;
}

double now_sec() {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    auto dt = clock::now() - t0;
    return std::chrono::duration<double>(dt).count();
}

string getenv_or(const char* key, const char* fallback) {
    const char* v = std::getenv(key);
    return v ? string(v) : string(fallback);
}

// Parse --key=value or --flag args from argv.
struct CliArgs {
    string family = "all";
    string model  = "all";
    string algo   = "all";
    int    L      = 0;       // 0 -> default per model
    long   sweeps = 0;       // 0 -> default per model
    long   warmup = 0;       // 0 -> at least sweeps
    double T = 1.0;
    int    repeats = 3;
    int    threads = 0;      // 0 -> leave OMP_NUM_THREADS / default alone
    string mode   = "auto";  // "serial" | "parallel" | "auto"
    bool   csv = false;
    bool   help = false;
};

CliArgs parse_args(int argc, char** argv) {
    CliArgs a;
    for (int i = 1; i < argc; ++i) {
        string s = argv[i];
        auto eat = [&](const string& key, auto& dst) -> bool {
            if (s.rfind(key + "=", 0) == 0) {
                std::stringstream ss(s.substr(key.size() + 1));
                ss >> dst;
                return true;
            }
            return false;
        };
        if (s == "--help" || s == "-h") { a.help = true; }
        else if (s == "--csv")          { a.csv = true; }
        else if (eat("--family", a.family)) {}
        else if (eat("--model",  a.model))  {}
        else if (eat("--algo",   a.algo))   {}
        else if (eat("--L",      a.L))      {}
        else if (eat("--sweeps", a.sweeps)) {}
        else if (eat("--warmup", a.warmup)) {}
        else if (eat("--T",      a.T))      {}
        else if (eat("--repeats",a.repeats)){}
        else if (eat("--threads",a.threads)){}
        else if (eat("--mode",   a.mode))   {}
        else {
            cerr << "unknown arg: " << s << "\n";
            a.help = true;
        }
    }
    return a;
}

void print_help() {
    cout <<
"bench_mc — Monte-Carlo kernel micro-benchmark\n"
"\n"
"Options:\n"
"  --family={lattice|mixed|strain|all}             default: all\n"
"  --model=<model-name|all>                        default: all\n"
"  --algo={metropolis|over|met_over|wolff|sw|all}  default: all\n"
"  --L=<int>                lattice linear dimension (default per model)\n"
"  --sweeps=<int>           measured sweeps (default per kernel)\n"
"  --warmup=<int>           warmup sweeps (default = max(1000, sweeps))\n"
"  --T=<float>              simulation temperature (default 1.0)\n"
"  --repeats=<int>          number of timed repeats (default 3)\n"
"  --threads=<int>          OMP threads (overrides OMP_NUM_THREADS; 0 = leave alone)\n"
"  --mode={serial|parallel|auto}\n"
"                           which kernel variant to call:\n"
"                             serial   : use the historical metropolis() / overrelaxation()\n"
"                             parallel : use metropolis_parallel() / overrelaxation_parallel()\n"
"                             auto     : pick parallel iff threads>1 (default)\n"
"  --csv                    print pure CSV (no decoration)\n"
"  -h, --help               show this help\n"
"\n"
"Models by family:\n"
"  lattice : heisenberg-honeycomb, kitaev-honeycomb, pyrochlore\n"
"  mixed   : tmfeo3-bilinear, tmfeo3-trilinear\n"
"  strain  : strain-honeycomb-bilin, strain-honeycomb-me\n";
}

// -----------------------------------------------------------------------------
// Algorithm enum + names
// -----------------------------------------------------------------------------

enum class Algo {
    Metropolis,        // pure metropolis sweep
    Overrelaxation,    // pure overrelaxation sweep (zero-T microcanonical)
    MetOverMix,        // 1 metropolis + 5 overrelaxation per "sweep" (typical)
    Wolff,             // single Wolff cluster update per "sweep" (Lattice only)
    SwendsenWang,      // one SW pass per "sweep" (Lattice only)
};

const char* algo_name(Algo a) {
    switch (a) {
        case Algo::Metropolis:     return "metropolis";
        case Algo::Overrelaxation: return "over";
        case Algo::MetOverMix:     return "met+5over";
        case Algo::Wolff:          return "wolff";
        case Algo::SwendsenWang:   return "swendsen_wang";
    }
    return "?";
}

bool is_cluster(Algo a) {
    return a == Algo::Wolff || a == Algo::SwendsenWang;
}

vector<Algo> parse_algo_set(const string& s) {
    if (s == "all") {
        return { Algo::Metropolis, Algo::Overrelaxation, Algo::MetOverMix,
                 Algo::Wolff, Algo::SwendsenWang };
    }
    if (s == "metropolis") return { Algo::Metropolis };
    if (s == "over")       return { Algo::Overrelaxation };
    if (s == "met_over")   return { Algo::MetOverMix };
    if (s == "wolff")      return { Algo::Wolff };
    if (s == "sw")         return { Algo::SwendsenWang };
    return {};
}

// -----------------------------------------------------------------------------
// Output formatting (one BenchRow per measurement, family-agnostic)
// -----------------------------------------------------------------------------

struct BenchRow {
    string family;
    string model;
    string algo;
    int    L;
    size_t lattice_size;     // total spin-update sites (SU2+SU3 for mixed)
    long   sweeps;
    int    repeat;
    double wall_sec;
    double sweeps_per_sec;
    double updates_per_sec;
    size_t vmpeak_kb;
    size_t vmhwm_kb;
};

void print_header(bool csv) {
    if (csv) {
        cout << "family,model,algo,L,N,sweeps,repeat,wall_s,sweeps_per_s,"
                "updates_per_s,vmpeak_kb,vmhwm_kb\n";
    } else {
        cout << left
             << setw(8)  << "family"
             << setw(28) << "model"
             << setw(15) << "algo"
             << setw(5)  << "L"
             << setw(10) << "N"
             << setw(10) << "sweeps"
             << setw(6)  << "rep"
             << right
             << setw(12) << "wall (s)"
             << setw(14) << "sweeps/s"
             << setw(16) << "updates/s"
             << setw(13) << "VmHWM (MB)"
             << "\n";
        cout << string(137, '-') << "\n";
    }
}

void print_row(const BenchRow& r, bool csv) {
    if (csv) {
        cout << r.family << "," << r.model << "," << r.algo << "," << r.L
             << "," << r.lattice_size << "," << r.sweeps << "," << r.repeat
             << "," << std::scientific << std::setprecision(6) << r.wall_sec
             << "," << r.sweeps_per_sec << "," << r.updates_per_sec << ","
             << r.vmpeak_kb << "," << r.vmhwm_kb << "\n";
    } else {
        cout << left
             << setw(8)  << r.family
             << setw(28) << r.model
             << setw(15) << r.algo
             << setw(5)  << r.L
             << setw(10) << r.lattice_size
             << setw(10) << r.sweeps
             << setw(6)  << r.repeat
             << right << std::scientific << std::setprecision(3)
             << setw(12) << r.wall_sec
             << setw(14) << r.sweeps_per_sec
             << setw(16) << r.updates_per_sec
             << std::fixed << std::setprecision(1)
             << setw(13) << (double(r.vmhwm_kb) / 1024.0)
             << "\n";
    }
}

bool family_selected(const string& want, const string& f) {
    return want == "all" || want == f;
}

bool model_selected(const string& want, const string& m) {
    if (want == "all") return true;
    if (want == m) return true;
    // short aliases for the historical lattice models
    if (want == "heisenberg" && m == "heisenberg-honeycomb") return true;
    if (want == "kitaev"     && m == "kitaev-honeycomb")     return true;
    if (want == "tmfeo3"     && (m == "tmfeo3-bilinear" || m == "tmfeo3-trilinear"))
        return true;
    if (want == "strain"     && (m == "strain-honeycomb-bilin" || m == "strain-honeycomb-me"))
        return true;
    return false;
}

// =============================================================================
// Family 1: Lattice (single species)
// =============================================================================

unique_ptr<Lattice> build_heisenberg_honeycomb_lat(int L, double /*T*/) {
    SpinConfig cfg;
    cfg.set_param("J", 1.0);
    cfg.set_param("K", 0.0);
    cfg.set_param("Gamma", 0.0);
    cfg.set_param("Gammap", 0.0);
    cfg.field_strength = 0.0;
    UnitCell uc = build_kitaev_honeycomb(cfg);
    auto lat = std::make_unique<Lattice>(uc, L, L, 1, 1.0f);
    lat->lattice_type = "heisenberg_honeycomb";
    lat->init_random();
    return lat;
}

unique_ptr<Lattice> build_kitaev_honeycomb_lat(int L, double /*T*/) {
    SpinConfig cfg;
    cfg.set_param("J", 0.0);
    cfg.set_param("K", -1.0);
    cfg.set_param("Gamma", 0.25);
    cfg.set_param("Gammap", -0.02);
    cfg.field_strength = 0.0;
    UnitCell uc = build_kitaev_honeycomb(cfg);
    auto lat = std::make_unique<Lattice>(uc, L, L, 1, 1.0f);
    lat->lattice_type = "honeycomb_kitaev";
    lat->init_random();
    return lat;
}

unique_ptr<Lattice> build_pyrochlore_lat(int L, double /*T*/) {
    SpinConfig cfg;
    cfg.set_param("Jxx", 1.0);
    cfg.set_param("Jyy", 1.0);
    cfg.set_param("Jzz", 1.0);
    cfg.field_strength = 0.0;
    UnitCell uc = build_pyrochlore(cfg);
    auto lat = std::make_unique<Lattice>(uc, L, L, L, 1.0f);
    lat->lattice_type = "pyrochlore";
    lat->init_random();
    return lat;
}

void run_lattice_kernel(Lattice& lat, Algo algo, double T, long n_sweeps, bool parallel) {
    switch (algo) {
        case Algo::Metropolis:
            if (parallel) for (long i = 0; i < n_sweeps; ++i) lat.metropolis_parallel(T, false, 60.0);
            else          for (long i = 0; i < n_sweeps; ++i) lat.metropolis(T, false, 60.0);
            break;
        case Algo::Overrelaxation:
            if (parallel) for (long i = 0; i < n_sweeps; ++i) lat.overrelaxation_parallel();
            else          for (long i = 0; i < n_sweeps; ++i) lat.overrelaxation();
            break;
        case Algo::MetOverMix:
            for (long i = 0; i < n_sweeps; ++i) {
                if (parallel) {
                    lat.metropolis_parallel(T, false, 60.0);
                    for (int k = 0; k < 5; ++k) lat.overrelaxation_parallel();
                } else {
                    lat.metropolis(T, false, 60.0);
                    for (int k = 0; k < 5; ++k) lat.overrelaxation();
                }
            }
            break;
        case Algo::Wolff:
            for (long i = 0; i < n_sweeps; ++i) lat.wolff_update(T, false);
            break;
        case Algo::SwendsenWang:
            for (long i = 0; i < n_sweeps; ++i) lat.swendsen_wang_sweep(T, false);
            break;
    }
}

// =============================================================================
// Family 2: MixedLattice (SU(2) + SU(3), bilinear / trilinear)
// =============================================================================

// TmFeO3 with the trilinear couplings turned ON. This is the workhorse
// benchmark for measuring the SU(2)-SU(2)-SU(3) trilinear hot loop.
unique_ptr<MixedLattice> build_tmfeo3_lat(int L, double /*T*/, bool with_trilinear) {
    SpinConfig cfg;
    cfg.field_strength = 0.0;
    cfg.spin_length     = 1.0f;
    cfg.spin_length_su3 = 1.0f;

    cfg.set_param("J1ab", 4.74);
    cfg.set_param("J1c",  5.15);
    cfg.set_param("J2ab", 0.15);
    cfg.set_param("J2c",  0.30);
    cfg.set_param("Ka", -0.16221);
    cfg.set_param("Kb",  0.0);
    cfg.set_param("Kc", -0.18318);
    cfg.set_param("D1",  0.12);
    cfg.set_param("D2",  0.0);
    cfg.set_param("e1",  0.97);
    cfg.set_param("e2",  3.97);

    if (with_trilinear) {
        // Realistic chi values produce the SU(2)-SU(2)-SU(3) mixed trilinear
        // couplings inside build_tmfeo3. Magnitudes are deliberately small so
        // the model stays physical but the trilinear loops are exercised.
        cfg.set_param("chi2x", 0.10);
        cfg.set_param("chi2y", 0.05);
        cfg.set_param("chi2z", 0.02);
        cfg.set_param("chi5x", 0.07);
        cfg.set_param("chi5y", 0.04);
        cfg.set_param("chi5z", 0.03);
        cfg.set_param("chi7x", 0.06);
        cfg.set_param("chi7y", 0.05);
        cfg.set_param("chi7z", 0.02);
    }

    MixedUnitCell mixed_uc = build_tmfeo3(cfg);
    auto lat = std::make_unique<MixedLattice>(
        mixed_uc, L, L, L, cfg.spin_length, cfg.spin_length_su3);
    lat->init_random();
    return lat;
}

unique_ptr<MixedLattice> build_tmfeo3_bilinear_lat(int L, double T) {
    return build_tmfeo3_lat(L, T, /*with_trilinear=*/false);
}

unique_ptr<MixedLattice> build_tmfeo3_trilinear_lat(int L, double T) {
    return build_tmfeo3_lat(L, T, /*with_trilinear=*/true);
}

void run_mixed_kernel(MixedLattice& lat, Algo algo, double T, long n_sweeps, bool parallel) {
    switch (algo) {
        case Algo::Metropolis:
            if (parallel) for (long i = 0; i < n_sweeps; ++i) lat.metropolis_parallel(T, false, 60.0);
            else          for (long i = 0; i < n_sweeps; ++i) lat.metropolis(T, false, 60.0);
            break;
        case Algo::Overrelaxation:
            if (parallel) for (long i = 0; i < n_sweeps; ++i) lat.overrelaxation_parallel();
            else          for (long i = 0; i < n_sweeps; ++i) lat.overrelaxation();
            break;
        case Algo::MetOverMix:
            for (long i = 0; i < n_sweeps; ++i) {
                if (parallel) {
                    lat.metropolis_parallel(T, false, 60.0);
                    for (int k = 0; k < 5; ++k) lat.overrelaxation_parallel();
                } else {
                    lat.metropolis(T, false, 60.0);
                    for (int k = 0; k < 5; ++k) lat.overrelaxation();
                }
            }
            break;
        case Algo::Wolff:
        case Algo::SwendsenWang:
            // MixedLattice has no cluster updates. Caller must skip these.
            break;
    }
}

// =============================================================================
// Family 3: StrainPhononLattice (spins + strain + magnetoelastic)
// =============================================================================

unique_ptr<StrainPhononLattice> build_strain_honeycomb_lat(int L, double /*T*/,
                                                           bool with_me) {
    SpinConfig cfg;
    cfg.set_param("J", 0.0);
    cfg.set_param("K", -1.0);
    cfg.set_param("Gamma", 0.25);
    cfg.set_param("Gammap", -0.02);
    cfg.set_param("J2_A", 0.0);
    cfg.set_param("J2_B", 0.0);
    cfg.set_param("J3", 0.0);
    cfg.field_strength = 0.0;

    UnitCell uc = build_strain_honeycomb(cfg);
    auto lat = std::make_unique<StrainPhononLattice>(uc, L, L, 1, 1.0f);

    MagnetoelasticParams me{};
    me.J = 0.0; me.K = -1.0; me.Gamma = 0.25; me.Gammap = -0.02;
    me.J2_A = 0.0; me.J2_B = 0.0; me.J3 = 0.0; me.J7 = 0.0;
    me.lambda_A1g = 0.0;
    me.lambda_Eg  = with_me ? 0.05 : 0.0;
    me.gamma_J7   = 0.0;

    ElasticParams el{};
    el.C11 = 1.0; el.C12 = 0.3; el.C44 = 0.5; el.M = 1.0;
    el.gamma_A1g = 0.0; el.gamma_Eg = 0.0;

    StrainDriveParams dr{};
    dr.E0_1 = 0.0; dr.E0_2 = 0.0;

    lat->set_parameters(me, el, dr);
    lat->init_random();
    return lat;
}

unique_ptr<StrainPhononLattice> build_strain_honeycomb_bilin_lat(int L, double T) {
    return build_strain_honeycomb_lat(L, T, /*with_me=*/false);
}

unique_ptr<StrainPhononLattice> build_strain_honeycomb_me_lat(int L, double T) {
    return build_strain_honeycomb_lat(L, T, /*with_me=*/true);
}

void run_strain_kernel(StrainPhononLattice& lat, Algo algo, double T, long n_sweeps, bool parallel) {
    switch (algo) {
        case Algo::Metropolis:
            if (parallel) for (long i = 0; i < n_sweeps; ++i) lat.metropolis_parallel(T, false, 60.0);
            else          for (long i = 0; i < n_sweeps; ++i) lat.metropolis(T, false, 60.0);
            break;
        case Algo::Overrelaxation:
            if (parallel) for (long i = 0; i < n_sweeps; ++i) lat.overrelaxation_parallel();
            else          for (long i = 0; i < n_sweeps; ++i) lat.overrelaxation();
            break;
        case Algo::MetOverMix:
            for (long i = 0; i < n_sweeps; ++i) {
                if (parallel) {
                    lat.metropolis_parallel(T, false, 60.0);
                    for (int k = 0; k < 5; ++k) lat.overrelaxation_parallel();
                } else {
                    lat.metropolis(T, false, 60.0);
                    for (int k = 0; k < 5; ++k) lat.overrelaxation();
                }
            }
            break;
        case Algo::Wolff:
        case Algo::SwendsenWang:
            // StrainPhononLattice has no cluster updates. Caller must skip.
            break;
    }
}

// =============================================================================
// Generic per-family driver
// =============================================================================

template <typename LatPtr>
struct GenericModelSpec {
    using LatticeT = typename LatPtr::element_type;

    string                                                    family;
    string                                                    name;
    int                                                       default_L;
    long                                                      default_sweeps;       // local
    long                                                      default_cluster_sweeps;
    bool                                                      supports_cluster;
    std::function<LatPtr(int, double)>                        build;
    std::function<size_t(const LatticeT&)>                    size_of;
    std::function<void(LatticeT&, Algo, double, long, bool)>  run_kernel;
};

template <typename LatPtr>
void run_family(const vector<GenericModelSpec<LatPtr>>& models,
                const CliArgs& args,
                const vector<Algo>& algos,
                bool parallel) {
    for (const auto& m : models) {
        if (!family_selected(args.family, m.family)) continue;
        if (!model_selected(args.model, m.name))     continue;

        const int L = (args.L > 0) ? args.L : m.default_L;
        LatPtr lat = m.build(L, args.T);
        if (!lat) {
            cerr << "Failed to build " << m.name << " L=" << L << "\n";
            continue;
        }
        const size_t N = m.size_of(*lat);  // size_of takes the lattice ref


        for (Algo algo : algos) {
            if (is_cluster(algo) && !m.supports_cluster) continue;

            const long sw_default =
                is_cluster(algo) ? m.default_cluster_sweeps : m.default_sweeps;
            const long sweeps = (args.sweeps > 0) ? args.sweeps : sw_default;
            const long warmup = (args.warmup > 0) ? args.warmup
                                                  : std::max(long(1000), sweeps);

            // Warmup (untimed). Brings the lattice to a stable thermalized
            // state and primes the allocator / branch predictor.
            m.run_kernel(*lat, algo, args.T, warmup, parallel);
            (void)warmup;

            for (int rep = 0; rep < args.repeats; ++rep) {
                double t0 = now_sec();
                m.run_kernel(*lat, algo, args.T, sweeps, parallel);
                double t1 = now_sec();
                double dt = t1 - t0;
                if (dt <= 0.0) dt = 1e-9;

                BenchRow row;
                row.family          = m.family;
                row.model           = m.name;
                row.algo            = algo_name(algo);
                row.L               = L;
                row.lattice_size    = N;
                row.sweeps          = sweeps;
                row.repeat          = rep;
                row.wall_sec        = dt;
                row.sweeps_per_sec  = double(sweeps) / dt;
                row.updates_per_sec = double(sweeps) * double(N) / dt;
                row.vmpeak_kb       = read_proc_status_kb("VmPeak");
                row.vmhwm_kb        = read_proc_status_kb("VmHWM");
                print_row(row, args.csv);
                std::cout.flush();
            }
        }
    }
}

vector<GenericModelSpec<unique_ptr<Lattice>>> default_lattice_models() {
    auto sz = [](const Lattice& l) { return l.lattice_size; };
    return {
        { "lattice", "heisenberg-honeycomb", 32, 20000, 5000, true,
          build_heisenberg_honeycomb_lat, sz, run_lattice_kernel },
        { "lattice", "kitaev-honeycomb",     32, 20000, 5000, true,
          build_kitaev_honeycomb_lat,     sz, run_lattice_kernel },
        { "lattice", "pyrochlore",            6, 10000, 3000, true,
          build_pyrochlore_lat,           sz, run_lattice_kernel },
    };
}

vector<GenericModelSpec<unique_ptr<MixedLattice>>> default_mixed_models() {
    // For Mixed, "size" is total spin-update sites = SU(2) + SU(3).
    auto sz = [](const MixedLattice& l) {
        return l.lattice_size_SU2 + l.lattice_size_SU3;
    };
    // Defaults: L=4 -> N_SU2 = N_SU3 = 4 atoms * 4^3 = 256 each, total 512
    // sites. Same memory footprint class as honeycomb at L=32 (sub-L1).
    // Sweep budget is lower because each Metropolis sweep does ~10x the
    // arithmetic of honeycomb (mixed bilinear + trilinear).
    return {
        { "mixed", "tmfeo3-bilinear",  4, 2000, 1, false,
          build_tmfeo3_bilinear_lat,
          sz,
          run_mixed_kernel },
        { "mixed", "tmfeo3-trilinear", 4,  500, 1, false,
          build_tmfeo3_trilinear_lat,
          sz,
          run_mixed_kernel },
    };
}

vector<GenericModelSpec<unique_ptr<StrainPhononLattice>>> default_strain_models() {
    auto sz = [](const StrainPhononLattice& l) { return l.lattice_size; };
    // Honeycomb at L=24 -> N = 1152 spins, fits in L1.
    // Sweep budget is modest because the ME path is ~5x more expensive than
    // pure bilinear honeycomb.
    return {
        { "strain", "strain-honeycomb-bilin", 24, 4000, 1, false,
          build_strain_honeycomb_bilin_lat, sz, run_strain_kernel },
        { "strain", "strain-honeycomb-me",    24, 1000, 1, false,
          build_strain_honeycomb_me_lat,    sz, run_strain_kernel },
    };
}

}  // anonymous namespace

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(int argc, char** argv) {
    CliArgs args = parse_args(argc, argv);
    if (args.help) { print_help(); return 0; }

#ifdef _OPENMP
    if (args.threads > 0) {
        omp_set_num_threads(args.threads);
    }
    const int n_threads = omp_get_max_threads();
#else
    const int n_threads = 1;
#endif

    bool parallel;
    if (args.mode == "auto") {
        parallel = (n_threads > 1);
    } else if (args.mode == "parallel") {
        parallel = true;
    } else if (args.mode == "serial") {
        parallel = false;
    } else {
        cerr << "Unknown --mode=" << args.mode << " (expected serial|parallel|auto)\n";
        return 1;
    }

    // Force the touch of now_sec()'s static t0 so the first measurement is
    // not skewed by the first call to the clock.
    (void)now_sec();

    if (!args.csv) {
        cout << "ClassicalSpin_Cpp Monte-Carlo benchmark\n";
        cout << "  CXX threads (OpenMP): " << n_threads << "\n";
        cout << "  Kernel mode:          " << (parallel ? "parallel (coloured)" : "serial") << "\n";
        cout << "  Compiler hint:        " <<
#if defined(__clang__)
        "clang " << __clang_major__ << "." << __clang_minor__
#elif defined(__GNUC__)
        "gcc " << __GNUC__ << "." << __GNUC_MINOR__
#else
        "unknown"
#endif
        << "\n";
        cout << "  Build type:           " << getenv_or("BUILD_TYPE", "Release") << "\n";
        cout << "\n";
    }

    print_header(args.csv);

    vector<Algo> algos = parse_algo_set(args.algo);
    if (algos.empty()) {
        cerr << "Unknown --algo=" << args.algo << "\n";
        return 1;
    }

    run_family(default_lattice_models(), args, algos, parallel);
    run_family(default_mixed_models(),   args, algos, parallel);
    run_family(default_strain_models(),  args, algos, parallel);

    return 0;
}

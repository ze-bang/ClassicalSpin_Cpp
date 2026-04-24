/**
 * bench_mc.cpp — micro-benchmark harness for the classical-spin Monte Carlo
 *                kernels.
 *
 * Reports the throughput of the local-update kernels (Metropolis,
 * overrelaxation, Wolff, Swendsen-Wang) in spin updates per second per core
 * and in wall time per sweep, on a few canonical models of increasing
 * difficulty:
 *
 *   - heisenberg-honeycomb : isotropic NN Heisenberg, 3 NN, SU(2)
 *   - kitaev-honeycomb     : Kitaev-Γ-Γ', 3 NN, anisotropic exchange
 *   - pyrochlore           : isotropic NN Heisenberg, 6 NN, frustrated 3D
 *
 * The output is intentionally machine-parseable (one row per measurement)
 * so the numbers can be fed straight into tables or CSV.
 *
 * Usage:
 *
 *   bench_mc                                       # default suite
 *   bench_mc --model=kitaev --L=24 --sweeps=20000 --algo=metropolis
 *
 * Notes on methodology:
 *   * Every measurement is preceded by a warmup that is at least as long as
 *     the measurement itself, so JIT-y effects (page faults, allocator
 *     warm-up, branch-predictor training) are amortized out.
 *   * `sweeps/sec` is reported as
 *         sweeps_done / wall_time
 *     and `updates/sec/site` as
 *         (sweeps_done * lattice_size) / wall_time.
 *   * Peak resident set size is read from /proc/self/status (VmPeak/VmHWM).
 */

#include "classical_spin/core/spin_config.h"
#include "classical_spin/core/unitcell.h"
#include "classical_spin/core/unitcell_builders.h"
#include "classical_spin/lattice/lattice.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
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
    string model = "all";
    string algo  = "all";
    int    L     = 0;       // 0 -> default per model
    long   sweeps = 0;       // 0 -> default per model
    long   warmup = 0;       // 0 -> at least sweeps
    double T = 1.0;
    int    repeats = 3;
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
        else if (eat("--model",  a.model))  {}
        else if (eat("--algo",   a.algo))   {}
        else if (eat("--L",      a.L))      {}
        else if (eat("--sweeps", a.sweeps)) {}
        else if (eat("--warmup", a.warmup)) {}
        else if (eat("--T",      a.T))      {}
        else if (eat("--repeats",a.repeats)){}
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
"  --model={heisenberg|kitaev|pyrochlore|all}   default: all\n"
"  --algo={metropolis|over|met_over|wolff|sw|all} default: all\n"
"  --L=<int>                lattice linear dimension (default per model)\n"
"  --sweeps=<int>           measured sweeps (default 5000-20000 per kernel)\n"
"  --warmup=<int>           warmup sweeps (default = max(1000, sweeps))\n"
"  --T=<float>              simulation temperature (default 1.0)\n"
"  --repeats=<int>          number of timed repeats (default 3)\n"
"  --csv                    print pure CSV (no decoration)\n"
"  -h, --help               show this help\n";
}

// -----------------------------------------------------------------------------
// Lattice builders for the benchmark suite
// -----------------------------------------------------------------------------

// Heisenberg honeycomb: isotropic NN J·S_i·S_j on the same honeycomb unit cell
// used by the Kitaev builder, with K=Γ=Γ'=h=0 and J=1.
unique_ptr<Lattice> build_heisenberg_honeycomb_lat(int L, double T) {
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
    (void)T;
    return lat;
}

unique_ptr<Lattice> build_kitaev_honeycomb_lat(int L, double T) {
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
    (void)T;
    return lat;
}

unique_ptr<Lattice> build_pyrochlore_lat(int L, double T) {
    SpinConfig cfg;
    cfg.set_param("Jxx", 1.0);
    cfg.set_param("Jyy", 1.0);
    cfg.set_param("Jzz", 1.0);
    cfg.field_strength = 0.0;
    UnitCell uc = build_pyrochlore(cfg);
    auto lat = std::make_unique<Lattice>(uc, L, L, L, 1.0f);
    lat->lattice_type = "pyrochlore";
    lat->init_random();
    (void)T;
    return lat;
}

// -----------------------------------------------------------------------------
// Algorithm dispatch
// -----------------------------------------------------------------------------

enum class Algo {
    Metropolis,        // pure metropolis sweep
    Overrelaxation,    // pure overrelaxation sweep (zero-T microcanonical)
    MetOverMix,        // 1 metropolis + 5 overrelaxation per "sweep" (typical)
    Wolff,             // single Wolff cluster update per "sweep"
    SwendsenWang,      // one SW pass per "sweep"
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

// Run `n_sweeps` of the chosen algorithm. We DO NOT measure inside the loop;
// the timing is done by the caller.
void run_kernel(Lattice& lat, Algo algo, double T, long n_sweeps) {
    switch (algo) {
        case Algo::Metropolis:
            for (long i = 0; i < n_sweeps; ++i) lat.metropolis(T, false, 60.0);
            break;
        case Algo::Overrelaxation:
            for (long i = 0; i < n_sweeps; ++i) lat.overrelaxation();
            break;
        case Algo::MetOverMix:
            for (long i = 0; i < n_sweeps; ++i) {
                lat.metropolis(T, false, 60.0);
                for (int k = 0; k < 5; ++k) lat.overrelaxation();
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

struct BenchRow {
    string model;
    string algo;
    int    L;
    size_t lattice_size;
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
        cout << "model,algo,L,N,sweeps,repeat,wall_s,sweeps_per_s,"
                "updates_per_s,vmpeak_kb,vmhwm_kb\n";
    } else {
        cout << left
             << setw(22) << "model"
             << setw(15) << "algo"
             << setw(5)  << "L"
             << setw(10) << "N"
             << setw(10) << "sweeps"
             << setw(8)  << "rep"
             << right
             << setw(12) << "wall (s)"
             << setw(14) << "sweeps/s"
             << setw(16) << "updates/s"
             << setw(14) << "VmHWM (MB)"
             << "\n";
        cout << string(126, '-') << "\n";
    }
}

void print_row(const BenchRow& r, bool csv) {
    if (csv) {
        cout << r.model << "," << r.algo << "," << r.L << "," << r.lattice_size
             << "," << r.sweeps << "," << r.repeat << ","
             << std::scientific << std::setprecision(6) << r.wall_sec << ","
             << r.sweeps_per_sec << "," << r.updates_per_sec << ","
             << r.vmpeak_kb << "," << r.vmhwm_kb << "\n";
    } else {
        cout << left
             << setw(22) << r.model
             << setw(15) << r.algo
             << setw(5)  << r.L
             << setw(10) << r.lattice_size
             << setw(10) << r.sweeps
             << setw(8)  << r.repeat
             << right << std::scientific << std::setprecision(3)
             << setw(12) << r.wall_sec
             << setw(14) << r.sweeps_per_sec
             << setw(16) << r.updates_per_sec
             << std::fixed << std::setprecision(1)
             << setw(14) << (double(r.vmhwm_kb) / 1024.0)
             << "\n";
    }
}

struct ModelSpec {
    string name;
    int    default_L;
    long   default_sweeps;     // for local-update kernels
    long   default_cluster_sweeps;
    std::function<unique_ptr<Lattice>(int, double)> build;
};

vector<ModelSpec> default_models() {
    return {
        { "heisenberg-honeycomb", 32, 20000, 5000, build_heisenberg_honeycomb_lat },
        { "kitaev-honeycomb",     32, 20000, 5000, build_kitaev_honeycomb_lat     },
        { "pyrochlore",            6, 10000, 3000, build_pyrochlore_lat           },
    };
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

bool model_selected(const string& want, const string& m) {
    if (want == "all") return true;
    if (want == m) return true;
    // allow short aliases
    if (want == "heisenberg" && m == "heisenberg-honeycomb") return true;
    if (want == "kitaev"     && m == "kitaev-honeycomb")     return true;
    return false;
}

bool is_cluster(Algo a) {
    return a == Algo::Wolff || a == Algo::SwendsenWang;
}

}  // anonymous namespace

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(int argc, char** argv) {
    CliArgs args = parse_args(argc, argv);
    if (args.help) { print_help(); return 0; }

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    // Force the touch of now_sec()'s static t0 so the first measurement is
    // not skewed by the first call to the clock.
    (void)now_sec();

    if (!args.csv) {
        cout << "ClassicalSpin_Cpp Monte-Carlo benchmark\n";
        cout << "  CXX threads (OpenMP): " << n_threads << "\n";
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

    for (const auto& m : default_models()) {
        if (!model_selected(args.model, m.name)) continue;

        const int  L = (args.L > 0) ? args.L : m.default_L;
        auto lat = m.build(L, args.T);
        if (!lat) {
            cerr << "Failed to build " << m.name << " L=" << L << "\n";
            continue;
        }
        const size_t N = lat->lattice_size;

        for (Algo algo : algos) {
            const long sw_default =
                is_cluster(algo) ? m.default_cluster_sweeps : m.default_sweeps;
            const long sweeps = (args.sweeps > 0) ? args.sweeps : sw_default;
            const long warmup = (args.warmup > 0) ? args.warmup
                                                  : std::max(long(1000), sweeps);

            // Warmup (untimed). Brings the lattice to a stable thermalized
            // state and primes the allocator / branch predictor.
            run_kernel(*lat, algo, args.T, warmup);

            for (int rep = 0; rep < args.repeats; ++rep) {
                double t0 = now_sec();
                run_kernel(*lat, algo, args.T, sweeps);
                double t1 = now_sec();
                double dt = t1 - t0;
                if (dt <= 0.0) dt = 1e-9;

                BenchRow row;
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

    return 0;
}

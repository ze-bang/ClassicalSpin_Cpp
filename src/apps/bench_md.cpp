/**
 * bench_md.cpp — micro-benchmark harness for the molecular-dynamics
 *                (Landau-Lifshitz / spin-strain ODE) RHS kernels.
 *
 * Reports the throughput of the ODE right-hand-side kernels in
 * RHS-calls per second per core and in equivalent "site updates per second"
 * across the three lattice families:
 *
 *   Lattice (single species):
 *     - heisenberg-honeycomb-su2 : SU(2) cross product LLG
 *     - kitaev-honeycomb-su2     : SU(2) cross product LLG, anisotropic exchange
 *     - tmfeo3-su3-only          : SU(3) sublattice only — exercises the
 *                                  sparse Gell-Mann cross product
 *
 *   MixedLattice:
 *     - tmfeo3-bilinear   : SU(2) + SU(3) RHS, bilinear couplings only
 *     - tmfeo3-trilinear  : full mixed-trilinear RHS path
 *     - tmfeo3-driven     : same as bilinear but with a non-zero pulse
 *                           amplitude (tests the drive-envelope hoist)
 *
 *   StrainPhononLattice:
 *     - strain-honeycomb-bilin : ME off, exercises the spin LLG path on the
 *                                strain lattice (no per-RHS heap copy after
 *                                Phase B)
 *     - strain-honeycomb-me    : ME on, exercises the fused magnetoelastic
 *                                field on every RHS call
 *     - strain-honeycomb-local : same as -me but with per-cell local strain
 *                                DOFs (this is the path with the largest
 *                                heap-copy savings from Phase B)
 *
 * The reported throughput is the number of full ode_system() (or
 * landau_lifshitz_flat()) RHS evaluations per second. For RK4 there are 4
 * RHS calls per integrator step, so dividing by 4 gives steps/sec.
 *
 * Usage:
 *   bench_md
 *   bench_md --family=mixed --L=4 --rhs=2000 --repeats=5
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
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <mpi.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

namespace {

double now_sec() {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    auto dt = clock::now() - t0;
    return std::chrono::duration<double>(dt).count();
}

struct CliArgs {
    string family = "all";
    string model  = "all";
    int    L      = 0;
    long   rhs    = 0;       // measured RHS calls (default per model)
    long   warmup = 0;
    int    repeats = 3;
    int    threads = 0;       // 0 = leave at OMP_NUM_THREADS / OpenMP default
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
        else if (eat("--L",      a.L))      {}
        else if (eat("--rhs",    a.rhs))    {}
        else if (eat("--warmup", a.warmup)) {}
        else if (eat("--repeats", a.repeats)){}
        else if (eat("--threads", a.threads)){}
        else { cerr << "unknown arg: " << s << "\n"; a.help = true; }
    }
    return a;
}

void print_help() {
    cout <<
"bench_md — molecular-dynamics RHS-kernel micro-benchmark\n"
"\n"
"Options:\n"
"  --family={lattice|mixed|strain|all}   default: all\n"
"  --model=<name|all>                    default: all\n"
"  --L=<int>                             lattice linear dim\n"
"  --rhs=<int>                           measured RHS calls\n"
"  --warmup=<int>                        warmup RHS calls\n"
"  --repeats=<int>                       timed repeats\n"
"  --threads=<int>                       OpenMP threads (0 = OMP default)\n"
"  --csv                                 CSV output\n"
"  -h, --help                            this help\n";
}

bool family_selected(const string& want, const string& f) {
    return want == "all" || want == f;
}
bool model_selected(const string& want, const string& m) {
    return want == "all" || want == m;
}

struct BenchRow {
    string family;
    string model;
    int    L;
    size_t lattice_size;
    long   rhs_calls;
    int    repeat;
    double wall_sec;
    double rhs_per_sec;
    double site_updates_per_sec;
    int    threads;
};

void print_header(bool csv) {
    if (csv) {
        cout << "family,model,L,N,rhs_calls,repeat,wall_s,rhs_per_s,site_updates_per_s,threads\n";
    } else {
        cout << left
             << setw(8)  << "family"
             << setw(28) << "model"
             << setw(5)  << "L"
             << setw(10) << "N"
             << setw(10) << "rhs"
             << setw(6)  << "rep"
             << right
             << setw(12) << "wall (s)"
             << setw(14) << "rhs/s"
             << setw(16) << "siteupd/s"
             << setw(6)  << "thr"
             << "\n";
        cout << string(115, '-') << "\n";
    }
}

void print_row(const BenchRow& r, bool csv) {
    if (csv) {
        cout << r.family << "," << r.model << "," << r.L
             << "," << r.lattice_size << "," << r.rhs_calls << "," << r.repeat
             << "," << std::scientific << std::setprecision(6) << r.wall_sec
             << "," << r.rhs_per_sec << "," << r.site_updates_per_sec
             << "," << r.threads << "\n";
    } else {
        cout << left
             << setw(8)  << r.family
             << setw(28) << r.model
             << setw(5)  << r.L
             << setw(10) << r.lattice_size
             << setw(10) << r.rhs_calls
             << setw(6)  << r.repeat
             << right << std::scientific << std::setprecision(3)
             << setw(12) << r.wall_sec
             << setw(14) << r.rhs_per_sec
             << setw(16) << r.site_updates_per_sec
             << setw(6)  << std::defaultfloat << r.threads
             << "\n";
    }
}

// ---------------------------------------------------------------------------
// Family 1: Lattice (SU(2) honeycomb + SU(3) tmfeo3-su3-only)
// ---------------------------------------------------------------------------

unique_ptr<Lattice> build_heisenberg_honeycomb(int L) {
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

unique_ptr<Lattice> build_kitaev_honeycomb_lat(int L) {
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

void bench_lattice_rhs(Lattice& lat, long n_rhs, double t0_drive) {
    // Allocate state buffers once. We feed the integrator's RHS directly to
    // measure the kernel cost without integrator overhead.
    const size_t N = lat.lattice_size;
    const size_t D = lat.spin_dim;
    std::vector<double> x(N * D), dxdt(N * D);
    // Pack current spins into x.
    for (size_t i = 0; i < N; ++i) {
        for (size_t d = 0; d < D; ++d) x[i*D + d] = lat.spins[i](d);
    }
    double t = t0_drive;
    for (long i = 0; i < n_rhs; ++i) {
        lat.landau_lifshitz_flat(x.data(), dxdt.data(), t);
        t += 1e-3;
    }
    // Touch dxdt to keep the compiler honest.
    volatile double sink = dxdt[0]; (void)sink;
}

// ---------------------------------------------------------------------------
// Family 2: MixedLattice
// ---------------------------------------------------------------------------

unique_ptr<MixedLattice> build_tmfeo3_lat(int L, bool with_trilinear) {
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

void bench_mixed_rhs(MixedLattice& lat, long n_rhs, bool driven) {
    if (driven) {
        // Switch on a pulse so the drive-envelope hoist is exercised. The
        // set_pulse signature is per-atom (one SpinVector per sublattice
        // atom in the unit cell), so we build N_atoms small drive vectors.
        std::vector<SpinVector> drive_in1(lat.N_atoms_SU2,
            SpinVector::Constant(lat.spin_dim_SU2, 0.01));
        std::vector<SpinVector> drive_in2 = drive_in1;
        lat.set_pulse_SU2(drive_in1, 0.0, drive_in2, 0.5, 1.0, 0.05, 5.0);
    }

    const size_t total = lat.lattice_size_SU2 * lat.spin_dim_SU2 +
                         lat.lattice_size_SU3 * lat.spin_dim_SU3;
    std::vector<double> x(total), dxdt(total);
    // Pack initial state.
    {
        size_t off = 0;
        for (size_t i = 0; i < lat.lattice_size_SU2; ++i) {
            for (size_t d = 0; d < lat.spin_dim_SU2; ++d) {
                x[off + i * lat.spin_dim_SU2 + d] = lat.spins_SU2[i](d);
            }
        }
        off = lat.lattice_size_SU2 * lat.spin_dim_SU2;
        for (size_t i = 0; i < lat.lattice_size_SU3; ++i) {
            for (size_t d = 0; d < lat.spin_dim_SU3; ++d) {
                x[off + i * lat.spin_dim_SU3 + d] = lat.spins_SU3[i](d);
            }
        }
    }
    double t = 0.0;
    for (long i = 0; i < n_rhs; ++i) {
        lat.landau_lifshitz(x, dxdt, t);
        t += 1e-3;
    }
    volatile double sink = dxdt[0]; (void)sink;
}

// ---------------------------------------------------------------------------
// Family 3: StrainPhononLattice
// ---------------------------------------------------------------------------

unique_ptr<StrainPhononLattice> build_strain_honeycomb_lat(int L,
                                                           bool with_me,
                                                           bool local_strain) {
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
    if (local_strain) {
        el.K_gradient = 0.5;
    }

    StrainDriveParams dr{};
    dr.E0_1 = 0.0; dr.E0_2 = 0.0;

    lat->set_parameters(me, el, dr);
    if (local_strain) {
        lat->init_local_strain();
    }
    lat->init_random();
    return lat;
}

void bench_strain_rhs(StrainPhononLattice& lat, long n_rhs) {
    const size_t total = lat.state_size;
    std::vector<double> x(total), dxdt(total);
    // Pack initial state. Spins go first; strain (or per-cell strain) is
    // packed via the lattice's PT extra-DOF interface, which matches the
    // ODE-state layout exactly.
    for (size_t i = 0; i < lat.lattice_size; ++i) {
        const size_t idx = i * lat.spin_dim;
        x[idx]   = lat.spins[i](0);
        x[idx+1] = lat.spins[i](1);
        x[idx+2] = lat.spins[i](2);
    }
    lat.pack_extra_dof(x.data() + lat.spin_dim * lat.lattice_size);
    double t = 0.0;
    for (long i = 0; i < n_rhs; ++i) {
        lat.ode_system(x, dxdt, t);
        t += 1e-3;
    }
    volatile double sink = dxdt[0]; (void)sink;
}

}  // namespace

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    // Initialize MPI: a few of the lattice setup paths (e.g. set_pulse_*)
    // call MPI_Comm_rank to localize per-rank initialization, so MPI must
    // be alive for the whole benchmark run even though we always run
    // single-rank.
    int mpi_already = 0;
    MPI_Initialized(&mpi_already);
    if (!mpi_already) MPI_Init(&argc, &argv);

    CliArgs args = parse_args(argc, argv);
    if (args.help) {
        print_help();
        if (!mpi_already) MPI_Finalize();
        return 0;
    }

#ifdef _OPENMP
    if (args.threads > 0) omp_set_num_threads(args.threads);
#endif

    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif

    (void)now_sec();

    if (!args.csv) {
        cout << "ClassicalSpin_Cpp molecular-dynamics RHS benchmark\n";
        cout << "  CXX threads (OpenMP): " << n_threads << "\n\n";
    }

    print_header(args.csv);

    // ---- lattice family ----
    struct LatticeBench {
        string name;
        int    default_L;
        long   default_rhs;
        std::function<unique_ptr<Lattice>(int)> build;
    };
    vector<LatticeBench> lattice_bench = {
        { "heisenberg-honeycomb-su2", 32, 4000, build_heisenberg_honeycomb },
        { "kitaev-honeycomb-su2",     32, 4000, build_kitaev_honeycomb_lat },
    };
    if (family_selected(args.family, "lattice")) {
        for (auto& m : lattice_bench) {
            if (!model_selected(args.model, m.name)) continue;
            int L = (args.L > 0) ? args.L : m.default_L;
            auto lat = m.build(L);
            const long n = (args.rhs > 0) ? args.rhs : m.default_rhs;
            const long warm = (args.warmup > 0) ? args.warmup : std::max(long(500), n / 4);
            bench_lattice_rhs(*lat, warm, 0.0);
            for (int rep = 0; rep < args.repeats; ++rep) {
                double t0 = now_sec();
                bench_lattice_rhs(*lat, n, 0.0);
                double t1 = now_sec();
                double dt = std::max(1e-9, t1 - t0);
                BenchRow r;
                r.family = "lattice"; r.model = m.name; r.L = L;
                r.lattice_size = lat->lattice_size;
                r.rhs_calls = n; r.repeat = rep;
                r.wall_sec = dt;
                r.rhs_per_sec = double(n) / dt;
                r.site_updates_per_sec = double(n) * double(r.lattice_size) / dt;
                r.threads = n_threads;
                print_row(r, args.csv);
                std::cout.flush();
            }
        }
    }

    // ---- mixed family ----
    struct MixedBench {
        string name;
        int    default_L;
        long   default_rhs;
        bool   trilinear;
        bool   driven;
    };
    vector<MixedBench> mixed_bench = {
        { "tmfeo3-bilinear",  4, 1500, false, false },
        { "tmfeo3-trilinear", 4,  300, true,  false },
        { "tmfeo3-driven",    4, 1500, false, true  },
    };
    if (family_selected(args.family, "mixed")) {
        for (auto& m : mixed_bench) {
            if (!model_selected(args.model, m.name)) continue;
            int L = (args.L > 0) ? args.L : m.default_L;
            auto lat = build_tmfeo3_lat(L, m.trilinear);
            const long n = (args.rhs > 0) ? args.rhs : m.default_rhs;
            const long warm = (args.warmup > 0) ? args.warmup : std::max(long(200), n / 4);
            bench_mixed_rhs(*lat, warm, m.driven);
            for (int rep = 0; rep < args.repeats; ++rep) {
                double t0 = now_sec();
                bench_mixed_rhs(*lat, n, m.driven);
                double t1 = now_sec();
                double dt = std::max(1e-9, t1 - t0);
                size_t N = lat->lattice_size_SU2 + lat->lattice_size_SU3;
                BenchRow r;
                r.family = "mixed"; r.model = m.name; r.L = L;
                r.lattice_size = N;
                r.rhs_calls = n; r.repeat = rep;
                r.wall_sec = dt;
                r.rhs_per_sec = double(n) / dt;
                r.site_updates_per_sec = double(n) * double(N) / dt;
                r.threads = n_threads;
                print_row(r, args.csv);
                std::cout.flush();
            }
        }
    }

    // ---- strain family ----
    struct StrainBench {
        string name;
        int    default_L;
        long   default_rhs;
        bool   with_me;
        bool   local_strain;
    };
    vector<StrainBench> strain_bench = {
        { "strain-honeycomb-bilin", 24, 2000, false, false },
        { "strain-honeycomb-me",    24, 1000, true,  false },
        { "strain-honeycomb-local", 24,  500, true,  true  },
    };
    if (family_selected(args.family, "strain")) {
        for (auto& m : strain_bench) {
            if (!model_selected(args.model, m.name)) continue;
            int L = (args.L > 0) ? args.L : m.default_L;
            auto lat = build_strain_honeycomb_lat(L, m.with_me, m.local_strain);
            const long n = (args.rhs > 0) ? args.rhs : m.default_rhs;
            const long warm = (args.warmup > 0) ? args.warmup : std::max(long(200), n / 4);
            bench_strain_rhs(*lat, warm);
            for (int rep = 0; rep < args.repeats; ++rep) {
                double t0 = now_sec();
                bench_strain_rhs(*lat, n);
                double t1 = now_sec();
                double dt = std::max(1e-9, t1 - t0);
                BenchRow r;
                r.family = "strain"; r.model = m.name; r.L = L;
                r.lattice_size = lat->lattice_size;
                r.rhs_calls = n; r.repeat = rep;
                r.wall_sec = dt;
                r.rhs_per_sec = double(n) / dt;
                r.site_updates_per_sec = double(n) * double(r.lattice_size) / dt;
                r.threads = n_threads;
                print_row(r, args.csv);
                std::cout.flush();
            }
        }
    }

    if (!mpi_already) MPI_Finalize();
    return 0;
}

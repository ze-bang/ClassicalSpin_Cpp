#ifndef LATTICE_H
#define LATTICE_H

#define _USE_MATH_DEFINES
#include "unitcell.h"
#include "simple_linear_alg.h"
#include <cmath>
#include <numbers>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <random>
#include <chrono>
#include <math.h>
#include <functional>
#include <mpi.h>
#include "binning_analysis.h"
#include <sstream>
#include <cstdint>
#include <deque>

template<size_t N, size_t N_ATOMS, size_t dim1, size_t dim2, size_t dim>
class lattice
{   
    public:

    typedef array<array<double,N>,N_ATOMS*dim1*dim2*dim> spin_config;
    typedef function<array<double,N>(const array<double,N> &, const array<double, N> &)> cross_product_method;
    typedef function<spin_config(double &, const spin_config &, const double, cross_product_method)> ODE_method;

    UnitCell<N, N_ATOMS> UC;
    size_t lattice_size;
    spin_config  spins;
    array<array<double,3>, N_ATOMS*dim1*dim2*dim> site_pos;
    array<array<double, N*N>, 3> twist_matrices;
    array<array<double, N>, 3> rotation_axis;

    //Lookup table for the lattice
    spin_config field;
    array<array<double, N>, N_ATOMS> field_drive_1;
    array<array<double, N>, N_ATOMS> field_drive_2;
    double field_drive_freq;
    double field_drive_amp;
    double field_drive_width;
    double t_B_1;
    double t_B_2;

    array<array<double, N * N>, N_ATOMS*dim1*dim2*dim> onsite_interaction;

    array<vector<array<double, N * N>>, N_ATOMS*dim1*dim2*dim> bilinear_interaction;
    array<vector<array<double, N * N * N>>, N_ATOMS*dim1*dim2*dim>  trilinear_interaction;

    array<vector<size_t>, N_ATOMS*dim1*dim2*dim> bilinear_partners;
    array<vector<array<size_t, 2>>, N_ATOMS*dim1*dim2*dim> trilinear_partners;

    size_t num_bi;
    size_t num_tri;
    size_t num_gen;
    float spin_length;

    // Optimization member variables for energy caching
    private:
    // Pre-allocated temporary arrays to avoid allocations in hot loops
    mutable array<double,N> temp_spin_array;
    mutable array<double,N> temp_local_field;
    
    // Twist-boundary Monte Carlo support
    // For each bilinear neighbor of a site, record whether the partner wraps across a boundary in x (dim1), y (dim2), z (dim)
    // Values are -1 for negative wrap, 0 for no wrap, +1 for positive wrap
    array<vector<array<int8_t,3>>, N_ATOMS*dim1*dim2*dim> bilinear_wrap_dir;
    
    // Boundary sites per lattice dimension (0->dim1, 1->dim2, 2->dim)
    array<vector<size_t>, 3> boundary_sites_per_dim;
    array<size_t, 3> boundary_thickness; // how many layers from each face are affected by TBC (max abs neighbor offset per dimension)
    
    public:
    // Configure twist axes per dimension (0:x,1:y,2:z index of lattice directions)
    void set_twist_axes(const array<array<double,N>,3>& axes){
        rotation_axis = axes;
    }
    const array<array<double,N>,3>& get_twist_axes() const { return rotation_axis; }
    const array<array<double,N*N>,3>& get_twist_matrices() const { return twist_matrices; }

    array<double,N> gen_random_spin(float spin_l){
        array<double,N> temp_spin;
        array<double,N-2> euler_angles;
        // double z = random_double(-1,1, gen);
        double z = random_double_lehman(-1,1);
        double r = sqrt(1.0 - z*z);

        for(size_t i = 0; i < N-2; ++i){
            // euler_angles[i] = random_double(0, 2*M_PI, gen);
            euler_angles[i] = random_double_lehman(0, 2*M_PI);
            temp_spin[i] = r;
            for(size_t j = 0; j < i; ++j){
                temp_spin[i] *= sin(euler_angles[j]);
            }
            if (i == N-3){
                temp_spin[i+1] = temp_spin[i]*sin(euler_angles[i]);
            }
            temp_spin[i] *= cos(euler_angles[i]);

        }
        temp_spin[N-1] = z;
        return temp_spin*spin_l;
    }


    size_t flatten_index(size_t i, size_t j, size_t k, size_t l){
        return i*dim2*dim*N_ATOMS+ j*dim*N_ATOMS+ k*N_ATOMS + l;
    }
    
    size_t periodic_boundary(int i, size_t D){
        if(i < 0){
            return size_t((D+i) % D);
        }
        else{
            return size_t(i % D);
        }
    }

    size_t flatten_index_periodic_boundary(int i, int j, int k, int l){
        return periodic_boundary(i, dim1)*dim2*dim*N_ATOMS+ periodic_boundary(j, dim2)*dim*N_ATOMS+ periodic_boundary(k, dim)*N_ATOMS + l;
    }

    // Helper for default twist_matrix argument
    array<array<double, N*N>, 3> default_twist_matrix() {
        array<array<double, N*N>, 3> twist_matrix = {};
        for (size_t d = 0; d < 3; ++d) {
            for (size_t i = 0; i < N; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    twist_matrix[d][i*N + j] = (i == j) ? 1.0 : 0.0;
                }
            }
        }
        return twist_matrix;
    }


    lattice(const UnitCell<N, N_ATOMS> *atoms, float spin_l=1, bool periodic = true)
        : UC(*atoms)
    {
        array<array<double,3>, N_ATOMS> basis;
        array<array<double,3>, 3> unit_vector;

        lattice_size = dim1*dim2*dim*N_ATOMS;
        basis = UC.lattice_pos;
        unit_vector = UC.lattice_vectors;
        spin_length = spin_l;

    set_pulse({{0}}, 0, {{0}}, 0, 0, 1, 0);
        srand (time(NULL));
        seed_lehman(rand()*2+1);

        size_t total_sites = dim1 * dim2 * dim * N_ATOMS;
        size_t processed_sites = 0;
        
        cout << "Initializing lattice sites..." << endl;
        cout << "Progress: [";
        for(int p = 0; p < 50; ++p) cout << " ";
        cout << "] 0%" << flush;
        
        // Precompute maximum neighbor offset to determine boundary thickness in each dimension
        boundary_thickness = {0,0,0};
        for (size_t l=0; l<N_ATOMS; ++l){
            auto bilinear_matched_pre = UC.bilinear_interaction.equal_range(l);
            for (auto m = bilinear_matched_pre.first; m != bilinear_matched_pre.second; ++m){
                bilinear<N> J = m->second;
                boundary_thickness[0] = std::max(boundary_thickness[0], size_t(std::abs(J.offset[0])));
                boundary_thickness[1] = std::max(boundary_thickness[1], size_t(std::abs(J.offset[1])));
                boundary_thickness[2] = std::max(boundary_thickness[2], size_t(std::abs(J.offset[2])));
            }
            auto trilinear_matched_pre = UC.trilinear_interaction.equal_range(l);
            for (auto m = trilinear_matched_pre.first; m != trilinear_matched_pre.second; ++m){
                trilinear<N> J = m->second;
                boundary_thickness[0] = std::max(boundary_thickness[0], size_t(std::max(std::abs(J.offset1[0]), std::abs(J.offset2[0]))));
                boundary_thickness[1] = std::max(boundary_thickness[1], size_t(std::max(std::abs(J.offset1[1]), std::abs(J.offset2[1]))));
                boundary_thickness[2] = std::max(boundary_thickness[2], size_t(std::max(std::abs(J.offset1[2]), std::abs(J.offset2[2]))));
            }
        }

        for (size_t i=0; i< dim1; ++i){
            for (size_t j=0; j< dim2; ++j){
                for(size_t k=0; k< dim;++k){
                    for (size_t l=0; l< N_ATOMS;++l){
                        size_t current_site_index = flatten_index(i,j,k,l);

                        site_pos[current_site_index]  = unit_vector[0]*int(i) + unit_vector[1]*int(j)  + unit_vector[2]*int(k)  + basis[l];
                        spins[current_site_index] = gen_random_spin(spin_length);
                        field[current_site_index] = UC.field[l];
                        onsite_interaction[current_site_index] = UC.onsite_interaction[l];
                        auto bilinear_matched = UC.bilinear_interaction.equal_range(l);
                        for (auto m = bilinear_matched.first; m != bilinear_matched.second; ++m){
                            bilinear<N> J = m->second;
                            int partner_i = int(i) + J.offset[0];
                            int partner_j = int(j) + J.offset[1];
                            int partner_k = int(k) + J.offset[2];
                            size_t partner = flatten_index_periodic_boundary(partner_i, partner_j, partner_k, J.partner);
                            if (periodic || partner_i < dim1 && partner_i >= 0 && partner_j < dim2 && partner_j >= 0 && partner_k < dim && partner_k >= 0) {
                                array<int8_t,3> wrap_dir = {0,0,0};
                                if (partner_i < 0) wrap_dir[0] = -1; else if (partner_i >= int(dim1)) wrap_dir[0] = +1;
                                if (partner_j < 0) wrap_dir[1] = -1; else if (partner_j >= int(dim2)) wrap_dir[1] = +1;
                                if (partner_k < 0) wrap_dir[2] = -1; else if (partner_k >= int(dim)) wrap_dir[2] = +1;
                                array<double, N * N> bilinear_matrix_here = J.bilinear_interaction;
                                bilinear_interaction[current_site_index].push_back(bilinear_matrix_here);
                                bilinear_partners[current_site_index].push_back(partner);
                                bilinear_wrap_dir[current_site_index].push_back(wrap_dir);
                                bilinear_interaction[partner].push_back(transpose2D<N, N>(bilinear_matrix_here));
                                bilinear_partners[partner].push_back(current_site_index);
                                array<int8_t,3> wrap_dir_partner = {int8_t(-wrap_dir[0]), int8_t(-wrap_dir[1]), int8_t(-wrap_dir[2])};
                                bilinear_wrap_dir[partner].push_back(wrap_dir_partner);
                            }else{
                                array<double, N * N> zero_matrix = {{{0}}};
                                bilinear_interaction[current_site_index].push_back(zero_matrix);
                                bilinear_partners[current_site_index].push_back(partner);
                                bilinear_wrap_dir[current_site_index].push_back({0,0,0});
                                bilinear_interaction[partner].push_back(zero_matrix);
                                bilinear_partners[partner].push_back(current_site_index);
                                bilinear_wrap_dir[partner].push_back({0,0,0});
                            }
                        }
                        auto trilinear_matched = UC.trilinear_interaction.equal_range(l);
                        for (auto m = trilinear_matched.first; m != trilinear_matched.second; ++m){
                            trilinear<N> J = m->second;
                            int partner1_i = int(i) + J.offset1[0];
                            int partner1_j = int(j) + J.offset1[1];
                            int partner1_k = int(k) + J.offset1[2];
                            int partner2_i = int(i) + J.offset2[0];
                            int partner2_j = int(j) + J.offset2[1];
                            int partner2_k = int(k) + J.offset2[2];

                            size_t partner1 = flatten_index_periodic_boundary(partner1_i, partner1_j, partner1_k, J.partner1);
                            size_t partner2 = flatten_index_periodic_boundary(partner2_i, partner2_j, partner2_k, J.partner2);
                            if (periodic || (partner1_i < dim1 && partner1_i >= 0 && partner1_j < dim2 && partner1_j >= 0 && partner1_k < dim && partner1_k >= 0) &&
                            (partner2_i < dim1 && partner2_i >= 0 && partner2_j < dim2 && partner2_j >= 0 && partner2_k < dim && partner2_k >= 0)) {
                                // Note: For trilinear terms, we will apply twist only using wrap relative to the current site.
                                array<int8_t,3> wrap1 = {0,0,0};
                                array<int8_t,3> wrap2 = {0,0,0};
                                if (partner1_i < 0) wrap1[0] = -1; else if (partner1_i >= int(dim1)) wrap1[0] = +1;
                                if (partner1_j < 0) wrap1[1] = -1; else if (partner1_j >= int(dim2)) wrap1[1] = +1;
                                if (partner1_k < 0) wrap1[2] = -1; else if (partner1_k >= int(dim)) wrap1[2] = +1;
                                if (partner2_i < 0) wrap2[0] = -1; else if (partner2_i >= int(dim1)) wrap2[0] = +1;
                                if (partner2_j < 0) wrap2[1] = -1; else if (partner2_j >= int(dim2)) wrap2[1] = +1;
                                if (partner2_k < 0) wrap2[2] = -1; else if (partner2_k >= int(dim)) wrap2[2] = +1;
                                
                                trilinear_interaction[current_site_index].push_back(J.trilinear_interaction);
                                trilinear_partners[current_site_index].push_back({partner1, partner2});
                                // We do not push wrap metadata arrays for trilinear explicitly; handled at evaluation time from current site perspective.

                                trilinear_interaction[partner1].push_back(transpose3D(J.trilinear_interaction, N, N, N));
                                trilinear_partners[partner1].push_back({partner2, current_site_index});

                                trilinear_interaction[partner2].push_back(transpose3D(transpose3D(J.trilinear_interaction, N, N, N), N, N, N));
                                trilinear_partners[partner2].push_back({current_site_index, partner1});
                            }else{
                                array<double, N*N*N> zero_matrix = {{{0}}};
                                trilinear_interaction[current_site_index].push_back(zero_matrix);
                                trilinear_partners[current_site_index].push_back({partner1, partner2});

                                trilinear_interaction[partner1].push_back(zero_matrix);
                                trilinear_partners[partner1].push_back({partner2, current_site_index});

                                trilinear_interaction[partner2].push_back(zero_matrix);
                                trilinear_partners[partner2].push_back({current_site_index, partner1});
                            }
                        }
                    
                        // Update progress bar
                        processed_sites++;
                        if (processed_sites % max(total_sites / 100, size_t(1)) == 0 || processed_sites == total_sites) {
                            double progress = 100.0 * processed_sites / total_sites;
                            int filled = static_cast<int>(progress / 2);
                            
                            cout << "\rProgress: [";
                            for(int p = 0; p < 50; ++p) {
                            if(p < filled) cout << "=";
                            else if(p == filled) cout << ">";
                            else cout << " ";
                            }
                            cout << "] " << fixed << setprecision(1) << progress << "%" << flush;
                        }
                    }
                }
            }
        }

        twist_matrices = default_twist_matrix();
        // Default twist rotation axes (z-axis). Users can override via set_twist_axes.
        for (size_t d=0; d<3; ++d) {
            for (size_t t=0; t<N; ++t) rotation_axis[d][t] = 0.0;
            if (N>=3) rotation_axis[d][2] = 1.0;
        }

        num_bi = bilinear_partners[0].size();
        num_tri = trilinear_partners[0].size();
        num_gen = spins[0].size();
        cout << "\nLattice initialization complete!" << endl;
        cout << "Total sites: " << lattice_size << endl;
        cout << "Bilinear interactions: " << num_bi << endl;
        cout << "Trilinear interactions: " << num_tri << endl;
        cout << "Spin dimension: " << num_gen << endl;
        build_boundary_sites();
    }

    lattice(const lattice<N, N_ATOMS, dim1, dim2, dim> *lattice_in){
        UC = lattice_in->UC;
        lattice_size = lattice_in->lattice_size;
        spins = lattice_in->spins;
        site_pos = lattice_in->site_pos;
        field = lattice_in->field;
        onsite_interaction = lattice_in->onsite_interaction;
        bilinear_interaction = lattice_in->bilinear_interaction;
        trilinear_interaction = lattice_in->trilinear_interaction;
        bilinear_partners = lattice_in->bilinear_partners;
        trilinear_partners = lattice_in->trilinear_partners;
        num_bi = lattice_in->num_bi;
        num_tri = lattice_in->num_tri;
        num_gen = lattice_in->num_gen;
    };

    void reset_lattice(const lattice<N, N_ATOMS, dim1, dim2, dim> *lattice_in){
        UC = lattice_in->UC;
        lattice_size = lattice_in->lattice_size;
        spins = lattice_in->spins;
        site_pos = lattice_in->site_pos;
        field = lattice_in->field;
        onsite_interaction = lattice_in->onsite_interaction;
        bilinear_interaction = lattice_in->bilinear_interaction;
        trilinear_interaction = lattice_in->trilinear_interaction;
        bilinear_partners = lattice_in->bilinear_partners;
        trilinear_partners = lattice_in->trilinear_partners;
        num_bi = lattice_in->num_bi;
        num_tri = lattice_in->num_tri;
        num_gen = lattice_in->num_gen;
    };

    
    void build_boundary_sites() {
        // Build boundary site lists based on boundary_thickness and lattice dimensions
        boundary_sites_per_dim[0].clear();
        boundary_sites_per_dim[1].clear();
        boundary_sites_per_dim[2].clear();
        for (size_t i=0; i< dim1; ++i){
            bool is_boundary_x = (dim1>1) && (i < boundary_thickness[0] || i >= dim1 - boundary_thickness[0]);
            for (size_t j=0; j< dim2; ++j){
                bool is_boundary_y = (dim2>1) && (j < boundary_thickness[1] || j >= dim2 - boundary_thickness[1]);
                for (size_t k=0; k< dim; ++k){
                    bool is_boundary_z = (dim>1) && (k < boundary_thickness[2] || k >= dim - boundary_thickness[2]);
                    for (size_t l=0; l<N_ATOMS; ++l){
                        size_t idx = flatten_index(i,j,k,l);
                        if (is_boundary_x) boundary_sites_per_dim[0].push_back(idx);
                        if (is_boundary_y) boundary_sites_per_dim[1].push_back(idx);
                        if (is_boundary_z) boundary_sites_per_dim[2].push_back(idx);
                    }
                }
            }
        }
    }

    
    static array<double, N*N> identityNN(){
        array<double, N*N> I = {};
        for (size_t i=0;i<N;++i) I[i*N + i] = 1.0;
        return I;
    }
    
    static array<double, N*N> rotation_from_axis_angle(const array<double, N>& axis_in, double angle){
        // Only defined for 3D spins; otherwise return identity
        if constexpr (N == 3) {
            double ax = axis_in[0], ay = axis_in[1], az = axis_in[2];
            double nrm = sqrt(ax*ax + ay*ay + az*az);
            if (nrm < 1e-12) return identityNN();
            ax /= nrm; ay /= nrm; az /= nrm;
            double c = cos(angle), s = sin(angle), C = 1.0 - c;
            array<double, N*N> R = { ax*ax*C + c,     ax*ay*C - az*s, ax*az*C + ay*s,
                                      ay*ax*C + az*s, ay*ay*C + c,    ay*az*C - ax*s,
                                      az*ax*C - ay*s, az*ay*C + ax*s, az*az*C + c };
            return R;
        } else {
            return identityNN();
        }
    }
    
    array<double,N> apply_twist_to_partner_spin(const array<double,N>& partner_spin,
                                                const array<int8_t,3>& wrap) const {
        array<double,N> s = partner_spin;
        for (size_t d=0; d<3; ++d){
            int8_t w = wrap[d];
            if (w == 0) continue;
            if (w > 0) {
                s = multiply<N,N>(twist_matrices[d], s);
            } else {
                // inverse (transpose) for rotations
                s = multiply<N,N>(transpose2D<N,N>(twist_matrices[d]), s);
            }
        }
        return s;
    }
    
    void set_pulse(const array<array<double,N>, N_ATOMS> &field_in, double t_B, const array<array<double,N>, N_ATOMS> &field_in_2, double t_B_2, double pulse_amp, double pulse_width, double pulse_freq){
        field_drive_1 = field_in;
        field_drive_2 = field_in_2;
        field_drive_amp = pulse_amp;
        field_drive_freq = pulse_freq;
        field_drive_width = pulse_width;
        t_B_1 = t_B;
        t_B_2 = t_B_2;
    }
    void reset_pulse(){
        field_drive_1 = {{0}};
        field_drive_2 = {{0}};
        field_drive_amp = 0;
        field_drive_freq = 0;
        field_drive_width = 1;
        t_B_1 = 0;
        t_B_2 = 0;
    }

    const array<double, N> drive_field_T(double currT, size_t ind){
        double factor1 = double(field_drive_amp*exp(-pow((currT-t_B_1)/(2*field_drive_width),2))*cos(2*M_PI*field_drive_freq*(currT-t_B_1)));
        double factor2 = double(field_drive_amp*exp(-pow((currT-t_B_2)/(2*field_drive_width),2))*cos(2*M_PI*field_drive_freq*(currT-t_B_2)));
        return field_drive_1[ind]*factor1 + field_drive_2[ind]*factor2;
    }


    void set_spin(size_t site_index, array<double, N> &spin_in){
        spins[site_index] = spin_in;
    }

    void read_spin_from_file(const string &filename){
        ifstream file;
        file.open(filename);
        if (!file){
            cout << "Unable to open file " << filename << endl;
            exit(1);
        }
        string line;
        size_t count = 0;
        while(getline(file, line)){
            istringstream iss(line);
            array<double, N> spin;
            for(size_t i = 0; i < N; ++i){
                iss >> spin[i];
            }
            spins[count] = spin;
            count++;
        }
        file.close();
    }

    double site_energy(array<double, N> &spin_here, size_t site_index){
        double single_site_energy = 0, double_site_energy = 0, triple_site_energy = 0;
        single_site_energy -= dot(spin_here, field[site_index]);
        single_site_energy += contract(spin_here, onsite_interaction[site_index], spin_here);
        #pragma omp simd
        for (size_t i=0; i< num_bi; ++i) {
            array<double,N> partner_spin_eff = apply_twist_to_partner_spin(spins[bilinear_partners[site_index][i]], bilinear_wrap_dir[site_index][i]);
            double_site_energy += contract(spin_here, bilinear_interaction[site_index][i], partner_spin_eff);
        }
        #pragma omp simd
        for (size_t i=0; i < num_tri; ++i){
            // For trilinear, approximate by applying twists from current site to both partners when they wrap.
            size_t p1 = trilinear_partners[site_index][i][0];
            size_t p2 = trilinear_partners[site_index][i][1];
            // We don't have stored wrap metadata for trilinear; as a fallback, use identity (no twist).
            // Extend here if trilinear twist is needed.
            triple_site_energy += contract_trilinear(trilinear_interaction[site_index][i], spin_here, spins[p1], spins[p2]);
        }
        return single_site_energy + double_site_energy/2 + triple_site_energy/3;
    }

    array<double, dim1*dim2*dim*N_ATOMS> local_energy_densities(spin_config &curr_spins){
        array<double, dim1*dim2*dim*N_ATOMS> local_energies;
        #pragma omp simd
        for(size_t i = 0; i < dim1*dim2*dim*N_ATOMS; ++i){
            local_energies[i] = site_energy(curr_spins[i], i);
        }
        return local_energies;
    }

    double site_energy_diff(const array<double, N> &new_spins, const array<double, N> &old_spins, const size_t site_index) const {
        double single_site_energy = 0, double_site_energy = 0, triple_site_energy = 0;
        single_site_energy -= dot(new_spins - old_spins, field[site_index]);
        single_site_energy += contract(new_spins, onsite_interaction[site_index], new_spins) - contract(old_spins, onsite_interaction[site_index], old_spins);
        #pragma omp simd
        for (size_t i=0; i< num_bi; ++i) {
            array<double,N> partner_spin_eff = apply_twist_to_partner_spin(spins[bilinear_partners[site_index][i]], bilinear_wrap_dir[site_index][i]);
            double_site_energy += contract(new_spins, bilinear_interaction[site_index][i], partner_spin_eff)
                     - contract(old_spins, bilinear_interaction[site_index][i], partner_spin_eff);
        }
        #pragma omp simd
        for (size_t i=0; i < num_tri; ++i){
            size_t p1 = trilinear_partners[site_index][i][0];
            size_t p2 = trilinear_partners[site_index][i][1];
            triple_site_energy += contract_trilinear(trilinear_interaction[site_index][i], new_spins, spins[p1], spins[p2])
                     - contract_trilinear(trilinear_interaction[site_index][i], old_spins, spins[p1], spins[p2]);
        }
        return single_site_energy + double_site_energy + triple_site_energy;
    }

    double total_energy(spin_config &curr_spins){
        double field_energy = 0.0;
        double onsite_energy = 0.0;
        double bilinear_energy = 0.0;
        double trilinear_energy = 0.0;
        
        size_t test_site_index = 0; // Define a test site index, e.g., 0
        #pragma omp simd
        for(size_t i = 0; i < lattice_size; ++i){
            field_energy -= dot(curr_spins[i], field[i]);
            onsite_energy += contract(curr_spins[i], onsite_interaction[i], curr_spins[i]);
            #pragma omp simd
            for (size_t j=0; j< num_bi; ++j) {
            size_t partner_idx = bilinear_partners[i][j];
            array<double,N> partner_spin_eff = apply_twist_to_partner_spin(curr_spins[partner_idx], bilinear_wrap_dir[i][j]);
            bilinear_energy += contract(curr_spins[i], bilinear_interaction[i][j], partner_spin_eff);
            }

            #pragma omp simd
            for (size_t j=0; j < num_tri; ++j){
            trilinear_energy += contract_trilinear(trilinear_interaction[i][j], curr_spins[i], curr_spins[trilinear_partners[i][j][0]], curr_spins[trilinear_partners[i][j][1]]);
            }
        }
        return field_energy + onsite_energy + bilinear_energy/2 + trilinear_energy/3;
    }

    double energy_density(spin_config &curr_spins){
        return total_energy(curr_spins)/lattice_size;
    }
    
    array<double, N>  get_local_field(size_t site_index){
        array<double,N> local_field;
        local_field = multiply<N, N>(onsite_interaction[site_index], spins[site_index])*2;
        #pragma omp simd
        for (size_t i=0; i< num_bi; ++i) {
            array<double,N> partner_spin_eff = apply_twist_to_partner_spin(spins[bilinear_partners[site_index][i]], bilinear_wrap_dir[site_index][i]);
            local_field = local_field + multiply<N, N>(bilinear_interaction[site_index][i], partner_spin_eff);
        }
        #pragma omp simd
        for (size_t i=0; i < num_tri; ++i){
            array<double, N> current_spin_SU2_partner1 = spins[trilinear_partners[site_index][i][0]];
            array<double, N> current_spin_SU2_partner2 = spins[trilinear_partners[site_index][i][1]];
            local_field = local_field + contract_trilinear_field<N, N, N>(trilinear_interaction[site_index][i], current_spin_SU2_partner1, current_spin_SU2_partner2);
        }
        return local_field-field[site_index];
    }

    array<double, N>  get_local_field_lattice(size_t site_index, const spin_config &current_spin){
        array<double,N> local_field;
        local_field =  multiply<N, N>(onsite_interaction[site_index], spins[site_index]);
        #pragma omp simd
        for (size_t i=0; i< num_bi; ++i) {
            array<double,N> partner_spin_eff = apply_twist_to_partner_spin(current_spin[bilinear_partners[site_index][i]], bilinear_wrap_dir[site_index][i]);
            local_field = local_field + multiply<N, N>(bilinear_interaction[site_index][i], partner_spin_eff);
        }
        #pragma omp simd
        for (size_t i=0; i < num_tri; ++i){
            array<double, N> current_spin_SU2_partner1 = spins[trilinear_partners[site_index][i][0]];
            array<double, N> current_spin_SU2_partner2 = spins[trilinear_partners[site_index][i][1]];
            local_field = local_field + contract_trilinear_field<N, N, N>(trilinear_interaction[site_index][i], current_spin_SU2_partner1, current_spin_SU2_partner2);
        }
        return local_field-field[site_index];
    }

    // Perform a Metropolis update for global twist matrices along relevant dimensions
    void metropolis_twist_sweep(double T){
        // For each dimension that has length > 1, attempt one global angle move
        for (size_t d=0; d<3; ++d){
            size_t Ld = (d==0? dim1 : (d==1? dim2 : dim));
            if (Ld <= 1) continue; // not relevant

            // Energy on boundary sites before the move
            double E_before = 0.0;
            for (size_t idx : boundary_sites_per_dim[d]){
                E_before += site_energy(spins[idx], idx);
            }

            // Propose a small rotation update
            double delta = random_double_lehman(0, 2*M_PI);
            array<double, N*N> R_new = rotation_from_axis_angle(rotation_axis[d], delta);

            // Temporarily apply
            auto saved_R = twist_matrices[d];
            twist_matrices[d] = R_new;

            double E_after = 0.0;
            for (size_t idx : boundary_sites_per_dim[d]){
                E_after += site_energy(spins[idx], idx);
            }

            double dE = E_after - E_before;
            bool accept = (dE < 0) || (random_double_lehman(0,1) < exp(-dE/T));
            if (!accept){
                twist_matrices[d] = saved_R;
            } 
        }
    }


    void deterministic_sweep(){
        size_t count = 0;
        int i;
        while(count < lattice_size){
            i = random_int_lehman(lattice_size);
            array<double,N> local_field = get_local_field(i);
            double norm = sqrt(dot(local_field, local_field));
            if(norm == 0){
                continue;
            }
            else{
                for(size_t j=0; j < N; ++j){
                    spins[i][j] = -local_field[j]/norm*spin_length;
                }
            }
            count++;
        }
    }
    
    array<double,N> gaussian_move(const array<double,N> &current_spin, double sigma=60){
        array<double,N> new_spin;
        new_spin = current_spin + gen_random_spin(spin_length)*sigma;
        return new_spin/sqrt(dot(new_spin, new_spin)) * spin_length;
    }

    void overrelaxation(){
        array<double,N> local_field;
        int i;
        double proj;
        size_t count = 0;
        while(count < lattice_size){
            // i = random_int(0, lattice_size-1, gen);
            i = random_int_lehman(lattice_size);
            local_field = get_local_field(i);
            double norm = dot(local_field, local_field);
            if(norm == 0){
                continue;
            }
            else{
                proj = 2* dot(spins[i], local_field)/norm;
                spins[i] = local_field*proj - spins[i];
            }
            count++;
        }
    }

    double metropolis(spin_config &curr_spin, double T, bool gaussian=false, double sigma=60){
        double dE, r;
        int i;
        array<double,N> new_spin;
        int accept = 0;
        size_t count = 0;
        while(count < lattice_size){
            i = random_int_lehman(lattice_size);
            new_spin = gaussian ? gaussian_move(curr_spin[i], sigma) 
                                : gen_random_spin(spin_length);
            dE = site_energy_diff(new_spin, curr_spin[i], i);
            
            if(dE < 0 || random_double_lehman(0,1) < exp(-dE/T)){
                curr_spin[i] = new_spin;
                accept++;
            }
            count++;
        }

        double acceptance_rate = double(accept)/double(lattice_size);
        return acceptance_rate;
    }

    // =========================
    // Cluster Monte Carlo (Wolff & Swendsen–Wang)
    // =========================
    // Notes:
    // - These implementations use an embedded-Ising projection: pick a random unit vector r in R^N,
    //   define sigma_i = sign(s_i · r). Bonds are activated with probability
    //       p_ij = 1 - exp(-2 beta K_ij (s_i·r)(s_j·r)) for (s_i·r)(s_j·r) > 0, else 0,
    //   where K_ij = r^T J_ij r with J_ij the bilinear interaction matrix.
    // - Only bilinear interactions are used for cluster construction; trilinear, onsite terms,
    //   and twist matrices are ignored in bond activation and should be disabled for strict detailed balance.
    // - If twist matrices differ from identity or nonzero fields/onsite/trilinear exist, the methods remain
    //   usable as heuristic large moves, but strict balance is not guaranteed.

    // Helper: generate a random unit vector in R^N
    array<double,N> random_unit_vector() const {
        array<double,N> v = {};
        // reuse generator that yields on the sphere but scaled by spin_length
        v = const_cast<lattice*>(this)->gen_random_spin(1.0);
        // ensure normalization (defensive)
        double n2 = dot(v, v);
        if (n2 <= 0) { v[0] = 1.0; return v; }
        return v * (1.0 / sqrt(n2));
    }

    // Helper: reflect a spin across the hyperplane perpendicular to r
    static array<double,N> reflect_across_plane(const array<double,N>& s, const array<double,N>& r_unit){
        double proj = dot(s, r_unit);
        return s - r_unit * (2.0 * proj);
    }

    // Helper: projected coupling along r for edge (i -> neighborIdx)
    inline double projected_coupling_r(size_t i, size_t neighborIdx, const array<double,N>& r_unit) const {
        const auto& Jij = bilinear_interaction[i][neighborIdx];
        // K = r^T J r
        array<double,N> Jr = multiply<N,N>(Jij, r_unit);
        return dot(r_unit, Jr);
    }

    // Check whether twist matrices are identity (within tolerance)
    bool twist_is_identity(double tol=1e-12) const {
        for (size_t d=0; d<3; ++d){
            for (size_t i=0; i<N; ++i){
                for (size_t j=0; j<N; ++j){
                    double want = (i==j)?1.0:0.0;
                    if (fabs(twist_matrices[d][i*N + j] - want) > tol) return false;
                }
            }
        }
        return true;
    }

    // Wolff single-cluster update; returns cluster size. If useGhostField is true, approximate field handling by ghost bonds.
    size_t wolff_update(double T, bool useGhostField=false){
        if (T <= 0) return 0;
        const double beta = 1.0 / T;
        // random seed site
        size_t seed = random_int_lehman(lattice_size);
        array<double,N> r = random_unit_vector();

        // Precompute projections s·r for all sites
        vector<double> proj(lattice_size);
        for (size_t i=0;i<lattice_size;++i) proj[i] = dot(spins[i], r);

        // BFS stack
        vector<uint8_t> in_cluster(lattice_size, 0);
        vector<size_t> stack;
        stack.reserve(lattice_size/10 + 1);
        bool attached_to_ghost = false;

        in_cluster[seed] = 1;
        stack.push_back(seed);

        while(!stack.empty()){
            size_t i = stack.back(); stack.pop_back();
            double si = proj[i];
            // Ghost-field bond (optional, heuristic): do not flip cluster if it attaches to ghost
            if (useGhostField){
                double hproj = dot(field[i], r); // Zeeman projection along r
                double arg = 2.0 * beta * hproj * std::max(0.0, si);
                if (arg > 0){
                    double pghost = 1.0 - exp(-arg);
                    if (random_double_lehman(0,1) < pghost) attached_to_ghost = true;
                }
            }

            // Visit neighbors
            for (size_t n=0; n<bilinear_partners[i].size(); ++n){
                size_t j = bilinear_partners[i][n];
                if (in_cluster[j]) continue;
                double sj = proj[j];
                if (si * sj <= 0) continue; // only like-signed projections can bond
                double K = projected_coupling_r(i, n, r);
                if (K <= 0) continue; // only ferromagnetic along r
                double p = 1.0 - exp(-2.0 * beta * K * si * sj);
                if (random_double_lehman(0,1) < p){
                    in_cluster[j] = 1;
                    stack.push_back(j);
                }
            }
        }

        // Flip (reflect) cluster spins if not attached to ghost
        size_t cluster_size = 0;
        if (!attached_to_ghost){
            for (size_t i=0;i<lattice_size;++i){
                if (in_cluster[i]){
                    spins[i] = reflect_across_plane(spins[i], r);
                    cluster_size++;
                }
            }
        }

        return cluster_size;
    }

    // Swendsen–Wang sweep: build all clusters and reflect each with probability 1/2.
    // Returns the number of clusters flipped.
    size_t swendsen_wang_sweep(double T, bool useGhostField=false){
        if (T <= 0) return 0;
        const double beta = 1.0 / T;
        array<double,N> r = random_unit_vector();

        // Precompute projections
        vector<double> proj(lattice_size);
        for (size_t i=0;i<lattice_size;++i) proj[i] = dot(spins[i], r);

        // Union-Find structures
        vector<int> parent(lattice_size), sz(lattice_size,1);
        iota(parent.begin(), parent.end(), 0);
        auto findp = [&](auto&& self, int x)->int { return parent[x]==x? x : parent[x] = self(self, parent[x]); };
        auto unite = [&](int a, int b){ a=findp(findp, a); b=findp(findp, b); if (a==b) return; if (sz[a]<sz[b]) swap(a,b); parent[b]=a; sz[a]+=sz[b]; };

        // Build bonds; to avoid double counting, process only j with j>i
        for (size_t i=0; i<lattice_size; ++i){
            double si = proj[i];
            for (size_t n=0; n<bilinear_partners[i].size(); ++n){
                size_t j = bilinear_partners[i][n];
                if (j <= i) continue;
                double sj = proj[j];
                if (si * sj <= 0) continue;
                double K = projected_coupling_r(i, n, r);
                if (K <= 0) continue;
                double p = 1.0 - exp(-2.0 * beta * K * si * sj);
                if (random_double_lehman(0,1) < p){
                    unite((int)i, (int)j);
                }
            }
        }

        // Optional ghost bonds (heuristic): mark cluster roots attached to ghost to prevent flipping
        vector<uint8_t> forbid_flip(lattice_size, 0);
        if (useGhostField){
            for (size_t i=0;i<lattice_size;++i){
                double si = proj[i];
                double hproj = dot(field[i], r);
                double arg = 2.0 * beta * hproj * std::max(0.0, si);
                if (arg > 0){
                    double pghost = 1.0 - exp(-arg);
                    if (random_double_lehman(0,1) < pghost){
                        int root = findp(findp, (int)i);
                        forbid_flip[root] = 1;
                    }
                }
            }
        }

        // Decide flips per cluster root
        vector<uint8_t> flip_root(lattice_size, 0);
        for (size_t i=0; i<lattice_size; ++i){
            int rroot = findp(findp, (int)i);
            if ((int)i == rroot){
                if (!forbid_flip[rroot] && random_double_lehman(0,1) < 0.5) flip_root[rroot] = 1;
            }
        }

        // Apply reflections
        size_t flipped_clusters = 0;
        for (size_t i=0; i<lattice_size; ++i){
            int rroot = findp(findp, (int)i);
            if (flip_root[rroot]){
                spins[i] = reflect_across_plane(spins[i], r);
            }
        }
        for (size_t i=0; i<lattice_size; ++i){
            if ((int)i == findp(findp, (int)i) && flip_root[i]) flipped_clusters++;
        }
        return flipped_clusters;
    }

    // Convenience: perform k Wolff clusters at temperature T. Returns total flipped spins.
    size_t wolff_sweep(double T, size_t k=1, bool useGhostField=false){
        size_t total=0; for(size_t c=0;c<k;++c) total += wolff_update(T, useGhostField); return total; }

    // Convenience: one SW sweep at temperature T. Returns flipped cluster count.
    size_t sw_sweep(double T, bool useGhostField=false){ return swendsen_wang_sweep(T, useGhostField); }

    // Simple cluster-based annealing (does not include overrelaxation/twist updates)
    void cluster_annealing(double T_start, double T_end, size_t n_anneal, size_t wolff_per_temp,
                           bool use_sw=false, bool useGhostField=false, double cooling_rate=0.9,
                           string out_dir=""){
        if (!out_dir.empty()) { filesystem::create_directory(out_dir); }
        double T = T_start;
        while (T > T_end){
            if (use_sw){
                for (size_t i=0;i<n_anneal;++i) { swendsen_wang_sweep(T, useGhostField); }
                cout << "[Cluster SA] T=" << T << " SW sweeps=" << n_anneal << endl;
            } else {
                size_t flipped = 0;
                for (size_t i=0;i<n_anneal;++i) { flipped += wolff_sweep(T, wolff_per_temp, useGhostField); }
                cout << "[Cluster SA] T=" << T << " Wolff clusters=" << (n_anneal*wolff_per_temp)
                     << " flipped_spins~=" << flipped << endl;
            }
            T *= cooling_rate;
        }
        if (!out_dir.empty()){
            write_to_file_spin(out_dir + "/spin.txt", spins);
            write_to_file_pos(out_dir + "/pos.txt");
        }
    }


    void write_to_file_spin(string filename, spin_config towrite){
        ofstream myfile;
        myfile.open(filename);
        for(size_t i = 0; i<lattice_size; ++i){
            for(size_t j = 0; j<N; ++j){
                myfile << towrite[i][j] << " ";
            }
            myfile << endl;
        }
        myfile.close();
    }

    void write_to_file(string filename, spin_config towrite){
        ofstream myfile;
        myfile.open(filename, ios::app);
        for(size_t i = 0; i<lattice_size; ++i){
            for(size_t j = 0; j<N; ++j){
                myfile << towrite[i][j] << " ";
            }
            myfile << endl;
        }
        myfile.close();
    }
    void write_to_file_magnetization_init(string filename, double towrite){
        ofstream myfile;
        myfile.open(filename);
        myfile << towrite << " ";
        myfile << endl;
        myfile.close();
    }
    void write_to_file_magnetization(string filename, double towrite){
        ofstream myfile;
        myfile.open(filename, ios::app);
        myfile << towrite << " ";
        myfile << endl;
        myfile.close();
    }

    void write_to_file_magnetization_local(string filename, array<double, N> towrite){
        ofstream myfile;
        myfile.open(filename, ios::app);
        for(size_t j = 0; j<N; ++j){
            myfile << towrite[j] << " ";
        }
        myfile << endl;
        myfile.close();
    }

    void write_to_file_2d_vector_array(string filename, vector<array<double, N>> towrite){
        ofstream myfile;
        myfile.open(filename, ios::app);
        for(size_t i = 0; i<towrite.size(); ++i){
            for(size_t j = 0; j<N; ++j){
                myfile << towrite[i][j] << " ";
            }
            myfile << endl;
        }
        myfile.close();
    }

    void write_column_vector(string filename, vector<double> towrite){
        ofstream myfile;
        myfile.open(filename);
        for(size_t j = 0; j<towrite.size(); ++j){
            myfile << towrite[j] << endl;
        }
        myfile.close();
    }

    void write_to_file_pos(string filename){
        ofstream myfile;
        myfile.open(filename);
        for(size_t i = 0; i<lattice_size; ++i){
            for(size_t j = 0; j<3; ++j){
                myfile << site_pos[i][j] << " ";
            }
            myfile << endl;
        }
        myfile.close();
    }

    void write_T_param(double T_end, size_t num_steps, string dir_name){
        ofstream myfile;
        myfile.open(dir_name);
        myfile << T_end << " " << num_steps << " " << lattice_size << endl;
        myfile.close();
    }
    void simulated_annealing(double T_start, double T_end, size_t n_anneal, size_t overrelaxation_rate, bool gaussian_move = false, bool boundary_update = false, double cooling_rate = 0.9, string out_dir = "", bool save_observables = false){    
        if (!out_dir.empty()){
            filesystem::create_directory(out_dir);
        }
        double T = T_start;
        double acceptance_rate = 0;
        double sigma = 1000;
        
        // Enhanced convergence parameters
        const double log_cooling = log(cooling_rate);
        const double log_temp_ratio = log(T_end/T_start);
        const size_t expected_temp_steps = static_cast<size_t>(log_temp_ratio / log_cooling) + 1;
        const size_t convergence_window = min(size_t(100), max(size_t(20), expected_temp_steps / 5));
        const double energy_tolerance = 1e-5;
        const double gradient_tolerance = 1e-3;
        const double variance_tolerance = 1e-3;
        
        // Ground state search parameters
        const size_t ground_state_checks = 3;
        const size_t final_optimization_sweeps = n_anneal * 10;
        const double ground_state_temp_factor = 0.1;
        const double ground_state_temp_threshold = T_end * ground_state_temp_factor;
        
        // Pre-allocate tracking variables with reserve
        vector<double> energy_history;
        vector<double> variance_history;
        vector<double> gradient_history;
        deque<double> recent_energies;
        
        // Autocorrelation tracking
        vector<double> energy_samples_for_autocorr;
        const size_t min_samples_for_autocorr = 100;
        const size_t max_samples_for_autocorr = 1000;
        double tau_int = 1.0;
        size_t effective_samples = 0;
        
        energy_history.reserve(expected_temp_steps * 2);
        variance_history.reserve(expected_temp_steps);
        gradient_history.reserve(expected_temp_steps);
        energy_samples_for_autocorr.reserve(max_samples_for_autocorr);
        
        bool converged = false;
        bool ground_state_found = false;
        
        // Best configuration tracking
        spin_config best_config = spins;
        double best_energy = std::numeric_limits<double>::max();
        size_t steps_at_best = 0;
        size_t steps_since_improvement = 0;
        
        // Adaptive parameters
        double adaptive_cooling_rate = cooling_rate;
        const double min_cooling_rate = 0.98;
        const double max_cooling_rate = 0.6;
        size_t plateau_count = 0;
        const size_t max_plateau_count = 5;
        
        // Energy landscape analysis
        vector<double> local_minima_energies;
        local_minima_energies.reserve(20);
        
        // Statistics
        size_t total_metropolis_steps = 0;
        size_t total_accepted_moves = 0;
        
        // Pre-compute temperature thresholds
        const double low_temp_threshold = T_start * 0.1;
        const double very_low_temp_threshold = T_end * 10;
        const double ground_state_verify_temp = T_end * 2;
        
        // Helper lambda for autocorrelation - capture by reference for efficiency
        auto compute_autocorrelation = [](const vector<double>& data, size_t max_lag) -> vector<double> {
            const size_t n = data.size();
            if (n < 2) return vector<double>(1, 1.0);
            
            double mean = 0;
            for (const auto& x : data) mean += x;
            mean /= n;
            
            double c0 = 0;
            for (const auto& x : data) {
                const double diff = x - mean;
                c0 += diff * diff;
            }
            c0 /= n;
            
            if (c0 < 1e-10) return vector<double>(1, 1.0);
            
            const size_t actual_max_lag = min(max_lag, n >> 2); // n/4
            vector<double> autocorr(actual_max_lag + 1);
            autocorr[0] = 1.0;
            
            for (size_t lag = 1; lag <= actual_max_lag; ++lag) {
                double c_lag = 0;
                const size_t n_minus_lag = n - lag;
                for (size_t i = 0; i < n_minus_lag; ++i) {
                    c_lag += (data[i] - mean) * (data[i + lag] - mean);
                }
                autocorr[lag] = (c_lag / n_minus_lag) / c0;
            }
            
            return autocorr;
        };
        
        // Helper lambda for tau_int computation
        auto compute_tau_int = [](const vector<double>& autocorr) -> double {
            if (autocorr.size() < 2) return 1.0;
            
            double tau = 0.5;
            const size_t size = autocorr.size();
            
            for (size_t i = 1; i < size; ++i) {
                if (autocorr[i] < 0.05 || i >= 5 * tau) break;
                tau += autocorr[i];
            }
            
            return max(1.0, tau);
        };
        
        cout << "\n=== Enhanced Simulated Annealing with Autocorrelation Analysis ===" << endl;
        cout << "Temperature range: [" << T_start << ", " << T_end << "]" << endl;
        cout << "Expected steps: " << expected_temp_steps << endl;
        cout << "Convergence window: " << convergence_window << endl;
        cout << "Energy tolerance: " << scientific << energy_tolerance << endl;
        cout << "Gradient tolerance: " << scientific << gradient_tolerance << endl;
        cout << "Variance tolerance: " << scientific << variance_tolerance << endl;
        cout << "Ground state verification: " << ground_state_checks << " checks" << endl;
        cout << "==================================================================\n" << endl;
        
        // Initialize random seed more efficiently
        {
            std::random_device rd;
            const auto time_val = chrono::high_resolution_clock::now().time_since_epoch().count();
            seed_lehman((rd() ^ static_cast<uint64_t>(time_val) ^ reinterpret_cast<uintptr_t>(this)) | 1ULL);
        }
        
        const auto start_time = chrono::steady_clock::now();
        size_t temp_step = 0;
        const size_t progress_interval = max(size_t(1), expected_temp_steps/100);
        
        // Pre-allocate work arrays for variance calculation
        double running_mean = 0.0;
        double running_sum = 0.0;
        double running_sum_sq = 0.0;
        
        // Pre-calculate inverse lattice size for efficiency
        const double inv_lattice_size = 1.0 / lattice_size;
        
        // Main annealing loop
        while(T > ground_state_temp_threshold && !ground_state_found){
            ++temp_step;
            double curr_accept = 0;
            size_t curr_total = 0;
            
            energy_samples_for_autocorr.clear();
            
            // Adaptive number of sweeps based on temperature
            size_t adaptive_n_anneal = n_anneal;
            if(T < low_temp_threshold){
                adaptive_n_anneal <<= 1; // *= 2
            }
            if(T < very_low_temp_threshold){
                adaptive_n_anneal *= 5;
            }
            
            // Adjust sampling rate based on estimated autocorrelation
            const size_t sample_interval = max(size_t(1), static_cast<size_t>(tau_int));
            
            // Metropolis/Overrelaxation sweeps with energy sampling
            if(overrelaxation_rate > 0){
                const size_t overrelax_cycles = adaptive_n_anneal / overrelaxation_rate;
                for(size_t cycle = 0; cycle < overrelax_cycles; ++cycle){
                    overrelaxation();
                    curr_accept += metropolis(spins, T, gaussian_move, sigma);
                    ++curr_total;
                    
                    if ((cycle % sample_interval == 0) && 
                        (energy_samples_for_autocorr.size() < max_samples_for_autocorr)) {
                        energy_samples_for_autocorr.push_back(total_energy(spins) * inv_lattice_size);
                    }
                }
                for(size_t i = overrelax_cycles * overrelaxation_rate; i < adaptive_n_anneal; ++i){
                    overrelaxation();
                }
            } else {
                for(size_t i = 0; i < adaptive_n_anneal; ++i){
                    curr_accept += metropolis(spins, T, gaussian_move, sigma);
                    ++curr_total;
                    
                    if ((i % sample_interval == 0) && 
                        (energy_samples_for_autocorr.size() < max_samples_for_autocorr)) {
                        energy_samples_for_autocorr.push_back(total_energy(spins) * inv_lattice_size);
                    }
                }
            }
            
            total_metropolis_steps += curr_total;
            total_accepted_moves += static_cast<size_t>(curr_accept * lattice_size);
            
            if(boundary_update){
                const size_t boundary_sweeps = min(size_t(100), adaptive_n_anneal/10);
                for(size_t i = 0; i < boundary_sweeps; ++i){
                    metropolis_twist_sweep(T);
                }
            }
            
            // Compute autocorrelation if we have enough samples
            double unbiased_variance = 0;
            double unbiased_mean = 0;
            if (energy_samples_for_autocorr.size() >= min_samples_for_autocorr) {
                const size_t max_lag = min(size_t(50), energy_samples_for_autocorr.size() >> 2);
                const vector<double> autocorr = compute_autocorrelation(energy_samples_for_autocorr, max_lag);
                
                tau_int = compute_tau_int(autocorr);
                effective_samples = energy_samples_for_autocorr.size() / (2 * tau_int);
                
                unbiased_mean = 0;
                for (const auto& e : energy_samples_for_autocorr) {
                    unbiased_mean += e;
                }
                unbiased_mean /= energy_samples_for_autocorr.size();
                
                double sum_sq_diff = 0;
                for (const auto& e : energy_samples_for_autocorr) {
                    const double diff = e - unbiased_mean;
                    sum_sq_diff += diff * diff;
                }
                
                if (effective_samples > 1) {
                    unbiased_variance = (sum_sq_diff / energy_samples_for_autocorr.size()) * 
                                       (2 * tau_int / (1 - 1.0/effective_samples));
                } else {
                    unbiased_variance = sum_sq_diff / max(1.0, static_cast<double>(energy_samples_for_autocorr.size() - 1));
                }
                
                variance_history.push_back(unbiased_variance);
            }
            
            const double current_energy = (unbiased_mean > 0) ? unbiased_mean : (total_energy(spins) * inv_lattice_size);
            energy_history.push_back(current_energy);
            recent_energies.push_back(current_energy);
            if(recent_energies.size() > convergence_window){
                recent_energies.pop_front();
            }
            
            double gradient = 0;
            if(energy_history.size() >= 2){
                const size_t n = energy_history.size();
                gradient = (energy_history[n-1] - energy_history[n-2]) / ((adaptive_cooling_rate-1) * T);
                gradient_history.push_back(gradient);
            }
            
            if(current_energy < best_energy){
                const double improvement = best_energy - current_energy;
                best_energy = current_energy;
                best_config = spins;
                steps_at_best = temp_step;
                steps_since_improvement = 0;
                
                if(improvement > energy_tolerance * fabs(best_energy)){
                    cout << "  >>> NEW BEST: E/N = " << fixed << setprecision(10) << best_energy 
                         << " (improvement: " << scientific << improvement 
                         << ", τ_int = " << fixed << setprecision(2) << tau_int << ")" << endl;
                }
            } else {
                ++steps_since_improvement;
            }
            
            const bool is_plateau = (steps_since_improvement > convergence_window * tau_int * 0.5) && 
                                   (unbiased_variance < variance_tolerance);
            
            if(is_plateau){
                ++plateau_count;
                if(plateau_count > max_plateau_count){
                    cout << "  >> Plateau detected (τ_int=" << fixed << setprecision(2) << tau_int 
                         << ", eff_samples=" << effective_samples << "). Attempting escape..." << endl;
                    local_minima_energies.push_back(current_energy);
                    
                    const double escape_temp = min(T * 2, T_start * 0.1);
                    const size_t escape_sweeps = n_anneal >> 1;
                    for(size_t i = 0; i < escape_sweeps; ++i){
                        metropolis(spins, escape_temp, gaussian_move, sigma * 2);
                    }
                    plateau_count = 0;
                }
            } else {
                plateau_count = 0;
            }
            
            if(curr_total > 0){
                acceptance_rate = curr_accept / curr_total;
            }
            
            if(energy_history.size() >= (convergence_window >> 2)){
                if(unbiased_variance < 1e-3 || acceptance_rate > 0.5){
                    adaptive_cooling_rate = min(min_cooling_rate, adaptive_cooling_rate * 0.9);
                    // Moving well, cool faster
                } else if(unbiased_variance > 1e-2 || acceptance_rate < 0.1){
                    adaptive_cooling_rate = max(max_cooling_rate, adaptive_cooling_rate / 0.9);
                    // Struggling, cool slower
                } 
            }
            
            // Multi-criteria convergence check
            bool energy_stable = false, gradient_small = false, variance_small = false;
            
            if(energy_history.size() >= convergence_window && effective_samples >= 10){
                const size_t n = energy_history.size();
                const size_t effective_window = max(size_t(2), 
                    min(convergence_window, static_cast<size_t>(effective_samples)));
                const size_t half_window = effective_window >> 1;
                
                if (n >= effective_window) {
                    double recent_mean = 0, old_mean = 0;
                    
                    for(size_t i = n - half_window; i < n; ++i){
                        recent_mean += energy_history[i];
                    }
                    recent_mean /= half_window;
                    
                    for(size_t i = n - effective_window; i < n - half_window; ++i){
                        old_mean += energy_history[i];
                    }
                    old_mean /= half_window;
                    
                    const double relative_change = fabs((recent_mean - old_mean) / fabs(old_mean));
                    energy_stable = (relative_change < energy_tolerance * sqrt(2 * tau_int));
                }
                
                if(!gradient_history.empty()){
                    double avg_gradient = 0;
                    const size_t count = min(size_t(10), gradient_history.size());
                    const size_t start_idx = gradient_history.size() - count;
                    for(size_t i = start_idx; i < gradient_history.size(); ++i){
                        avg_gradient += gradient_history[i];
                    }
                    avg_gradient = fabs(avg_gradient / count);
                    gradient_small = (avg_gradient < sqrt(unbiased_variance / effective_samples));
                }
                
                variance_small = (unbiased_variance < variance_tolerance);
                
                converged = energy_stable && gradient_small && variance_small && 
                           (acceptance_rate < 0.01) && (effective_samples >= 20);
            }
            
            if(temp_step % progress_interval == 0 || converged){
                cout << "Step " << setw(5) << temp_step 
                     << " | T=" << fixed << setprecision(6) << T
                     << " | E/N=" << setprecision(10) << current_energy
                     << " | Best=" << setprecision(10) << best_energy
                     << " | Accept=" << setprecision(4) << acceptance_rate
                     << " | τ_int=" << setprecision(2) << tau_int
                     << " | N_eff=" << setw(4) << effective_samples
                     << " | Cool=" << setprecision(4) << adaptive_cooling_rate;
                if(unbiased_variance > 0){
                    cout << " | Var=" << scientific << setprecision(2) << unbiased_variance;
                }
                if(!gradient_history.empty()){
                    cout << " | Grad=" << scientific << setprecision(2) << gradient_history.back();
                }
                if(converged) cout << " [CONVERGED]";
                cout << endl;
            }
            
            if(T <= ground_state_verify_temp && !ground_state_found){
                cout << "\n>>> Entering ground state verification phase <<<" << endl;
                cout << "Current τ_int = " << tau_int << ", effective samples = " << effective_samples << endl;
                
                vector<double> verification_energies;
                verification_energies.reserve(ground_state_checks);
                
                for(size_t check = 0; check < ground_state_checks; ++check){
                    spins = best_config;
                    
                    const size_t perturbation_sites = lattice_size / 100;
                    for(size_t i = 0; i < perturbation_sites; ++i){
                        const size_t site = random_int_lehman(lattice_size);
                        const array<double,N> perturbation = gen_random_spin(spin_length * 0.01);
                        spins[site] = spins[site] + perturbation;
                        const double norm = sqrt(dot(spins[site], spins[site]));
                        spins[site] = spins[site] * (spin_length / norm);
                    }
                    
                    const double search_temp = T_end * ground_state_temp_factor;
                    const size_t search_sweeps = static_cast<size_t>(final_optimization_sweeps * tau_int);
                    for(size_t i = 0; i < search_sweeps; ++i){
                        metropolis(spins, search_temp, false, sigma/100);
                        if((i & 0x63) == 0){
                            overrelaxation();
                        }
                    }
                    
                    const double check_energy = total_energy(spins) * inv_lattice_size;
                    verification_energies.push_back(check_energy);
                    
                    if(check_energy < best_energy){
                        best_energy = check_energy;
                        best_config = spins;
                        cout << "  Check " << check+1 << ": Found lower energy = " 
                             << fixed << setprecision(12) << best_energy << endl;
                    } else {
                        cout << "  Check " << check+1 << ": Energy = " 
                             << fixed << setprecision(12) << check_energy << endl;
                    }
                }
                
                double mean_verification = 0, var_verification = 0;
                for(const auto& e : verification_energies) mean_verification += e;
                mean_verification /= verification_energies.size();
                for(const auto& e : verification_energies){
                    const double diff = e - mean_verification;
                    var_verification += diff * diff;
                }
                var_verification /= (verification_energies.size() - 1);
                
                cout << "Verification statistics:" << endl;
                cout << "  Mean E/N = " << fixed << setprecision(12) << mean_verification << endl;
                cout << "  Std Dev = " << scientific << setprecision(4) << sqrt(var_verification) << endl;
                cout << "  Best E/N = " << fixed << setprecision(12) << best_energy << endl;
                
                const double mean_tolerance = energy_tolerance * fabs(mean_verification) * sqrt(2 * tau_int);
                ground_state_found = (sqrt(var_verification) < mean_tolerance)
                                    && (fabs(best_energy - mean_verification) < mean_tolerance);
                
                if(ground_state_found){
                    cout << ">>> GROUND STATE FOUND (within tolerance corrected for τ_int) <<<" << endl;
                    spins = best_config;
                    break;
                } else {
                    cout << "Ground state not yet confirmed. Continuing search..." << endl;
                    spins = best_config;
                }
            }
            
            T *= adaptive_cooling_rate;
        }
        
        if(!ground_state_found){
            cout << "\n=== FINAL GROUND STATE OPTIMIZATION ===" << endl;
            
            spins = best_config;
            const double final_T = T_end * ground_state_temp_factor;
            
            cout << "Phase 1: Deterministic optimization..." << endl;
            for(size_t i = 0; i < 100; ++i){
                deterministic_sweep();
                if((i % 10) == 0){
                    const double e = total_energy(spins) * inv_lattice_size;
                    cout << "  Deterministic " << i << ": E/N = " << fixed << setprecision(12) << e << endl;
                    if(e < best_energy){
                        best_energy = e;
                        best_config = spins;
                    }
                }
            }
            
            cout << "Phase 2: Ultra-low temperature Monte Carlo..." << endl;
            spins = best_config;
            const size_t ultra_low_sweeps = static_cast<size_t>(n_anneal * 10 * max(1.0, tau_int));
            for(size_t cycle = 0; cycle < 10; ++cycle){
                const double cycle_temp = final_T / (cycle + 1);
                for(size_t i = 0; i < ultra_low_sweeps; ++i){
                    metropolis(spins, cycle_temp, false, sigma/1000);
                    if((i & 0x63) == 0) overrelaxation();
                }
                const double e = total_energy(spins) * inv_lattice_size;
                cout << "  Cycle " << cycle+1 << ": E/N = " << fixed << setprecision(12) << e << endl;
                if(e < best_energy){
                    best_energy = e;
                    best_config = spins;
                }
            }
            
            cout << "Phase 3: Gradient-based refinement..." << endl;
            spins = best_config;
            const size_t gradient_steps = lattice_size * 10;
            for(size_t i = 0; i < gradient_steps; ++i){
                const size_t site = i % lattice_size;
                const array<double,N> local_field = get_local_field(site);
                const double norm = sqrt(dot(local_field, local_field));
                if(norm > 1e-10){
                    const array<double,N> new_spin = local_field * (-spin_length / norm);
                    const double dE = site_energy_diff(new_spin, spins[site], site);
                    if(dE < 0){
                        spins[site] = new_spin;
                    }
                }
            }
            
            const double final_energy = total_energy(spins) * inv_lattice_size;
            if(final_energy < best_energy){
                best_energy = final_energy;
                best_config = spins;
                cout << "  Gradient descent improved: E/N = " << fixed << setprecision(12) << best_energy << endl;
            }
        }
        
        spins = best_config;
        
        const auto end_time = chrono::steady_clock::now();
        const auto elapsed = chrono::duration_cast<chrono::seconds>(end_time - start_time).count();
        
        cout << "\n==========================================================" << endl;
        cout << "=== SIMULATED ANNEALING COMPLETE ===" << endl;
        cout << "Ground State Found: " << (ground_state_found ? "YES" : "ATTEMPTED") << endl;
        cout << "Final Energy/Site: " << fixed << setprecision(12) << best_energy << endl;
        cout << "Final τ_int: " << fixed << setprecision(2) << tau_int << endl;
        cout << "Total Temperature Steps: " << temp_step << endl;
        cout << "Total Metropolis Steps: " << total_metropolis_steps * lattice_size << endl;
        cout << "Total Accepted Moves: " << total_accepted_moves << endl;
        cout << "Overall Acceptance Rate: " << fixed << setprecision(4) 
             << double(total_accepted_moves) / (total_metropolis_steps * lattice_size) << endl;
        cout << "Total Time: " << elapsed << " seconds" << endl;
        cout << "==========================================================" << endl;
        
        const double verification_energy = total_energy(spins) * inv_lattice_size;
        if(fabs(verification_energy - best_energy) > 1e-10){
            cout << "WARNING: Verification energy differs from best: " 
                 << fixed << setprecision(12) << verification_energy << endl;
        }
        
        if(!out_dir.empty()){
            write_to_file_spin(out_dir + "/spin.txt", spins);
            write_to_file_pos(out_dir + "/pos.txt");
            
            ofstream info_file(out_dir + "/annealing_info.txt");
            info_file << "Ground State Found: " << (ground_state_found ? "Yes" : "Attempted") << endl;
            info_file << "Final Energy/Site: " << fixed << setprecision(15) << best_energy << endl;
            info_file << "Final Integrated Autocorrelation Time: " << fixed << setprecision(3) << tau_int << endl;
            info_file << "Temperature Steps: " << temp_step << endl;
            info_file << "Total Time (s): " << elapsed << endl;
            info_file << "Final Cooling Rate: " << adaptive_cooling_rate << endl;
            info_file << "Acceptance Rate: " << double(total_accepted_moves) / (total_metropolis_steps * lattice_size) << endl;
            
            if(!local_minima_energies.empty()){
                info_file << "\nLocal Minima Detected:" << endl;
                for(size_t i = 0; i < local_minima_energies.size(); ++i){
                    info_file << "  " << i+1 << ": " << fixed << setprecision(12) 
                             << local_minima_energies[i] << endl;
                }
            }
            info_file.close();
            
            ofstream energy_file(out_dir + "/energy_history.txt");
            for(const auto& e : energy_history){
                energy_file << fixed << setprecision(12) << e << endl;
            }
            energy_file.close();
        }
    }

    void simulated_annealing_deterministic(double T_start, double T_end, size_t n_anneal, size_t n_deterministics, size_t overrelaxation_rate, string dir_name, bool gaussian_move = false){
        simulated_annealing(T_start, T_end, n_anneal, overrelaxation_rate, gaussian_move);
        for(size_t i = 0; i<n_deterministics; ++i){
            deterministic_sweep();
        }   

        if(dir_name != ""){
            filesystem::create_directory(dir_name);
            write_to_file_spin(dir_name + "/spin.txt", spins);
            write_to_file_pos(dir_name + "/pos.txt");
        }
    }

    void adaptive_restarted_simulated_annealing(
        double T_start, double T_end, size_t n_anneal, size_t overrelaxation_rate,
        size_t max_restarts, size_t stagnation_patience, bool gaussian_move = false,
        string out_dir = "")
    {
        if (!out_dir.empty()) {
            filesystem::create_directory(out_dir);
        }

        spin_config best_spins_so_far = spins;
        double best_energy_so_far = total_energy(spins);
        cout << "Initial Energy: " << best_energy_so_far / lattice_size << endl;

        size_t restarts_done = 0;
        bool restart_triggered = true;

        while (restarts_done <= max_restarts) {
            if (restart_triggered) {
                cout << "\n--- Starting Annealing Cycle " << restarts_done + 1 << "/" << max_restarts + 1 << " ---" << endl;
                if (restarts_done > 0) {
                    // Re-initialize spins from the best state found so far
                    spins = best_spins_so_far;
                    
                    // Add a perturbation to escape the local minimum
                    cout << "Applying perturbation to escape local minimum..." << endl;
                    double perturbation_temperature = T_start * 1.5; // High temperature kick
                    size_t perturbation_sweeps = 20; // A few sweeps to shake things up
                    for(size_t i = 0; i < perturbation_sweeps; ++i) {
                        metropolis(spins, perturbation_temperature, gaussian_move, 1000.0);
                    }
                }
            }

            double T = T_start;
            double cooling_rate = 0.9; // Start with a standard cooling rate
            double sigma = 1000.0;
            size_t stagnation_counter = 0;
            double last_energy = total_energy(spins);

            while (T > T_end) {
                double total_acceptance_in_step = 0;
                size_t metropolis_calls = 0;

                for (size_t i = 0; i < n_anneal; ++i) {
                    if (overrelaxation_rate > 0 && i % overrelaxation_rate == 0) {
                        overrelaxation();
                    }
                    total_acceptance_in_step += metropolis(spins, T, gaussian_move, sigma);
                    metropolis_calls++;
                }

                double acceptance_rate = total_acceptance_in_step / metropolis_calls;
                cout << "T: " << T << ", Acceptance: " << acceptance_rate;

                // Adaptive sigma for Gaussian moves
                if (gaussian_move && acceptance_rate < 0.5){
                    sigma = sigma * 0.5 / (1-acceptance_rate); 
                    cout << "Sigma is adjusted to: " << sigma << endl;   
                }

                // Adaptive cooling rate
                if (acceptance_rate > 0.8) cooling_rate = 0.85; // Cool faster
                else if (acceptance_rate < 0.1) cooling_rate = 0.98; // Cool slower
                else cooling_rate = 0.9; // Default
                cout << ", Cooling Rate: " << cooling_rate << endl;

                double current_energy = total_energy(spins);
                if (current_energy < best_energy_so_far) {
                    best_energy_so_far = current_energy;
                    best_spins_so_far = spins;
                    stagnation_counter = 0; // Reset stagnation counter on improvement
                    cout << "New best energy found: " << best_energy_so_far / lattice_size << endl;
                } else {
                    // Check for stagnation if energy hasn't improved significantly
                    if (abs(current_energy - last_energy) / abs(last_energy) < 1e-5) {
                        stagnation_counter++;
                    } else {
                        stagnation_counter = 0; // Reset if there's some change
                    }
                }
                last_energy = current_energy;

                if (stagnation_counter >= stagnation_patience) {
                    cout << "Stagnation detected. Triggering restart." << endl;
                    restart_triggered = true;
                    break; // Exit inner while loop to restart
                } else {
                    restart_triggered = false;
                }

                T *= cooling_rate;
            }

            restarts_done++;
            if (!restart_triggered) {
                cout << "\nAnnealing cycle finished without stagnation. Concluding." << endl;
                break; // Exit outer while loop
            }
        }

        cout << "\n--- Simulated Annealing Finished ---" << endl;
        cout << "Final best energy: " << best_energy_so_far / lattice_size << endl;
        spins = best_spins_so_far;

        if (!out_dir.empty()) {
            write_to_file_spin(out_dir + "/spin_final.txt", spins);
            write_to_file_pos(out_dir + "/pos.txt");
        }
    }

    void simulated_annealing_zigzag(double T_start, double T_end, size_t n_anneal, size_t overrelaxation_rate, bool gaussian_move = false, double cooling_rate = 0.9, string out_dir = "", bool save_observables = false){
        // Set initial spin configuration to zigzag order
        array<double, N> spin_up = {0};
        if (N > 0) {
            spin_up[N-1] = spin_length;
        }

        for (size_t i=0; i< dim1; ++i){
            for (size_t j=0; j< dim2; ++j){
                for(size_t k=0; k< dim;++k){
                    for (size_t l=0; l< N_ATOMS;++l){
                        size_t current_site_index = flatten_index(i,j,k,l);
                        if ((i + j + k) % 2 == 0) {
                            spins[current_site_index] = spin_up;
                        } else {
                            spins[current_site_index] = -spin_up;
                        }
                    }
                }
            }
        }

        cout << "Starting simulated annealing with zigzag initial configuration." << endl;
        // Call the existing simulated annealing method
        simulated_annealing(T_start, T_end, n_anneal, overrelaxation_rate, gaussian_move, cooling_rate, out_dir, save_observables);
    }
    

    // Advanced temperature ladder spacing algorithm using heat capacity and acceptance rate optimization
    vector<double> generate_optimal_temperature_ladder(double T_min, double T_max, size_t n_temps, 
                                                       size_t n_test_sweeps = 1000, 
                                                       double target_swap_rate = 0.23) {
        cout << "=== Generating Optimal Temperature Ladder ===" << endl;
        cout << "Target temperature range: [" << T_min << ", " << T_max << "]" << endl;
        cout << "Number of temperatures: " << n_temps << endl;
        cout << "Target swap acceptance rate: " << target_swap_rate << endl;
        
        // Initialize with geometric spacing as starting point
        vector<double> temps(n_temps);
        double ratio = pow(T_max/T_min, 1.0/(n_temps-1));
        for(size_t i = 0; i < n_temps; ++i) {
            temps[i] = T_min * pow(ratio, i);
        }
        
        // Adaptive refinement based on energy overlap
        cout << "\nPhase 1: Energy-based refinement..." << endl;
        
        // Estimate energy distributions at each temperature
        vector<double> mean_energies(n_temps);
        vector<double> energy_variances(n_temps);
        vector<spin_config> test_configs(n_temps);
        
        // Initialize test configurations
        for(size_t t = 0; t < n_temps; ++t) {
            test_configs[t] = spins;
            // Pre-equilibrate at each temperature
            for(size_t s = 0; s < n_test_sweeps/10; ++s) {
                metropolis(test_configs[t], temps[t], false);
            }
        }
        
        // Collect energy statistics
        for(size_t t = 0; t < n_temps; ++t) {
            vector<double> energies;
            for(size_t s = 0; s < n_test_sweeps; ++s) {
                metropolis(test_configs[t], temps[t], false);
                if(s % 10 == 0) {
                    energies.push_back(total_energy(test_configs[t]));
                }
            }
            
            // Calculate mean and variance
            double mean = 0, var = 0;
            for(auto e : energies) mean += e;
            mean /= energies.size();
            for(auto e : energies) var += (e - mean) * (e - mean);
            var /= (energies.size() - 1);
            
            mean_energies[t] = mean;
            energy_variances[t] = var;
            
            cout << "  T[" << t << "] = " << fixed << setprecision(4) << temps[t] 
                 << " | <E>/N = " << mean/lattice_size 
                 << " | σ²(E)/N² = " << var/(lattice_size*lattice_size) << endl;
        }
        
        // Calculate overlap integrals and adjust spacing
        cout << "\nPhase 2: Overlap optimization..." << endl;
        
        auto gaussian_overlap = [](double E1, double var1, double E2, double var2) -> double {
            // Estimate overlap between two Gaussian distributions
            double sigma1 = sqrt(var1);
            double sigma2 = sqrt(var2);
            double sigma_sum = sigma1*sigma1 + sigma2*sigma2;
            if(sigma_sum <= 0) return 0;
            
            double overlap = exp(-(E1-E2)*(E1-E2)/(2*sigma_sum));
            return overlap;
        };
        
        // Iterative temperature adjustment
        const size_t max_iterations = 10;
        for(size_t iter = 0; iter < max_iterations; ++iter) {
            vector<double> swap_probs(n_temps-1);
            vector<double> new_temps = temps;
            
            // Calculate expected swap acceptance rates
            for(size_t t = 0; t < n_temps-1; ++t) {
                double overlap = gaussian_overlap(mean_energies[t], energy_variances[t],
                                                 mean_energies[t+1], energy_variances[t+1]);
                
                // Estimate swap probability using detailed balance
                double beta1 = 1.0/temps[t];
                double beta2 = 1.0/temps[t+1];
                double dE = mean_energies[t+1] - mean_energies[t];
                swap_probs[t] = min(1.0, exp((beta1-beta2)*dE)) * overlap;
            }
            
            // Adjust temperatures to equalize swap probabilities
            double avg_swap_prob = 0;
            for(auto p : swap_probs) avg_swap_prob += p;
            avg_swap_prob /= swap_probs.size();
            
            cout << "  Iteration " << iter+1 << " | Avg swap prob: " << avg_swap_prob << endl;
            
            // Stop if close enough to target
            if(fabs(avg_swap_prob - target_swap_rate) < 0.02) break;
            
            // Redistribute temperatures
            for(size_t t = 1; t < n_temps-1; ++t) {
                double local_prob = (swap_probs[t-1] + swap_probs[t])/2;
                double adjustment = 1.0;
                
                if(local_prob < target_swap_rate) {
                    // Temperatures too far apart, bring closer
                    adjustment = 1.0 - 0.1*(target_swap_rate - local_prob)/target_swap_rate;
                } else if(local_prob > target_swap_rate) {
                    // Temperatures too close, spread apart
                    adjustment = 1.0 + 0.1*(local_prob - target_swap_rate)/target_swap_rate;
                }
                
                // Interpolate between neighbors with adjustment
                double t_prev = (t > 0) ? new_temps[t-1] : new_temps[t];
                double t_next = (t < n_temps-1) ? new_temps[t+1] : new_temps[t];
                new_temps[t] = temps[t] * adjustment;
                
                // Ensure monotonicity
                new_temps[t] = max(t_prev * 1.01, min(new_temps[t], t_next * 0.99));
            }
            
            temps = new_temps;
            
            // Re-equilibrate and update statistics for changed temperatures
            for(size_t t = 1; t < n_temps-1; ++t) {
                vector<double> energies;
                for(size_t s = 0; s < n_test_sweeps/2; ++s) {
                    metropolis(test_configs[t], temps[t], false);
                    if(s % 10 == 0) {
                        energies.push_back(total_energy(test_configs[t]));
                    }
                }
                
                double mean = 0, var = 0;
                for(auto e : energies) mean += e;
                mean /= energies.size();
                for(auto e : energies) var += (e - mean) * (e - mean);
                var /= (energies.size() - 1);
                
                mean_energies[t] = mean;
                energy_variances[t] = var;
            }
        }
        
        // Phase 3: Critical point detection and refinement
        cout << "\nPhase 3: Critical point refinement..." << endl;
        
        // Estimate specific heat to find critical regions
        vector<double> specific_heats(n_temps);
        for(size_t t = 0; t < n_temps; ++t) {
            specific_heats[t] = energy_variances[t]/(temps[t]*temps[t]*lattice_size);
        }
        
        // Find peak in specific heat (critical point)
        size_t critical_idx = 0;
        double max_cv = 0;
        for(size_t t = 1; t < n_temps-1; ++t) {
            if(specific_heats[t] > max_cv) {
                max_cv = specific_heats[t];
                critical_idx = t;
            }
        }
        
        if(critical_idx > 0 && critical_idx < n_temps-1) {
            cout << "  Critical region detected around T = " << temps[critical_idx] << endl;
            
            // Increase density around critical point
            vector<double> final_temps;
            final_temps.push_back(temps[0]);
            
            for(size_t t = 1; t < n_temps-1; ++t) {
                double distance_to_critical = fabs(double(t) - double(critical_idx));
                double weight = exp(-distance_to_critical*distance_to_critical/(n_temps*n_temps/9.0));
                
                // Add extra point near critical region
                if(distance_to_critical < n_temps/4 && final_temps.size() < n_temps-1) {
                    double t_interp = (temps[t-1] + temps[t])/2;
                    if(weight > 0.5 && t_interp > final_temps.back()*1.001) {
                        final_temps.push_back(t_interp);
                    }
                }
                
                if(final_temps.size() < n_temps) {
                    final_temps.push_back(temps[t]);
                }
            }
            
            // Ensure we have exactly n_temps
            if(final_temps.size() < n_temps) {
                final_temps.push_back(temps.back());
            }
            while(final_temps.size() > n_temps) {
                // Remove points with smallest gaps
                size_t min_gap_idx = 0;
                double min_gap = final_temps[1] - final_temps[0];
                for(size_t i = 1; i < final_temps.size()-2; ++i) {
                    double gap = final_temps[i+1] - final_temps[i];
                    if(gap < min_gap) {
                        min_gap = gap;
                        min_gap_idx = i;
                    }
                }
                final_temps.erase(final_temps.begin() + min_gap_idx);
            }
            
            temps = final_temps;
        }
        
        // Final output
        cout << "\n=== Optimized Temperature Ladder ===" << endl;
        for(size_t t = 0; t < temps.size(); ++t) {
            cout << "  T[" << setw(2) << t << "] = " << fixed << setprecision(6) << temps[t];
            if(t > 0) {
                cout << " | ΔT/T = " << setprecision(4) << (temps[t]-temps[t-1])/temps[t];
            }
            if(t == critical_idx) {
                cout << " [*critical*]";
            }
            cout << endl;
        }
        
        return temps;
    }
    
    // Simplified interface using exponential spacing with automatic critical point detection
    vector<double> generate_temperature_ladder_exponential(double T_min, double T_max, size_t n_temps,
                                                          double critical_enhancement = 1.5) {
        vector<double> temps(n_temps);
        
        // Estimate critical temperature (geometric mean as first guess)
        double T_crit_estimate = sqrt(T_min * T_max);
        
        // Use a modified exponential spacing that's denser near T_crit
        double log_min = log(T_min);
        double log_max = log(T_max);
        double log_crit = log(T_crit_estimate);
        
        for(size_t i = 0; i < n_temps; ++i) {
            double x = double(i)/(n_temps-1); // 0 to 1
            
            // Standard exponential
            double log_T = log_min + x*(log_max - log_min);
            
            // Add Gaussian enhancement near critical point
            double crit_x = (log_crit - log_min)/(log_max - log_min);
            double enhancement = critical_enhancement * exp(-20*(x-crit_x)*(x-crit_x));
            
            // Modify spacing
            log_T = log_T - 0.1*enhancement*(log_T - log_crit);
            
            temps[i] = exp(log_T);
        }
        
        // Ensure strict ordering and bounds
        temps[0] = T_min;
        temps[n_temps-1] = T_max;
        for(size_t i = 1; i < n_temps-1; ++i) {
            temps[i] = max(temps[i-1]*1.001, min(temps[i], temps[i+1]*0.999));
        }
        
        return temps;
    }

    void parallel_tempering(vector<double> temp, size_t n_anneal, size_t n_measure, size_t overrelaxation_rate, size_t swap_rate, size_t probe_rate, string dir_name, const vector<int> rank_to_write, bool gaussian_move = false){

        int swap_accept = 0;
        double curr_accept = 0;
        int overrelaxation_flag = overrelaxation_rate > 0 ? overrelaxation_rate : 1;
        int initialized;
        MPI_Initialized(&initialized);
        if (!initialized){
            MPI_Init(NULL, NULL);
        }
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        // Improved random seeding
        std::random_device rd;
        seed_lehman((rd() + rank * 1000) | 1);
        
        double E, T_partner, E_partner;
        bool accept;        
        spin_config new_spins;
        double curr_Temp = temp[rank];
        vector<double> heat_capacity, dHeat;
        if (rank == 0){
            heat_capacity.resize(size);
            dHeat.resize(size);
        }   
        
        // Pre-allocate data collection vectors with reserve
        vector<double> energies;
        vector<array<double,N>> magnetizations;
        vector<spin_config> spin_configs_at_temp;
        const size_t expected_samples = n_measure / probe_rate;
        energies.reserve(expected_samples);
        magnetizations.reserve(expected_samples);
        spin_configs_at_temp.reserve(expected_samples);

        // Use circular buffers for history tracking
        const size_t convergence_window = 100;
        const size_t max_history = 1000; // Reduced from 10000
        std::deque<double> energy_history;
        std::deque<double> acceptance_history;
        std::deque<double> swap_acceptance_history;
        
        const size_t convergence_check_interval = 1000;
        const double energy_convergence_tolerance = 1e-5;
        const double swap_rate_convergence_tolerance = 0.05;
        const size_t min_converged_rounds = 3;
        
        bool local_converged = false;
        bool global_converged = false;
        size_t local_converged_rounds = 0;
        size_t convergence_step = 0;
        
        const size_t max_extensions = 10;
        const size_t extension_steps = n_measure / 2;
        const size_t min_steps_before_extension = n_anneal + n_measure;
        size_t current_extensions = 0;
        size_t total_steps = n_anneal + n_measure;
        
        // Optimized running statistics (Welford's algorithm)
        double running_mean_energy = 0;
        double M2_energy = 0; // Sum of squares of differences from current mean
        size_t stats_count = 0;
        
        // Recent swap rates with fixed-size circular buffer
        std::deque<double> recent_swap_rates;
        
        // Temperature-specific convergence criteria
        double temp_specific_tolerance = energy_convergence_tolerance;
        if(curr_Temp > 1.0) {
            temp_specific_tolerance *= (1.0 + log(curr_Temp));
        }
        
        cout << "Process " << rank << " initialized with T=" << curr_Temp << endl;

        // Convergence diagnostics file
        ofstream conv_file;
        if(!dir_name.empty()){
            filesystem::create_directory(dir_name);
            conv_file.open(dir_name + "/convergence_rank" + to_string(rank) + ".txt");
            conv_file << "# Step Energy AcceptRate SwapRate RunningMean RunningVar LocalConverged GlobalConverged Extension\n";
        }

        size_t i = 0;
        while(i < total_steps && !global_converged){

            // Metropolis with optimized acceptance tracking
            double step_accept = 0;
            if(overrelaxation_rate > 0 && i % overrelaxation_rate == 0){
                overrelaxation();
                step_accept = metropolis(spins, curr_Temp, gaussian_move);
                curr_accept += step_accept;
            }
            else if(overrelaxation_rate == 0){
                step_accept = metropolis(spins, curr_Temp, gaussian_move);
                curr_accept += step_accept;
            }
            else{
                overrelaxation();
            }
            
            E = total_energy(spins);
            
            // Optimized Welford's online algorithm
            ++stats_count;
            double delta = E - running_mean_energy;
            running_mean_energy += delta / stats_count;
            M2_energy += delta * (E - running_mean_energy);

            // Maintain circular buffers efficiently
            energy_history.push_back(E);
            if(energy_history.size() > max_history){
                energy_history.pop_front();
            }
            
            if(step_accept > 0){
                acceptance_history.push_back(step_accept);
                if(acceptance_history.size() > convergence_window){
                    acceptance_history.pop_front();
                }
            }

            // Replica exchange with optimized MPI communication
            if (i % swap_rate == 0){
                accept = false;
                const bool even_swap = ((i / swap_rate) & 1) == 0;
                const int partner_rank = even_swap ? 
                    (rank & 1 ? rank - 1 : rank + 1) : 
                    (rank & 1 ? rank + 1 : rank - 1);
                
                if (partner_rank >= 0 && partner_rank < size){
                    T_partner = temp[partner_rank];
                    
                    // Optimized MPI communication pattern
                    if ((partner_rank & 1) == 0){
                        MPI_Send(&E, 1, MPI_DOUBLE, partner_rank, 0, MPI_COMM_WORLD);
                        MPI_Recv(&E_partner, 1, MPI_DOUBLE, partner_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    } else{
                        MPI_Recv(&E_partner, 1, MPI_DOUBLE, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        MPI_Send(&E, 1, MPI_DOUBLE, partner_rank, 1, MPI_COMM_WORLD);
                    }
                    
                    // Metropolis criterion
                    const double delta_beta = 1.0/curr_Temp - 1.0/T_partner;
                    const double delta_E = E - E_partner;
                    
                    if ((partner_rank & 1) == 0){
                        accept = (delta_beta * delta_E < 0) || (random_double_lehman(0,1) < exp(delta_beta * delta_E));
                        MPI_Send(&accept, 1, MPI_C_BOOL, partner_rank, 2, MPI_COMM_WORLD);
                    } else{
                        MPI_Recv(&accept, 1, MPI_C_BOOL, partner_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                    
                    if (accept){
                        // Exchange configurations
                        if ((partner_rank & 1) == 0){
                            MPI_Sendrecv(spins.data(), N*lattice_size, MPI_DOUBLE, partner_rank, 4,
                                        new_spins.data(), N*lattice_size, MPI_DOUBLE, partner_rank, 3,
                                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        } else{
                            MPI_Sendrecv(spins.data(), N*lattice_size, MPI_DOUBLE, partner_rank, 3,
                                        new_spins.data(), N*lattice_size, MPI_DOUBLE, partner_rank, 4,
                                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        }
                        spins = std::move(new_spins);
                        E = E_partner;
                        ++swap_accept;
                    }
                }
                
                swap_acceptance_history.push_back(accept ? 1.0 : 0.0);
                if(swap_acceptance_history.size() > convergence_window){
                    swap_acceptance_history.pop_front();
                }
                
                // Update recent swap rates
                if(swap_acceptance_history.size() >= convergence_window){
                    double recent_rate = std::accumulate(swap_acceptance_history.begin(), 
                                                        swap_acceptance_history.end(), 0.0) / swap_acceptance_history.size();
                    recent_swap_rates.push_back(recent_rate);
                    if(recent_swap_rates.size() > convergence_window/2){
                        recent_swap_rates.pop_front();
                    }
                }
            }

            // Optimized convergence check
            if(i > n_anneal && (i % convergence_check_interval) == 0){
                bool current_check_converged = false;
                
                // Energy convergence check using circular buffer
                bool energy_converged = false;
                if(energy_history.size() >= convergence_window * 2){
                    double recent_sum = 0, old_sum = 0;
                    auto it = energy_history.rbegin();
                    
                    // Recent window
                    for(size_t j = 0; j < convergence_window && it != energy_history.rend(); ++j, ++it){
                        recent_sum += *it;
                    }
                    // Old window
                    for(size_t j = 0; j < convergence_window && it != energy_history.rend(); ++j, ++it){
                        old_sum += *it;
                    }
                    
                    double recent_mean = recent_sum / convergence_window;
                    double old_mean = old_sum / convergence_window;
                    double relative_change = fabs((recent_mean - old_mean) / old_mean);
                    energy_converged = (relative_change < temp_specific_tolerance);
                }
                
                // Swap rate stability check
                bool swap_rate_stable = false;
                if(recent_swap_rates.size() >= convergence_window/4){
                    double mean_rate = std::accumulate(recent_swap_rates.begin(), recent_swap_rates.end(), 0.0) 
                                      / recent_swap_rates.size();
                    
                    if(mean_rate > 0){
                        double variance = 0;
                        for(const auto& rate : recent_swap_rates){
                            double diff = rate - mean_rate;
                            variance += diff * diff;
                        }
                        variance /= (recent_swap_rates.size() - 1);
                        double cv = sqrt(variance) / mean_rate;
                        swap_rate_stable = (cv < swap_rate_convergence_tolerance);
                    }
                }
                
                // Acceptance rate check
                double recent_accept_rate = acceptance_history.empty() ? 0 : 
                    std::accumulate(acceptance_history.begin(), acceptance_history.end(), 0.0) / acceptance_history.size();
                
                // Combined criteria
                current_check_converged = energy_converged && swap_rate_stable && (recent_accept_rate < 0.3);
                
                local_converged_rounds = current_check_converged ? local_converged_rounds + 1 : 0;
                local_converged = (local_converged_rounds >= min_converged_rounds);
                
                // Global convergence check
                int local_conv_int = local_converged ? 1 : 0;
                int global_conv_sum = 0;
                MPI_Allreduce(&local_conv_int, &global_conv_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
                
                // Majority convergence (75% of ranks)
                global_converged = (global_conv_sum >= (size * 3 / 4));
                
                if(global_converged){
                    convergence_step = i;
                }
                
                // Output diagnostics
                if(rank == 0 && (i % (convergence_check_interval * 5)) == 0){
                    cout << "[Step " << i << "] Converged: " << global_conv_sum << "/" << size 
                         << " | Extensions: " << current_extensions << "/" << max_extensions << endl;
                }
                
                // Extension logic
                if(!global_converged && i >= min_steps_before_extension && i >= total_steps - convergence_check_interval){
                    if(current_extensions < max_extensions){
                        total_steps += extension_steps;
                        ++current_extensions;
                        
                        if(rank == 0){
                            cout << "*** Extension " << current_extensions << "/" << max_extensions 
                                 << " | New total: " << total_steps << " ***" << endl;
                        }
                    }
                }
                
                // Write convergence data
                if(conv_file.is_open() && stats_count > 1){
                    double variance = M2_energy / (stats_count - 1);
                    double current_swap_rate = swap_accept / double(i/swap_rate + 1);
                    
                    conv_file << i << " " << E/lattice_size << " " 
                             << curr_accept/(i+1)*overrelaxation_flag << " "
                             << current_swap_rate << " "
                             << running_mean_energy/lattice_size << " "
                             << variance/(lattice_size*lattice_size) << " "
                             << local_converged << " "
                             << global_converged << " "
                             << current_extensions << "\n";
                }
            }

            // Data collection (optimized)
            if (i >= n_anneal && (i % probe_rate) == 0 && !dir_name.empty()){
                energies.push_back(E);
                magnetizations.push_back(magnetization_local(spins));
                spin_configs_at_temp.push_back(spins);
            }
            
            ++i;
        }
        
        // Final statistics and output
        if(!energies.empty()){
            auto [varE, dVarE] = binning_analysis(energies, max(1, int(energies.size()/10)));
            double curr_heat_capacity = varE / (curr_Temp * curr_Temp * lattice_size);
            double curr_dHeat = dVarE / (curr_Temp * curr_Temp * lattice_size);
            
            MPI_Gather(&curr_heat_capacity, 1, MPI_DOUBLE, heat_capacity.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gather(&curr_dHeat, 1, MPI_DOUBLE, dHeat.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        
        // Final output
        double final_variance = stats_count > 1 ? M2_energy / (stats_count - 1) : 0;
        cout << "\n=== Rank " << rank << " Summary ===" << endl;
        cout << "T=" << curr_Temp << " | Steps=" << i << " | E/N=" << running_mean_energy/lattice_size 
             << " | σ(E)/N=" << sqrt(final_variance)/lattice_size << endl;
        cout << "Accept=" << curr_accept/i*overrelaxation_flag 
             << " | Swap=" << double(swap_accept)/(i/swap_rate) 
             << " | Converged=" << (global_converged ? "Yes" : "No") << endl;
        
        if(conv_file.is_open()) conv_file.close();
        
        // Write output files
        if(!dir_name.empty()){
            for(const auto& r : rank_to_write){
                if (rank == r){
                    write_to_file_2d_vector_array(dir_name + "/magnetization" + to_string(rank) + ".txt", magnetizations);
                    write_column_vector(dir_name + "/energy" + to_string(rank) + ".txt", energies);
                }
            }
            
            if (rank == 0){
                write_to_file_pos(dir_name + "/pos.txt");
                ofstream myfile(dir_name + "/heat_capacity.txt", ios::app);
                for(size_t j = 0; j < size; ++j){
                    myfile << temp[j] << " " << heat_capacity[j] << " " << dHeat[j] << "\n";
                }
            }
        }
    }


    void parallel_tempering_with_twist(vector<double> temp, size_t n_anneal, size_t n_measure, 
                                       size_t overrelaxation_rate, size_t swap_rate, size_t probe_rate,
                                       size_t twist_update_rate, string dir_name, 
                                       const vector<int> rank_to_write, bool gaussian_move = false) {
        
        int swap_accept = 0;
        int twist_swap_accept = 0;
        double curr_accept = 0;
        int overrelaxation_flag = overrelaxation_rate > 0 ? overrelaxation_rate : 1;
        
        int initialized;
        MPI_Initialized(&initialized);
        if (!initialized) {
            MPI_Init(NULL, NULL);
        }
        
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        // Improved random seeding
        std::random_device rd;
        seed_lehman((rd() + rank * 1000) | 1);
        
        double E, T_partner, E_partner;
        bool accept;
        spin_config new_spins;
        double curr_Temp = temp[rank];
        
        // Initialize twist parameters per replica
        array<double, 3> twist_angles = {0, 0, 0};
        array<double, 3> twist_angle_step = {M_PI/20, M_PI/20, M_PI/20};
        
        // Pre-calculate which dimensions are active
        array<bool, 3> active_dim = {dim1 > 1, dim2 > 1, dim > 1};
        
        // Set initial twist matrices based on rank
        for (size_t d = 0; d < 3; ++d) {
            if (active_dim[d]) {
                twist_angles[d] = (2.0 * M_PI * rank / size) * (d == 0 ? 1.0 : (d == 1 ? 0.5 : 0.25));
                twist_matrices[d] = rotation_from_axis_angle(rotation_axis[d], twist_angles[d]);
            }
        }
        
        vector<double> heat_capacity, dHeat;
        if (rank == 0) {
            heat_capacity.resize(size);
            dHeat.resize(size);
        }
        
        // Pre-allocate data collection vectors
        const size_t expected_samples = n_measure / probe_rate;
        vector<double> energies;
        vector<array<double, N>> magnetizations;
        vector<spin_config> spin_configs_at_temp;
        vector<array<double, 3>> twist_history;
        
        energies.reserve(expected_samples);
        magnetizations.reserve(expected_samples);
        spin_configs_at_temp.reserve(expected_samples);
        twist_history.reserve(expected_samples);
        
        // Use circular buffer for convergence tracking
        std::deque<double> energy_history;
        const size_t convergence_window = 100;
        const size_t max_history = 1000;
        const double energy_tolerance = 1e-5;
        bool converged = false;
        
        // Running statistics (Welford's algorithm)
        double running_mean_energy = 0;
        double M2_energy = 0;
        size_t stats_count = 0;
        
        cout << "Process " << rank << " initialized with T=" << curr_Temp 
             << " and twist angles=(" << twist_angles[0] << ", " 
             << twist_angles[1] << ", " << twist_angles[2] << ")" << endl;
        
        if (!dir_name.empty()) {
            filesystem::create_directory(dir_name);
        }
        
        // Cache frequently used values
        const size_t total_steps = n_anneal + n_measure;
        const size_t convergence_check_interval = 1000;
        const size_t progress_interval = 1000;
        const double inv_temp = 1.0 / curr_Temp;
        
        // Pre-compute partner ranks for different swap schemes
        const int temp_partner_even = (rank & 1) ? rank - 1 : rank + 1;
        const int temp_partner_odd = (rank & 1) ? rank + 1 : rank - 1;
        const int twist_partner = (rank + size/2) % size;
        
        // Main parallel tempering loop
        for (size_t i = 0; i < total_steps; ++i) {
            
            // Standard Metropolis updates
            double step_accept = 0;
            if (overrelaxation_rate == 0) {
                step_accept = metropolis(spins, curr_Temp, gaussian_move);
                curr_accept += step_accept;
            } else if (i % overrelaxation_rate == 0) {
                overrelaxation();
                step_accept = metropolis(spins, curr_Temp, gaussian_move);
                curr_accept += step_accept;
            } else {
                overrelaxation();
            }
            
            // Update twist angles with reduced boundary calculations
            if (i % twist_update_rate == 0) {
                for (size_t d = 0; d < 3; ++d) {
                    if (!active_dim[d]) continue;
                    
                    // Use metropolis_twist_sweep for efficiency
                    metropolis_twist_sweep(curr_Temp);
                    break; // Update only one dimension per cycle to reduce overhead
                }
            }
            
            E = total_energy(spins);
            
            // Update running statistics
            ++stats_count;
            double delta = E - running_mean_energy;
            running_mean_energy += delta / stats_count;
            M2_energy += delta * (E - running_mean_energy);
            
            // Temperature swap attempts with pre-computed partners
            if (i % swap_rate == 0) {
                const bool even_swap = ((i / swap_rate) & 1) == 0;
                const int partner_rank = even_swap ? temp_partner_even : temp_partner_odd;
                
                if (partner_rank >= 0 && partner_rank < size) {
                    T_partner = temp[partner_rank];
                    
                    // Optimized MPI communication pattern
                    if ((partner_rank & 1) == 0) {
                        MPI_Send(&E, 1, MPI_DOUBLE, partner_rank, 0, MPI_COMM_WORLD);
                        MPI_Recv(&E_partner, 1, MPI_DOUBLE, partner_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    } else {
                        MPI_Recv(&E_partner, 1, MPI_DOUBLE, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        MPI_Send(&E, 1, MPI_DOUBLE, partner_rank, 1, MPI_COMM_WORLD);
                    }
                    
                    // Metropolis criterion
                    const double delta_beta = inv_temp - 1.0/T_partner;
                    const double delta_E = E - E_partner;
                    
                    if ((partner_rank & 1) == 0) {
                        accept = (delta_beta * delta_E < 0) || (random_double_lehman(0, 1) < exp(delta_beta * delta_E));
                        MPI_Send(&accept, 1, MPI_C_BOOL, partner_rank, 2, MPI_COMM_WORLD);
                    } else {
                        MPI_Recv(&accept, 1, MPI_C_BOOL, partner_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                    
                    if (accept) {
                        // Use MPI_Sendrecv for atomic exchange
                        if ((partner_rank & 1) == 0) {
                            MPI_Sendrecv(spins.data(), N*lattice_size, MPI_DOUBLE, partner_rank, 4,
                                        new_spins.data(), N*lattice_size, MPI_DOUBLE, partner_rank, 3,
                                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        } else {
                            MPI_Sendrecv(spins.data(), N*lattice_size, MPI_DOUBLE, partner_rank, 3,
                                        new_spins.data(), N*lattice_size, MPI_DOUBLE, partner_rank, 4,
                                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        }
                        spins = std::move(new_spins);
                        E = E_partner;
                        ++swap_accept;
                    }
                }
            }
            
            // Simplified twist parameter exchange (less frequent)
            if (i % (swap_rate * 4) == swap_rate && twist_partner != rank) {
                array<double, 3> partner_twist_angles;
                
                MPI_Sendrecv(twist_angles.data(), 3, MPI_DOUBLE, twist_partner, 10,
                            partner_twist_angles.data(), 3, MPI_DOUBLE, twist_partner, 10,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // Simplified acceptance: always accept if partner has lower energy twist
                if (rank < twist_partner) {
                    accept = random_double_lehman(0, 1) < 0.5; // 50% acceptance for mixing
                    MPI_Send(&accept, 1, MPI_C_BOOL, twist_partner, 11, MPI_COMM_WORLD);
                } else {
                    MPI_Recv(&accept, 1, MPI_C_BOOL, twist_partner, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                
                if (accept) {
                    for (size_t d = 0; d < 3; ++d) {
                        if (active_dim[d]) {
                            twist_angles[d] = partner_twist_angles[d];
                            twist_matrices[d] = rotation_from_axis_angle(rotation_axis[d], twist_angles[d]);
                        }
                    }
                    ++twist_swap_accept;
                }
            }
            
            // Data collection phase
            if (i >= n_anneal && (i % probe_rate) == 0) {
                energies.push_back(E);
                magnetizations.push_back(magnetization_local(spins));
                spin_configs_at_temp.push_back(spins);
                twist_history.push_back(twist_angles);
            }
            
            // Efficient convergence check using circular buffer
            if (i >= n_anneal) {
                energy_history.push_back(E);
                if (energy_history.size() > max_history) {
                    energy_history.pop_front();
                }
                
                if ((i % convergence_check_interval) == 0 && energy_history.size() >= convergence_window * 2) {
                    double recent_sum = 0, old_sum = 0;
                    auto it = energy_history.rbegin();
                    
                    for (size_t j = 0; j < convergence_window && it != energy_history.rend(); ++j, ++it) {
                        recent_sum += *it;
                    }
                    for (size_t j = 0; j < convergence_window && it != energy_history.rend(); ++j, ++it) {
                        old_sum += *it;
                    }
                    
                    double recent_mean = recent_sum / convergence_window;
                    double old_mean = old_sum / convergence_window;
                    converged = (fabs((recent_mean - old_mean) / old_mean) < energy_tolerance);
                }
            }
            
            // Progress output
            if ((i % progress_interval) == 0 && i > 0) {
                double acceptance_rate = overrelaxation_flag * curr_accept / i;
                double swap_rate_actual = double(swap_accept) / (i / swap_rate + 1);
                double twist_swap_rate_actual = double(twist_swap_accept) / (i / (swap_rate * 4) + 1);
                
                cout << "Rank " << rank << " Step " << i 
                     << " | T=" << curr_Temp 
                     << " | E/N=" << E/lattice_size
                     << " | Accept=" << acceptance_rate
                     << " | TSwap=" << swap_rate_actual
                     << " | TWSwap=" << twist_swap_rate_actual
                     << (converged ? " [CONVERGED]" : "") << endl;
            }
        }
        
        // Calculate final statistics
        if (!energies.empty()) {
            auto [varE, dVarE] = binning_analysis(energies, max(1, int(energies.size() / 10)));
            double curr_heat_capacity = varE / (curr_Temp * curr_Temp * lattice_size);
            double curr_dHeat = dVarE / (curr_Temp * curr_Temp * lattice_size);
            
            MPI_Gather(&curr_heat_capacity, 1, MPI_DOUBLE, heat_capacity.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Gather(&curr_dHeat, 1, MPI_DOUBLE, dHeat.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        
        // Final output
        double final_variance = stats_count > 1 ? M2_energy / (stats_count - 1) : 0;
        cout << "\n=== Final Statistics [Rank " << rank << "] ===" << endl;
        cout << "Temperature: " << curr_Temp << endl;
        cout << "Final Energy/site: " << running_mean_energy/lattice_size << endl;
        cout << "σ(E)/site: " << sqrt(final_variance)/lattice_size << endl;
        cout << "Acceptance rate: " << curr_accept/total_steps*overrelaxation_flag << endl;
        cout << "Temperature swap rate: " << double(swap_accept)/(total_steps/swap_rate) << endl;
        cout << "Twist swap rate: " << double(twist_swap_accept)/(total_steps/(swap_rate*4)) << endl;
        cout << "Converged: " << (converged ? "Yes" : "No") << endl;
        
        // Write output files
        if (!dir_name.empty()) {
            for (const auto& r : rank_to_write) {
                if (rank == r) {
                    write_to_file_2d_vector_array(dir_name + "/magnetization" + to_string(rank) + ".txt", magnetizations);
                    write_column_vector(dir_name + "/energy" + to_string(rank) + ".txt", energies);
                    
                    // Write twist angle history
                    ofstream twist_file(dir_name + "/twist_history" + to_string(rank) + ".txt");
                    for (const auto& angles : twist_history) {
                        twist_file << angles[0] << " " << angles[1] << " " << angles[2] << "\n";
                    }
                }
            }
            
            if (rank == 0) {
                write_to_file_pos(dir_name + "/pos.txt");
                
                ofstream myfile(dir_name + "/heat_capacity.txt");
                for (size_t j = 0; j < size; ++j) {
                    myfile << temp[j] << " " << heat_capacity[j] << " " << dHeat[j] << "\n";
                }
            }
        }
    }

    void measurement(string toread, double T, size_t n_measure, size_t prob_rate, size_t overrelaxation_rate, bool gaussian_move, const vector<int> rank_to_write , string dir_name){
        vector<double> energies;
        vector<array<double,N>> magnetizations;
        vector<double> energy;
        vector<double> magnetization;
        int initialized;
        MPI_Initialized(&initialized);
        if (!initialized){
            MPI_Init(NULL, NULL);
        }
        int rank, size, partner_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        energies.resize(int(n_measure/prob_rate*size));
        magnetizations.resize(int(n_measure/prob_rate*size));

        read_spin_from_file(toread);

        double curr_accept = 0;
        for(size_t i=0; i < n_measure; ++i){
            if(overrelaxation_rate > 0){
                overrelaxation();
                if (i%overrelaxation_rate == 0){
                    curr_accept += metropolis(spins, T, gaussian_move);
                }
            }
            else{
                curr_accept += metropolis(spins, T, gaussian_move);
            }
            if (i % prob_rate == 0){
                magnetization.push_back(magnetization_local(spins));
                energy.push_back(total_energy(spins));
            }
        }


        MPI_Gather(&energy, int(n_measure/prob_rate), MPI_DOUBLE, energies.data(), int(n_measure/prob_rate), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&magnetization, int(n_measure/prob_rate), MPI_DOUBLE, magnetizations.data(), int(n_measure/prob_rate), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if(dir_name != ""){
            filesystem::create_directory(dir_name);
            for(size_t i=0; i<rank_to_write.size(); ++i){
                if (rank == rank_to_write[i]){
                    write_to_file_2d_vector_array(dir_name + "/magnetization" + to_string(rank) + ".txt", magnetizations);
                    write_column_vector(dir_name + "/energy" + to_string(rank) + ".txt", energies);
                }
            }
            if (rank == 0){
                std::tuple<double,double> varE = binning_analysis(energies, int(energies.size()/10));
                double curr_heat_capacity = 1/(T*T)*get<0>(varE)/lattice_size;
                double curr_dHeat = 1/(T*T)*get<1>(varE)/lattice_size;
                write_to_file_pos(dir_name + "/pos.txt");
                ofstream myfile;
                myfile.open(dir_name + "/heat_capacity.txt", ios::app);
                myfile << T << " " << curr_heat_capacity << " " << curr_dHeat << endl;
                myfile.close();
            }
        }

        int finalized;
        if (!MPI_Finalized(&finalized)){
            MPI_Finalize();
        }
    }

    void print_2D(const spin_config &a){
        for(size_t i = 0; i<N_ATOMS*dim1*dim2*dim; ++i){
            for(size_t j = 0; j<N; ++j){
                cout << a[i][j] << " ";
            }
            cout << endl;
        }
    }

    spin_config landau_lifshitz(const spin_config &current_spin, const double &curr_time, cross_product_method cross_prod){
        spin_config dS;
        #pragma omp simd
        for(size_t i = 0; i<lattice_size; ++i){
            dS[i] = cross_prod(get_local_field_lattice(i, current_spin)- drive_field_T(curr_time, i % N_ATOMS), current_spin[i]);
        }
        return dS;
    }

    spin_config RK4_step(double &step_size, const spin_config &curr_spins, const double &curr_time, const double tol, cross_product_method cross_prod){
        spin_config k1 = landau_lifshitz(curr_spins, curr_time, cross_prod);
        spin_config lval_k2 = curr_spins + k1*(0.5*step_size);
        spin_config k2 = landau_lifshitz(lval_k2, curr_time + step_size*0.5,cross_prod);
        spin_config lval_k3 = curr_spins + k2*(0.5*step_size);
        spin_config k3 = landau_lifshitz(lval_k3, curr_time + step_size*0.5, cross_prod);
        spin_config lval_k4 = curr_spins + k3*step_size;
        spin_config k4 = landau_lifshitz(lval_k4, curr_time + step_size, cross_prod);
        spin_config new_spins  = curr_spins + (k1+ k2 * 2 + k3 * 2 + k4)*(step_size/6);
        return new_spins;
    }

    spin_config RK45_step(double &step_size, const spin_config &curr_spins, const double &curr_time, const double tol, cross_product_method cross_prod){
        spin_config k1 = landau_lifshitz(curr_spins, curr_time, cross_prod)*step_size;
        spin_config k2 = landau_lifshitz(curr_spins + k1*(1.0/4.0), curr_time + step_size*(1/4) ,cross_prod)*step_size;
        spin_config k3 = landau_lifshitz(curr_spins + k1*(3.0/32.0) + k2*(9.0/32.0), curr_time + step_size*(3/8) ,cross_prod)*step_size;
        spin_config k4 = landau_lifshitz(curr_spins + k1*(1932.0/2197.0) + k2*(-7200.0/2197.0) + k3*(7296.0/2197.0), curr_time + step_size*(12/13), cross_prod)*step_size;
        spin_config k5 = landau_lifshitz(curr_spins + k1*(439.0/216.0) + k2*(-8.0) + k3*(3680.0/513.0) + k4*(-845.0/4104.0), curr_time + step_size, cross_prod)*step_size;
        spin_config k6 = landau_lifshitz(curr_spins + k1*(-8.0/27.0) + k2*(2.0) + k3*(-3544.0/2565.0)+ k4*(1859.0/4104.0)+ k5*(-11.0/40.0), curr_time + step_size/2,cross_prod)*step_size;

        spin_config y = curr_spins + k1*(25.0/216.0) + k3*(1408.0/2565.0) + k4*(2197.0/4101.0) - k5*(1.0/5.0);
        spin_config z = curr_spins + k1*(16.0/135.0) + k3*(6656.0/12825.0) + k4*(28561.0/56430.0) - k5*(9.0/50.0) + k6*(2.0/55.0);

        double error = norm_average_2D(y-z);
        step_size *= 0.9*pow(tol/error, 0.2);
        if (error < tol){
            return z;
        }
        else{
            return RK45_step(step_size, curr_spins, curr_time, tol, cross_prod);
        }
        return z;
    }
    spin_config RK45_step_fixed(double &step_size, const spin_config &curr_spins, const double &curr_time, const double tol, cross_product_method cross_prod){
        spin_config k1 = landau_lifshitz(curr_spins, curr_time, cross_prod)*step_size;
        spin_config k2 = landau_lifshitz(curr_spins + k1*(1.0/4.0), curr_time + step_size/4 ,cross_prod)*step_size;
        spin_config k3 = landau_lifshitz(curr_spins + k1*(3.0/32.0) + k2*(9.0/32.0), curr_time + step_size/8*3 ,cross_prod)*step_size;
        spin_config k4 = landau_lifshitz(curr_spins + k1*(1932.0/2197.0) + k2*(-7200.0/2197.0) + k3*(7296.0/2197.0), curr_time + step_size/13*12, cross_prod)*step_size;
        spin_config k5 = landau_lifshitz(curr_spins + k1*(439.0/216.0) + k2*(-8.0) + k3*(3680.0/513.0) + k4*(-845.0/4104.0), curr_time + step_size,cross_prod)*step_size;
        spin_config k6 = landau_lifshitz(curr_spins + k1*(-8.0/27.0) + k2*(2.0) + k3*(-3544.0/2565.0)+ k4*(1859.0/4104.0)+ k5*(-11.0/40.0), curr_time + step_size/2,cross_prod)*step_size;
        spin_config z = curr_spins + k1*(16.0/135.0) + k3*(6656.0/12825.0) + k4*(28561.0/56430.0) - k5*(9.0/50.0) + k6*(2.0/55.0);
        return z;
    }
    spin_config euler_step(const double step_size, const spin_config &curr_spins, const double &curr_time, const double tol, cross_product_method cross_prod){
        spin_config dS = landau_lifshitz(curr_spins, curr_time, cross_prod);
        spin_config new_spins = curr_spins + dS*step_size;
        return new_spins;
    }

    spin_config SSPRK53_step(const double step_size, const spin_config &curr_spins, const double &curr_time, const double tol, cross_product_method cross_prod){
        double a30 = 0.355909775063327;
        double a32 = 0.644090224936674;
        double a40 = 0.367933791638137;
        double a43 = 0.632066208361863;
        double a52 = 0.237593836598569;
        double a54 = 0.762406163401431;
        double b10 = 0.377268915331368;
        double b21 = 0.377268915331368;
        double b32 = 0.242995220537396;
        double b43 = 0.238458932846290;
        double b54 =0.287632146308408;
        double c1 = 0.377268915331368;
        double c2 = 0.754537830662736;
        double c3 = 0.728985661612188;
        double c4 = 0.699226135931670;

        spin_config tmp = curr_spins + landau_lifshitz(curr_spins, curr_time, cross_prod) * b10 * step_size;
        spin_config k = landau_lifshitz(tmp, curr_time + c1*step_size, cross_prod);
        spin_config u = tmp + k * step_size * b21;
        //u3
        k = landau_lifshitz(u, curr_time + c2*step_size, cross_prod);
        tmp = curr_spins * a30 + u * a32 + k * step_size * b32;
        k = landau_lifshitz(tmp, curr_time + c3*step_size, cross_prod);
        //u4
        tmp = curr_spins * a40 + tmp * a43 + k * step_size * b43;
        k = landau_lifshitz(tmp, curr_time + c4*step_size, cross_prod);
        u = u * a52 + tmp * a54 + k * step_size * b54;
        return u;
    }


    void molecular_dynamics(double T_start, double T_end, double step_size, string dir_name){
        // simulated_annealing(Temp_start, Temp_end, n_anneal, overrelaxation_rate, gaussian_move);
        if (dir_name != ""){
            filesystem::create_directory(dir_name);
        }
        write_to_file_pos(dir_name + "/pos.txt");
        write_to_file_spin(dir_name + "/spin_t.txt", spins);
        spin_config spin_t = spins;
        cross_product_method cross_prod;
        if constexpr(N==3){
            cross_prod = cross_prod_SU2;
        }else if constexpr(N==8){
            cross_prod = cross_prod_SU3;
        }

        double tol = 1e-8;

        int check_frequency = 10;
        double currT = T_start;
        size_t count = 1;
        vector<double> time;

        time.push_back(currT);
        while(currT < T_end){
            spin_t = RK45_step(step_size, spin_t, currT, tol, cross_prod);
            write_to_file(dir_name + "/spin_t.txt", spin_t);
            currT = currT + step_size;
            cout << "Time: " << currT << endl;
            time.push_back(currT);
            count++;
        }

        ofstream time_sections;
        time_sections.open(dir_name + "/Time_steps.txt");
        for(size_t i = 0; i<count; ++i){
            time_sections << time[i] << endl;
        }
        time_sections.close();
    }



    array<double,N>  magnetization(const spin_config &current_spins, array<array<double, N>, N_ATOMS>  x, array<array<double, N>, N_ATOMS>  y, array<array<double, N>, N_ATOMS>  z){
        array<double,N> mag = {{0}};
        for (size_t i=0; i< dim1; ++i){
            for (size_t j=0; j< dim2; ++j){
                for(size_t k=0; k< dim;++k){
                    for (size_t l=0; l< N_ATOMS;++l){
                        size_t current_site_index = flatten_index(i,j,k,l);
                        
                        mag = x[l] * current_spins[current_site_index][0] 
                            + y[l] * current_spins[current_site_index][1]
                            + z[l] * current_spins[current_site_index][2];

                    }
                }
            }
        }
        return mag/double(lattice_size);
    }

    array<double,N> magnetization_local(const spin_config &current_spins){
        array<double,N> mag = {{0}};
        for (size_t i=0; i< lattice_size; ++i){
            mag = mag + current_spins[i];
        }
        return mag/double(lattice_size);
    }

    array<double,N> magnetization_local_antiferro(const spin_config &current_spins){
        array<double,N> mag = {{0}};
        for (size_t i=0; i< lattice_size; ++i){
            mag = mag + current_spins[i]*pow(-1,i);
        }
        return mag/double(lattice_size);
    }


    void M_B_t(array<array<double,N>, N_ATOMS> &field_in, double t_B, double pulse_amp, double pulse_width, double pulse_freq, double T_start, double T_end, double step_size, string dir_name){
        spin_config spin_t = spins;
        if (dir_name != ""){
            filesystem::create_directory(dir_name);
        }
        double tol = 1e-6;

        double currT = T_start;
        size_t count = 1;
        vector<double> time;
        time.push_back(currT);
        write_to_file_magnetization_local(dir_name + "/M_t.txt", magnetization_local(spin_t));
        // write_to_file_spin(dir_name + "/spin_t.txt", spin_t);
        // ofstream pulse_info;
        // pulse_info.open(dir_name + "/pulse_t.txt");
        cross_product_method cross_prod;
        if constexpr(N==3){
            cross_prod = cross_prod_SU2;
        }else if constexpr(N==8){
            cross_prod = cross_prod_SU3;
        }

        set_pulse(field_in, t_B, {{0}}, 0, pulse_amp, pulse_width, pulse_freq);
        while(currT < T_end){
            // double factor = double(pulse_amp*exp(-pow((currT+t_B)/(2*pulse_width),2))*cos(2*M_PI*pulse_freq*(currT+t_B)));
            // pulse_info << "Current Time: " << currT << " Pulse Time: " << t_B << " Factor: " << factor << " Field: " endl;
            spin_t = RK45_step_fixed(step_size, spin_t, currT, tol, cross_prod);
            write_to_file_magnetization_local(dir_name + "/M_t.txt", magnetization_local_antiferro(spin_t));
            write_to_file_magnetization_local(dir_name + "/M_t_f.txt", magnetization_local(spin_t));

            // write_to_file(dir_name + "/spin_t.txt", spin_t);
            currT = currT + step_size;
            time.push_back(currT);
            count++;
        }
        reset_pulse();
        // pulse_info.close();dfsbf
        ofstream time_sections;
        time_sections.open(dir_name + "/Time_steps.txt");
        for(size_t i = 0; i<count; ++i){
            time_sections << time[i] << endl;
        }
        time_sections.close();      
    };

    void M_BA_BB_t(array<array<double,N>, N_ATOMS> &field_in_1, double t_B_1, array<array<double,N>, N_ATOMS> &field_in_2, double t_B_2, double pulse_amp, double pulse_width, double pulse_freq, double T_start, double T_end, double step_size, string dir_name){
        // simulated_annealing(Temp_start, Temp_end, n_anneal, overrelaxation_rate);
        // write_to_file_pos(dir_name + "/pos.txt");
        // write_to_file_spin(dir_name + "/spin_t.txt", spins);
        cross_product_method cross_prod;
        if constexpr(N==3){
            cross_prod = cross_prod_SU2;
        }else if constexpr(N==8){
            cross_prod = cross_prod_SU3;
        }
        spin_config spin_t = spins;
        if (dir_name != ""){
            filesystem::create_directory(dir_name);
        }
        double tol = 1e-6;
        double currT = T_start;
        size_t count = 1;
        vector<double> time;

        time.push_back(currT);
        write_to_file_magnetization_local(dir_name + "/M_t.txt", magnetization_local_antiferro(spin_t));
        write_to_file_magnetization_local(dir_name + "/M_t_f.txt", magnetization_local(spin_t));

        set_pulse(field_in_1, t_B_1, field_in_2, t_B_2, pulse_amp, pulse_width, pulse_freq);
        while(currT < T_end){
            spin_t = RK45_step_fixed(step_size, spin_t, currT, tol, cross_prod);
            write_to_file_magnetization_local(dir_name + "/M_t.txt", magnetization_local_antiferro(spin_t));
            write_to_file_magnetization_local(dir_name + "/M_t_f.txt", magnetization_local(spin_t));
            
            currT = currT + step_size;
            time.push_back(currT);
            count++;
        }  
        reset_pulse();
        ofstream time_sections;
        time_sections.open(dir_name + "/Time_steps.txt");
        for(size_t i = 0; i<count; ++i){
            time_sections << time[i] << endl;
        }
        time_sections.close();     
    }

};
#endif // LATTICE_H
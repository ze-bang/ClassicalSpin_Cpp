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
        return single_site_energy + double_site_energy/2 + triple_site_energy/3;
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

    void simulated_annealing(double T_start, double T_end, size_t n_anneal, size_t overrelaxation_rate, bool gaussian_move = false, double cooling_rate = 0.9, string out_dir = "", bool boundary_update = false, bool save_observables = false){    
        if (out_dir != ""){
            filesystem::create_directory(out_dir);
        }
        double T = T_start;
        double acceptance_rate = 0;
        double sigma = 1000;
        // cout << "Gaussian Move: " << gaussian_move << endl;
        srand (time(NULL));
        seed_lehman(rand()*2+1);
        while(T > T_end){
            double curr_accept = 0;
            for(size_t i = 0; i<n_anneal; ++i){
                if(overrelaxation_rate > 0){
                    overrelaxation();
                    if (i%overrelaxation_rate == 0){
                        curr_accept += metropolis(spins, T, gaussian_move, sigma);
                    }
                }
                else{
                    curr_accept += metropolis(spins, T, gaussian_move, sigma);
                }
            }
            if (boundary_update) {
                for (size_t i = 0; i < 1e2; ++i){
                    metropolis_twist_sweep(T);
                }
            }
            if (overrelaxation_rate > 0){
                acceptance_rate = curr_accept/n_anneal*overrelaxation_rate;
                cout << "Temperature: " << T << " Acceptance rate: " << acceptance_rate << endl;
            }else{
                acceptance_rate = curr_accept/n_anneal;
                cout << "Temperature: " << T << " Acceptance rate: " << acceptance_rate << endl;
            }
            if (gaussian_move && acceptance_rate < 0.5){
                sigma = sigma * 0.5 / (1-acceptance_rate); 
                cout << "Sigma is adjusted to: " << sigma << endl;   
            }
            if(save_observables){
                vector<double> energies;
                for(size_t i = 0; i<1e7; ++i){
                    if(overrelaxation_rate > 0){
                        overrelaxation();
                        if (i%overrelaxation_rate == 0){
                            metropolis(spins, T, gaussian_move, sigma);
                        }
                    }
                    else{
                        metropolis(spins, T, gaussian_move, sigma);
                    }
                    if (i % 1000 == 0){
                        energies.push_back(total_energy(spins));
                    }
                }
                double k_B = 1.380649e-23;
                double N_A = 6.02214076e23;
                std::tuple<double,double> varE = binning_analysis(energies, int(energies.size()/10));
                double curr_heat_capacity = 1/(T*T)*get<0>(varE)/lattice_size * 2 * k_B * N_A;
                double curr_dHeat = 1/(T*T)*get<1>(varE)/lattice_size * 2 * k_B * N_A;
                ofstream myfile;
                myfile.open(out_dir + "/specific_heat.txt", ios::app);
                myfile << T << " " << curr_heat_capacity << " " << curr_dHeat;
                myfile << endl;
                myfile.close();
            }
            T *= cooling_rate;
        }
        if(out_dir != ""){
            write_to_file_spin(out_dir + "/spin.txt", spins);
            write_to_file_pos(out_dir + "/pos.txt");
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

    void parallel_tempering(vector<double> temp, size_t n_anneal, size_t n_measure, size_t overrelaxation_rate, size_t swap_rate, size_t probe_rate, string dir_name, const vector<int> rank_to_write, bool boundary_update = false, bool gaussian_move = true){

        int swap_accept = 0;
        double curr_accept = 0;
        int overrelaxation_flag = overrelaxation_rate > 0 ? overrelaxation_rate : 1;
        int initialized;
        MPI_Initialized(&initialized);
        if (!initialized){
            MPI_Init(NULL, NULL);
        }
        int rank, size, partner_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        srand (time(NULL));
        seed_lehman(rand()*2+1);
        double E, T_partner, E_partner;
        bool accept;        
        spin_config new_spins;
        double curr_Temp = temp[rank];
        vector<double> heat_capacity, dHeat;
        if (rank == 0){
            heat_capacity.resize(size);
            dHeat.resize(size);
        }   
        vector<double> energies;
        vector<array<double,N>> magnetizations;
        vector<spin_config> spin_configs_at_temp;
        
        // Twist angle coordination variables
        array<array<double, N*N>, 3> shared_twist_matrices = twist_matrices;
        array<array<double, N>, 3> shared_rotation_axes = rotation_axis;
        size_t twist_update_interval = 100; // Update twist angles every N iterations
        size_t twist_consensus_interval = 500; // Global consensus every M iterations
        
        cout << "Initialized Process on rank: " << rank << " with temperature: " << curr_Temp << endl;

        // Prepare progress reporting artifacts if an output directory is provided
        const size_t n_total_iters = n_anneal + n_measure;
        string progress_log_path;
        string progress_now_path;
        if (!dir_name.empty()) {
            try {
                filesystem::create_directory(dir_name);
            } catch (...) {
                // Best-effort; ignore if directory exists or can't be created
            }
            progress_log_path = dir_name + "/progress_rank" + to_string(rank) + ".log";
            progress_now_path = dir_name + "/progress_rank" + to_string(rank) + ".now";
            // Initialize the per-rank progress log with a header
            {
                ofstream flog(progress_log_path, ios::out | ios::trunc);
                flog << "# iter total_iters progress temp energy local_accept_est swap_accept_est timestamp" << '\n';
            }
        }

        for(size_t i=0; i < n_anneal+n_measure; ++i){

            // Metropolis with local twist updates
            if(overrelaxation_rate > 0){
                overrelaxation();
                if (i%overrelaxation_rate == 0){
                    curr_accept += metropolis(spins, curr_Temp, gaussian_move);
                }
            }
            else{
                curr_accept += metropolis(spins, curr_Temp, gaussian_move);
            }
            
            // Coordinated twist angle updates
            if (boundary_update && i % twist_update_interval == 0) {
                // Strategy 1: Temperature-weighted consensus
                // Lower temperature processes have more weight in determining global twist
                if (i % twist_consensus_interval == 0) {
                    // Gather all local twist matrices weighted by inverse temperature
                    array<double, 3*N*N> local_twist_flat;
                    array<double, 3*N*N> global_twist_sum = {0};
                    
                    // Flatten local twist matrices with temperature weighting
                    double weight = 1.0 / curr_Temp; // Lower T = higher weight
                    for (size_t d = 0; d < 3; ++d) {
                        for (size_t j = 0; j < N*N; ++j) {
                            local_twist_flat[d*N*N + j] = twist_matrices[d][j] * weight;
                        }
                    }
                    
                    // All-reduce to sum weighted matrices across all processes
                    MPI_Allreduce(local_twist_flat.data(), global_twist_sum.data(), 
                                  3*N*N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                    
                    // Compute total weight
                    double local_weight = weight;
                    double total_weight = 0;
                    MPI_Allreduce(&local_weight, &total_weight, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                    
                    // Normalize and orthogonalize the consensus matrices
                    for (size_t d = 0; d < 3; ++d) {
                        for (size_t j = 0; j < N*N; ++j) {
                            shared_twist_matrices[d][j] = global_twist_sum[d*N*N + j] / total_weight;
                        }
                        // Re-orthogonalize using Gram-Schmidt if needed (for rotation matrices)
                        if constexpr (N == 3) {
                            shared_twist_matrices[d] = orthogonalize_rotation_matrix(shared_twist_matrices[d]);
                        }
                    }
                    
                    // Update local twist matrices to consensus
                    twist_matrices = shared_twist_matrices;
                }
                
                // Strategy 2: Stochastic local updates with broadcast from lowest T
                else if (boundary_update && rank == 0) { // Lowest temperature process leads
                    // Perform twist update only at lowest temperature
                    metropolis_twist_sweep(curr_Temp);
                    
                    // Broadcast the updated twist matrices to all other processes
                    for (size_t d = 0; d < 3; ++d) {
                        MPI_Bcast(twist_matrices[d].data(), N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    }
                } else {
                    // Receive broadcast from rank 0
                    for (size_t d = 0; d < 3; ++d) {
                        MPI_Bcast(twist_matrices[d].data(), N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                    }
                }
            }
            
            E = total_energy(spins);

            if ((i % swap_rate == 0)){
                accept = false;
                if ((i / swap_rate) % 2 ==0){
                    partner_rank = rank % 2 == 0 ? rank + 1 : rank - 1;
                }else{
                    partner_rank = rank % 2 == 0 ? rank - 1 : rank + 1;
                }
                if ((partner_rank >= 0) && (partner_rank < size)){
                    T_partner = temp[partner_rank];
                    if (partner_rank % 2 == 0){
                        MPI_Send(&E, 1, MPI_DOUBLE, partner_rank, 0, MPI_COMM_WORLD);
                        MPI_Recv(&E_partner, 1, MPI_DOUBLE, partner_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    } else{
                        MPI_Recv(&E_partner, 1, MPI_DOUBLE, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                        MPI_Send(&E, 1, MPI_DOUBLE, partner_rank, 1, MPI_COMM_WORLD);
                    }
                    if (partner_rank % 2 == 0){
                        accept = min(double(1.0), exp((1/curr_Temp-1/T_partner)*(E - E_partner))) > random_double_lehman(0,1);
                        MPI_Send(&accept, 1, MPI_C_BOOL, partner_rank, 2, MPI_COMM_WORLD);
                    } else{
                        MPI_Recv(&accept, 1, MPI_C_BOOL, partner_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    }
                    if (accept){
                        // Exchange both spins AND twist matrices during replica exchange
                        if (partner_rank % 2 == 0){
                            MPI_Send(&spins, N*lattice_size, MPI_DOUBLE, partner_rank, 4, MPI_COMM_WORLD);
                            MPI_Recv(&new_spins, N*lattice_size, MPI_DOUBLE, partner_rank, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            if (boundary_update){
                                // Also exchange twist matrices to maintain consistency
                                for (size_t d = 0; d < 3; ++d) {
                                    array<double, N*N> temp_twist;
                                    MPI_Send(twist_matrices[d].data(), N*N, MPI_DOUBLE, partner_rank, 10+d, MPI_COMM_WORLD);
                                    MPI_Recv(temp_twist.data(), N*N, MPI_DOUBLE, partner_rank, 13+d, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                                    twist_matrices[d] = temp_twist;
                                }
                            }
                           
                        } else{
                            MPI_Recv(&new_spins, N*lattice_size, MPI_DOUBLE, partner_rank, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            MPI_Send(&spins, N*lattice_size, MPI_DOUBLE, partner_rank, 3, MPI_COMM_WORLD);
                            if (boundary_update){
                                for (size_t d = 0; d < 3; ++d) {
                                    array<double, N*N> temp_twist;
                                    MPI_Recv(temp_twist.data(), N*N, MPI_DOUBLE, partner_rank, 10+d, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                                    MPI_Send(twist_matrices[d].data(), N*N, MPI_DOUBLE, partner_rank, 13+d, MPI_COMM_WORLD);
                                    twist_matrices[d] = temp_twist;
                                }
                            }
                        }
                        copy(new_spins.begin(), new_spins.end(), spins.begin());
                        E = E_partner;
                        swap_accept++;
                    }
                }
            }

            // Lightweight, periodic progress reporting that can be tailed during long runs
            if (!dir_name.empty()) {
                // Reuse probe_rate as a sensible default cadence; always report on last iteration
                if ((probe_rate > 0 && (i % probe_rate == 0)) || (i + 1 == n_total_iters)) {
                    const double denom = static_cast<double>(i + 1);
                    // Estimate local and swap acceptance per attempt, consistent with end-of-run reporting
                    const double local_accept_est = (denom > 0.0)
                        ? (static_cast<double>(curr_accept) / denom) * overrelaxation_flag
                        : 0.0;
                    const double swap_accept_est = (denom > 0.0)
                        ? (static_cast<double>(swap_accept) / denom) * swap_rate * overrelaxation_flag
                        : 0.0;
                    const double progress = denom / static_cast<double>(n_total_iters);
                    const std::time_t ts = std::time(nullptr);

                    // Append a line to the per-rank progress log
                    {
                        ofstream flog(progress_log_path, ios::out | ios::app);
                        flog << i+1 << ' '
                             << n_total_iters << ' '
                             << progress << ' '
                             << curr_Temp << ' '
                             << E << ' '
                             << local_accept_est << ' '
                             << swap_accept_est << ' '
                             << ts << '\n';
                    }
                    // Write the latest status to a small file (overwritten) for quick checks
                    {
                        ofstream fnow(progress_now_path, ios::out | ios::trunc);
                        fnow << "iter=" << i+1
                            << " total=" << n_total_iters
                            << " progress=" << progress
                            << " T=" << curr_Temp
                            << " E=" << E
                            << " local_acc_est=" << local_accept_est
                            << " swap_acc_est=" << swap_accept_est
                            << " ts=" << ts << '\n';
                    }
                    // Rank 0 can also emit a concise console heartbeat
                    if (rank == 0) {
                        cout << "[PT] iter " << i+1 << "/" << n_total_iters
                             << " (" << int(progress * 100.0) << "%)"
                             << " T0=" << curr_Temp
                             << " E0=" << E
                             << " acc_local~" << local_accept_est
                             << " acc_swap~" << swap_accept_est << endl;
                    }
                }
            }

            if (i >= n_anneal){
                if (i % probe_rate == 0){
                    if(dir_name != ""){
                        magnetizations.push_back(magnetization_local(spins));
                        spin_configs_at_temp.push_back(spins);
                        energies.push_back(E);
                    }
                }
            }
        }
        
        std::tuple<double,double> varE = binning_analysis(energies, int(energies.size()/10));
        double curr_heat_capacity = 1/(curr_Temp*curr_Temp)*get<0>(varE)/lattice_size;
        double curr_dHeat = 1/(curr_Temp*curr_Temp)*get<1>(varE)/lattice_size;
        MPI_Gather(&curr_heat_capacity, 1, MPI_DOUBLE, heat_capacity.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&curr_dHeat, 1, MPI_DOUBLE, dHeat.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        cout << "Process finished on rank: " << rank << " with temperature: " << curr_Temp << " with local acceptance rate: " << double(curr_accept)/double(n_anneal+n_measure)*overrelaxation_flag << " Swap Acceptance rate: " << double(swap_accept)/double(n_anneal+n_measure)*swap_rate*overrelaxation_flag << endl;
        if(dir_name != ""){
            filesystem::create_directory(dir_name);
            for(size_t i=0; i<rank_to_write.size(); ++i){
                if (rank == rank_to_write[i]){
                    write_to_file_2d_vector_array(dir_name + "/magnetization" + to_string(rank) + ".txt", magnetizations);
                    write_column_vector(dir_name + "/energy" + to_string(rank) + ".txt", energies);
                }
            }
            if (rank == 0){
                write_to_file_pos(dir_name + "/pos.txt");
                ofstream myfile;
                myfile.open(dir_name + "/heat_capacity.txt", ios::app);
                for(size_t j = 0; j<size; ++j){
                    myfile << temp[j] << " " << heat_capacity[j] << " " << dHeat[j] << endl;
                }
                myfile.close();
            }
        }
    }
    
    // Helper function to orthogonalize rotation matrix (Gram-Schmidt)
    array<double, N*N> orthogonalize_rotation_matrix(const array<double, N*N>& M) {
        if constexpr (N == 3) {
            // Extract columns
            array<double, 3> c0 = {M[0], M[3], M[6]};
            array<double, 3> c1 = {M[1], M[4], M[7]};
            array<double, 3> c2 = {M[2], M[5], M[8]};
            
            // Normalize first column
            double norm0 = sqrt(c0[0]*c0[0] + c0[1]*c0[1] + c0[2]*c0[2]);
            c0 = c0 * (1.0/norm0);
            
            // Make c1 orthogonal to c0 and normalize
            double dot01 = c0[0]*c1[0] + c0[1]*c1[1] + c0[2]*c1[2];
            c1 = c1 - c0 * dot01;
            double norm1 = sqrt(c1[0]*c1[0] + c1[1]*c1[1] + c1[2]*c1[2]);
            c1 = c1 * (1.0/norm1);
            
            // c2 = c0 x c1 (cross product for proper rotation matrix)
            c2[0] = c0[1]*c1[2] - c0[2]*c1[1];
            c2[1] = c0[2]*c1[0] - c0[0]*c1[2];
            c2[2] = c0[0]*c1[1] - c0[1]*c1[0];
            
            return {c0[0], c1[0], c2[0],
                    c0[1], c1[1], c2[1],
                    c0[2], c1[2], c2[2]};
        }
        return M;
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
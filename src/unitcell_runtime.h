#ifndef UNITCELL_RUNTIME_H
#define UNITCELL_RUNTIME_H

#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include "lattice_config.h"

using namespace std;

// Runtime version of bilinear interaction (no template)
struct Bilinear {
    vector<double> bilinear_interaction;  // Flattened N*N matrix
    size_t partner;
    array<int, 3> offset;
    size_t N;  // Dimension stored at runtime

    Bilinear(size_t n) : N(n), partner(static_cast<size_t>(-1)), offset({0,0,0}) {
        bilinear_interaction.resize(N * N, 0.0);
    }

    Bilinear(const vector<vector<double>>& b_set, size_t p_set, size_t n) 
        : N(n), partner(p_set), offset({0,0,0}) {
        bilinear_interaction.resize(N * N);
        for(size_t i=0; i<N; i++){
            for (size_t j=0; j<N; j++){
                bilinear_interaction[i*N+j] = b_set[i][j];
            }
        }
    }

    Bilinear(const vector<vector<double>>& b_set, size_t p_set, const array<int, 3>& o_set, size_t n)
        : N(n), partner(p_set), offset(o_set) {
        bilinear_interaction.resize(N * N);
        for(size_t i=0; i<N; i++){
            for (size_t j=0; j<N; j++){
                bilinear_interaction[i*N+j] = b_set[i][j];
            }
        }
    }
};

// Runtime version of trilinear interaction (no template)
struct Trilinear {
    vector<double> trilinear_interaction;  // Flattened N*N*N tensor
    size_t partner1;
    size_t partner2;
    array<int, 3> offset1;
    array<int, 3> offset2;
    size_t N;  // Dimension stored at runtime

    Trilinear(size_t n) : N(n), partner1(static_cast<size_t>(-1)), partner2(static_cast<size_t>(-1)),
                          offset1({0,0,0}), offset2({0,0,0}) {
        trilinear_interaction.resize(N * N * N, 0.0);
    }

    Trilinear(const vector<vector<vector<double>>>& b_set, size_t p1, size_t p2, size_t n)
        : N(n), partner1(p1), partner2(p2), offset1({0,0,0}), offset2({0,0,0}) {
        trilinear_interaction.resize(N * N * N);
        for (size_t i=0; i<N; i++){
            for (size_t j=0; j<N; j++){
                for (size_t l=0; l<N; l++){
                    trilinear_interaction[i*N*N+j*N+l] = b_set[i][j][l];
                }
            }
        }
    }

    Trilinear(const vector<vector<vector<double>>>& b_set, size_t p1, size_t p2,
              const array<int, 3>& o1, const array<int, 3>& o2, size_t n)
        : N(n), partner1(p1), partner2(p2), offset1(o1), offset2(o2) {
        trilinear_interaction.resize(N * N * N);
        for (size_t i=0; i<N; i++){
            for (size_t j=0; j<N; j++){
                for (size_t l=0; l<N; l++){
                    trilinear_interaction[i*N*N+j*N+l] = b_set[i][j][l];
                }
            }
        }
    }
};

// Mixed bilinear interaction between different dimensions
struct MixedBilinear {
    vector<double> bilinear_interaction;  // Flattened N_SU3*N_SU2 matrix
    size_t partner;
    array<int, 3> offset;
    size_t N_SU3;
    size_t N_SU2;

    MixedBilinear(size_t n_su3, size_t n_su2) 
        : N_SU3(n_su3), N_SU2(n_su2), partner(static_cast<size_t>(-1)), offset({0,0,0}) {
        bilinear_interaction.resize(N_SU3 * N_SU2, 0.0);
    }

    MixedBilinear(const vector<vector<double>>& b_set, size_t p, size_t n_su3, size_t n_su2)
        : N_SU3(n_su3), N_SU2(n_su2), partner(p), offset({0,0,0}) {
        bilinear_interaction.resize(N_SU3 * N_SU2);
        for(size_t i=0; i<N_SU3; i++){
            for (size_t j=0; j<N_SU2; j++){
                bilinear_interaction[i*N_SU2+j] = b_set[i][j];
            }
        }
    }

    MixedBilinear(const vector<vector<double>>& b_set, size_t p, const array<int, 3>& o,
                  size_t n_su3, size_t n_su2)
        : N_SU3(n_su3), N_SU2(n_su2), partner(p), offset(o) {
        bilinear_interaction.resize(N_SU3 * N_SU2);
        for(size_t i=0; i<N_SU3; i++){
            for (size_t j=0; j<N_SU2; j++){
                bilinear_interaction[i*N_SU2+j] = b_set[i][j];
            }
        }
    }
};

// Mixed trilinear interaction
struct MixedTrilinear {
    vector<double> trilinear_interaction;  // Flattened N_SU2*N_SU2*N_SU3 tensor
    size_t partner1;
    size_t partner2;
    array<int, 3> offset1;
    array<int, 3> offset2;
    size_t N_SU2;
    size_t N_SU3;

    MixedTrilinear(size_t n_su2, size_t n_su3)
        : N_SU2(n_su2), N_SU3(n_su3), 
          partner1(static_cast<size_t>(-1)), partner2(static_cast<size_t>(-1)),
          offset1({0,0,0}), offset2({0,0,0}) {
        trilinear_interaction.resize(N_SU3 * N_SU2 * N_SU2, 0.0);
    }

    MixedTrilinear(const vector<vector<vector<double>>>& b_set, size_t p1, size_t p2,
                   size_t n_su2, size_t n_su3)
        : N_SU2(n_su2), N_SU3(n_su3), partner1(p1), partner2(p2),
          offset1({0,0,0}), offset2({0,0,0}) {
        trilinear_interaction.resize(N_SU3 * N_SU2 * N_SU2);
        for(size_t i=0; i<N_SU3; i++){
            for (size_t j=0; j<N_SU2; j++){
                for (size_t l=0; l<N_SU2; l++){
                    trilinear_interaction[i*N_SU2*N_SU2+j*N_SU2+l] = b_set[i][j][l];
                }
            }
        }
    }

    MixedTrilinear(const vector<vector<vector<double>>>& b_set, size_t p1, size_t p2,
                   const array<int, 3>& o1, const array<int, 3>& o2,
                   size_t n_su2, size_t n_su3)
        : N_SU2(n_su2), N_SU3(n_su3), partner1(p1), partner2(p2),
          offset1(o1), offset2(o2) {
        trilinear_interaction.resize(N_SU3 * N_SU2 * N_SU2);
        for(size_t i=0; i<N_SU3; i++){
            for (size_t j=0; j<N_SU2; j++){
                for (size_t l=0; l<N_SU2; l++){
                    trilinear_interaction[i*N_SU2*N_SU2+j*N_SU2+l] = b_set[i][j][l];
                }
            }
        }
    }
};

// Runtime version of UnitCell (no templates)
class UnitCell {
public:
    size_t N;          // Spin dimension
    size_t N_ATOMS;    // Number of atoms

    vector<array<double,3>> lattice_pos;
    array<array<double,3>, 3> lattice_vectors;
    vector<vector<vector<double>>> sublattice_frames;  // N_ATOMS x N x N

    vector<vector<double>> field;  // N_ATOMS x N
    vector<vector<double>> onsite_interaction;  // N_ATOMS x (N*N)
    multimap<int, Bilinear> bilinear_interaction;
    multimap<int, Trilinear> trilinear_interaction;

    UnitCell(size_t n, size_t n_atoms, const vector<array<double,3>>& spos,
             const array<array<double,3>, 3>& svec)
        : N(n), N_ATOMS(n_atoms), lattice_pos(spos), lattice_vectors(svec) {
        
        // Initialize field
        field.resize(N_ATOMS, vector<double>(N, 0.0));
        
        // Initialize onsite interaction
        onsite_interaction.resize(N_ATOMS, vector<double>(N * N, 0.0));
        
        // Initialize sublattice frames as identity matrices
        sublattice_frames.resize(N_ATOMS);
        for (size_t atom = 0; atom < N_ATOMS; ++atom) {
            sublattice_frames[atom].resize(N, vector<double>(N, 0.0));
            for (size_t i = 0; i < N; ++i) {
                sublattice_frames[atom][i][i] = 1.0;
            }
        }
    }

    UnitCell(size_t n, size_t n_atoms) : N(n), N_ATOMS(n_atoms) {
        // Initialize with default values
        field.resize(N_ATOMS, vector<double>(N, 0.0));
        onsite_interaction.resize(N_ATOMS, vector<double>(N * N, 0.0));
        lattice_pos.resize(N_ATOMS, {0.0, 0.0, 0.0});
        lattice_vectors = {{{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}};
        
        // Initialize sublattice frames as identity matrices
        sublattice_frames.resize(N_ATOMS);
        for (size_t atom = 0; atom < N_ATOMS; ++atom) {
            sublattice_frames[atom].resize(N, vector<double>(N, 0.0));
            for (size_t i = 0; i < N; ++i) {
                sublattice_frames[atom][i][i] = 1.0;
            }
        }
    }

    void set_field(const vector<double>& f, size_t index) {
        if (index < N_ATOMS && f.size() == N) {
            field[index] = f;
        }
    }

    void set_bilinear_interaction(const vector<vector<double>>& bin, size_t source, 
                                  size_t partner, const array<int, 3>& offset) {
        Bilinear b_set(bin, partner, offset, N);
        bilinear_interaction.insert(make_pair(source, b_set));
    }

    void set_trilinear_interaction(const vector<vector<vector<double>>>& tin, size_t source,
                                   size_t partner1, size_t partner2,
                                   const array<int, 3>& offset1, const array<int, 3>& offset2) {
        Trilinear t_set(tin, partner1, partner2, offset1, offset2, N);
        trilinear_interaction.insert(make_pair(source, t_set));
    }

    void set_onsite_interaction(const vector<double>& oin, size_t index) {
        if (index < N_ATOMS && oin.size() == N * N) {
            onsite_interaction[index] = oin;
        }
    }

    void set_sublattice_frames(const vector<vector<double>>& frames, size_t index) {
        if (index < N_ATOMS && frames.size() == N) {
            sublattice_frames[index] = frames;
        }
    }

    void print() const {
        cout << "--- UnitCell Information ---" << endl;
        cout << "Number of atoms: " << N_ATOMS << endl;
        cout << "Spin dimension (N): " << N << endl;
        
        cout << "\nLattice Positions:" << endl;
        for (size_t i = 0; i < N_ATOMS; ++i) {
            cout << "  Atom " << i << ": (" << lattice_pos[i][0] << ", " 
                 << lattice_pos[i][1] << ", " << lattice_pos[i][2] << ")" << endl;
        }
        
        cout << "\nLattice Vectors:" << endl;
        for (size_t i = 0; i < 3; ++i) {
            cout << "  v" << i + 1 << ": (" << lattice_vectors[i][0] << ", " 
                 << lattice_vectors[i][1] << ", " << lattice_vectors[i][2] << ")" << endl;
        }
        cout << "--- End of UnitCell Information ---" << endl;
    }
};

// Runtime version of mixed UnitCell
class MixedUnitCell {
public:
    UnitCell SU2;
    UnitCell SU3;
    multimap<int, MixedTrilinear> trilinear_SU2_SU3;
    multimap<int, MixedBilinear> bilinear_SU2_SU3;

    MixedUnitCell(size_t n_su2, size_t n_atoms_su2, size_t n_su3, size_t n_atoms_su3)
        : SU2(n_su2, n_atoms_su2), SU3(n_su3, n_atoms_su3) {
    }

    MixedUnitCell(const UnitCell& su2_cell, const UnitCell& su3_cell)
        : SU2(su2_cell), SU3(su3_cell) {
    }

    void set_mix_trilinear_interaction(const vector<vector<vector<double>>>& tin, size_t source,
                                      size_t partner1, size_t partner2,
                                      const array<int, 3>& offset1, const array<int, 3>& offset2) {
        MixedTrilinear t_set(tin, partner1, partner2, offset1, offset2, SU2.N, SU3.N);
        trilinear_SU2_SU3.insert(make_pair(source, t_set));
    }

    void set_mix_bilinear_interaction(const vector<vector<double>>& bin, size_t source,
                                     size_t partner, const array<int, 3>& offset) {
        MixedBilinear b_set(bin, partner, offset, SU3.N, SU2.N);
        bilinear_SU2_SU3.insert(make_pair(source, b_set));
    }

    void set_sublattice_frames_SU2(const vector<vector<double>>& frames, size_t index) {
        SU2.set_sublattice_frames(frames, index);
    }

    void set_sublattice_frames_SU3(const vector<vector<double>>& frames, size_t index) {
        SU3.set_sublattice_frames(frames, index);
    }
};

// Example specific lattice types (without templates)
class HoneyComb : public UnitCell {
public:
    HoneyComb(size_t n) : UnitCell(n, 2, 
        {{{0,0,0}}, {{0,1/double(sqrt(3)),0}}},
        {{{1,0,0}}, {{0.5,double(sqrt(3))/2,0}}, {{0,0,1}}}) {
        
        vector<double> field0(n, 0.0);
        set_field(field0, 0);
        set_field(field0, 1);
    }
};

class Pyrochlore : public UnitCell {
public:
    Pyrochlore(size_t n) : UnitCell(n, 4,
        {{{0.125,0.125,0.125}}, {{0.125,-0.125,-0.125}}, 
         {{-0.125,0.125,-0.125}}, {{-0.125,-0.125,0.125}}},
        {{{0,0.5,0.5}}, {{0.5,0,0.5}}, {{0.5,0.5,0}}}) {
        
        vector<double> field0(n, 0.0);
        for (size_t i = 0; i < 4; ++i) {
            set_field(field0, i);
        }
        
        // Set sublattice frames for Pyrochlore symmetry
        array<double,3> z1 = {1/sqrt(3), 1/sqrt(3), 1/sqrt(3)};
        array<double,3> z2 = {1/sqrt(3),-1/sqrt(3),-1/sqrt(3)};
        array<double,3> z3 = {-1/sqrt(3),1/sqrt(3),-1/sqrt(3)};
        array<double,3> z4 = {-1/sqrt(3),-1/sqrt(3),1/sqrt(3)};

        array<double, 3> y1 = {0,-1/sqrt(2),1/sqrt(2)};
        array<double, 3> y2 = {0,1/sqrt(2),-1/sqrt(2)};
        array<double, 3> y3 = {0,-1/sqrt(2),-1/sqrt(2)};
        array<double, 3> y4 = {0,1/sqrt(2),1/sqrt(2)};

        array<double, 3> x1 = {-2/sqrt(6),1/sqrt(6),1/sqrt(6)};
        array<double, 3> x2 = {-2/sqrt(6),-1/sqrt(6),-1/sqrt(6)};
        array<double, 3> x3 = {2/sqrt(6),1/sqrt(6),-1/sqrt(6)};
        array<double, 3> x4 = {2/sqrt(6),-1/sqrt(6),1/sqrt(6)};

        if (n >= 3) {
            set_sublattice_frames({{x1[0], x1[1], x1[2]}, {y1[0], y1[1], y1[2]}, {z1[0], z1[1], z1[2]}}, 0);
            set_sublattice_frames({{x2[0], x2[1], x2[2]}, {y2[0], y2[1], y2[2]}, {z2[0], z2[1], z2[2]}}, 1);
            set_sublattice_frames({{x3[0], x3[1], x3[2]}, {y3[0], y3[1], y3[2]}, {z3[0], z3[1], z3[2]}}, 2);
            set_sublattice_frames({{x4[0], x4[1], x4[2]}, {y4[0], y4[1], y4[2]}, {z4[0], z4[1], z4[2]}}, 3);
        }
    }
};

#endif // UNITCELL_RUNTIME_H

#ifndef UNITCELL_REFACTORED_H
#define UNITCELL_REFACTORED_H

#include "simple_linear_alg.h"
#include <iostream>
#include <vector>
#include <map>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// Bilinear interaction structure (replaces template version)
struct Bilinear {
    SpinMatrix interaction;  // N×N matrix
    size_t partner;
    Vector3i offset;  // Offset in lattice coordinates
    
    Bilinear() : partner(-1), offset(Vector3i::Zero()) {}
    
    Bilinear(const SpinMatrix& J, size_t p) 
        : interaction(J), partner(p), offset(Vector3i::Zero()) {}
    
    Bilinear(const SpinMatrix& J, size_t p, const Vector3i& o)
        : interaction(J), partner(p), offset(o) {}
};

// Trilinear interaction structure (replaces template version)
struct Trilinear {
    SpinTensor3 interaction;  // N×N×N tensor stored as vector of matrices
    size_t partner1;
    size_t partner2;
    Vector3i offset1;
    Vector3i offset2;
    
    Trilinear() : partner1(-1), partner2(-1), 
                  offset1(Vector3i::Zero()), offset2(Vector3i::Zero()) {}
    
    Trilinear(const SpinTensor3& K, size_t p1, size_t p2)
        : interaction(K), partner1(p1), partner2(p2),
          offset1(Vector3i::Zero()), offset2(Vector3i::Zero()) {}
    
    Trilinear(const SpinTensor3& K, size_t p1, size_t p2, 
              const Vector3i& o1, const Vector3i& o2)
        : interaction(K), partner1(p1), partner2(p2), offset1(o1), offset2(o2) {}
};

// Mixed bilinear for different spin dimensions (e.g., SU2-SU3 coupling)
struct MixedBilinear {
    SpinMatrix interaction;  // N_SU3 × N_SU2 matrix
    size_t partner;
    Vector3i offset;
    
    MixedBilinear() : partner(-1), offset(Vector3i::Zero()) {}
    
    MixedBilinear(const SpinMatrix& J, size_t p, const Vector3i& o)
        : interaction(J), partner(p), offset(o) {}
};

// Mixed trilinear for different spin dimensions
struct MixedTrilinear {
    SpinTensor3 interaction;  // Tensor with mixed dimensions
    size_t partner1;
    size_t partner2;
    Vector3i offset1;
    Vector3i offset2;
    
    MixedTrilinear() : partner1(-1), partner2(-1),
                       offset1(Vector3i::Zero()), offset2(Vector3i::Zero()) {}
    
    MixedTrilinear(const SpinTensor3& K, size_t p1, size_t p2,
                   const Vector3i& o1, const Vector3i& o2)
        : interaction(K), partner1(p1), partner2(p2), offset1(o1), offset2(o2) {}
};

// Unit cell class (replaces template version)
class UnitCell {
public:
    size_t N;  // Spin dimension (3 for SU2, 8 for SU3, etc.)
    size_t N_atoms;  // Number of atoms in unit cell
    
    vector<Vector3d> lattice_pos;  // Atomic positions
    vector<Vector3d> lattice_vectors;  // 3 lattice vectors
    vector<SpinMatrix> sublattice_frames;  // Local coordinate frames for each atom
    
    vector<SpinVector> field;  // External field per atom
    vector<SpinMatrix> onsite_interaction;  // On-site interactions per atom
    
    multimap<int, Bilinear> bilinear_interaction;
    multimap<int, Trilinear> trilinear_interaction;
    
    // Constructors
    UnitCell(size_t spin_dim, size_t num_atoms,
             const vector<Vector3d>& positions,
             const vector<Vector3d>& vectors)
        : N(spin_dim), N_atoms(num_atoms),
          lattice_pos(positions), lattice_vectors(vectors) {
        
        if (positions.size() != num_atoms) {
            throw std::invalid_argument("Number of positions must match N_atoms");
        }
        if (vectors.size() != 3) {
            throw std::invalid_argument("Must provide exactly 3 lattice vectors");
        }
        
        // Initialize fields and interactions to zero
        field.resize(N_atoms, SpinVector::Zero(N));
        onsite_interaction.resize(N_atoms, SpinMatrix::Zero(N, N));
        sublattice_frames.resize(N_atoms, SpinMatrix::Identity(N, N));
    }
    
    UnitCell(size_t spin_dim, size_t num_atoms)
        : N(spin_dim), N_atoms(num_atoms) {
        lattice_pos.resize(num_atoms, Vector3d::Zero());
        lattice_vectors.resize(3, Vector3d::Zero());
        field.resize(N_atoms, SpinVector::Zero(N));
        onsite_interaction.resize(N_atoms, SpinMatrix::Zero(N, N));
        sublattice_frames.resize(N_atoms, SpinMatrix::Identity(N, N));
    }
    
    // Setters
    void set_lattice_pos(const Vector3d& pos, size_t index) {
        if (index >= N_atoms) throw out_of_range("Atom index out of range");
        lattice_pos[index] = pos;
    }
    
    void set_lattice_vector(const Vector3d& vec, size_t index) {
        if (index >= 3) throw out_of_range("Lattice vector index must be 0-2");
        lattice_vectors[index] = vec;
    }
    
    void set_field(const SpinVector& f, size_t index) {
        if (index >= N_atoms) throw out_of_range("Atom index out of range");
        if (f.size() != N) throw invalid_argument("Field dimension mismatch");
        field[index] = f;
    }
    
    void set_bilinear_interaction(const SpinMatrix& J, size_t source, 
                                  size_t partner, const Vector3i& offset) {
        if (J.rows() != N || J.cols() != N) {
            throw invalid_argument("Bilinear matrix dimension mismatch");
        }
        bilinear_interaction.insert(make_pair(source, Bilinear(J, partner, offset)));
    }
    
    void set_trilinear_interaction(const SpinTensor3& K, size_t source,
                                   size_t partner1, size_t partner2,
                                   const Vector3i& offset1, const Vector3i& offset2) {
        if (K.size() != N) {
            throw invalid_argument("Trilinear tensor dimension mismatch");
        }
        trilinear_interaction.insert(make_pair(source, 
            Trilinear(K, partner1, partner2, offset1, offset2)));
    }
    
    void set_onsite_interaction(const SpinMatrix& A, size_t index) {
        if (index >= N_atoms) throw out_of_range("Atom index out of range");
        if (A.rows() != N || A.cols() != N) {
            throw invalid_argument("Onsite matrix dimension mismatch");
        }
        onsite_interaction[index] = A;
    }
    
    void set_sublattice_frame(const SpinMatrix& frame, size_t index) {
        if (index >= N_atoms) throw out_of_range("Atom index out of range");
        if (frame.rows() != N || frame.cols() != N) {
            throw invalid_argument("Frame matrix dimension mismatch");
        }
        sublattice_frames[index] = frame;
    }
    
    // Print method for debugging
    void print() const {
        cout << "--- UnitCell Information ---" << endl;
        cout << "Number of atoms: " << N_atoms << endl;
        cout << "Spin dimension (N): " << N << endl;
        
        cout << "\nLattice Positions:" << endl;
        for (size_t i = 0; i < N_atoms; ++i) {
            cout << "  Atom " << i << ": " << lattice_pos[i].transpose() << endl;
        }
        
        cout << "\nLattice Vectors:" << endl;
        for (size_t i = 0; i < 3; ++i) {
            cout << "  v" << i + 1 << ": " << lattice_vectors[i].transpose() << endl;
        }
        
        cout << "\nFields:" << endl;
        for (size_t i = 0; i < N_atoms; ++i) {
            cout << "  Atom " << i << ": " << field[i].transpose() << endl;
        }
        
        cout << "\nBilinear Interactions: " << bilinear_interaction.size() << " total" << endl;
        cout << "Trilinear Interactions: " << trilinear_interaction.size() << " total" << endl;
        cout << "--- End of UnitCell Information ---" << endl;
    }
};

// Mixed unit cell for systems with multiple spin types (e.g., TmFeO3 with Fe and Tm)
class MixedUnitCell {
public:
    UnitCell SU2_cell;
    UnitCell SU3_cell;
    
    multimap<int, MixedTrilinear> trilinear_SU2_SU3;
    multimap<int, MixedBilinear> bilinear_SU2_SU3;
    
    MixedUnitCell(const UnitCell& su2, const UnitCell& su3)
        : SU2_cell(su2), SU3_cell(su3) {}
    
    void set_mixed_trilinear(const SpinTensor3& K, size_t source,
                            size_t partner1, size_t partner2,
                            const Vector3i& offset1, const Vector3i& offset2) {
        trilinear_SU2_SU3.insert(make_pair(source,
            MixedTrilinear(K, partner1, partner2, offset1, offset2)));
    }
    
    void set_mixed_bilinear(const SpinMatrix& J, size_t source,
                           size_t partner, const Vector3i& offset) {
        bilinear_SU2_SU3.insert(make_pair(source,
            MixedBilinear(J, partner, offset)));
    }
};

// Predefined lattice structures

// Honeycomb lattice
class HoneyComb : public UnitCell {
public:
    HoneyComb(size_t spin_dim) 
        : UnitCell(spin_dim, 2,
                   {Vector3d(0, 0, 0), Vector3d(0, 1/sqrt(3.0), 0)},
                   {Vector3d(1, 0, 0), Vector3d(0.5, sqrt(3.0)/2, 0), Vector3d(0, 0, 1)}) {}
};


class HoneyComb_alt : public UnitCell {
public:
    HoneyComb_alt(size_t spin_dim) 
        : UnitCell(spin_dim, 2,
                   {Vector3d(0, 0, 0), Vector3d(1/sqrt(3.0), 0, 0)},
                   {Vector3d(0, 1, 0), Vector3d(sqrt(3.0)/2, 0.5, 0), Vector3d(0, 0, 1)}) {}
};

// Pyrochlore lattice
class Pyrochlore : public UnitCell {
public:
    Pyrochlore(size_t spin_dim)
        : UnitCell(spin_dim, 4,
                   {Vector3d(0.125, 0.125, 0.125),
                    Vector3d(0.125, -0.125, -0.125),
                    Vector3d(-0.125, 0.125, -0.125),
                    Vector3d(-0.125, -0.125, 0.125)},
                   {Vector3d(0, 0.5, 0.5), Vector3d(0.5, 0, 0.5), Vector3d(0.5, 0.5, 0)}) {
        
        // Set local coordinate frames for pyrochlore sublattices
        if (spin_dim == 3) {  // Only for SU(2)
            Vector3d z1(1/sqrt(3.0), 1/sqrt(3.0), 1/sqrt(3.0));
            Vector3d z2(1/sqrt(3.0), -1/sqrt(3.0), -1/sqrt(3.0));
            Vector3d z3(-1/sqrt(3.0), 1/sqrt(3.0), -1/sqrt(3.0));
            Vector3d z4(-1/sqrt(3.0), -1/sqrt(3.0), 1/sqrt(3.0));
            
            Vector3d y1(0, -1/sqrt(2.0), 1/sqrt(2.0));
            Vector3d y2(0, 1/sqrt(2.0), -1/sqrt(2.0));
            Vector3d y3(0, -1/sqrt(2.0), -1/sqrt(2.0));
            Vector3d y4(0, 1/sqrt(2.0), 1/sqrt(2.0));
            
            Vector3d x1(-2/sqrt(6.0), 1/sqrt(6.0), 1/sqrt(6.0));
            Vector3d x2(-2/sqrt(6.0), -1/sqrt(6.0), -1/sqrt(6.0));
            Vector3d x3(2/sqrt(6.0), 1/sqrt(6.0), -1/sqrt(6.0));
            Vector3d x4(2/sqrt(6.0), -1/sqrt(6.0), 1/sqrt(6.0));
            
            SpinMatrix frame0(3, 3);
            frame0.col(0) = x1; frame0.col(1) = y1; frame0.col(2) = z1;
            set_sublattice_frame(frame0, 0);
            
            SpinMatrix frame1(3, 3);
            frame1.col(0) = x2; frame1.col(1) = y2; frame1.col(2) = z2;
            set_sublattice_frame(frame1, 1);
            
            SpinMatrix frame2(3, 3);
            frame2.col(0) = x3; frame2.col(1) = y3; frame2.col(2) = z3;
            set_sublattice_frame(frame2, 2);
            
            SpinMatrix frame3(3, 3);
            frame3.col(0) = x4; frame3.col(1) = y4; frame3.col(2) = z4;
            set_sublattice_frame(frame3, 3);
        }
    }
};

// TmFeO3 Iron sublattice
class TmFeO3_Fe : public UnitCell {
public:
    TmFeO3_Fe(size_t spin_dim)
        : UnitCell(spin_dim, 4,
                   {Vector3d(0, 0.5, 0.5), Vector3d(0.5, 0, 0.5),
                    Vector3d(0.5, 0, 0), Vector3d(0, 0.5, 0)},
                   {Vector3d(1, 0, 0), Vector3d(0, 1, 0), Vector3d(0, 0, 1)}) {
        
        if (spin_dim == 3) {  // Set local frames for Fe sites
            SpinMatrix frame0 = SpinMatrix::Identity(3, 3);
            set_sublattice_frame(frame0, 0);
            
            SpinMatrix frame1(3, 3);
            frame1 << 1, 0, 0, 0, -1, 0, 0, 0, -1;
            set_sublattice_frame(frame1, 1);
            
            SpinMatrix frame2(3, 3);
            frame2 << -1, 0, 0, 0, 1, 0, 0, 0, -1;
            set_sublattice_frame(frame2, 2);
            
            SpinMatrix frame3(3, 3);
            frame3 << -1, 0, 0, 0, -1, 0, 0, 0, 1;
            set_sublattice_frame(frame3, 3);
        }
    }
};

// TmFeO3 Thulium sublattice
class TmFeO3_Tm : public UnitCell {
public:
    TmFeO3_Tm(size_t spin_dim)
        : UnitCell(spin_dim, 4,
                   {Vector3d(0.02111, 0.92839, 0.75), Vector3d(0.52111, 0.57161, 0.25),
                    Vector3d(0.47889, 0.42839, 0.75), Vector3d(0.97889, 0.07161, 0.25)},
                   {Vector3d(1, 0, 0), Vector3d(0, 1, 0), Vector3d(0, 0, 1)}) {}
};

#endif // UNITCELL_REFACTORED_H

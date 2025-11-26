#ifndef LATTICE_CONFIG_H
#define LATTICE_CONFIG_H

#include <cstddef>

// Runtime configuration structure to replace template parameters
struct LatticeConfig {
    size_t N;           // Spin dimension
    size_t N_ATOMS;     // Number of atoms per unit cell
    size_t dim1;        // Lattice dimension 1
    size_t dim2;        // Lattice dimension 2
    size_t dim;         // Lattice dimension 3
    
    LatticeConfig(size_t n, size_t n_atoms, size_t d1, size_t d2, size_t d3)
        : N(n), N_ATOMS(n_atoms), dim1(d1), dim2(d2), dim(d3) {}
    
    size_t lattice_size() const {
        return dim1 * dim2 * dim * N_ATOMS;
    }
};

// Configuration for mixed lattices with two sublattices
struct MixedLatticeConfig {
    size_t N_SU2;           // SU2 spin dimension
    size_t N_ATOMS_SU2;     // Number of SU2 atoms
    size_t N_SU3;           // SU3 spin dimension
    size_t N_ATOMS_SU3;     // Number of SU3 atoms
    size_t dim1;            // Lattice dimension 1
    size_t dim2;            // Lattice dimension 2
    size_t dim;             // Lattice dimension 3
    
    MixedLatticeConfig(size_t n_su2, size_t n_atoms_su2, size_t n_su3, size_t n_atoms_su3,
                       size_t d1, size_t d2, size_t d3)
        : N_SU2(n_su2), N_ATOMS_SU2(n_atoms_su2), N_SU3(n_su3), N_ATOMS_SU3(n_atoms_su3),
          dim1(d1), dim2(d2), dim(d3) {}
    
    size_t lattice_size_SU2() const {
        return dim1 * dim2 * dim * N_ATOMS_SU2;
    }
    
    size_t lattice_size_SU3() const {
        return dim1 * dim2 * dim * N_ATOMS_SU3;
    }
};

#endif // LATTICE_CONFIG_H

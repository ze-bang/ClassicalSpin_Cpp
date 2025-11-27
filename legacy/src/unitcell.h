#ifndef UNITCELL_H
#define UNITCELL_H

#include <iostream>
#include <array>
#include <vector>
#include <map>
#include <cmath>
using namespace std;



template <size_t N> 
struct bilinear{
    array<double, N*N> bilinear_interaction;
    size_t partner;
    array<int, 3> offset;

    // Constructor

    bilinear(){
        partner = -1;
        for(size_t i =0; i<N*N; i++){
            this->bilinear_interaction[i] = 0;
        }
        for(size_t i=0; i<3; i++) {
            this->offset[i] = 0;
        }
    }

    bilinear(const array<array<double,N>,N>  &b_set, const int p_set) : partner(p_set) {
        for(size_t i=0; i<N; i++){
            for (size_t j=0; j<N; j++){
                this->bilinear_interaction[i*N+j] = b_set[i][j];
            }
        }
        for(size_t i=0; i<3; i++) {
            this->offset[i] = 0;
        }
    };


    bilinear(const array<array<double,N>,N>  &b_set, int p_set, const array<int, 3> &o_set) : partner(p_set) {
        for(size_t i=0; i<N; i++){
            for (size_t j=0; j<N; j++){
                this->bilinear_interaction[i*N+j] = b_set[i][j];
            }
        }
        for(size_t i=0; i<3; i++) {
            this->offset[i] = o_set[i];
        }
    };
};


template <size_t N> 
struct trilinear{
    array<double,N*N*N> trilinear_interaction;
    size_t partner1;
    size_t partner2;
    array<int, 3> offset1;
    array<int, 3> offset2;

    // Constructor

    trilinear(){
        partner1 = -1;
        partner2 = -1;
        for(size_t i =0; i<N; i++){
            for (size_t j=0; j<N; j++){
                for (size_t l =0; l<N; l++){
                    this->trilinear_interaction[i*N*N+j*N+l] = 0;
                }
            }
        }
        for(size_t i=0; i<3; i++) {
            this->offset1[i] = 0;
            this->offset2[i] = 0;
        }
    }

    trilinear(const array<array<array<double,N>, N>,N> &b_set, const int partner1, const int partner2) : partner1(partner1), partner2(partner2) {
        for(size_t i=0; i<3; i++) {
            this->offset1[i] = 0;
            this->offset2[i] = 0;
        }
        for (size_t i=0; i<N; i++){
            for (size_t j=0; j<N; j++){
                for (size_t l =0; l<N; l++){
                    this->trilinear_interaction[i*N*N+j*N+l] = b_set[i][j][l];
                }
            }
        }
    };

    trilinear(const array<array<array<double,N>, N>,N> &b_set, const  int partner1, const int partner2, const array<int, 3> &offset1, const array<int, 3> &offset2) : partner1(partner1), partner2(partner2) {
        for(size_t i=0; i<3; i++) {
            this->offset1[i] = offset1[i];
            this->offset2[i] = offset2[i];
        }
        for (size_t i=0; i<N; i++){
            for (size_t j=0; j<N; j++){
                for (size_t l =0; l<N; l++){
                    this->trilinear_interaction[i*N*N+j*N+l] = b_set[i][j][l];
                }
            }
        }
    };
};


template <size_t N_SU2, size_t N_SU3> 
struct mixed_trilinear{
    array<double,N_SU2*N_SU2*N_SU3> trilinear_interaction;
    size_t partner1;
    size_t partner2;
    array<int, 3> offset1;
    array<int, 3> offset2;

    // Constructor

    mixed_trilinear(){
        partner1 = -1;
        partner2 = -1;
        for(size_t i =0; i<N_SU3; i++){
            for (size_t j=0; j<N_SU2; j++){
                for (size_t l =0; l<N_SU2; l++){
                    this->trilinear_interaction[i*N_SU2*N_SU2*+j*N_SU2+l] = 0;
                }
            }
        }
        for(int i=0; i<3; i++) {
            this->offset1[i] = 0;
            this->offset2[i] = 0;
        }
    }

    mixed_trilinear(array<array<array<double,N_SU2>, N_SU2>,N_SU3> &b_set, int partner1, int partner2) : partner1(partner1), partner2(partner2) {
        for(int i=0; i<3; i++) {
            this->offset1[i] = 0;
            this->offset2[i] = 0;
        }
        for(size_t i=0; i<N_SU3; i++){
            for (size_t j=0; j<N_SU2; j++){
                for (size_t l=0; l<N_SU2; l++){
                    this->trilinear_interaction[i*N_SU2*N_SU2+j*N_SU2+l] = b_set[i][j][l];
                }
            }
        }
    };

    mixed_trilinear(array<array<array<double,N_SU2>, N_SU2>,N_SU3> &b_set, int partner1, int partner2, const array<int, 3> &offset1, const array<int, 3> &offset2) : partner1(partner1), partner2(partner2) {
        for(size_t i=0; i<3; i++) {
            this->offset1[i] = offset1[i];
            this->offset2[i] = offset2[i];
        }
        for(size_t i=0; i<N_SU3; i++){
            for (size_t j=0; j<N_SU2; j++){
                for (size_t l=0; l<N_SU2; l++){
                    this->trilinear_interaction[i*N_SU2*N_SU2+j*N_SU2+l] = b_set[i][j][l];
                }
            }
        }
    };
};


template <size_t N_SU3, size_t N_SU2> 
struct mixed_bilinear{
    array<double,N_SU2*N_SU3> bilinear_interaction;
    size_t partner;
    array<int, 3> offset;

    // Constructor

    mixed_bilinear(){
        partner = -1;
        for(size_t i =0; i<N_SU3; i++){
            for (size_t j=0; j<N_SU2; j++){
                this->bilinear_interaction[i*N_SU2+j] = 0;
            }
        }
        for(int i=0; i<3; i++) {
            this->offset[i] = 0;
        }
    }

    mixed_bilinear(array<array<double,N_SU2>, N_SU3> &b_set, int partner1, int partner2) : partner(partner) {
        for(int i=0; i<3; i++) {
            this->offset[i] = 0;
        }
        for(size_t i=0; i<N_SU3; i++){
            for (size_t j=0; j<N_SU2; j++){
                this->bilinear_interaction[i*N_SU2+j] = b_set[i][j];
                }
            }
        }

    mixed_bilinear(array<array<double,N_SU2>, N_SU3> &b_set, int partner, const array<int, 3> &offset) : partner(partner), offset(offset) {
        for(size_t i=0; i<N_SU3; i++){
            for (size_t j=0; j<N_SU2; j++){
                this->bilinear_interaction[i*N_SU2+j] = b_set[i][j];
            }
        }
    };
};


template<size_t N, size_t N_ATOMS>
struct UnitCell{

    array<array<double,3>, N_ATOMS> lattice_pos;
    array<array<double,3>, 3> lattice_vectors;
    array<array<array<double, N>, N>, N_ATOMS> sublattice_frames;

    array<array<double, N>, N_ATOMS> field;
    array<array<double, N * N>, N_ATOMS> onsite_interaction;
    multimap<int, bilinear<N>> bilinear_interaction;
    multimap<int, trilinear<N>> trilinear_interaction;

    UnitCell(const array<array<double,3>, N_ATOMS> &spos,const array<array<double,3>, 3> &svec) : lattice_pos(spos), lattice_vectors(svec) {
        field = {{{0}}};
        onsite_interaction = {{{0}}};
        // Initialize sublattice_frames with identity matrices
        for (size_t atom = 0; atom < N_ATOMS; ++atom) {
            for (size_t i = 0; i < N; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    sublattice_frames[atom][i][j] = (i == j) ? 1.0 : 0.0;
                }
            }
        }
    };

    UnitCell(){
        field = {{{0}}};
        lattice_pos = {{{0}}};
        lattice_vectors = {{{0}}};
        onsite_interaction = {{{0}}};
        // Initialize sublattice_frames with identity matrices
        for (size_t atom = 0; atom < N_ATOMS; ++atom) {
            for (size_t i = 0; i < N; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    sublattice_frames[atom][i][j] = (i == j) ? 1.0 : 0.0;
                }
            }
        }
    };  

    void set_UnitCell(const UnitCell<N, N_ATOMS> *atoms){
        lattice_pos = atoms->lattice_pos;
        lattice_vectors = atoms->lattice_vectors;
        field = atoms->field;
        bilinear_interaction = atoms->bilinear_interaction;
        trilinear_interaction = atoms->trilinear_interaction;
        onsite_interaction = atoms->onsite_interaction;
    };

    void set_lattice_pos(array<double,N> &pos, int index){
        lattice_pos[index] = pos;
    };
    void set_lattice_vectors(array<double,N> &vectors, int index){
        lattice_vectors[index] = vectors;
    };

    void set_field(const array<double,N> &f, int index){
        field[index] = f;
    };

    void set_bilinear_interaction(const array<array<double,N>, N> &bin, int source, int partner, const array<int, 3> &offset){
        bilinear<N> b_set(bin, partner, offset);
        bilinear_interaction.insert(make_pair(source, b_set));
    };
    
    void set_trilinear_interaction(const array<array<array<double,N>, N>,N> &tin, int source, int partner1, int partner2, const array<int, 3> &offset1, const array<int, 3> &offset2){
        trilinear<N> t_set(tin, partner1, partner2, offset1, offset2);
        trilinear_interaction.insert(make_pair(source, t_set));
    };

    void set_onsite_interaction(const array<double,N * N> &oin, size_t index){
        onsite_interaction[index] = oin;
    };

    void set_sublattice_frames(const array<array<double, N>, N> &frames, size_t index) {
        sublattice_frames[index] = frames;
    }

    void print() const {
        cout << "--- UnitCell Information ---" << endl;
        cout << "Number of atoms: " << N_ATOMS << endl;
        cout << "Spin dimension (N): " << N << endl;

        cout << "\nLattice Positions:" << endl;
        for (size_t i = 0; i < N_ATOMS; ++i) {
            cout << "  Atom " << i << ": (" << lattice_pos[i][0] << ", " << lattice_pos[i][1] << ", " << lattice_pos[i][2] << ")" << endl;
        }

        cout << "\nLattice Vectors:" << endl;
        for (size_t i = 0; i < 3; ++i) {
            cout << "  v" << i + 1 << ": (" << lattice_vectors[i][0] << ", " << lattice_vectors[i][1] << ", " << lattice_vectors[i][2] << ")" << endl;
        }

        cout << "\nFields:" << endl;
        for (size_t i = 0; i < N_ATOMS; ++i) {
            cout << "  Atom " << i << ": [";
            for (size_t j = 0; j < N; ++j) {
                cout << field[i][j] << (j == N - 1 ? "" : ", ");
            }
            cout << "]" << endl;
        }

        cout << "\nOn-site Interactions:" << endl;
        for (size_t i = 0; i < N_ATOMS; ++i) {
            cout << "  Atom " << i << " matrix:" << endl;
            for (size_t row = 0; row < N; ++row) {
                cout << "    [";
                for (size_t col = 0; col < N; ++col) {
                    cout << onsite_interaction[i][row * N + col] << (col == N - 1 ? "" : ", ");
                }
                cout << "]" << endl;
            }
        }

        cout << "\nBilinear Interactions:" << endl;
        if (bilinear_interaction.empty()) {
            cout << "  None" << endl;
        } else {
            for (const auto& pair : bilinear_interaction) {
                cout << "  Source Atom " << pair.first << " -> Partner Atom " << pair.second.partner << endl;
                cout << "    Offset: (" << pair.second.offset[0] << ", " << pair.second.offset[1] << ", " << pair.second.offset[2] << ")" << endl;
                cout << "    Interaction Matrix:" << endl;
                for (size_t i = 0; i < N; ++i) {
                    cout << "      [";
                    for (size_t j = 0; j < N; ++j) {
                        cout << pair.second.bilinear_interaction[i * N + j] << (j == N - 1 ? "" : ", ");
                    }
                    cout << "]" << endl;
                }
            }
        }

        cout << "\nTrilinear Interactions:" << endl;
        if (trilinear_interaction.empty()) {
            cout << "  None" << endl;
        } else {
            for (const auto& pair : trilinear_interaction) {
                cout << "  Source Atom " << pair.first << " -> Partner1 Atom " << pair.second.partner1 << ", Partner2 Atom " << pair.second.partner2 << endl;
                cout << "    Offset1: (" << pair.second.offset1[0] << ", " << pair.second.offset1[1] << ", " << pair.second.offset1[2] << ")" << endl;
                cout << "    Offset2: (" << pair.second.offset2[0] << ", " << pair.second.offset2[1] << ", " << pair.second.offset2[2] << ")" << endl;
                cout << "    Interaction Tensor:" << endl;
                for (size_t i = 0; i < N; ++i) {
                    cout << "      Slice " << i << ":" << endl;
                    for (size_t j = 0; j < N; ++j) {
                        cout << "        [";
                        for (size_t l = 0; l < N; ++l) {
                            cout << pair.second.trilinear_interaction[i * N * N + j * N + l] << (l == N - 1 ? "" : ", ");
                        }
                        cout << "]" << endl;
                    }
                }
            }
        }
        cout << "--- End of UnitCell Information ---" << endl;
    }
};

template<size_t N_SU2, size_t N_ATOMS_SU2, size_t N_SU3, size_t N_ATOMS_SU3>
struct mixed_UnitCell{
    UnitCell<N_SU2, N_ATOMS_SU2> SU2;
    UnitCell<N_SU3, N_ATOMS_SU3> SU3;
    multimap<int, mixed_trilinear<N_SU2, N_SU3>> trilinear_SU2_SU3;
    multimap<int, mixed_bilinear<N_SU3, N_SU2>> bilinear_SU2_SU3;

    mixed_UnitCell() : SU2(), SU3() {
        trilinear_SU2_SU3.clear();
        bilinear_SU2_SU3.clear();
    };

    mixed_UnitCell(UnitCell<N_SU2, N_ATOMS_SU2> *SU2, UnitCell<N_SU3, N_ATOMS_SU3> *SU3) : SU2(*SU2), SU3(*SU3) {
    };

    mixed_UnitCell(const mixed_UnitCell<N_SU2, N_ATOMS_SU2, N_SU3, N_ATOMS_SU3> *atoms) : SU2(atoms->SU2), SU3(atoms->SU3) {
        trilinear_SU2_SU3 = atoms->trilinear_SU2_SU3;
    };

    void set_mix_trilinear_interaction(array<array<array<double,N_SU2>, N_SU2>,N_SU3> &tin, int source, int partner1, int partner2, const array<int, 3> & offset1, const array<int, 3> & offset2){
        mixed_trilinear<N_SU2, N_SU3> t_set(tin, partner1, partner2, offset1, offset2);
        trilinear_SU2_SU3.insert(make_pair(source, t_set));
    };

    void set_mix_bilinear_interaction(array<array<double,N_SU2>, N_SU3> &bin, int source, int partner, const array<int, 3> & offset){
        mixed_bilinear<N_SU3, N_SU2> b_set(bin, partner, offset);
        bilinear_SU2_SU3.insert(make_pair(source, b_set));
    };

    void set_sublattice_frames_SU2(const array<array<double, N_SU2>, N_SU2> &frames, size_t index) {
        SU2.set_sublattice_frames(frames, index);
    }

    void set_sublattice_frames_SU3(const array<array<double, N_SU3>, N_SU3> &frames, size_t index) {
        SU3.set_sublattice_frames(frames, index);
    }


};


template<size_t N>
struct HoneyComb : UnitCell<N, 2>{
    HoneyComb() : UnitCell<N, 2>({{{0,0,0},{0,1/double(sqrt(3)),0}}}, {{{1,0,0},{0.5,double(sqrt(3))/2,0},{0,0,1}}}) {
        array<double,N> field0 = {0};
        this->set_field(field0, 0);
        this->set_field(field0, 1);
    };
};

template<size_t N>
struct HoneyComb_standarx : UnitCell<N, 2>{
    HoneyComb_standarx() : UnitCell<N, 2>({{{0,0,0},{1/double(sqrt(3)),0,0}}}, {{{0,1,0},{double(sqrt(3))/2, 0.5,0},{0,0,1}}}) {
        array<double,N> field0 = {0};
        this->set_field(field0, 0);
        this->set_field(field0, 1);
    };
};

template<size_t N>
struct TmFeO3_Fe : UnitCell<N, 4>{
    TmFeO3_Fe() : UnitCell<N, 4>({{{0, 0.5, 0.5},{0.5, 0, 0.5},{0.5, 0, 0},{0, 0.5, 0}}}, {{{1,0,0},{0,1,0},{0,0,1}}}) {
        array<double,N> field0 = {0};
        this->set_field(field0, 0);
        this->set_field(field0, 1);
        this->set_field(field0, 2);
        this->set_field(field0, 3);
        this->set_sublattice_frames({{{1,0,0},{0,1,0},{0,0,1}}}, 0);
        this->set_sublattice_frames({{{1,0,0},{0,-1,0},{0,0,-1}}}, 1);
        this->set_sublattice_frames({{{-1,0,0},{0,1,0},{0,0,-1}}}, 2);
        this->set_sublattice_frames({{{-1,0,0},{0,-1,0},{0,0,1}}}, 3);
    };
};
template<size_t N>
struct TmFeO3_Tm : UnitCell<N, 4>{
    TmFeO3_Tm() : UnitCell<N, 4>({{{0.02111, 0.92839, 0.75},{0.52111, 0.57161, 0.25},{0.47889, 0.42839, 0.75},{0.97889, 0.07161, 0.25}}}, {{{1,0,0},{0,1,0},{0,0,1}}}) {
        array<double,N> field0 = {0};
        this->set_field(field0, 0);
        this->set_field(field0, 1);
        this->set_field(field0, 2);
        this->set_field(field0, 3);
    };
};

template<size_t N_SU2, size_t N_SU3>
struct TmFeO3 : mixed_UnitCell<N_SU2, 4, N_SU3, 4>{
    TmFeO3() : mixed_UnitCell<N_SU2, 4, N_SU3, 4>(new TmFeO3_Fe<N_SU2>, new TmFeO3_Tm<N_SU3>) {
    };
    TmFeO3(TmFeO3_Fe<N_SU2> *Fe, TmFeO3_Tm<N_SU3> *Tm) : mixed_UnitCell<N_SU2, 4, N_SU3, 4>(Fe, Tm) {
    };
};


template<size_t N>
struct Pyrochlore : UnitCell<N, 4>{
    Pyrochlore() : UnitCell<N, 4>({{{0.125,0.125,0.125},{0.125,-0.125,-0.125},{-0.125,0.125,-0.125},{-0.125,-0.125,0.125}}}, {{{0,0.5,0.5},{0.5,0,0.5},{0.5,0.5,0}}}) {
        array<double,N> field0 = {0};
        this->set_field(field0, 0);
        this->set_field(field0, 1);
        this->set_field(field0, 2);
        this->set_field(field0, 3);

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

        this->set_sublattice_frames({{{x1[0], x1[1], x1[2]}, {y1[0], y1[1], y1[2]}, {z1[0], z1[1], z1[2]}}}, 0);
        this->set_sublattice_frames({{{x2[0], x2[1], x2[2]}, {y2[0], y2[1], y2[2]}, {z2[0], z2[1], z2[2]}}}, 1);
        this->set_sublattice_frames({{{x3[0], x3[1], x3[2]}, {y3[0], y3[1], y3[2]}, {z3[0], z3[1], z3[2]}}}, 2);
        this->set_sublattice_frames({{{x4[0], x4[1], x4[2]}, {y4[0], y4[1], y4[2]}, {z4[0], z4[1], z4[2]}}}, 3);
    };
};


#endif // UNITCELL_H    
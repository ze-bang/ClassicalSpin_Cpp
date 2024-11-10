#ifndef UNITCELL_H
#define UNITCELL_H

#include <iostream>
#include <array>
#include <map>
#include <cmath>
using namespace std;



template <size_t N> 
struct bilinear{
    array<array<float,N>, N> bilinear_interaction;
    int partner;
    int offset[3];

    // Constructor

    bilinear(){
        partner = -1;
        for(int i =0; i<N; i++){
            for (int j=0; j<N; j++){
                this->bilinear_interaction[i][j] = 0;
            }
        }
        for(int i=0; i<3; i++) {
            this->offset[i] = 0;
        }
    }

    bilinear(array<array<float,N>, N>  &b_set, int p_set) : bilinear_interaction(b_set), partner(p_set) {
        for(int i=0; i<3; i++) {
            this->offset[i] = 0;
        }
    };


    bilinear(array<array<float,N>, N>  &b_set, int p_set, int* o_set) : bilinear_interaction(b_set), partner(p_set) {
        for(int i=0; i<3; i++) {
            this->offset[i] = o_set[i];
        }
    };
};


template <size_t N> 
struct trilinear{
    array<array<array<float,N>, N>,N>   trilinear_interaction;
    int partner1;
    int partner2;
    int offset1[3];
    int offset2[3];

    // Constructor

    trilinear(){
        partner1 = -1;
        partner2 = -1;
        for(int i =0; i<N; i++){
            for (int j=0; j<N; j++){
                for (int l =0; l<N; l++){
                    this->trilinear_interaction[i][j][l] = 0;
                }
            }
        }
        for(int i=0; i<3; i++) {
            this->offset1[i] = 0;
            this->offset2[i] = 0;
        }
    }

    trilinear(array<array<array<float,N>, N>,N> &b_set, int partner1, int partner2) : trilinear_interaction(b_set), partner1(partner1), partner2(partner2) {
        for(int i=0; i<3; i++) {
            this->offset1[i] = 0;
            this->offset2[i] = 0;
        }
    };

    trilinear(array<array<array<float,N>, N>,N> b_set, int partner1, int partner2, int* offset1, int* offset2) : trilinear_interaction(b_set), partner1(partner1), partner2(partner2) {
        for(int i=0; i<3; i++) {
            this->offset1[i] = offset1[i];
            this->offset2[i] = offset2[i];
        }
    };
};

template<size_t N, size_t N_ATOMS>
struct UnitCell{

    array<array<float,3>, N_ATOMS> lattice_pos;
    array<array<float,3>, 3> lattice_vectors;

    array<array<float, N>, N_ATOMS> field;
    multimap<int, bilinear<N>> bilinear_interaction;
    multimap<int, trilinear<N>> trilinear_interaction;

    UnitCell(const array<array<float,3>, N_ATOMS> &spos,const array<array<float,3>, 3> &svec) : lattice_pos(spos), lattice_vectors(svec) {
    };

    void set_lattice_pos(array<float,N> &pos, int index){
        lattice_pos[index] = pos;
    };
    void set_lattice_vectors(array<float,N> &vectors, int index){
        lattice_vectors[index] = vectors;
    };

    void set_field(const array<float,N> &f, int index){
        field[index] = f;
    };

    void set_bilinear_interaction(array<array<float,N>, N> &bin, int source, int partner, int* offset){
        bilinear<N> b_set(bin, partner, offset);
        bilinear_interaction.insert(make_pair(source, b_set));
    };
    
    void set_trilinear_interaction(array<array<array<float,N>, N>,N> &tin, int source, int partner1, int partner2, int* offset1, int* offset2){
        trilinear<N> t_set(tin, partner1, partner2, offset1, offset2);
        trilinear_interaction.insert(make_pair(source, t_set));
    };

};


template<size_t N>
struct HoneyComb : UnitCell<N,2>{
    HoneyComb() : UnitCell<N,2>({{{0,0,0},{0,1/float(sqrt(3)),0}}}, {{{1,0,0},{0.5,float(sqrt(3))/2,0},{0,0,1}}}) {
        array<float,N> field0 = {0};
        this->set_field(field0, 0);
        this->set_field(field0, 1);
    };
};

#endif // UNITCELL_H    
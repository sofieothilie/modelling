#ifndef BUFFER_H
#define BUFFER_H

#include "simulation.h"
#include "types.h"

// #define PADDING_BOTTOM_INDEX 0
// #define PADDING_BOTTOM_SIZE (d_Nx + PADDING) * (d_Ny + PADDING) * PADDING

// #define PADDING_SIDE_INDEX PADDING_BOTTOM_SIZE // the side indexing starts right after the BOTTOM
// #define PADDING_SIDE_SIZE (d_Ny + PADDING) * d_Nz *PADDING

// #define PADDING_FRONT_INDEX                                                                        \
//     PADDING_SIDE_INDEX + PADDING_SIDE_SIZE // FRONT starts indexing right after side
// #define PADDING_FRONT_SIZE d_Nx *d_Nz *PADDING

extern Aux_variable d_phi_x_prv, d_phi_y_prv, d_phi_z_prv;
extern Aux_variable d_psi_x_prv, d_psi_y_prv, d_psi_z_prv;

extern Aux_variable d_phi_x, d_phi_y, d_phi_z;
extern Aux_variable d_psi_x, d_psi_y, d_psi_z;

#define pml_indexing(buffer, i, j, k) (get_buf(buffer, i, j, k))

#define d_Psi_x(i, j, k) pml_indexing(d_psi_x, i, j, k)
#define d_Psi_y(i, j, k) pml_indexing(d_psi_y, i, j, k)
#define d_Psi_z(i, j, k) pml_indexing(d_psi_z, i, j, k)
#define d_Phi_x(i, j, k) pml_indexing(d_phi_x, i, j, k)
#define d_Phi_y(i, j, k) pml_indexing(d_phi_y, i, j, k)
#define d_Phi_z(i, j, k) pml_indexing(d_phi_z, i, j, k)

#define SIGMA 1

 real_t PML(int i,
                      int j,
                      int k,
                      real_t *d_buffer,
                      Aux_variable d_phi_x,
                      Aux_variable d_phi_y,
                      Aux_variable d_phi_z,
                      Aux_variable d_psi_x,
                      Aux_variable d_psi_y,
                      Aux_variable d_psi_z);

 void step_all_aux_var();

#endif
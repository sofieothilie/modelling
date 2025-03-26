#ifndef BUFFER_H
#define BUFFER_H

#include "simulation.h"

// #define PADDING_BOTTOM_INDEX 0
// #define PADDING_BOTTOM_SIZE (d_Nx + PADDING) * (d_Ny + PADDING) * PADDING

// #define PADDING_SIDE_INDEX PADDING_BOTTOM_SIZE // the side indexing starts right after the BOTTOM
// #define PADDING_SIDE_SIZE (d_Ny + PADDING) * d_Nz *PADDING

// #define PADDING_FRONT_INDEX                                                                        \
//     PADDING_SIDE_INDEX + PADDING_SIDE_SIZE // FRONT starts indexing right after side
// #define PADDING_FRONT_SIZE d_Nx *d_Nz *PADDING


extern real_t *d_phi_x_prv, *d_phi_y_prv, *d_phi_z_prv;
extern real_t *d_psi_x_prv, *d_psi_y_prv, *d_psi_z_prv;

// might be possible to avoid using this, but it would be a mess
extern real_t *d_phi_x, *d_phi_y, *d_phi_z;
extern real_t *d_psi_x, *d_psi_y, *d_psi_z;


#define pml_indexing(buffer, i, j, k) (boundary_at(buffer, i, j, k))
#define d_Psi_x(i, j, k) pml_indexing(d_psi_x, i, j, k)
#define d_Psi_y(i, j, k) pml_indexing(d_psi_y, i, j, k)
#define d_Psi_z(i, j, k) pml_indexing(d_psi_z, i, j, k)
#define d_Phi_x(i, j, k) pml_indexing(d_phi_x, i, j, k)
#define d_Phi_y(i, j, k) pml_indexing(d_phi_y, i, j, k)
#define d_Phi_z(i, j, k) pml_indexing(d_phi_z, i, j, k)



__global__ void aux_variable_step_z(const real_t *d_buffer,
    real_t *d_phi_z_prv,
    real_t *d_psi_z_prv,
    real_t *d_phi_z,
    real_t *d_psi_z) ;


__device__ real_t PML(int i,
    int j,
    int k,
    real_t *d_buffer,
    real_t *d_phi_x,
    real_t *d_phi_y,
    real_t *d_phi_z,
    real_t *d_psi_x,
    real_t *d_psi_y,
    real_t *d_psi_z);

__global__ void aux_variable_step_x(const real_t *d_buffer,
    real_t *d_phi_x_prv,
    real_t *d_psi_x_prv,
    real_t *d_phi_x,
    real_t *d_psi_x);

__global__ void aux_variable_step_y(const real_t *d_buffer,
    real_t *d_phi_y_prv,
    real_t *d_psi_y_prv,
    real_t *d_phi_y,
    real_t *d_psi_y);

#endif
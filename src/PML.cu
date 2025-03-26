
#include "PML.h"
#include "simulation.h"

#include <stdio.h>

Aux_variable d_phi_x_prv, d_phi_y_prv, d_phi_z_prv;
Aux_variable d_psi_x_prv, d_psi_y_prv, d_psi_z_prv;

Aux_variable d_phi_x, d_phi_y, d_phi_z;
Aux_variable d_psi_x, d_psi_y, d_psi_z;

__device__ real_t PML(int i,
                      int j,
                      int k,
                      real_t *d_buffer,
                      Aux_variable d_phi_x,
                      Aux_variable d_phi_y,
                      Aux_variable d_phi_z,
                      Aux_variable d_psi_x,
                      Aux_variable d_psi_y,
                      Aux_variable d_psi_z) {
    real_t result =
        -(d_Phi_x(i - 1, j, k) * sigma_x(i - 1, j, k) - d_Psi_x(i + 1, j, k) * sigma_x(i, j, k))
            / (d_dx * d_dx)
        - (d_Phi_y(i, j - 1, k) * sigma_y(i, j - 1, k) - d_Psi_y(i, j + 1, k) * sigma_y(i, j, k))
              / (d_dy * d_dy)
        - (d_Phi_z(i, j, k - 1) * sigma_z(i, j, k - 1) - d_Psi_z(i, j, k + 1) * sigma_z(i, j, k))
              / (d_dz * d_dz);
    return K(i, j, k) * K(i, j, k) * result;
}

__device__ real_t sigma_z(int_t i, int_t j, int_t k) {
    if(k < 0 || k >= PADDING)
        return 0;
    return SIGMA;
}
__device__ real_t sigma_x(int_t i, int_t j, int_t k) {
    if(j < 0 || j >= PADDING)
        return 0;
    return SIGMA;
}
__device__ real_t sigma_y(int_t i, int_t j, int_t k) {
    if(j < 0 || j >= PADDING)
        return 0;
    return SIGMA;
}

#define padded_index_bottom(i, j, k)                                                               \
    (i * (d_Nx + PADDING) * (d_Ny + PADDING) + j * (d_Ny + PADDING) + k)
#define padded_index_side(i, j, k) (i + j * PADDING + k * (d_Ny + PADDING) * PADDING)
#define padded_index_front(i, j, k) (i * PADDING + j * PADDING * (d_Nx + PADDING) + k)

__device__ real_t get_buf_bottom(real_t *buffer, int_t i, int_t j, int_t k) {
    if(k < 0 || k >= PADDING)
        return 0;
    return buffer[padded_index_bottom(i, j, k)];
}
__device__ void set_buf_bottom(real_t *buffer, real_t value, int_t i, int_t j, int_t k) {
    if(k < 0 || k >= PADDING) {
        printf("ERROR trying to write at illegal access. check your code.\n");
        return;
    }
    buffer[padded_index_bottom(i, j, k)] = value;
}

__device__ real_t get_buf_side(real_t *buffer, int_t i, int_t j, int_t k) {
    if(i < 0 || i >= PADDING)
        return 0;
    return buffer[padded_index_side(i, j, k)];
}
__device__ void set_buf_side(real_t *buffer, real_t value, int_t i, int_t j, int_t k) {
    if(i < 0 || i >= PADDING) {
        printf("ERROR trying to write at illegal access. check your code.\n");
        return;
    }
    buffer[padded_index_side(i, j, k)] = value;
}

__device__ real_t get_buf_front(real_t *buffer, int_t i, int_t j, int_t k) {
    if(j < 0 || j >= PADDING)
        return 0;
    return buffer[padded_index_front(i, j, k)];
}
__device__ void set_buf_front(real_t *buffer, real_t value, int_t i, int_t j, int_t k) {
    if(j < 0 || j >= PADDING) {
        printf("ERROR trying to write at illegal access. check your code.\n");
        return;
    }
    buffer[padded_index_front(i, j, k)] = value;
}

// interface that finds the correct place in memory from the index
__device__ int get_buf(Aux_variable buffer, int_t i, int_t j, int_t k) {

    if(k >= d_Nz) // on the BOTTOM boundary ! axis go downwards
    {
        k -= d_Nz; // bring bottom layer back up
        return get_buf_bottom(buffer.bottom, i, j, k);
        // dimensions of bottom layer: (Nx + PADDING, Ny + PADDING, PADDING)
    }
    if(i >= d_Nx) // on the SIDE boundary
    {
        i -= d_Nx; // bring side layer to left
        return get_buf_side(buffer.side, i, j, k);
        // dimensions of side layer: (PADDING, Ny + PADDING, Nz)
    }
    if(j >= d_Ny) // on the FRONT boundary!
    {
        j -= d_Ny; // bring front layer back
        return get_buf_front(buffer.front, i, j, k);
        // dimensions of front layer: (Nx, PADDING, Nz)
    }

    // this happens when I'm not in a boundary anymore, what to do then ? I guess that can happen,
    // but will never be processed further, so just ignore this
    return 0;
}

__device__ int set_buf(Aux_variable buffer, real_t value, int_t i, int_t j, int_t k) {

    if(k >= d_Nz) // on the BOTTOM boundary ! axis go downwards
    {
        k -= d_Nz; // bring bottom layer back up
        set_buf_bottom(buffer.bottom, value, i, j, k);
        return;
        // dimensions of bottom layer: (Nx + PADDING, Ny + PADDING, PADDING)
    }
    if(i >= d_Nx) // on the SIDE boundary
    {
        i -= d_Nx; // bring side layer to left
        set_buf_side(buffer.side, value, i, j, k);
        return;
        // dimensions of side layer: (PADDING, Ny + PADDING, Nz)
    }
    if(j >= d_Ny) // on the FRONT boundary!
    {
        j -= d_Ny; // bring front layer back
        set_buf_front(buffer.front, value, i, j, k);
        return;
        // dimensions of front layer: (Nx, PADDING, Nz)
    }

    // this happens when I'm not in a boundary anymore, what to do then ? I guess that can happen,
    // but will never be processed further, so just ignore this
    return 0;
}

/*

    Variable stepping

*/

// splits the boundary into 3 parts and calls 3 kernels
__host__ void step_all_aux_var() {

    int block_x = 8;
    int block_y = 8;
    int block_z = 8;
    dim3 blockSize(block_x, block_y, block_z);
    dim3 gridSize((Nx + PADDING + block_x - 1) / block_x,
                  (Ny + PADDING + block_y - 1) / block_y,
                  (Nz + PADDING + block_z - 1) / block_z);
    dim3 pml_bottom_gridSize((Nx + PADDING + block_x - 1) / block_x,
                             (Ny + PADDING + block_y - 1) / block_y,
                             (PADDING + block_z - 1) / block_z);
    dim3 pml_side_gridSize((PADDING + block_x - 1) / block_x,
                           (Ny + PADDING + block_y - 1) / block_y,
                           (Nz + block_z - 1) / block_z);
    dim3 pml_front_gridSize((Nx + block_x - 1) / block_x,
                            (PADDING + block_y - 1) / block_y,
                            (Nz + block_z - 1) / block_z);

    aux_variable_step_front<<<pml_front_gridSize, blockSize>>>(d_buffer,
                                                               d_phi_y_prv,
                                                               d_psi_y_prv,
                                                               d_phi_y,
                                                               d_psi_y,
                                                               d_phi_x_prv,
                                                               d_psi_x_prv,
                                                               d_phi_x,
                                                               d_psi_x,
                                                               d_phi_z_prv,
                                                               d_psi_z_prv,
                                                               d_phi_z,
                                                               d_psi_z);

    aux_variable_step_side<<<pml_side_gridSize, blockSize>>>(d_buffer,
                                                             d_phi_y_prv,
                                                             d_psi_y_prv,
                                                             d_phi_y,
                                                             d_psi_y,
                                                             d_phi_x_prv,
                                                             d_psi_x_prv,
                                                             d_phi_x,
                                                             d_psi_x,
                                                             d_phi_z_prv,
                                                             d_psi_z_prv,
                                                             d_phi_z,
                                                             d_psi_z);

    aux_variable_step_bottom<<<pml_bottom_gridSize, blockSize>>>(d_buffer,
                                                                d_phi_y_prv,
                                                                d_psi_y_prv,
                                                                d_phi_y,
                                                                d_psi_y,
                                                                d_phi_x_prv,
                                                                d_psi_x_prv,
                                                                d_phi_x,
                                                                d_psi_x,
                                                                d_phi_z_prv,
                                                                d_psi_z_prv,
                                                                d_phi_z,
                                                                d_psi_z);
}

// psi and phi formula
__device__ void update_all_aux_var_at_ijk(const real_t *d_buffer,
                                          Aux_variable d_phi_y_prv,
                                          Aux_variable d_psi_y_prv,
                                          Aux_variable d_phi_y,
                                          Aux_variable d_psi_y,
                                          Aux_variable d_phi_x_prv,
                                          Aux_variable d_psi_x_prv,
                                          Aux_variable d_phi_x,
                                          Aux_variable d_psi_x,
                                          Aux_variable d_phi_z_prv,
                                          Aux_variable d_psi_z_prv,
                                          Aux_variable d_phi_z,
                                          Aux_variable d_psi_z,
                                          int_t i,
                                          int_t j,
                                          int_t k) {

    /*
       I need to keep using the general get_buf (access to all possible sides), because I'm
      accessing neighbors, and might access other sides but it will diverge less, because only the
      border cells will give different results at if statements

    => I can't simply use one buffer side
   */

    real_t next_phi_y = K(i, j + d_Ny, k)
                          * (-0.5 / d_dy
                                 * (sigma_y(i, j - 1, k) * get_buf(d_phi_y_prv, i, j - 1, k)
                                    + sigma_y(i, j, k) * get_buf(d_phi_y_prv, i, j, k))
                             - 0.5 / d_dy * (d_P(i, j + 1, k) - d_P(i, j - 1, k)))
                          * d_dt
                      + get_buf(d_phi_y_prv, i, j, k);

    real_t next_psi_y = K(i, j + d_Ny, k)
                          * (-0.5 / d_dy
                                 * (sigma_y(i, j - 1, k) * get_buf(d_psi_y_prv, i, j, k)
                                    + sigma_y(i, j, k) * get_buf(d_psi_y_prv, i, j + 1, k))
                             - 0.5 / d_dy * (d_P(i, j + 1, k) - d_P(i, j - 1, k)))
                          * d_dt
                      + get_buf(d_psi_y_prv, i, j, k);

    real_t next_phi_x = K(i + d_Nx, j + PADDING, k)
                          * (-0.5 / d_dx
                                 * (sigma_x(i - 1, j, k) * get_buf(d_phi_x_prv, i - 1, j, k)
                                    + sigma_x(i, j, k) * get_buf(d_phi_x_prv, i, j, k))
                             - 0.5 / d_dx * (d_P(i + 1, j, k) - d_P(i - 1, j, k)))
                          * d_dt
                      + get_buf(d_phi_x_prv, i, j, k);

    real_t next_psi_x = K(i + d_Nx, j + PADDING, k)
                          * (-0.5 / d_dx
                                 * (sigma_x(i - 1, j, k) * get_buf(d_psi_x_prv, i, j, k)
                                    + sigma_x(i, j, k) * get_buf(d_psi_x_prv, i + 1, j, k))
                             - 0.5 / d_dx * (d_P(i + 1, j, k) - d_P(i - 1, j, k)))
                          * d_dt
                      + get_buf(d_psi_x_prv, i, j, k);

    real_t next_phi_z = K(i, j, k + d_Nz)
                          * (-0.5 / d_dz
                                 * (sigma_z(i, j, k - 1) * get_buf(d_phi_z_prv, i, j, k - 1)
                                    + sigma_z(i, j, k) * get_buf(d_phi_z_prv, i, j, k))
                             - 0.5 / d_dz * (d_P(i, j, k + 1) - d_P(i, j, k - 1)))
                          * d_dt
                      + get_buf(d_phi_z_prv, i, j, k);

    real_t next_psi_z = K(i, j, k + d_Nz)
                          * (-0.5 / d_dz
                                 * (sigma_z(i, j, k - 1) * get_buf(d_psi_z_prv, i, j, k)
                                    + sigma_z(i, j, k) * get_buf(d_psi_z_prv, i, j, k + 1))
                             - 0.5 / d_dz * (d_P(i, j, k + 1) - d_P(i, j, k - 1)))
                          * d_dt
                      + get_buf(d_psi_z_prv, i, j, k);

    set_buf(d_phi_y, next_phi_y, i, j, k);
    set_buf(d_psi_y, next_psi_y, i, j, k);
    set_buf(d_phi_x, next_phi_x, i, j, k);
    set_buf(d_psi_x, next_psi_x, i, j, k);
    set_buf(d_phi_z, next_phi_z, i, j, k);
    set_buf(d_psi_z, next_psi_z, i, j, k);
}

/*NOTE
    The three following function all do the same thing, they are just calling the step formula for different parts of
   the border for warp and indexing
*/
__global__ void aux_variable_step_front(const real_t *d_buffer,
                                        Aux_variable d_phi_y_prv,
                                        Aux_variable d_psi_y_prv,
                                        Aux_variable d_phi_y,
                                        Aux_variable d_psi_y,
                                        Aux_variable d_phi_x_prv,
                                        Aux_variable d_psi_x_prv,
                                        Aux_variable d_phi_x,
                                        Aux_variable d_psi_x,
                                        Aux_variable d_phi_z_prv,
                                        Aux_variable d_psi_z_prv,
                                        Aux_variable d_phi_z,
                                        Aux_variable d_psi_z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i >= d_Nx || j >= PADDING || k >= d_Nz) {
        return;
    }

    update_all_aux_var_at_ijk(d_buffer,
                              d_phi_y_prv,
                              d_psi_y_prv,
                              d_phi_y,
                              d_psi_y,
                              d_phi_x_prv,
                              d_psi_x_prv,
                              d_phi_x,
                              d_psi_x,
                              d_phi_z_prv,
                              d_psi_z_prv,
                              d_phi_z,
                              d_psi_z,
                              i,
                              j,
                              k);
}

__global__ void aux_variable_step_side(const real_t *d_buffer,
                                       Aux_variable d_phi_y_prv,
                                       Aux_variable d_psi_y_prv,
                                       Aux_variable d_phi_y,
                                       Aux_variable d_psi_y,
                                       Aux_variable d_phi_x_prv,
                                       Aux_variable d_psi_x_prv,
                                       Aux_variable d_phi_x,
                                       Aux_variable d_psi_x,
                                       Aux_variable d_phi_z_prv,
                                       Aux_variable d_psi_z_prv,
                                       Aux_variable d_phi_z,
                                       Aux_variable d_psi_z) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i >= PADDING || j >= d_Ny + PADDING || k >= d_Nz) {
        return;
    }

    update_all_aux_var_at_ijk(d_buffer,
                              d_phi_y_prv,
                              d_psi_y_prv,
                              d_phi_y,
                              d_psi_y,
                              d_phi_x_prv,
                              d_psi_x_prv,
                              d_phi_x,
                              d_psi_x,
                              d_phi_z_prv,
                              d_psi_z_prv,
                              d_phi_z,
                              d_psi_z,
                              i,
                              j,
                              k);
}

__global__ void aux_variable_step_bottom(const real_t *d_buffer,
                                         Aux_variable d_phi_y_prv,
                                         Aux_variable d_psi_y_prv,
                                         Aux_variable d_phi_y,
                                         Aux_variable d_psi_y,
                                         Aux_variable d_phi_x_prv,
                                         Aux_variable d_psi_x_prv,
                                         Aux_variable d_phi_x,
                                         Aux_variable d_psi_x,
                                         Aux_variable d_phi_z_prv,
                                         Aux_variable d_psi_z_prv,
                                         Aux_variable d_phi_z,
                                         Aux_variable d_psi_z) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i >= d_Nx + PADDING || j >= d_Ny + PADDING || k >= PADDING) {
        return;
    }

    update_all_aux_var_at_ijk(d_buffer,
                              d_phi_y_prv,
                              d_psi_y_prv,
                              d_phi_y,
                              d_psi_y,
                              d_phi_x_prv,
                              d_psi_x_prv,
                              d_phi_x,
                              d_psi_x,
                              d_phi_z_prv,
                              d_psi_z_prv,
                              d_phi_z,
                              d_psi_z,
                              i,
                              j,
                              k);
}
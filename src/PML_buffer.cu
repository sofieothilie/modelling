
#include "simulation.h"

#include "PML_buffer.h"

// the indexing is weird, because these values only exist for the boundaries, so looks like a
// corner, of depth PADDING
real_t *d_phi_x_prv, *d_phi_y_prv, *d_phi_z_prv;
real_t *d_psi_x_prv, *d_psi_y_prv, *d_psi_z_prv;

// might be possible to avoid using this, but it would be a mess
real_t *d_phi_x, *d_phi_y, *d_phi_z;
real_t *d_psi_x, *d_psi_y, *d_psi_z;

__device__ real_t PML(int i,
    int j,
    int k,
    real_t *d_buffer,
    real_t *d_phi_x,
    real_t *d_phi_y,
    real_t *d_phi_z,
    real_t *d_psi_x,
    real_t *d_psi_y,
    real_t *d_psi_z) {
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
    return 1;
}

#define padded_index_z(i, j, k) (i * (d_Nx + PADDING) * (d_Ny + PADDING) + j * (d_Ny + PADDING) + k)
__device__ real_t buf_at_z(real_t *buffer, int_t i, int_t j, int_t k) {
    if(k < 0 || k >= PADDING)
        return 0;
    return buffer[padded_index_z(i, j, k)];
}

__device__ void set_buf_z(real_t *buffer, real_t value, int_t i, int_t j, int_t k) {
    if(k < 0 || k >= PADDING)
        return;
    buffer[padded_index_z(i, j, k)] = value;
}

__global__ void aux_variable_step_z(const real_t *d_buffer,
                                    real_t *d_phi_z_prv,
                                    real_t *d_psi_z_prv,
                                    real_t *d_phi_z,
                                    real_t *d_psi_z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i >= d_Nx + PADDING || j >= d_Ny + PADDING || k >= PADDING) {
        return;
    }

    real_t next_phi_z = K(i, j, k + d_Nz)
                          * (-0.5 / d_dz
                                 * (sigma_z(i, j, k - 1) * buf_at_z(d_phi_z_prv, i, j, k - 1)
                                    + sigma_z(i, j, k) * buf_at_z(d_phi_z_prv, i, j, k))
                             - 0.5 / d_dz * (d_P(i, j, k + 1) - d_P(i, j, k - 1)))
                          * d_dt
                      + buf_at_z(d_phi_z_prv, i, j, k);

    real_t next_psi_z = K(i, j, k + d_Nz)
                          * (-0.5 / d_dz
                                 * (sigma_z(i, j, k - 1) * buf_at_z(d_psi_z_prv, i, j, k)
                                    + sigma_z(i, j, k) * buf_at_z(d_psi_z_prv, i, j, k + 1))
                             - 0.5 / d_dz * (d_P(i, j, k + 1) - d_P(i, j, k - 1)))
                          * d_dt
                      + buf_at_z(d_psi_z_prv, i, j, k);

    set_buf_z(d_phi_z, next_phi_z, i, j, k);
    set_buf_z(d_psi_z, next_psi_z, i, j, k);
}

__device__ real_t sigma_x(int_t i, int_t j, int_t k) {
    if(j < 0 || j >= PADDING)
        return 0;
    return 1;
}

#define padded_index_x(i, j, k) (i + j * PADDING + k * (d_Ny + PADDING) * PADDING)
__device__ real_t buf_at_x(real_t *buffer, int_t i, int_t j, int_t k) {
    if(i < 0 || i >= PADDING)
        return 0;
    return buffer[padded_index_x(i, j, k)];
}

__device__ void set_buf_x(real_t *buffer, real_t value, int_t i, int_t j, int_t k) {
    if(i < 0 || i >= PADDING)
        return;
    buffer[padded_index_x(i, j, k)] = value;
}

__global__ void aux_variable_step_x(const real_t *d_buffer,
                                    real_t *d_phi_x_prv,
                                    real_t *d_psi_x_prv,
                                    real_t *d_phi_x,
                                    real_t *d_psi_x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i >= PADDING || j >= d_Ny || k >= d_Nz + PADDING) {
        return;
    }

    real_t next_phi_x = K(i + d_Nx, j + PADDING, k)
                          * (-0.5 / d_dx
                                 * (sigma_x(i - 1, j, k) * buf_at_x(d_phi_x_prv, i - 1, j, k)
                                    + sigma_x(i, j, k) * buf_at_x(d_phi_x_prv, i, j, k))
                             - 0.5 / d_dx * (d_P(i + 1, j, k) - d_P(i - 1, j, k)))
                          * d_dt
                      + buf_at_x(d_phi_x_prv, i, j, k);

    real_t next_psi_x = K(i + d_Nx, j + PADDING, k)
                          * (-0.5 / d_dx
                                 * (sigma_x(i - 1, j, k) * buf_at_x(d_psi_x_prv, i, j, k)
                                    + sigma_x(i, j, k) * buf_at_x(d_psi_x_prv, i + 1, j, k))
                             - 0.5 / d_dx * (d_P(i + 1, j, k) - d_P(i - 1, j, k)))
                          * d_dt
                      + buf_at_x(d_psi_x_prv, i, j, k);

    set_buf_x(d_phi_x, next_phi_x, i, j, k);
    set_buf_x(d_psi_x, next_psi_x, i, j, k);
}

__device__ real_t sigma_y(int_t i, int_t j, int_t k) {
    if(j < 0 || j >= PADDING)
        return 0;
    return 1;
}

#define padded_index_y(i, j, k) (i * PADDING + j * PADDING * (d_Nx + PADDING) + k)
__device__ real_t buf_at_y(real_t *buffer, int_t i, int_t j, int_t k) {
    if(j < 0 || j >= PADDING)
        return 0;
    return buffer[padded_index_y(i, j, k)];
}

__device__ void set_buf_y(real_t *buffer, real_t value, int_t i, int_t j, int_t k) {
    if(j < 0 || j >= PADDING)
        return;
    buffer[padded_index_y(i, j, k)] = value;
}

__global__ void aux_variable_step_y(const real_t *d_buffer,
                                    real_t *d_phi_y_prv,
                                    real_t *d_psi_y_prv,
                                    real_t *d_phi_y,
                                    real_t *d_psi_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i >= d_Nx || j >= PADDING || k >= d_Nz) {
        return;
    }

    real_t next_phi_y = K(i, j + d_Ny, k)
                          * (-0.5 / d_dy
                                 * (sigma_y(i, j - 1, k) * buf_at_y(d_phi_y_prv, i, j - 1, k)
                                    + sigma_y(i, j, k) * buf_at_y(d_phi_y_prv, i, j, k))
                             - 0.5 / d_dy * (d_P(i, j + 1, k) - d_P(i, j - 1, k)))
                          * d_dt
                      + buf_at_y(d_phi_y_prv, i, j, k);

    real_t next_psi_y = K(i, j + d_Ny, k)
                          * (-0.5 / d_dy
                                 * (sigma_y(i, j - 1, k) * buf_at_y(d_psi_y_prv, i, j, k)
                                    + sigma_y(i, j, k) * buf_at_y(d_psi_y_prv, i, j + 1, k))
                             - 0.5 / d_dy * (d_P(i, j + 1, k) - d_P(i, j - 1, k)))
                          * d_dt
                      + buf_at_y(d_psi_y_prv, i, j, k);

    set_buf_y(d_phi_y, next_phi_y, i, j, k);
    set_buf_y(d_psi_y, next_psi_y, i, j, k);
}

// When calling this, consider the warp geometry to reduce divergence. more info in my notebook
__device__ int boundary_at(real_t *buffer, int_t i, int_t j, int_t k) {
    // doing some weird indexing to retrieve the correct boundary

    // assuming x is on the side, y axis pointing you, I guess I could have done better to have the
    // axes in reverse order but ok

    if(k >= d_Nz) // on the BOTTOM boundary ! axis go downwards
    {
        k -= d_Nz; // bring bottom layer back up
        return buf_at_z(buffer, i, j, k);
        // dimensions of bottom layer: (Nx + PADDING, Ny + PADDING, PADDING)
        size_t idx = i * (d_Ny + PADDING) * PADDING + j * PADDING + k;
        return buffer[idx];
    }
    if(i >= d_Nx) // on the SIDE boundary
    {
        i -= d_Nx; // bring side layer to left
        return buf_at_x(buffer, i, j, k);
        // dimensions of side layer: (PADDING, Ny + PADDING, Nz)
        size_t idx = i * (d_Ny + PADDING) * d_Nz + j * d_Nz + k;
        return buffer[idx];
    }
    if(j >= d_Ny) // on the FRONT boundary!
    {
        j -= d_Ny; // bring front layer back
        return buf_at_z(buffer, i, j, k);
        // dimensions of front layer: (Nx, PADDING, Nz)
        size_t idx = i * PADDING * d_Nz + j * d_Nz + k;
        return buffer[idx];
    }

    // this happens when I'm not in a boundary anymore, what to do then ? I guess that can happen,
    // but will never be processed further, so just ignore this
    return 0;
}


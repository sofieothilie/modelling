#include "simulation.h"
#include "PML_buffer.h"

__device__ real_t gauss_seidel_formula(int i,
    int j,
    int k,
    real_t *d_buffer,
    real_t *d_buffer_prv,
    real_t *d_buffer_prv_prv,
    real_t *d_phi_x,
    real_t *d_phi_y,
    real_t *d_phi_z,
    real_t *d_psi_x,
    real_t *d_psi_y,
    real_t *d_psi_z) {
real_t PML_val = PML(i, j, k, d_buffer, d_phi_x, d_phi_y, d_phi_z, d_psi_x, d_psi_y, d_psi_z);

real_t result =
(d_dt * d_dt)
* (2 * (-K(i - 1, j, k) / (2 * d_dx) + K(i + 1, j, k) / (2 * d_dx))
* (-d_P(i - 1, j, k) / (2 * d_dx) + d_P(i + 1, j, k) / (2 * d_dx)) * K(i, j, k)
+ 2 * (-K(i, j - 1, k) / (2 * d_dy) + K(i, j + 1, k) / (2 * d_dy))
* (-d_P(i, j - 1, k) / (2 * d_dy) + d_P(i, j + 1, k) / (2 * d_dy)) * K(i, j, k)
+ 2 * (-K(i, j, k - 1) / (2 * d_dz) + K(i, j, k + 1) / (2 * d_dz))
* (-d_P(i, j, k - 1) / (2 * d_dz) + d_P(i, j, k + 1) / (2 * d_dz)) * K(i, j, k)
+ (-2 * d_P(i, j, k) / (d_dx * d_dx) + d_P(i - 1, j, k) / (d_dx * d_dx)
+ d_P(i + 1, j, k) / (d_dx * d_dx))
* (K(i, j, k) * K(i, j, k))
+ (-2 * d_P(i, j, k) / (d_dy * d_dy) + d_P(i, j - 1, k) / (d_dy * d_dy)
+ d_P(i, j + 1, k) / (d_dy * d_dy))
* (K(i, j, k) * K(i, j, k))
+ (-2 * d_P(i, j, k) / (d_dz * d_dz) + d_P(i, j, k - 1) / (d_dz * d_dz)
+ d_P(i, j, k + 1) / (d_dz * d_dz))
* (K(i, j, k) * K(i, j, k))
+ PML_val)
+ 2 * d_P_prv(i, j, k) - d_P_prv_prv(i, j, k);

return result;
}

__global__ void gauss_seidel_red(real_t *d_buffer,
real_t *d_buffer_prv,
real_t *d_buffer_prv_prv,
real_t *d_phi_x,
real_t *d_phi_y,
real_t *d_phi_z,
real_t *d_psi_x,
real_t *d_psi_y,
real_t *d_psi_z) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;
int k = blockIdx.z * blockDim.z + threadIdx.z;

if(i >= d_Nx + PADDING || j >= d_Ny + PADDING || k >= d_Nz + PADDING)
return;

if((i + j + k) % 2 == 0) {
d_P(i, j, k) = gauss_seidel_formula(i,
         j,
         k,
         d_buffer,
         d_buffer_prv,
         d_buffer_prv_prv,
         d_phi_x,
         d_phi_y,
         d_phi_z,
         d_psi_x,
         d_psi_y,
         d_psi_z);
}
}

__global__ void gauss_seidel_black(real_t *d_buffer,
real_t *d_buffer_prv,
real_t *d_buffer_prv_prv,
real_t *d_phi_x,
real_t *d_phi_y,
real_t *d_phi_z,
real_t *d_psi_x,
real_t *d_psi_y,
real_t *d_psi_z) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;
int k = blockIdx.z * blockDim.z + threadIdx.z;

if(i >= d_Nx + PADDING || j >= d_Ny + PADDING || k >= d_Nz + PADDING)
return;

if((i + j + k) % 2 == 1) {
d_P(i, j, k) = gauss_seidel_formula(i,
         j,
         k,
         d_buffer,
         d_buffer_prv,
         d_buffer_prv_prv,
         d_phi_x,
         d_phi_y,
         d_phi_z,
         d_psi_x,
         d_psi_y,
         d_psi_z);
}
}

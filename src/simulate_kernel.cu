#define _USE_MATH_DEFINES
#include "simulate_kernel.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

// #define MODEL_LY (double)1//width of model in meters
// #define MODEL_LZ 0.2//depth of model in centimeters

// #define RESERVOIR_OFFSET .5//just water on each side of the reservoir

// total size of simulation
// 5x1x1 cm tube of water
//  #define SIM_LX 0.01
//  #define SIM_LY 0.01
//  #define SIM_LZ 0.01//need to add height of sensors, but thats a parameter

// source and receiver at start and end of tube
//  #define SOURCE_X 0
//  #define SOURCE_Y SIM_LY / 2
//  #define SOURCE_Z SIM_LZ / 2

// #define RECEIVER_X SIM_LX
// #define RECEIVER_Y SOURCE_X
// #define RECEIVER_Z SOURCE_Z

#define PADDING 5

#define WATER_K 1500
#define PLASTIC_K 2270

// lame_parameters params_at(real_t x, real_t y, real_t z);
__device__ double K(int_t i, int_t j, int_t k);
void show_model();
static bool init_cuda();

#define WALLTIME(t) ((double) (t).tv_sec + 1e-6 * (double) (t).tv_usec)

int_t max_iteration;
int_t snapshot_freq;
double sensor_height;

#define MODEL_NX 1201
#define MODEL_NY 401

double *model;

double *signature_wave;
int signature_len;
int sampling_freq;

int_t Nx, Ny, Nz; // they must be parsed in the main before used anywhere, I guess that's a very bad
                  // way of doing it, but the template did it like this
double sim_Lx, sim_Ly, sim_Lz;

double dt;
double dx, dy, dz;

// first index is the dimension(xyz direction of vector), second is the time step
//  real_t *buffers[3] = { NULL, NULL, NULL };
real_t *saved_buffer = NULL;

#define MODEL_AT(i, j) model[i + j * MODEL_NX]

// CUDA elements
//  real_t *d_buffer_prv, *d_buffer, *d_buffer_nxt;
real_t *d_buffer_prv_prv, *d_buffer_prv, *d_buffer;

// the indexing is weird, because these values only exist for the boundaries, so looks like a
// corner, of depth PADDING
real_t *d_phi_x_prv, *d_phi_y_prv, *d_phi_z_prv;
real_t *d_psi_x_prv, *d_psi_y_prv, *d_psi_z_prv;

// might be possible to avoid using this, but it would be a mess
real_t *d_phi_x, *d_phi_y, *d_phi_z;
real_t *d_psi_x, *d_psi_y, *d_psi_z;

#define PADDING_BOTTOM_INDEX 0
#define PADDING_BOTTOM_SIZE (d_Nx + PADDING) * (d_Ny + PADDING) * PADDING

#define PADDING_SIDE_INDEX PADDING_BOTTOM_SIZE // the side indexing starts right after the BOTTOM
#define PADDING_SIDE_SIZE (d_Ny + PADDING) * d_Nz *PADDING

#define PADDING_FRONT_INDEX                                                                        \
    PADDING_SIDE_INDEX + PADDING_SIDE_SIZE // FRONT starts indexing right after side
#define PADDING_FRONT_SIZE d_Nx *d_Nz *PADDING

// account for borders, (PADDING: ghost values)

#define per(i, n) ((i + n) % n)

#define padded_index(i, j, k)                                                                      \
    per(i, d_Nx) * ((d_Ny) * (d_Nz)) + (per(j, d_Ny)) * ((d_Nz)) + (per(k, d_Nz))

#define d_P_prv_prv(i, j, k) d_buffer_prv_prv[padded_index(i, j, k)]
#define d_P_prv(i, j, k) d_buffer_prv[padded_index(i, j, k)]
#define d_P(i, j, k) d_buffer[padded_index(i, j, k)]

__constant__ int_t d_Nx, d_Ny, d_Nz;
__constant__ double d_dt, d_dx, d_dy, d_dz;
__constant__ double d_sim_Lx, d_sim_Ly, d_sim_Lz;

#define cudaErrorCheck(ans)                                                                        \
    {                                                                                              \
        gpuAssert((ans), __FILE__, __LINE__);                                                      \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if(code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if(abort)
            exit(code);
    }
}

// Rotate the time step buffers for each dimension
void move_buffer_window() {

    real_t *temp = d_buffer_prv_prv;
    d_buffer_prv_prv = d_buffer_prv;
    d_buffer_prv = d_buffer;
    d_buffer = temp;

    // move auxiliary variables, I guess
    temp = d_phi_x_prv;
    d_phi_x_prv = d_phi_x;
    d_phi_x = temp;

    temp = d_phi_y_prv;
    d_phi_y_prv = d_phi_y;
    d_phi_y = temp;

    temp = d_phi_z_prv;
    d_phi_z_prv = d_phi_z;
    d_phi_z = temp;

    temp = d_psi_x_prv;
    d_psi_x_prv = d_psi_x;
    d_psi_x = temp;

    temp = d_psi_y_prv;
    d_psi_y_prv = d_psi_y;
    d_psi_y = temp;

    temp = d_psi_z_prv;
    d_psi_z_prv = d_psi_z;
    d_psi_z = temp;
}

// Save the present time step in a numbered file under 'data/'
void domain_save(int_t step) {
    char filename[256];
    sprintf(filename, "wave_data/%.5d.dat", step);
    FILE *out = fopen(filename, "wb");
    if(!out) {
        printf("[ERROR] File pointer is NULL!\n");
        exit(EXIT_FAILURE);
    }

    // cudaMemcpy2D() I can use that if I take other axis than YZ maybe ? not sure...
    size_t buffer_size = (Nx + PADDING) * (Ny + PADDING) * (Nz + PADDING);
    cudaErrorCheck(
        cudaMemcpy(saved_buffer, d_buffer, sizeof(real_t) * buffer_size, cudaMemcpyDeviceToHost));

    for(int j = 0; j < Ny; j++) {
        for(int_t i = 0; i < Nx; i++) {
            int_t k = Nz / 2;
            int w = fwrite(
                &saved_buffer[i * ((Ny + PADDING) * (Nz + PADDING)) + j * (Nz + PADDING) + k],
                sizeof(real_t),
                1,
                out); // take horizontal slice from middle, around yz axis
            if(w != 1)
                printf("could write all\n");
        }
    }

    fclose(out);
}

__global__ void init_buffers(real_t *d_buffer_prv, real_t *d_buffer) {
    int x_center = 3 * d_Nx / 4;
    int y_center = d_Ny / 2;
    int n = 10;
    for(int i = x_center - n; i <= x_center + n; i++) {
        for(int j = y_center - n; j <= y_center + n; j++) {
            for(int k = d_Nz / 2 - n; k <= d_Nz / 2 + 1; k++) {
                // dst to center
                real_t delta = ((i - x_center) * (i - x_center) / (double) d_Nx
                                + (j - y_center) * (j - y_center) / (double) d_Ny
                                + (k - d_Nz / 2.) * (k - d_Nz / 2.) / (double) d_Nz);
                // printf("%d\n", delta);
                d_P_prv(i, j, k) = d_P(i, j, k) = exp(-4.0 * delta);
            }
        }
    }
}

// Set up our three buffers, and fill two with an initial perturbation
void domain_initialize() {
    size_t buffer_size = (Nx + PADDING) * (Ny + PADDING) * (Nz + PADDING);
    // alloc cpu memory for saving image
    saved_buffer = (real_t *) calloc(buffer_size, sizeof(real_t));
    if(!saved_buffer) {
        fprintf(stderr, "[ERROR] could not allocate cpu memory\n");
        exit(EXIT_FAILURE);
    }

    cudaErrorCheck(cudaMalloc(&d_buffer_prv_prv, buffer_size));
    cudaErrorCheck(cudaMalloc(&d_buffer_prv, buffer_size));
    cudaErrorCheck(cudaMalloc(&d_buffer, buffer_size));

    size_t border_size_z = (Nx + PADDING) * (Ny + PADDING) * PADDING;
    size_t border_size_x = (Ny + PADDING) * Nz * PADDING;
    size_t border_size_y = Nx * Nz * PADDING;
    cudaErrorCheck(cudaMalloc(&d_phi_x_prv, border_size_x));
    cudaErrorCheck(cudaMalloc(&d_phi_y_prv, border_size_y));
    cudaErrorCheck(cudaMalloc(&d_phi_z_prv, border_size_z));

    cudaErrorCheck(cudaMalloc(&d_psi_x_prv, border_size_x));
    cudaErrorCheck(cudaMalloc(&d_psi_y_prv, border_size_y));
    cudaErrorCheck(cudaMalloc(&d_psi_z_prv, border_size_z));

    cudaErrorCheck(cudaMalloc(&d_phi_x, border_size_x));
    cudaErrorCheck(cudaMalloc(&d_phi_y, border_size_y));
    cudaErrorCheck(cudaMalloc(&d_phi_z, border_size_z));

    cudaErrorCheck(cudaMalloc(&d_psi_x, border_size_x));
    cudaErrorCheck(cudaMalloc(&d_psi_y, border_size_y));
    cudaErrorCheck(cudaMalloc(&d_psi_z, border_size_z));

    // set it all to 0 (memset only works for int!)
    cudaErrorCheck(cudaMemset(d_buffer_prv_prv, 0, buffer_size));
    cudaErrorCheck(cudaMemset(d_buffer_prv, 0, buffer_size));
    cudaErrorCheck(cudaMemset(d_buffer, 0, buffer_size));

    cudaErrorCheck(cudaMemset(d_phi_x, 0, border_size_x));
    cudaErrorCheck(cudaMemset(d_phi_y, 0, border_size_y));
    cudaErrorCheck(cudaMemset(d_phi_z, 0, border_size_z));

    cudaErrorCheck(cudaMemset(d_psi_x, 0, border_size_x));
    cudaErrorCheck(cudaMemset(d_psi_y, 0, border_size_y));
    cudaErrorCheck(cudaMemset(d_psi_z, 0, border_size_z));

    cudaErrorCheck(cudaMemcpyToSymbol(d_Nx, &Nx, sizeof(int_t)));
    cudaErrorCheck(cudaMemcpyToSymbol(d_Ny, &Ny, sizeof(int_t)));
    cudaErrorCheck(cudaMemcpyToSymbol(d_Nz, &Nz, sizeof(int_t)));

    cudaErrorCheck(cudaMemcpyToSymbol(d_dx, &dx, sizeof(double)));
    cudaErrorCheck(cudaMemcpyToSymbol(d_dy, &dy, sizeof(double)));
    cudaErrorCheck(cudaMemcpyToSymbol(d_dz, &dz, sizeof(double)));
    cudaErrorCheck(cudaMemcpyToSymbol(d_dt, &dt, sizeof(double)));

    cudaErrorCheck(cudaMemcpyToSymbol(d_sim_Lx, &sim_Lx, sizeof(double)));
    cudaErrorCheck(cudaMemcpyToSymbol(d_sim_Ly, &sim_Ly, sizeof(double)));
    cudaErrorCheck(cudaMemcpyToSymbol(d_sim_Lz, &sim_Lz, sizeof(double)));
}

// Get rid of all the memory allocations
void domain_finalize(void) {
    cudaFree(d_buffer_prv);
    cudaFree(d_buffer_prv);
    cudaFree(d_buffer);

    cudaFree(d_phi_x_prv);
    cudaFree(d_phi_y_prv);
    cudaFree(d_phi_z_prv);

    cudaFree(d_psi_x_prv);
    cudaFree(d_psi_y_prv);
    cudaFree(d_psi_z_prv);

    free(saved_buffer);
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
        return buffer[PADDING_BOTTOM_INDEX + idx];
    }
    if(i >= d_Nx) // on the SIDE boundary
    {
        i -= d_Nx; // bring side layer to left
        return buf_at_x(buffer, i, j, k);
        // dimensions of side layer: (PADDING, Ny + PADDING, Nz)
        size_t idx = i * (d_Ny + PADDING) * d_Nz + j * d_Nz + k;
        return buffer[PADDING_SIDE_INDEX + idx];
    }
    if(j >= d_Ny) // on the FRONT boundary!
    {
        j -= d_Ny; // bring front layer back
        return buf_at_z(buffer, i, j, k);
        // dimensions of front layer: (Nx, PADDING, Nz)
        size_t idx = i * PADDING * d_Nz + j * d_Nz + k;
        return buffer[PADDING_FRONT_INDEX + idx];
    }

    // this happens when I'm not in a boundary anymore, what to do then ? I guess that can happen,
    // but will never be processed further, so just ignore this
    return 0;
}

__global__ void emit_source(real_t *d_buffer, double t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int sine_x = d_Nx / 4;
    double freq = 1e6; // 1MHz

    if(i == sine_x && j == d_Ny / 2 && k == d_Nz / 2 && t * freq < 1) {
        d_P(sine_x, d_Ny / 2, d_Nz / 2) = sin(2 * M_PI * t * freq);
    }
}

__global__ void time_step(real_t *d_buffer_prv,
                          real_t *d_buffer,
                          real_t *d_buffer_nxt,
                          real_t *d_phi_x,
                          real_t *d_phi_y,
                          real_t *d_phi_z,
                          real_t *d_psi_x,
                          real_t *d_psi_y,
                          real_t *d_psi_z,
                          double t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i >= d_Nx || j >= d_Ny || k >= d_Nz)
        return; // out of bounds. maybe try better way to deal with this, that induce less waste

    return;
}

#define pml_indexing(buffer, i, j, k) (boundary_at(buffer, i, j, k))
#define d_Psi_x(i, j, k) pml_indexing(d_psi_x, i, j, k)
#define d_Psi_y(i, j, k) pml_indexing(d_psi_y, i, j, k)
#define d_Psi_z(i, j, k) pml_indexing(d_psi_z, i, j, k)
#define d_Phi_x(i, j, k) pml_indexing(d_phi_x, i, j, k)
#define d_Phi_y(i, j, k) pml_indexing(d_phi_y, i, j, k)
#define d_Phi_z(i, j, k) pml_indexing(d_phi_z, i, j, k)

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

// Main time integration.
void simulation_loop(void) {
    // Go through each time step
    // I think we should not think in terms of iteration but in term of time

    for(int_t iteration = 0; iteration < max_iteration; iteration++) {

        if((iteration % snapshot_freq) == 0) {
            printf("iteration %d/%d\n", iteration, max_iteration);
            cudaDeviceSynchronize();
            domain_save(iteration / snapshot_freq);
        }

        cudaDeviceSynchronize();

        int block_x = 8;
        int block_y = 8;
        int block_z = 8;
        dim3 blockSize(block_x, block_y, block_z);
        dim3 gridSize((Nx + PADDING + block_x - 1) / block_x,
                      (Ny + PADDING + block_y - 1) / block_y,
                      (Nz + PADDING + block_z - 1) / block_z);
        dim3 pml_z_gridSize((Nx + PADDING + block_x - 1) / block_x,
                            (Ny + PADDING + block_y - 1) / block_y,
                            (PADDING + block_z - 1) / block_z);
        dim3 pml_x_gridSize((PADDING + block_x - 1) / block_x,
                            (Ny + block_y - 1) / block_y,
                            (Nz + PADDING + block_z - 1) / block_z);
        dim3 pml_y_gridSize((Nx + block_x - 1) / block_x,
                            (PADDING + block_y - 1) / block_y,
                            (Nz + block_z - 1) / block_z);

        emit_source<<<gridSize, blockSize>>>(d_buffer_prv, iteration * dt);

        aux_variable_step_z<<<pml_z_gridSize, blockSize>>>(d_buffer,
                                                           d_phi_z_prv,
                                                           d_psi_z_prv,
                                                           d_phi_z,
                                                           d_psi_z);

        aux_variable_step_x<<<pml_x_gridSize, blockSize>>>(d_buffer,
                                                           d_phi_x_prv,
                                                           d_psi_x_prv,
                                                           d_phi_x,
                                                           d_psi_x);

        aux_variable_step_y<<<pml_y_gridSize, blockSize>>>(d_buffer,
                                                           d_phi_y_prv,
                                                           d_psi_y_prv,
                                                           d_phi_y,
                                                           d_psi_y);

        for(size_t iter = 0; iter < 10; iter++) {
            gauss_seidel_red<<<gridSize, blockSize>>>(d_buffer,
                                                      d_buffer_prv,
                                                      d_buffer_prv_prv,
                                                      d_phi_x,
                                                      d_phi_y,
                                                      d_phi_z,
                                                      d_psi_x,
                                                      d_psi_y,
                                                      d_psi_z);
            gauss_seidel_black<<<gridSize, blockSize>>>(d_buffer,
                                                        d_buffer_prv,
                                                        d_buffer_prv_prv,
                                                        d_phi_x,
                                                        d_phi_y,
                                                        d_phi_z,
                                                        d_psi_x,
                                                        d_psi_y,
                                                        d_psi_z);
        }
        move_buffer_window();
    }
}

extern "C" int simulate_wave(simulation_parameters p) {
    dt = p.dt;
    max_iteration = p.max_iter;
    snapshot_freq = p.snapshot_freq;
    sensor_height = p.sensor_height;
    // SIM_LZ = MODEL_LZ + RESERVOIR_OFFSET + sensor_height;//need to add height of sensors, but
    // thats a parameter

    // model_Nx = r_model_nx;
    // model_Ny = r_model_ny;

    Nx = p.Nx;
    Ny = p.Ny;
    Nz = p.Nz;

    sim_Lx = p.sim_Lx;
    sim_Ly = p.sim_Ly;
    sim_Lz = p.sim_Lz;

    // the simulation size is fixed, and resolution is a parameter. the resolution should make sense
    // I guess
    dx = sim_Lx / Nx;
    dy = sim_Ly / Ny;
    dz = sim_Lz / Nz;
    // dx = 0.0001;//I'll need to make sure these are always small enough.
    // dy = 0.0001;
    // dz = 0.0001;
    printf("dx = %.4f, dy = %.4f, dz = %.4f\n", dx, dy, dz);

    // I need to create dx, dy, dz from the resolution given, knowing the shape of the reservoir
    // (which is fixed) and adjust to that

    // FIRST PARSE AND SETUP SIMULATION PARAMETERS (done in domain_initialize)
    model = p.model_data;

    init_cuda();

    // Set up the initial state of the domain
    domain_initialize();

    struct timeval t_start, t_end;

    // show_model();
    gettimeofday(&t_start, NULL);
    simulation_loop();
    gettimeofday(&t_end, NULL);

    printf("Total elapsed time: %lf seconds\n", WALLTIME(t_end) - WALLTIME(t_start));

    // Clean up and shut down
    domain_finalize();
    printf("dx = %.4f, dy = %.4f, dz = %.4f\n", dx, dy, dz);

    exit(EXIT_SUCCESS);
}

__device__ double K(int_t i, int_t j, int_t k) {
    if(i < d_Nx / 2 && i > d_Nx / 6)
        return PLASTIC_K;
    return WATER_K;

    double x = i * d_dx, y = j * d_dy, z = k * d_dz;
    printf("K is called, thats not right\n");
    // if(j < 60){
    //     return WATER_K;
    // }else if(j > 65){
    //     return PLASTIC_K;
    // }else{
    //     double close_to_plastic = ((double)j - 60.)/5.;
    //     return close_to_plastic * PLASTIC_K + (1-close_to_plastic)*WATER_K;
    // }

    // to test in smaller space
    //  if(j > 300){
    //      return PLASTIC_K;
    //  }
    return WATER_K;

    // printf("x = %.4f, y = %.4f, z = %.4f\n", x, y, z);

    // 1. am I (xy) on the model ?
    //  if(RESERVOIR_OFFSET < x && x < MODEL_LX + RESERVOIR_OFFSET &&
    //      RESERVOIR_OFFSET < y && y < MODEL_LY + RESERVOIR_OFFSET){
    //      //yes!
    //      //printf("on the model, z = %lf\n", z);

    //     //2. am I IN the model ?

    //     //figure out closest indices (approximated for now)
    //     int_t x_idx = (int_t)((x - RESERVOIR_OFFSET) * (double)model_Nx / MODEL_LX);
    //     int_t y_idx = (int_t)((y - RESERVOIR_OFFSET) * (double)model_Ny / MODEL_LY);

    //     //model height at this point (assume RESERVOIR_OFFSET below model)
    //     //model stores negative value of depth, so I invert it
    //     // if(MODEL_AT(x_idx, y_idx) != 0){
    //     //     //printf("model value: %lf\n", MODEL_AT(x_idx, y_idx));
    //     // }
    //     double model_bottom = RESERVOIR_OFFSET - MODEL_AT(x_idx, y_idx);
    //     //printf("min: %lf, max: %lf\n", model_bottom, RESERVOIR_OFFSET + MODEL_LZ);

    //     // if(model_bottom <= z && z < RESERVOIR_OFFSET + MODEL_LZ){
    //     //     // printf("x = %lf, y = %lf, RESERVOIR_OFFSET = %lf, MODEL_LX = %lf, MODEL_LY =
    //     %lf, model_Nx = %d, model_Ny = %d\n", x, y, RESERVOIR_OFFSET, MODEL_LX, MODEL_LY,
    //     model_Nx, model_Ny);
    //     //     //printf("x_idx = %d, y_idx = %d\n", x_idx, y_idx);

    //     //     //I am in the model !
    //     //     //printf("in the model !\n");
    //     //     return PLASTIC_K;
    //     // }

    // }

    return WATER_K;
}

static bool init_cuda() {
    // BEGIN: T2
    int dev_count;
    cudaErrorCheck(cudaGetDeviceCount(&dev_count));

    if(dev_count == 0) {
        fprintf(stderr, "No CUDA-compatible devices found.\n");
        return false;
    }

    cudaErrorCheck(cudaSetDevice(0));

    cudaDeviceProp prop;
    cudaErrorCheck(cudaGetDeviceProperties(&prop, 0));

    // Print the device properties
    printf("Device count: %d\n", dev_count);
    printf("Using device 0: %s\n", prop.name);
    printf("\tCompute capability: %d.%d\n", prop.major, prop.minor);
    printf("\tMultiprocessors: %d\n", prop.multiProcessorCount);
    printf("\tWarp size: %d\n", prop.warpSize);
    printf("\tGlobal memory: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
    printf("\tPer-block shared memory: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("\tPer-block registers: %d\n", prop.regsPerBlock);

    return true;

    // END: T2
}

#ifndef SIMULATION_H
#define SIMULATION_H

#include <stdint.h>

#define PADDING 5

#define WATER_K 1500
#define PLASTIC_K 2270

#define MODEL_NX 1201
#define MODEL_NY 401

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

typedef int32_t int_t;
typedef double real_t;

#define MODEL_AT(i, j) model[i + j * MODEL_NX]

#define per(i, n) ((i + n) % n)

#define padded_index(i, j, k)                                                                      \
    per(i, d_Nx) * ((d_Ny) * (d_Nz)) + (per(j, d_Ny)) * ((d_Nz)) + (per(k, d_Nz))

#define d_P_prv_prv(i, j, k) d_buffer_prv_prv[padded_index(i, j, k)]
#define d_P_prv(i, j, k) d_buffer_prv[padded_index(i, j, k)]
#define d_P(i, j, k) d_buffer[padded_index(i, j, k)]

#define WALLTIME(t) ((double) (t).tv_sec + 1e-6 * (double) (t).tv_usec)


typedef struct {
    double* model_data; 
    //int_t model_nx, model_ny;//this will be hardcoded for the 2d model. need to see for 3d model
    int_t Nx, Ny, Nz;//number of cells in each dimension
    double sim_Lx, sim_Ly, sim_Lz;//real life dimensions of simulation
    double dt; 
    int max_iter, snapshot_freq; 
} simulation_parameters;

extern __constant__ int_t d_Nx, d_Ny, d_Nz;
extern __constant__ double d_dt, d_dx, d_dy, d_dz;
extern __constant__ double d_sim_Lx, d_sim_Ly, d_sim_Lz;

__device__ double K(int_t i, int_t j, int_t k);

__global__ void gauss_seidel_red(real_t *d_buffer,
    real_t *d_buffer_prv,
    real_t *d_buffer_prv_prv,
    real_t *d_phi_x,
    real_t *d_phi_y,
    real_t *d_phi_z,
    real_t *d_psi_x,
    real_t *d_psi_y,
    real_t *d_psi_z);

__global__ void gauss_seidel_black(real_t *d_buffer,
    real_t *d_buffer_prv,
    real_t *d_buffer_prv_prv,
    real_t *d_phi_x,
    real_t *d_phi_y,
    real_t *d_phi_z,
    real_t *d_psi_x,
    real_t *d_psi_y,
    real_t *d_psi_z);


#ifdef __cplusplus
extern "C" {
#endif

int simulate_wave(simulation_parameters p);

#ifdef __cplusplus
}
#endif



#endif
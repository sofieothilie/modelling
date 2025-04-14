#pragma once
#include <stdint.h>

typedef int32_t int_t;
typedef float real_t;

typedef struct {
    int_t Nx;
    int_t Ny;
    int_t Nz;
    int_t padding;
    real_t dh[3];
    real_t dt;
} Dimensions;

typedef struct {
    real_t *model_data;
    Dimensions dimensions;
    real_t sim_Lx, sim_Ly, sim_Lz;
    real_t dt;
    int max_iter, snapshot_freq;
} simulation_parameters;

#ifdef __cplusplus
extern "C" {
#endif

int simulate_wave(simulation_parameters p);

#ifdef __cplusplus
}
#endif

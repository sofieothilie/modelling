#ifndef SIMULATION_H
#define SIMULATION_H

#include <stdint.h>

typedef int32_t int_t;
typedef double real_t;

typedef struct {
    real_t *bottom;
    real_t *side;
    real_t *front;
} Aux_variable;

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



typedef struct {
    double *model_data;
    // int_t model_nx, model_ny;//this will be hardcoded for the 2d model. need to see for 3d model
    int_t Nx, Ny, Nz;              // number of cells in each dimension
    double sim_Lx, sim_Ly, sim_Lz; // real life dimensions of simulation
    double dt;
    int max_iter, snapshot_freq;
} simulation_parameters;

#ifdef __cplusplus
extern "C" {
#endif

int simulate_wave(simulation_parameters p);

#ifdef __cplusplus
}
#endif

#endif

#ifndef MODELING_H
#define MODELING_H

#include <stdint.h>

typedef int32_t int_t;
typedef float real_t;

typedef struct {
    double* model_data; 
    //int_t model_nx, model_ny;//this will be hardcoded for the 2d model. need to see for 3d model
    int_t Nx, Ny, Nz;//number of cells in each dimension
    double sim_Lx, sim_Ly, sim_Lz;//real life dimensions of simulation
    double dt; 
    int max_iter, snapshot_freq; 
    double sensor_height; //not necessary for now
    
} simulation_parameters;


#ifdef __cplusplus
extern "C" {
#endif
    int simulate_wave(simulation_parameters p);
#ifdef __cplusplus
}
#endif

#endif // MODELING_H
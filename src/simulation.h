#pragma once
#include <cuda_runtime.h> // <-- this is important
#include <stdint.h>

typedef int32_t int_t;
typedef float real_t;

typedef struct {
    int_t Nx;
    int_t Ny;
    int_t Nz;
    int_t padding;
    real_t dh;
    real_t dt;
} Dimensions;

typedef struct {
    real_t *model_data;
    Dimensions dimensions;
    real_t sim_Lx, sim_Ly, sim_Lz;
    real_t dt;
    int max_iter, snapshot_freq;
} simulation_parameters;

typedef enum { BOTTOM, TOP, LEFT, RIGHT, FRONT, BACK } Side;
#define N_SIDES (6)

typedef enum { X, Y, Z } Component;
#define N_COMPONENTS (3)

// consists of 6 buffers: the whole shell around our rectangle
typedef struct {
    real_t *side[N_SIDES];
} PML_Variable;

typedef struct {
    real_t *U;
    real_t *V;
    PML_Variable Phi;
    PML_Variable Psi;
} SimulationState;

typedef struct {
    real_t k;
    real_t rho;
} MediumParameters;

typedef struct {
    int_t x;
    int_t y;
    int_t z;
} Coords;

#ifdef __cplusplus
extern "C" {
#endif

int_t get_domain_size(const Dimensions dimensions);
int simulate_wave(simulation_parameters p);
SimulationState allocate_simulation_state(const Dimensions dimensions);
void free_simulation_state(SimulationState s);

#ifdef __cplusplus
}
#endif

#define WATER_PARAMETERS ((MediumParameters) { .k = 1500.0, .rho = 998.0 })
#define PLASTIC_PARAMETERS ((MediumParameters) { .k = 2270.0, .rho = 1185.0 })

#define SRC_FREQUENCY ((real_t) 1.0e6)
#define SRC_SAMPLE_RATE ((real_t) (8.0 * SRC_FREQUENCY))

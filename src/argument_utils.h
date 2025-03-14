#ifndef _ARGUMENT_UTILS_H_
#define _ARGUMENT_UTILS_H_

#include <stdint.h>
#include "simulate_kernel.h"


typedef struct options_struct {
    real_t sim_Lx, sim_Ly, sim_Lz;
    real_t dt;
    int_t Nx, Ny, Nz;
    int_t max_iteration;
    int_t snapshot_frequency;
} OPTIONS;


OPTIONS *parse_args ( int argc, char **argv );

void help ( char const *exec, char const opt, char const *optarg );

#endif

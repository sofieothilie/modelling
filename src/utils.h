#pragma once
#include "argument_utils.h"
#include "simulation.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define WALLTIME(t) ((double) (t).tv_sec + 1e-6 * (double) (t).tv_usec)

#define incSide(s) s = static_cast<Side>(static_cast<int>(s) + 1)
#define incComp(c) c = static_cast<Component>(static_cast<int>(c) + 1)

inline void gpuAssert(const cudaError_t code, const char *file, const int line) {
    if(code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}
#define cudaErrorCheck(ans)                                                                        \
    {                                                                                              \
        gpuAssert((ans), __FILE__, __LINE__);                                                      \
    }

int init_cuda();

void print_progress_bar(int current_iteration,
                        int total_iterations,
                        struct timeval start,
                        struct timeval now);
Coords PositionToCoords(Position p, Dimensions d, Position source);

#ifdef __cplusplus
extern "C" {
#endif

void print_start_info(Dimensions dimensions);
real_t RTT(double *model, OPTIONS *options, simulation_parameters *p);

#ifdef __cplusplus
}
#endif
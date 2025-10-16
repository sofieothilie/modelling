#pragma once

#include "simulation.h"
#include <stdint.h>

typedef struct options_struct {
    real_t sim_Lx, sim_Ly, sim_Lz;
    Position sensor;
    real_t dt;
    real_t ppw;
    int_t padding;
    int_t max_iteration;
    int_t snapshot_frequency;
    int print_info;
    int_t RTM;
} OPTIONS;

OPTIONS *parse_args(int argc, char **argv);

void help(char const *exec, char const opt, char const *optarg);

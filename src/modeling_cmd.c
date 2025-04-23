#include "argument_utils.h"
#include "simulation.h"
#include <stdio.h>
#include <stdlib.h>

#define MODEL_Nx 1201
#define MODEL_Ny 401

int main(int argc, char **argv) {
    OPTIONS *options = parse_args(argc, argv);

    FILE *model_file = fopen("data/model.bin", "rb");
    if(!model_file) {
        perror("Failed to open file");
        return 1;
    }

    printf("allocate %lu bytes\n", MODEL_Nx * MODEL_Ny * sizeof(double));
    double *model = calloc(MODEL_Nx * MODEL_Ny, sizeof(double));

    if(model == NULL) {
        perror("Failed to allocate memory");
        fclose(model_file);
        return 1;
    }
    printf("alloced memory, at %p\n", model);

    // Read the binary data into the array
    size_t read_size = fread(model, sizeof(double), 1, model_file);
    if(read_size != 1) {
        perror("Failed to read file");
        free(model);
        fclose(model_file);
        return 1;
    }

    fclose(model_file); // data is safe in array, we don't need the file anymore

    simulation_parameters p = {
        .dimensions = {
            .Nx = options->Nx,
            .Ny = options->Ny,
            .Nz = options->Nz,
            .padding = 10,
            .dh = {
                options->sim_Lx / options->Nx,
                options->sim_Ly / options->Ny,
                options->sim_Lz / options->Nz
            },
            .dt = options->dt,
        },
        .sim_Lx = options->sim_Lx,
        .sim_Ly = options->sim_Ly,
        .sim_Lz = options->sim_Lz,
        .max_iter = options->max_iteration,
        .snapshot_freq = options->snapshot_frequency,
        .dt = options->dt,
    };
    int err = simulate_wave(p);

    return 0;
}

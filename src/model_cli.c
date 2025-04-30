#include "argument_utils.h"
#include "simulation.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MODEL_Nx 1201
#define MODEL_Ny 401

int main(int argc, char **argv) {
    OPTIONS *options = parse_args(argc, argv);

    // first compute dh.
    real_t smallest_wavelength = WATER_PARAMETERS.k / SRC_FREQUENCY;
    real_t dh = smallest_wavelength / options->ppw;

    real_t best_dt = 0.9 * dh / (PLASTIC_PARAMETERS.k * sqrt(3));
    options->dt = best_dt;

    int_t Nx = (int) ceil(options->sim_Lx / dh);
    int_t Ny = (int) ceil(options->sim_Ly / dh);
    int_t Nz = (int) ceil(options->sim_Lz / dh);

    Dimensions d = {
        .Nx = Nx,
        .Ny = Ny,
        .Nz = Nz,
        .padding = options->padding,
        .dh = dh,
        .dt = options->dt,
    };

    print_start_info(d);

    if(options->print_info) {
        // do not run, only print launch informations
        return 0;
    }

    FILE *model_file = fopen("data/model.bin", "rb");
    if(!model_file) {
        perror("Failed to open file");
        return 1;
    }

    double *model = calloc(MODEL_Nx * MODEL_Ny, sizeof(double));

    if(model == NULL) {
        perror("Failed to allocate memory");
        fclose(model_file);
        return 1;
    }

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
        .dimensions = d,
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

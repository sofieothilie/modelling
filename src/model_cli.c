#include "argument_utils.h"
#include "simulation.h"
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

    printf("-----------------------------------\n");
    printf("------ Simulation Parameters ------\n");
    printf("-----------------------------------\n");
    printf("dh: %lf\n", dh);
    printf("dt: %e\n", best_dt);
    printf("Grid dimensions: Nx = %d, Ny = %d, Nz = %d\n", Nx, Ny, Nz);
    printf("Padding cells: %d\n", options->padding);
    printf("\n-----------------------------------\n");

    printf("---- Memory Allocation Details ----\n");
    printf("-----------------------------------\n");
    int_t padding = options->padding;
    size_t tot_x = Nx + 2 * padding + 2;
    size_t tot_y = Ny + 2 * padding + 2;
    size_t tot_z = Nz + 2 * padding + 2;

    size_t full_buffer_size = tot_x * tot_y * tot_z * sizeof(real_t);
    size_t PML_shell_size = 2 * (Nx + 2 * padding + 2) * (Ny + 2 * padding + 2) * (padding + 1)
                          + 2 * (padding + 1) * (Ny + 2 * padding + 2) * (Nz + 2 * padding + 2)
                          + 2 * (Nx + 2 * padding + 2) * (padding + 1) * (Nz + 2 * padding + 2);

    PML_shell_size *= sizeof(real_t);

    printf("3 sim state x (2 full buffers + 1 PML shell)\n");
    printf("1 buffer size: (%zux%zux%zu) x %zu bytes = %.2lf MB\n",
           tot_x,
           tot_y,
           tot_z,
           sizeof(real_t),
           full_buffer_size / (1000. * 1000.));
    printf("PML shell size: %lf MB\n", PML_shell_size / (1000. * 1000.));
    printf(" => total allocated size = %lf MB\n",
           3. * (2. * full_buffer_size + PML_shell_size) / (1000. * 1000.));
    printf("-----------------------------------\n\n");

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

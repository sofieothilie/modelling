#include "argument_utils.h"
#include "simulate_kernel.h"
#include <stdio.h>
#include <stdlib.h>

#define MODEL_Nx 1201
#define MODEL_Ny 401

int main(int argc, char** argv){
    OPTIONS *options = parse_args(argc, argv);

    FILE* model_file = fopen("data/model.bin", "rb");
    if(!model_file){
        perror("Failed to open file");
        return 1;
    }

    printf("allocate %d bytes\n", MODEL_Nx*MODEL_Ny*sizeof(double));
    double* model = calloc(MODEL_Nx*MODEL_Ny, sizeof(double));

    if (model == NULL) {
        perror("Failed to allocate memory");
        fclose(model_file);
        return 1;
    }
    printf("alloced memory, at %p\n", model);

    // Read the binary data into the array
    size_t read_size = fread(model, sizeof(double), 1, model_file);
    if (read_size != 1) {
        perror("Failed to read file");
        free(model);
        fclose(model_file);
        return 1;
    }

    fclose(model_file);//data is safe in array, we don't need the file anymore

    int err = simulate_wave(model, options->Nx, options->Ny, options->Nz, options->dt, options->max_iteration, options->snapshot_frequency, 0, MODEL_Nx, MODEL_Ny);

    return 0;
}
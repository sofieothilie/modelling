#include "argument_utils.h"
#include <stdio.h>

#define MODEL_Nx 1201
#define MODEL_Ny 401

int main(int argc, char** argv){
    OPTIONS *options = parse_args(argc, argv);

    FILE* model_file = fopen("model.bin", "rb");
    if(!model_file){
        perror("Failed to open file");
    }

    double* model = calloc(MODEL_Nx*MODEL_Ny, sizeof(double));
    if (model == NULL) {
        perror("Failed to allocate memory");
        fclose(model_file);
        return 1;
    }

    // Read the binary data into the array
    size_t read_size = fread(model, sizeof(double), MODEL_Nx*MODEL_Ny, model_file);
    if (read_size != MODEL_Nx*MODEL_Ny) {
        perror("Failed to read file");
        free(model);
        fclose(model_file);
        return 1;
    }

    fclose(model_file);//data is safe in array, we don't need the file anymore

    int err = simulate(model, options->Nx, options->Ny, options->Nz, options->dt, options->max_iteration, options->snapshot_frequency, 0, MODEL_Nx, MODEL_Ny);

    return 0;
}
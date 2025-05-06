#include "simulation.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// this function cannot be used in general because it counts the ghost cells in the memory alloc
__host__ __device__ int_t get_alloc_side_size(const Dimensions dimensions, const Side side) {
    int_t Nx = dimensions.Nx;
    int_t Ny = dimensions.Ny;
    int_t Nz = dimensions.Nz;
    int_t padding = dimensions.padding;

    switch(side) {
        case BOTTOM:
        case TOP:
            return (Nx + 2 * padding + 2) * (Ny + 2 * padding + 2) * (padding + 1);
        case LEFT:
        case RIGHT:
            return (padding + 1) * (Ny + 2 * padding + 2) * (Nz + 2 * padding + 2);
        case FRONT:
        case BACK:
            return (Nx + 2 * padding + 2) * (padding + 1) * (Nz + 2 * padding + 2);
        default:
            printf("invalid side\n");
            return -1;
    }
}

int_t get_domain_size(const Dimensions dimensions) {
    int_t Nx = dimensions.Nx;
    int_t Ny = dimensions.Ny;
    int_t Nz = dimensions.Nz;
    int_t padding = dimensions.padding;

    return (Nx + 2 * padding + 2) * (Ny + 2 * padding + 2) * (Nz + 2 * padding + 2);
}

PML_Variable allocate_pml_variables(const Dimensions dimensions) {
    PML_Variable variable = { 0 };

    for(Side side = BOTTOM; side < N_SIDES; incSide(side)) {
            size_t size = get_alloc_side_size(dimensions, side);

            cudaErrorCheck(cudaMalloc(&(variable.side[side]), size * sizeof(real_t)));
            cudaErrorCheck(cudaMemset(variable.side[side], 0, size * sizeof(real_t)));
        }

    return variable;
}

void free_pml_variables(PML_Variable var) {
    for(Side side = BOTTOM; side < N_SIDES; incSide(side)) {
        cudaErrorCheck(cudaFree(var.side[side]));
    }
}

real_t *allocate_domain(const Dimensions dimensions) {
    real_t *result = NULL;

    int_t size = get_domain_size(dimensions);

    cudaErrorCheck(cudaMalloc(&result, size * sizeof(real_t)));
    cudaErrorCheck(cudaMemset(result, 0, size * sizeof(real_t)));

    return result;
}

SimulationState allocate_simulation_state(const Dimensions dimensions) {
    return SimulationState { .U = allocate_domain(dimensions),
                             .V = allocate_domain(dimensions),
                             .Phi = allocate_pml_variables(dimensions),
                             .Psi = allocate_pml_variables(dimensions) };
}

void free_domain(real_t *buf) {
    cudaErrorCheck(cudaFree(buf));
}

void free_simulation_state(SimulationState s) {
    free_domain(s.U);
    free_domain(s.V);
    free_pml_variables(s.Psi);
    free_pml_variables(s.Phi);
}

double* open_model(const char* filename){
    FILE *model_file = fopen(filename, "rb");
    if(!model_file) {
        perror("Failed to open file");
        return NULL;
    }

    double *model = (double*) calloc(MODEL_NX * MODEL_NY, sizeof(double));

    if(model == NULL) {
        perror("Failed to allocate model memory");
        fclose(model_file);
        return NULL;
    }

    // Read the binary data into the array
    size_t read_size = fread(model, sizeof(double), MODEL_NX * MODEL_NY, model_file);
    if(read_size != MODEL_NX * MODEL_NY) {
        perror("Failed to read file");
        free(model);
        fclose(model_file);
        return NULL;
    }

    fclose(model_file); // data is safe in array, we don't need the file anymore

    return model;
}

void free_model(double* model){
    free(model);
} 
#include "simulation.h"
#include "cuda_utils.h"
#include "PML_buffer.h"

#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>



__constant__ int_t d_Nx, d_Ny, d_Nz;
__constant__ double d_dt, d_dx, d_dy, d_dz;
__constant__ double d_sim_Lx, d_sim_Ly, d_sim_Lz;

// first index is the dimension(xyz direction of vector), second is the time step
//  real_t *buffers[3] = { NULL, NULL, NULL };
real_t *saved_buffer = NULL;

double *model;

// CUDA elements
//  real_t *d_buffer_prv, *d_buffer, *d_buffer_nxt;
real_t *d_buffer_prv_prv, *d_buffer_prv, *d_buffer;


int_t Nx, Ny, Nz; // they must be parsed in the main before used anywhere, I guess that's a very bad
                  // way of doing it, but the template did it like this
double sim_Lx, sim_Ly, sim_Lz;

double dt;
double dx, dy, dz;

int_t max_iteration;
int_t snapshot_freq;



extern "C" int simulate_wave(simulation_parameters p) {
    dt = p.dt;
    max_iteration = p.max_iter;
    snapshot_freq = p.snapshot_freq;
    // SIM_LZ = MODEL_LZ + RESERVOIR_OFFSET + sensor_height;//need to add height of sensors, but
    // thats a parameter

    // model_Nx = r_model_nx;
    // model_Ny = r_model_ny;

    Nx = p.Nx;
    Ny = p.Ny;
    Nz = p.Nz;

    sim_Lx = p.sim_Lx;
    sim_Ly = p.sim_Ly;
    sim_Lz = p.sim_Lz;

    // the simulation size is fixed, and resolution is a parameter. the resolution should make sense
    // I guess
    dx = sim_Lx / Nx;
    dy = sim_Ly / Ny;
    dz = sim_Lz / Nz;
    // dx = 0.0001;//I'll need to make sure these are always small enough.
    // dy = 0.0001;
    // dz = 0.0001;
    printf("dx = %.4f, dy = %.4f, dz = %.4f\n", dx, dy, dz);

    // I need to create dx, dy, dz from the resolution given, knowing the shape of the reservoir
    // (which is fixed) and adjust to that

    // FIRST PARSE AND SETUP SIMULATION PARAMETERS (done in domain_initialize)
    model = p.model_data;

    init_cuda();

    // Set up the initial state of the domain
    domain_initialize();

    struct timeval t_start, t_end;

    // show_model();
    gettimeofday(&t_start, NULL);
    simulation_loop();
    gettimeofday(&t_end, NULL);

    printf("Total elapsed time: %lf seconds\n", WALLTIME(t_end) - WALLTIME(t_start));

    // Clean up and shut down
    domain_finalize();
    printf("dx = %.4f, dy = %.4f, dz = %.4f\n", dx, dy, dz);

    exit(EXIT_SUCCESS);
}

void simulation_loop(void) {
    // Go through each time step
    // I think we should not think in terms of iteration but in term of time

    for(int_t iteration = 0; iteration < max_iteration; iteration++) {

        if((iteration % snapshot_freq) == 0) {
            printf("iteration %d/%d\n", iteration, max_iteration);
            cudaDeviceSynchronize();
            domain_save(iteration / snapshot_freq);
        }

        cudaDeviceSynchronize();

        int block_x = 8;
        int block_y = 8;
        int block_z = 8;
        dim3 blockSize(block_x, block_y, block_z);
        dim3 gridSize((Nx + PADDING + block_x - 1) / block_x,
                      (Ny + PADDING + block_y - 1) / block_y,
                      (Nz + PADDING + block_z - 1) / block_z);
        dim3 pml_z_gridSize((Nx + PADDING + block_x - 1) / block_x,
                            (Ny + PADDING + block_y - 1) / block_y,
                            (PADDING + block_z - 1) / block_z);
        dim3 pml_x_gridSize((PADDING + block_x - 1) / block_x,
                            (Ny + block_y - 1) / block_y,
                            (Nz + PADDING + block_z - 1) / block_z);
        dim3 pml_y_gridSize((Nx + block_x - 1) / block_x,
                            (PADDING + block_y - 1) / block_y,
                            (Nz + block_z - 1) / block_z);

        emit_source<<<gridSize, blockSize>>>(d_buffer_prv, iteration * dt);

        aux_variable_step_z<<<pml_z_gridSize, blockSize>>>(d_buffer,
                                                           d_phi_z_prv,
                                                           d_psi_z_prv,
                                                           d_phi_z,
                                                           d_psi_z);

        aux_variable_step_x<<<pml_x_gridSize, blockSize>>>(d_buffer,
                                                           d_phi_x_prv,
                                                           d_psi_x_prv,
                                                           d_phi_x,
                                                           d_psi_x);

        aux_variable_step_y<<<pml_y_gridSize, blockSize>>>(d_buffer,
                                                           d_phi_y_prv,
                                                           d_psi_y_prv,
                                                           d_phi_y,
                                                           d_psi_y);

        for(size_t iter = 0; iter < 10; iter++) {
            gauss_seidel_red<<<gridSize, blockSize>>>(d_buffer,
                                                      d_buffer_prv,
                                                      d_buffer_prv_prv,
                                                      d_phi_x,
                                                      d_phi_y,
                                                      d_phi_z,
                                                      d_psi_x,
                                                      d_psi_y,
                                                      d_psi_z);
            gauss_seidel_black<<<gridSize, blockSize>>>(d_buffer,
                                                        d_buffer_prv,
                                                        d_buffer_prv_prv,
                                                        d_phi_x,
                                                        d_phi_y,
                                                        d_phi_z,
                                                        d_psi_x,
                                                        d_psi_y,
                                                        d_psi_z);
        }
        move_buffer_window();
    }
}

__device__ double K(int_t i, int_t j, int_t k) {
    if(i < d_Nx / 2 && i > d_Nx / 6)
        return PLASTIC_K;
    return WATER_K;

    double x = i * d_dx, y = j * d_dy, z = k * d_dz;
    printf("K is called, thats not right\n");
    // if(j < 60){
    //     return WATER_K;
    // }else if(j > 65){
    //     return PLASTIC_K;
    // }else{
    //     double close_to_plastic = ((double)j - 60.)/5.;
    //     return close_to_plastic * PLASTIC_K + (1-close_to_plastic)*WATER_K;
    // }

    // to test in smaller space
    //  if(j > 300){
    //      return PLASTIC_K;
    //  }
    return WATER_K;

    // printf("x = %.4f, y = %.4f, z = %.4f\n", x, y, z);

    // 1. am I (xy) on the model ?
    //  if(RESERVOIR_OFFSET < x && x < MODEL_LX + RESERVOIR_OFFSET &&
    //      RESERVOIR_OFFSET < y && y < MODEL_LY + RESERVOIR_OFFSET){
    //      //yes!
    //      //printf("on the model, z = %lf\n", z);

    //     //2. am I IN the model ?

    //     //figure out closest indices (approximated for now)
    //     int_t x_idx = (int_t)((x - RESERVOIR_OFFSET) * (double)model_Nx / MODEL_LX);
    //     int_t y_idx = (int_t)((y - RESERVOIR_OFFSET) * (double)model_Ny / MODEL_LY);

    //     //model height at this point (assume RESERVOIR_OFFSET below model)
    //     //model stores negative value of depth, so I invert it
    //     // if(MODEL_AT(x_idx, y_idx) != 0){
    //     //     //printf("model value: %lf\n", MODEL_AT(x_idx, y_idx));
    //     // }
    //     double model_bottom = RESERVOIR_OFFSET - MODEL_AT(x_idx, y_idx);
    //     //printf("min: %lf, max: %lf\n", model_bottom, RESERVOIR_OFFSET + MODEL_LZ);

    //     // if(model_bottom <= z && z < RESERVOIR_OFFSET + MODEL_LZ){
    //     //     // printf("x = %lf, y = %lf, RESERVOIR_OFFSET = %lf, MODEL_LX = %lf, MODEL_LY =
    //     %lf, model_Nx = %d, model_Ny = %d\n", x, y, RESERVOIR_OFFSET, MODEL_LX, MODEL_LY,
    //     model_Nx, model_Ny);
    //     //     //printf("x_idx = %d, y_idx = %d\n", x_idx, y_idx);

    //     //     //I am in the model !
    //     //     //printf("in the model !\n");
    //     //     return PLASTIC_K;
    //     // }

    // }

    return WATER_K;
}

__global__ void emit_source(real_t *d_buffer, double t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int sine_x = d_Nx / 4;
    double freq = 1e6; // 1MHz

    if(i == sine_x && j == d_Ny / 2 && k == d_Nz / 2 && t * freq < 1) {
        d_P(sine_x, d_Ny / 2, d_Nz / 2) = sin(2 * M_PI * t * freq);
    }
}

// Set up our three buffers, and fill two with an initial perturbation
void domain_initialize() {
    size_t buffer_size = (Nx + PADDING) * (Ny + PADDING) * (Nz + PADDING);
    // alloc cpu memory for saving image
    saved_buffer = (real_t *) calloc(buffer_size, sizeof(real_t));
    if(!saved_buffer) {
        fprintf(stderr, "[ERROR] could not allocate cpu memory\n");
        exit(EXIT_FAILURE);
    }

    cudaErrorCheck(cudaMalloc(&d_buffer_prv_prv, buffer_size));
    cudaErrorCheck(cudaMalloc(&d_buffer_prv, buffer_size));
    cudaErrorCheck(cudaMalloc(&d_buffer, buffer_size));

    size_t border_size_z = (Nx + PADDING) * (Ny + PADDING) * PADDING;
    size_t border_size_x = (Ny + PADDING) * Nz * PADDING;
    size_t border_size_y = Nx * Nz * PADDING;
    cudaErrorCheck(cudaMalloc(&d_phi_x_prv, border_size_x));
    cudaErrorCheck(cudaMalloc(&d_phi_y_prv, border_size_y));
    cudaErrorCheck(cudaMalloc(&d_phi_z_prv, border_size_z));

    cudaErrorCheck(cudaMalloc(&d_psi_x_prv, border_size_x));
    cudaErrorCheck(cudaMalloc(&d_psi_y_prv, border_size_y));
    cudaErrorCheck(cudaMalloc(&d_psi_z_prv, border_size_z));

    cudaErrorCheck(cudaMalloc(&d_phi_x, border_size_x));
    cudaErrorCheck(cudaMalloc(&d_phi_y, border_size_y));
    cudaErrorCheck(cudaMalloc(&d_phi_z, border_size_z));

    cudaErrorCheck(cudaMalloc(&d_psi_x, border_size_x));
    cudaErrorCheck(cudaMalloc(&d_psi_y, border_size_y));
    cudaErrorCheck(cudaMalloc(&d_psi_z, border_size_z));

    // set it all to 0 (memset only works for int!)
    cudaErrorCheck(cudaMemset(d_buffer_prv_prv, 0, buffer_size));
    cudaErrorCheck(cudaMemset(d_buffer_prv, 0, buffer_size));
    cudaErrorCheck(cudaMemset(d_buffer, 0, buffer_size));

    cudaErrorCheck(cudaMemset(d_phi_x, 0, border_size_x));
    cudaErrorCheck(cudaMemset(d_phi_y, 0, border_size_y));
    cudaErrorCheck(cudaMemset(d_phi_z, 0, border_size_z));

    cudaErrorCheck(cudaMemset(d_psi_x, 0, border_size_x));
    cudaErrorCheck(cudaMemset(d_psi_y, 0, border_size_y));
    cudaErrorCheck(cudaMemset(d_psi_z, 0, border_size_z));

    cudaErrorCheck(cudaMemcpyToSymbol(d_Nx, &Nx, sizeof(int_t)));
    cudaErrorCheck(cudaMemcpyToSymbol(d_Ny, &Ny, sizeof(int_t)));
    cudaErrorCheck(cudaMemcpyToSymbol(d_Nz, &Nz, sizeof(int_t)));

    cudaErrorCheck(cudaMemcpyToSymbol(d_dx, &dx, sizeof(double)));
    cudaErrorCheck(cudaMemcpyToSymbol(d_dy, &dy, sizeof(double)));
    cudaErrorCheck(cudaMemcpyToSymbol(d_dz, &dz, sizeof(double)));
    cudaErrorCheck(cudaMemcpyToSymbol(d_dt, &dt, sizeof(double)));

    cudaErrorCheck(cudaMemcpyToSymbol(d_sim_Lx, &sim_Lx, sizeof(double)));
    cudaErrorCheck(cudaMemcpyToSymbol(d_sim_Ly, &sim_Ly, sizeof(double)));
    cudaErrorCheck(cudaMemcpyToSymbol(d_sim_Lz, &sim_Lz, sizeof(double)));
}

// Get rid of all the memory allocations
void domain_finalize(void) {
    cudaFree(d_buffer_prv);
    cudaFree(d_buffer_prv);
    cudaFree(d_buffer);

    cudaFree(d_phi_x_prv);
    cudaFree(d_phi_y_prv);
    cudaFree(d_phi_z_prv);

    cudaFree(d_psi_x_prv);
    cudaFree(d_psi_y_prv);
    cudaFree(d_psi_z_prv);

    free(saved_buffer);
}

// Rotate the time step buffers for each dimension
void move_buffer_window() {

    real_t *temp = d_buffer_prv_prv;
    d_buffer_prv_prv = d_buffer_prv;
    d_buffer_prv = d_buffer;
    d_buffer = temp;

    // move auxiliary variables, I guess
    temp = d_phi_x_prv;
    d_phi_x_prv = d_phi_x;
    d_phi_x = temp;

    temp = d_phi_y_prv;
    d_phi_y_prv = d_phi_y;
    d_phi_y = temp;

    temp = d_phi_z_prv;
    d_phi_z_prv = d_phi_z;
    d_phi_z = temp;

    temp = d_psi_x_prv;
    d_psi_x_prv = d_psi_x;
    d_psi_x = temp;

    temp = d_psi_y_prv;
    d_psi_y_prv = d_psi_y;
    d_psi_y = temp;

    temp = d_psi_z_prv;
    d_psi_z_prv = d_psi_z;
    d_psi_z = temp;
}

// Save the present time step in a numbered file under 'data/'
void domain_save(int_t step) {
    char filename[256];
    sprintf(filename, "wave_data/%.5d.dat", step);
    FILE *out = fopen(filename, "wb");
    if(!out) {
        printf("[ERROR] File pointer is NULL!\n");
        exit(EXIT_FAILURE);
    }

    // cudaMemcpy2D() I can use that if I take other axis than YZ maybe ? not sure...
    size_t buffer_size = (Nx + PADDING) * (Ny + PADDING) * (Nz + PADDING);
    cudaErrorCheck(
        cudaMemcpy(saved_buffer, d_buffer, sizeof(real_t) * buffer_size, cudaMemcpyDeviceToHost));

    for(int j = 0; j < Ny; j++) {
        for(int_t i = 0; i < Nx; i++) {
            int_t k = Nz / 2;
            int w = fwrite(
                &saved_buffer[i * ((Ny + PADDING) * (Nz + PADDING)) + j * (Nz + PADDING) + k],
                sizeof(real_t),
                1,
                out); // take horizontal slice from middle, around yz axis
            if(w != 1)
                printf("could write all\n");
        }
    }

    fclose(out);
}

//put a gaussian spike in the middle of the simulation, might be outdated
__global__ void init_buffers(real_t *d_buffer_prv, real_t *d_buffer) {
    int x_center = 3 * d_Nx / 4;
    int y_center = d_Ny / 2;
    int n = 10;
    for(int i = x_center - n; i <= x_center + n; i++) {
        for(int j = y_center - n; j <= y_center + n; j++) {
            for(int k = d_Nz / 2 - n; k <= d_Nz / 2 + 1; k++) {
                // dst to center
                real_t delta = ((i - x_center) * (i - x_center) / (double) d_Nx
                                + (j - y_center) * (j - y_center) / (double) d_Ny
                                + (k - d_Nz / 2.) * (k - d_Nz / 2.) / (double) d_Nz);
                // printf("%d\n", delta);
                d_P_prv(i, j, k) = d_P(i, j, k) = exp(-4.0 * delta);
            }
        }
    }
}


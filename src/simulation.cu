#include "simulation.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define cudaErrorCheck(ans)                                                                        \
    {                                                                                              \
        gpuAssert((ans), __FILE__, __LINE__);                                                      \
    }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true);

#define PADDING 5

#define WATER_K 1500
#define PLASTIC_K 2270

#define MODEL_NX 1201
#define MODEL_NY 401

typedef int32_t int_t;
typedef double real_t;

#define WALLTIME(t) ((double) (t).tv_sec + 1e-6 * (double) (t).tv_usec)

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

#define per(i, n) (((i) + (n)) % (n))

#define padded_index(i, j, k)                                                                      \
    per((i), (d_Nx + PADDING)) * ((d_Ny + PADDING) * (d_Nz + PADDING))                             \
        + (per((j), d_Ny + PADDING)) * ((d_Nz + PADDING)) + (per((k), d_Nz + PADDING))

#define pml_indexing(buffer, i, j, k) (get_buf(buffer, i, j, k))

#define d_Psi_x(i, j, k) pml_indexing(d_psi_x, i, j, k)
#define d_Psi_y(i, j, k) pml_indexing(d_psi_y, i, j, k)
#define d_Psi_z(i, j, k) pml_indexing(d_psi_z, i, j, k)
#define d_Phi_x(i, j, k) pml_indexing(d_phi_x, i, j, k)
#define d_Phi_y(i, j, k) pml_indexing(d_phi_y, i, j, k)
#define d_Phi_z(i, j, k) pml_indexing(d_phi_z, i, j, k)

#define d_P_prv_prv(i, j, k) d_buffer_prv_prv[padded_index(i, j, k)]
#define d_P_prv(i, j, k) d_buffer_prv[padded_index(i, j, k)]
#define d_P(i, j, k) d_buffer[padded_index(i, j, k)]

#define SIGMA (2 / d_dt)

bool init_cuda();
void move_buffer_window();
void domain_initialize();
void domain_finalize();
void domain_save(int_t);
void simulation_loop(void);

__device__ void set_buf(Aux_variable buffer, real_t value, int_t i, int_t j, int_t k);
__device__ real_t get_buf(Aux_variable buffer, int_t i, int_t j, int_t k);
__device__ real_t sigma_x(int_t i, int_t j, int_t k);
__device__ real_t sigma_y(int_t i, int_t j, int_t k);
__device__ real_t sigma_z(int_t i, int_t j, int_t k);
__global__ void aux_variable_step_front(const real_t *d_buffer,
                                        real_t *d_phi_y_prv,
                                        real_t *d_psi_y_prv,
                                        real_t *d_phi_y,
                                        real_t *d_psi_y,
                                        real_t *d_phi_x_prv,
                                        real_t *d_psi_x_prv,
                                        real_t *d_phi_x,
                                        real_t *d_psi_x,
                                        real_t *d_phi_z_prv,
                                        real_t *d_psi_z_prv,
                                        real_t *d_phi_z,
                                        real_t *d_psi_z);
__global__ void aux_variable_step_side(const real_t *d_buffer,
                                       real_t *d_phi_y_prv,
                                       real_t *d_psi_y_prv,
                                       real_t *d_phi_y,
                                       real_t *d_psi_y,
                                       real_t *d_phi_x_prv,
                                       real_t *d_psi_x_prv,
                                       real_t *d_phi_x,
                                       real_t *d_psi_x,
                                       real_t *d_phi_z_prv,
                                       real_t *d_psi_z_prv,
                                       real_t *d_phi_z,
                                       real_t *d_psi_z);
__global__ void aux_variable_step_bottom(const real_t *d_buffer,
                                         real_t *d_phi_y_prv,
                                         real_t *d_psi_y_prv,
                                         real_t *d_phi_y,
                                         real_t *d_psi_y,
                                         real_t *d_phi_x_prv,
                                         real_t *d_psi_x_prv,
                                         real_t *d_phi_x,
                                         real_t *d_psi_x,
                                         real_t *d_phi_z_prv,
                                         real_t *d_psi_z_prv,
                                         real_t *d_phi_z,
                                         real_t *d_psi_z);

__device__ double K(int_t i, int_t j, int_t k) {
    return WATER_K;
    if(i < d_Nx / 2 && i > d_Nx / 6)
        return PLASTIC_K;

    double x = i * d_dx, y = j * d_dy, z = k * d_dz;

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

Aux_variable d_phi_x_prv, d_phi_y_prv, d_phi_z_prv;
Aux_variable d_psi_x_prv, d_psi_y_prv, d_psi_z_prv;

Aux_variable d_phi_x, d_phi_y, d_phi_z;
Aux_variable d_psi_x, d_psi_y, d_psi_z;

__device__ real_t PML(int i,
                      int j,
                      int k,
                      real_t *d_buffer,
                      Aux_variable d_phi_x,
                      Aux_variable d_phi_y,
                      Aux_variable d_phi_z,
                      Aux_variable d_psi_x,
                      Aux_variable d_psi_y,
                      Aux_variable d_psi_z) {
    if (i < d_Nx - 1 || j < d_Ny - 1 || k < d_Nz - 1)
        return 0.0f;

    real_t result =
        -(d_Phi_x(i - 1, j, k) * sigma_x(i - 1, j, k) - d_Psi_x(i + 1, j, k) * sigma_x(i, j, k))
            / (d_dx * d_dx)
        - (d_Phi_y(i, j - 1, k) * sigma_y(i, j - 1, k) - d_Psi_y(i, j + 1, k) * sigma_y(i, j, k))
              / (d_dy * d_dy)
        - (d_Phi_z(i, j, k - 1) * sigma_z(i, j, k - 1) - d_Psi_z(i, j, k + 1) * sigma_z(i, j, k))
              / (d_dz * d_dz);
    return K(i, j, k) * K(i, j, k) * result;
}

__device__ real_t sigma_z(int_t i, int_t j, int_t k) {
    if(k < 0 || k >= PADDING)
        return 0;
    return SIGMA;
}

__device__ real_t sigma_x(int_t i, int_t j, int_t k) {
    if(j < 0 || j >= PADDING)
        return 0;
    return SIGMA;
}

__device__ real_t sigma_y(int_t i, int_t j, int_t k) {
    if(j < 0 || j >= PADDING)
        return 0;
    return SIGMA;
}

#define padded_index_bottom(i, j, k)                                                               \
    (per(i, d_Nx + PADDING) * (d_Ny + PADDING) * PADDING + per(j, d_Nz + PADDING) * PADDING        \
     + per(k, PADDING))
#define padded_index_side(i, j, k)                                                                 \
    (per(i, PADDING) + per(j, d_Ny + PADDING) * PADDING + per(k, d_Nz) * (d_Ny + PADDING) * PADDING)
#define padded_index_front(i, j, k)                                                                \
    (per(i, d_Nx) * PADDING + per(j, d_Nz) * PADDING * (d_Nx + PADDING) + per(k, PADDING))

__device__ real_t get_buf_bottom(real_t *buffer, int_t i, int_t j, int_t k) {
    if(k < 0 || k >= PADDING)
        return 0;
    return buffer[padded_index_bottom(i, j, k)];
}
__device__ void set_buf_bottom(real_t *buffer, real_t value, int_t i, int_t j, int_t k) {
    if(k < 0 || k >= PADDING) {
        printf("ERROR trying to write at illegal access. check your code.\n");
        return;
    }
    buffer[padded_index_bottom(i, j, k)] = value;
}

__device__ real_t get_buf_side(real_t *buffer, int_t i, int_t j, int_t k) {
    if(i < 0 || i >= PADDING)
        return 0;
    return buffer[padded_index_side(i, j, k)];
}
__device__ void set_buf_side(real_t *buffer, real_t value, int_t i, int_t j, int_t k) {
    if(i < 0 || i >= PADDING) {
        printf("ERROR trying to write at illegal access. check your code.\n");
        return;
    }
    buffer[padded_index_side(i, j, k)] = value;
}

__device__ real_t get_buf_front(real_t *buffer, int_t i, int_t j, int_t k) {
    if(j < 0 || j >= PADDING)
        return 0;
    return buffer[padded_index_front(i, j, k)];
}
__device__ void set_buf_front(real_t *buffer, real_t value, int_t i, int_t j, int_t k) {
    if(j < 0 || j >= PADDING) {
        printf("ERROR trying to write at illegal access. check your code.\n");
        return;
    }
    buffer[padded_index_front(i, j, k)] = value;
}

// interface that finds the correct place in memory from the index
__device__ real_t get_buf(Aux_variable buffer, int_t i, int_t j, int_t k) {
    if(k >= d_Nz) // on the BOTTOM boundary ! axis go downwards
    {
        k -= d_Nz; // bring bottom layer back up
        return get_buf_bottom(buffer.bottom, i, j, k);
        // dimensions of bottom layer: (Nx + PADDING, Ny + PADDING, PADDING)
    }
    if(i >= d_Nx) // on the SIDE boundary
    {
        i -= d_Nx; // bring side layer to left
        return get_buf_side(buffer.side, i, j, k);
        // dimensions of side layer: (PADDING, Ny + PADDING, Nz)
    }
    if(j >= d_Ny) // on the FRONT boundary!
    {
        j -= d_Ny; // bring front layer back
        return get_buf_front(buffer.front, i, j, k);
        // dimensions of front layer: (Nx, PADDING, Nz)
    }
    // thishappens when I'm not in a boundary anymore, what to do then ? I guess that can happen,
    // but will never be processed further, so just ignore this
    return 0;
}

__device__ void set_buf(Aux_variable buffer, real_t value, int_t i, int_t j, int_t k) {

    if(k >= d_Nz) // on the BOTTOM boundary ! axis go downwards
    {
        k -= d_Nz; // bring bottom layer back up
        set_buf_bottom(buffer.bottom, value, i, j, k);
        return;
        // dimensions of bottom layer: (Nx + PADDING, Ny + PADDING, PADDING)
    }
    if(i >= d_Nx) // on the SIDE boundary
    {
        i -= d_Nx; // bring side layer to left
        set_buf_side(buffer.side, value, i, j, k);
        return;
        // dimensions of side layer: (PADDING, Ny + PADDING, Nz)
    }
    if(j >= d_Ny) // on the FRONT boundary!
    {
        j -= d_Ny; // bring front layer back
        set_buf_front(buffer.front, value, i, j, k);
        return;
        // dimensions of front layer: (Nx, PADDING, Nz)
    }

    // this happens when I'm not in a boundary anymore, what to do then ? I guess that can happen,
    // but will never be processed further, so just ignore this
    return;
}

/*

    Variable stepping

*/

// splits the boundary into 3 parts and calls 3 kernels
__host__ void step_all_aux_var() {

    int block_x = 8;
    int block_y = 8;
    int block_z = 8;
    dim3 blockSize(block_x, block_y, block_z);
    dim3 gridSize((Nx + PADDING + block_x - 1) / block_x,
                  (Ny + PADDING + block_y - 1) / block_y,
                  (Nz + PADDING + block_z - 1) / block_z);
    dim3 pml_bottom_gridSize((Nx + PADDING + block_x - 1) / block_x,
                             (Ny + PADDING + block_y - 1) / block_y,
                             (PADDING + block_z - 1) / block_z);
    dim3 pml_side_gridSize((PADDING + block_x - 1) / block_x,
                           (Ny + PADDING + block_y - 1) / block_y,
                           (Nz + block_z - 1) / block_z);
    dim3 pml_front_gridSize((Nx + block_x - 1) / block_x,
                            (PADDING + block_y - 1) / block_y,
                            (Nz + block_z - 1) / block_z);

    aux_variable_step_front<<<pml_front_gridSize, blockSize>>>(d_buffer,
                                                               d_phi_y_prv.front,
                                                               d_psi_y_prv.front,
                                                               d_phi_y.front,
                                                               d_psi_y.front,
                                                               d_phi_x_prv.front,
                                                               d_psi_x_prv.front,
                                                               d_phi_x.front,
                                                               d_psi_x.front,
                                                               d_phi_z_prv.front,
                                                               d_psi_z_prv.front,
                                                               d_phi_z.front,
                                                               d_psi_z.front);

    aux_variable_step_side<<<pml_side_gridSize, blockSize>>>(d_buffer,
                                                             d_phi_y_prv.side,
                                                             d_psi_y_prv.side,
                                                             d_phi_y.side,
                                                             d_psi_y.side,
                                                             d_phi_x_prv.side,
                                                             d_psi_x_prv.side,
                                                             d_phi_x.side,
                                                             d_psi_x.side,
                                                             d_phi_z_prv.side,
                                                             d_psi_z_prv.side,
                                                             d_phi_z.side,
                                                             d_psi_z.side);

    aux_variable_step_bottom<<<pml_bottom_gridSize, blockSize>>>(d_buffer,
                                                                 d_phi_y_prv.bottom,
                                                                 d_psi_y_prv.bottom,
                                                                 d_phi_y.bottom,
                                                                 d_psi_y.bottom,
                                                                 d_phi_x_prv.bottom,
                                                                 d_psi_x_prv.bottom,
                                                                 d_phi_x.bottom,
                                                                 d_psi_x.bottom,
                                                                 d_phi_z_prv.bottom,
                                                                 d_psi_z_prv.bottom,
                                                                 d_phi_z.bottom,
                                                                 d_psi_z.bottom);
}

// psi and phi formula
__device__ void update_all_aux_var_at_ijk_front(const real_t *d_buffer,
                                                real_t *d_phi_y_prv,
                                                real_t *d_psi_y_prv,
                                                real_t *d_phi_y,
                                                real_t *d_psi_y,
                                                real_t *d_phi_x_prv,
                                                real_t *d_psi_x_prv,
                                                real_t *d_phi_x,
                                                real_t *d_psi_x,
                                                real_t *d_phi_z_prv,
                                                real_t *d_psi_z_prv,
                                                real_t *d_phi_z,
                                                real_t *d_psi_z,
                                                int_t i,
                                                int_t j,
                                                int_t k) {

    /*
       I need to keep using the general get_buf (access to all possible sides), because I'm
      accessing neighbors, and might access other sides but it will diverge less, because only the
      border cells will give different results at if statements

    => I can't simply use one buffer side
   */

    // (i >= d_Nx || j >= PADDING || k >= d_Nz)
    // global coordinates
    int_t g_i = i;
    int_t g_j = j + d_Ny;
    int_t g_k = k;

    real_t next_phi_y = K(i, j + d_Ny, k)
                          * (-0.5 / d_dy
                                 * (sigma_y(i, j - 1, k) * get_buf_front(d_phi_y_prv, i, j - 1, k)
                                    + sigma_y(i, j, k) * get_buf_front(d_phi_y_prv, i, j, k))
                             - 0.5 / d_dy * (d_P(i, j + 1, k) - d_P(i, j - 1, k)))
                          * d_dt
                      + get_buf_front(d_phi_y_prv, i, j, k);

    real_t next_psi_y = K(i, j + d_Ny, k)
                          * (-0.5 / d_dy
                                 * (sigma_y(i, j - 1, k) * get_buf_front(d_psi_y_prv, i, j, k)
                                    + sigma_y(i, j, k) * get_buf_front(d_psi_y_prv, i, j + 1, k))
                             - 0.5 / d_dy * (d_P(i, j + 1, k) - d_P(i, j - 1, k)))
                          * d_dt
                      + get_buf_front(d_psi_y_prv, i, j, k);

    real_t next_phi_x = K(i + d_Nx, j + PADDING, k)
                          * (-0.5 / d_dx
                                 * (sigma_x(i - 1, j, k) * get_buf_front(d_phi_x_prv, i - 1, j, k)
                                    + sigma_x(i, j, k) * get_buf_front(d_phi_x_prv, i, j, k))
                             - 0.5 / d_dx * (d_P(i + 1, j, k) - d_P(i - 1, j, k)))
                          * d_dt
                      + get_buf_front(d_phi_x_prv, i, j, k);

    real_t next_psi_x = K(i + d_Nx, j + PADDING, k)
                          * (-0.5 / d_dx
                                 * (sigma_x(i - 1, j, k) * get_buf_front(d_psi_x_prv, i, j, k)
                                    + sigma_x(i, j, k) * get_buf_front(d_psi_x_prv, i + 1, j, k))
                             - 0.5 / d_dx * (d_P(i + 1, j, k) - d_P(i - 1, j, k)))
                          * d_dt
                      + get_buf_front(d_psi_x_prv, i, j, k);

    real_t next_phi_z = K(i, j, k + d_Nz)
                          * (-0.5 / d_dz
                                 * (sigma_z(i, j, k - 1) * get_buf_front(d_phi_z_prv, i, j, k - 1)
                                    + sigma_z(i, j, k) * get_buf_front(d_phi_z_prv, i, j, k))
                             - 0.5 / d_dz * (d_P(i, j, k + 1) - d_P(i, j, k - 1)))
                          * d_dt
                      + get_buf_front(d_phi_z_prv, i, j, k);

    real_t next_psi_z = K(i, j, k + d_Nz)
                          * (-0.5 / d_dz
                                 * (sigma_z(i, j, k - 1) * get_buf_front(d_psi_z_prv, i, j, k)
                                    + sigma_z(i, j, k) * get_buf_front(d_psi_z_prv, i, j, k + 1))
                             - 0.5 / d_dz * (d_P(i, j, k + 1) - d_P(i, j, k - 1)))
                          * d_dt
                      + get_buf_front(d_psi_z_prv, i, j, k);

    set_buf_front(d_phi_y, next_phi_y, i, j, k);
    set_buf_front(d_psi_y, next_psi_y, i, j, k);
    set_buf_front(d_phi_x, next_phi_x, i, j, k);
    set_buf_front(d_psi_x, next_psi_x, i, j, k);
    set_buf_front(d_phi_z, next_phi_z, i, j, k);
    set_buf_front(d_psi_z, next_psi_z, i, j, k);
}

// psi and phi formula
__device__ void update_all_aux_var_at_ijk_side(const real_t *d_buffer,
                                               real_t *d_phi_y_prv,
                                               real_t *d_psi_y_prv,
                                               real_t *d_phi_y,
                                               real_t *d_psi_y,
                                               real_t *d_phi_x_prv,
                                               real_t *d_psi_x_prv,
                                               real_t *d_phi_x,
                                               real_t *d_psi_x,
                                               real_t *d_phi_z_prv,
                                               real_t *d_psi_z_prv,
                                               real_t *d_phi_z,
                                               real_t *d_psi_z,
                                               int_t i,
                                               int_t j,
                                               int_t k) {

    /*
       I need to keep using the general get_buf (access to all possible sides), because I'm
      accessing neighbors, and might access other sides but it will diverge less, because only the
      border cells will give different results at if statements

    => I can't simply use one buffer side
   */

    // (i >= PADDING || j >= d_Ny + PADDING || k >= d_Nz)
    // global coordinates
    int_t g_i = i + d_Nx;
    int_t g_j = j;
    int_t g_k = k;

    real_t next_phi_y = K(i, j + d_Ny, k)
                          * (-0.5 / d_dy
                                 * (sigma_y(i, j - 1, k) * get_buf_side(d_phi_y_prv, i, j - 1, k)
                                    + sigma_y(i, j, k) * get_buf_side(d_phi_y_prv, i, j, k))
                             - 0.5 / d_dy * (d_P(i, j + 1, k) - d_P(i, j - 1, k)))
                          * d_dt
                      + get_buf_side(d_phi_y_prv, i, j, k);

    real_t next_psi_y = K(i, j + d_Ny, k)
                          * (-0.5 / d_dy
                                 * (sigma_y(i, j - 1, k) * get_buf_side(d_psi_y_prv, i, j, k)
                                    + sigma_y(i, j, k) * get_buf_side(d_psi_y_prv, i, j + 1, k))
                             - 0.5 / d_dy * (d_P(i, j + 1, k) - d_P(i, j - 1, k)))
                          * d_dt
                      + get_buf_side(d_psi_y_prv, i, j, k);

    real_t next_phi_x = K(i + d_Nx, j + PADDING, k)
                          * (-0.5 / d_dx
                                 * (sigma_x(i - 1, j, k) * get_buf_side(d_phi_x_prv, i - 1, j, k)
                                    + sigma_x(i, j, k) * get_buf_side(d_phi_x_prv, i, j, k))
                             - 0.5 / d_dx * (d_P(i + 1, j, k) - d_P(i - 1, j, k)))
                          * d_dt
                      + get_buf_side(d_phi_x_prv, i, j, k);

    real_t next_psi_x = K(i + d_Nx, j + PADDING, k)
                          * (-0.5 / d_dx
                                 * (sigma_x(i - 1, j, k) * get_buf_side(d_psi_x_prv, i, j, k)
                                    + sigma_x(i, j, k) * get_buf_side(d_psi_x_prv, i + 1, j, k))
                             - 0.5 / d_dx * (d_P(i + 1, j, k) - d_P(i - 1, j, k)))
                          * d_dt
                      + get_buf_side(d_psi_x_prv, i, j, k);

    real_t next_phi_z = K(i, j, k + d_Nz)
                          * (-0.5 / d_dz
                                 * (sigma_z(i, j, k - 1) * get_buf_side(d_phi_z_prv, i, j, k - 1)
                                    + sigma_z(i, j, k) * get_buf_side(d_phi_z_prv, i, j, k))
                             - 0.5 / d_dz * (d_P(i, j, k + 1) - d_P(i, j, k - 1)))
                          * d_dt
                      + get_buf_side(d_phi_z_prv, i, j, k);

    real_t next_psi_z = K(i, j, k + d_Nz)
                          * (-0.5 / d_dz
                                 * (sigma_z(i, j, k - 1) * get_buf_side(d_psi_z_prv, i, j, k)
                                    + sigma_z(i, j, k) * get_buf_side(d_psi_z_prv, i, j, k + 1))
                             - 0.5 / d_dz * (d_P(i, j, k + 1) - d_P(i, j, k - 1)))
                          * d_dt
                      + get_buf_side(d_psi_z_prv, i, j, k);

    set_buf_side(d_phi_y, next_phi_y, i, j, k);
    set_buf_side(d_psi_y, next_psi_y, i, j, k);
    set_buf_side(d_phi_x, next_phi_x, i, j, k);
    set_buf_side(d_psi_x, next_psi_x, i, j, k);
    set_buf_side(d_phi_z, next_phi_z, i, j, k);
    set_buf_side(d_psi_z, next_psi_z, i, j, k);
}

// psi and phi formula
__device__ void update_all_aux_var_at_ijk_bottom(const real_t *d_buffer,
                                                 real_t *d_phi_y_prv,
                                                 real_t *d_psi_y_prv,
                                                 real_t *d_phi_y,
                                                 real_t *d_psi_y,
                                                 real_t *d_phi_x_prv,
                                                 real_t *d_psi_x_prv,
                                                 real_t *d_phi_x,
                                                 real_t *d_psi_x,
                                                 real_t *d_phi_z_prv,
                                                 real_t *d_psi_z_prv,
                                                 real_t *d_phi_z,
                                                 real_t *d_psi_z,
                                                 int_t i,
                                                 int_t j,
                                                 int_t k) {

    /*
       I need to keep using the general get_buf (access to all possible sides), because I'm
      accessing neighbors, and might access other sides but it will diverge less, because only the
      border cells will give different results at if statements

    => I can't simply use one buffer side
   */

    // (i >= d_Nx + PADDING || j >= d_Ny + PADDING || k >= PADDING)
    // global coordinates
    int_t g_i = i;
    int_t g_j = j;
    int_t g_k = k + d_Nz;

    real_t next_phi_y = K(i, j + d_Ny, k)
                          * (-0.5 / d_dy
                                 * (sigma_y(i, j - 1, k) * get_buf_bottom(d_phi_y_prv, i, j - 1, k)
                                    + sigma_y(i, j, k) * get_buf_bottom(d_phi_y_prv, i, j, k))
                             - 0.5 / d_dy * (d_P(i, j + 1, k) - d_P(i, j - 1, k)))
                          * d_dt
                      + get_buf_bottom(d_phi_y_prv, i, j, k);

    real_t next_psi_y = K(i, j + d_Ny, k)
                          * (-0.5 / d_dy
                                 * (sigma_y(i, j - 1, k) * get_buf_bottom(d_psi_y_prv, i, j, k)
                                    + sigma_y(i, j, k) * get_buf_bottom(d_psi_y_prv, i, j + 1, k))
                             - 0.5 / d_dy * (d_P(i, j + 1, k) - d_P(i, j - 1, k)))
                          * d_dt
                      + get_buf_bottom(d_psi_y_prv, i, j, k);

    real_t next_phi_x = K(i + d_Nx, j + PADDING, k)
                          * (-0.5 / d_dx
                                 * (sigma_x(i - 1, j, k) * get_buf_bottom(d_phi_x_prv, i - 1, j, k)
                                    + sigma_x(i, j, k) * get_buf_bottom(d_phi_x_prv, i, j, k))
                             - 0.5 / d_dx * (d_P(i + 1, j, k) - d_P(i - 1, j, k)))
                          * d_dt
                      + get_buf_bottom(d_phi_x_prv, i, j, k);

    real_t next_psi_x = K(i + d_Nx, j + PADDING, k)
                          * (-0.5 / d_dx
                                 * (sigma_x(i - 1, j, k) * get_buf_bottom(d_psi_x_prv, i, j, k)
                                    + sigma_x(i, j, k) * get_buf_bottom(d_psi_x_prv, i + 1, j, k))
                             - 0.5 / d_dx * (d_P(i + 1, j, k) - d_P(i - 1, j, k)))
                          * d_dt
                      + get_buf_bottom(d_psi_x_prv, i, j, k);

    real_t next_phi_z = K(i, j, k + d_Nz)
                          * (-0.5 / d_dz
                                 * (sigma_z(i, j, k - 1) * get_buf_bottom(d_phi_z_prv, i, j, k - 1)
                                    + sigma_z(i, j, k) * get_buf_bottom(d_phi_z_prv, i, j, k))
                             - 0.5 / d_dz * (d_P(i, j, k + 1) - d_P(i, j, k - 1)))
                          * d_dt
                      + get_buf_bottom(d_phi_z_prv, i, j, k);

    real_t next_psi_z = K(i, j, k + d_Nz)
                          * (-0.5 / d_dz
                                 * (sigma_z(i, j, k - 1) * get_buf_bottom(d_psi_z_prv, i, j, k)
                                    + sigma_z(i, j, k) * get_buf_bottom(d_psi_z_prv, i, j, k + 1))
                             - 0.5 / d_dz * (d_P(i, j, k + 1) - d_P(i, j, k - 1)))
                          * d_dt
                      + get_buf_bottom(d_psi_z_prv, i, j, k);

    set_buf_bottom(d_phi_y, next_phi_y, i, j, k);
    set_buf_bottom(d_psi_y, next_psi_y, i, j, k);
    set_buf_bottom(d_phi_x, next_phi_x, i, j, k);
    set_buf_bottom(d_psi_x, next_psi_x, i, j, k);
    set_buf_bottom(d_phi_z, next_phi_z, i, j, k);
    set_buf_bottom(d_psi_z, next_psi_z, i, j, k);
}

/*NOTE
    The three following function all do the same thing, they are just calling the step formula for
   different parts of the border for warp and indexing
*/
__global__ void aux_variable_step_front(const real_t *d_buffer,
                                        real_t *d_phi_y_prv,
                                        real_t *d_psi_y_prv,
                                        real_t *d_phi_y,
                                        real_t *d_psi_y,
                                        real_t *d_phi_x_prv,
                                        real_t *d_psi_x_prv,
                                        real_t *d_phi_x,
                                        real_t *d_psi_x,
                                        real_t *d_phi_z_prv,
                                        real_t *d_psi_z_prv,
                                        real_t *d_phi_z,
                                        real_t *d_psi_z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i >= d_Nx || j >= PADDING || k >= d_Nz) {
        return;
    }

    update_all_aux_var_at_ijk_front(d_buffer,
                                    d_phi_y_prv,
                                    d_psi_y_prv,
                                    d_phi_y,
                                    d_psi_y,
                                    d_phi_x_prv,
                                    d_psi_x_prv,
                                    d_phi_x,
                                    d_psi_x,
                                    d_phi_z_prv,
                                    d_psi_z_prv,
                                    d_phi_z,
                                    d_psi_z,
                                    i,
                                    j,
                                    k);
}

__global__ void aux_variable_step_side(const real_t *d_buffer,
                                       real_t *d_phi_y_prv,
                                       real_t *d_psi_y_prv,
                                       real_t *d_phi_y,
                                       real_t *d_psi_y,
                                       real_t *d_phi_x_prv,
                                       real_t *d_psi_x_prv,
                                       real_t *d_phi_x,
                                       real_t *d_psi_x,
                                       real_t *d_phi_z_prv,
                                       real_t *d_psi_z_prv,
                                       real_t *d_phi_z,
                                       real_t *d_psi_z) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i >= PADDING || j >= d_Ny + PADDING || k >= d_Nz) {
        return;
    }

    update_all_aux_var_at_ijk_side(d_buffer,
                                   d_phi_y_prv,
                                   d_psi_y_prv,
                                   d_phi_y,
                                   d_psi_y,
                                   d_phi_x_prv,
                                   d_psi_x_prv,
                                   d_phi_x,
                                   d_psi_x,
                                   d_phi_z_prv,
                                   d_psi_z_prv,
                                   d_phi_z,
                                   d_psi_z,
                                   i,
                                   j,
                                   k);
}

__global__ void aux_variable_step_bottom(const real_t *d_buffer,
                                         real_t *d_phi_y_prv,
                                         real_t *d_psi_y_prv,
                                         real_t *d_phi_y,
                                         real_t *d_psi_y,
                                         real_t *d_phi_x_prv,
                                         real_t *d_psi_x_prv,
                                         real_t *d_phi_x,
                                         real_t *d_psi_x,
                                         real_t *d_phi_z_prv,
                                         real_t *d_psi_z_prv,
                                         real_t *d_phi_z,
                                         real_t *d_psi_z) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i >= d_Nx + PADDING || j >= d_Ny + PADDING || k >= PADDING) {
        return;
    }

    update_all_aux_var_at_ijk_bottom(d_buffer,
                                     d_phi_y_prv,
                                     d_psi_y_prv,
                                     d_phi_y,
                                     d_psi_y,
                                     d_phi_x_prv,
                                     d_psi_x_prv,
                                     d_phi_x,
                                     d_psi_x,
                                     d_phi_z_prv,
                                     d_psi_z_prv,
                                     d_phi_z,
                                     d_psi_z,
                                     i,
                                     j,
                                     k);
}

__device__ real_t gauss_seidel_formula(int i,
                                       int j,
                                       int k,
                                       real_t *d_buffer,
                                       real_t *d_buffer_prv,
                                       real_t *d_buffer_prv_prv,
                                       Aux_variable d_phi_x,
                                       Aux_variable d_phi_y,
                                       Aux_variable d_phi_z,
                                       Aux_variable d_psi_x,
                                       Aux_variable d_psi_y,
                                       Aux_variable d_psi_z) {
    real_t PML_val = PML(i, j, k, d_buffer, d_phi_x, d_phi_y, d_phi_z, d_psi_x, d_psi_y, d_psi_z);

    real_t result =
        (d_dt * d_dt)
            * (2 * (-K(i - 1, j, k) / (2 * d_dx) + K(i + 1, j, k) / (2 * d_dx))
                   * (-d_P(i - 1, j, k) / (2 * d_dx) + d_P(i + 1, j, k) / (2 * d_dx)) * K(i, j, k)
               + 2 * (-K(i, j - 1, k) / (2 * d_dy) + K(i, j + 1, k) / (2 * d_dy))
                     * (-d_P(i, j - 1, k) / (2 * d_dy) + d_P(i, j + 1, k) / (2 * d_dy)) * K(i, j, k)
               + 2 * (-K(i, j, k - 1) / (2 * d_dz) + K(i, j, k + 1) / (2 * d_dz))
                     * (-d_P(i, j, k - 1) / (2 * d_dz) + d_P(i, j, k + 1) / (2 * d_dz)) * K(i, j, k)
               + (-2 * d_P(i, j, k) / (d_dx * d_dx) + d_P(i - 1, j, k) / (d_dx * d_dx)
                  + d_P(i + 1, j, k) / (d_dx * d_dx))
                     * (K(i, j, k) * K(i, j, k))
               + (-2 * d_P(i, j, k) / (d_dy * d_dy) + d_P(i, j - 1, k) / (d_dy * d_dy)
                  + d_P(i, j + 1, k) / (d_dy * d_dy))
                     * (K(i, j, k) * K(i, j, k))
               + (-2 * d_P(i, j, k) / (d_dz * d_dz) + d_P(i, j, k - 1) / (d_dz * d_dz)
                  + d_P(i, j, k + 1) / (d_dz * d_dz))
                     * (K(i, j, k) * K(i, j, k))
               + PML_val)
        + 2 * d_P_prv(i, j, k) - d_P_prv_prv(i, j, k);

    return result;
}

__global__ void gauss_seidel_red(real_t *d_buffer,
                                 real_t *d_buffer_prv,
                                 real_t *d_buffer_prv_prv,
                                 Aux_variable d_phi_x,
                                 Aux_variable d_phi_y,
                                 Aux_variable d_phi_z,
                                 Aux_variable d_psi_x,
                                 Aux_variable d_psi_y,
                                 Aux_variable d_psi_z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i >= d_Nx + PADDING || j >= d_Ny + PADDING || k >= d_Nz + PADDING)
        return;

    if((i + j + k) % 2 == 0) {
        d_P(i, j, k) = gauss_seidel_formula(i,
                                            j,
                                            k,
                                            d_buffer,
                                            d_buffer_prv,
                                            d_buffer_prv_prv,
                                            d_phi_x,
                                            d_phi_y,
                                            d_phi_z,
                                            d_psi_x,
                                            d_psi_y,
                                            d_psi_z);
    }
}

__global__ void gauss_seidel_black(real_t *d_buffer,
                                   real_t *d_buffer_prv,
                                   real_t *d_buffer_prv_prv,
                                   Aux_variable d_phi_x,
                                   Aux_variable d_phi_y,
                                   Aux_variable d_phi_z,
                                   Aux_variable d_psi_x,
                                   Aux_variable d_psi_y,
                                   Aux_variable d_psi_z) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if(i >= d_Nx + PADDING || j >= d_Ny + PADDING || k >= d_Nz + PADDING)
        return;

    if((i + j + k) % 2 == 1) {
        d_P(i, j, k) = gauss_seidel_formula(i,
                                            j,
                                            k,
                                            d_buffer,
                                            d_buffer_prv,
                                            d_buffer_prv_prv,
                                            d_phi_x,
                                            d_phi_y,
                                            d_phi_z,
                                            d_psi_x,
                                            d_psi_y,
                                            d_psi_z);
    }
}

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

        emit_source<<<gridSize, blockSize>>>(d_buffer, iteration * dt);

        step_all_aux_var();

        for(size_t iter = 0; iter < 5; iter++) {
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

static void alloc_aux_var(Aux_variable *v) {
    size_t border_size_bottom = (Nx + PADDING) * (Ny + PADDING) * PADDING;
    size_t border_size_side = (Ny + PADDING) * Nz * PADDING;
    size_t border_size_front = Nx * Nz * PADDING;

    cudaErrorCheck(cudaMalloc(&(v->bottom), border_size_bottom * sizeof(real_t)));
    cudaErrorCheck(cudaMalloc(&(v->side), border_size_side * sizeof(real_t)));
    cudaErrorCheck(cudaMalloc(&(v->front), border_size_front * sizeof(real_t)));

    // also a memset for safety

    cudaErrorCheck(cudaMemset(v->bottom, 0, border_size_bottom));
    cudaErrorCheck(cudaMemset(v->side, 0, border_size_side));
    cudaErrorCheck(cudaMemset(v->front, 0, border_size_front));
}

static void free_aux_var(Aux_variable *v) {
    size_t border_size_bottom = (Nx + PADDING) * (Ny + PADDING) * PADDING;
    size_t border_size_side = (Ny + PADDING) * Nz * PADDING;
    size_t border_size_front = Nx * Nz * PADDING;

    cudaErrorCheck(cudaFree(v->bottom));
    cudaErrorCheck(cudaFree(v->side));
    cudaErrorCheck(cudaFree(v->front));

    // also a memset for safety
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

    cudaErrorCheck(cudaMalloc(&d_buffer_prv_prv, buffer_size * sizeof(real_t)));
    cudaErrorCheck(cudaMalloc(&d_buffer_prv, buffer_size * sizeof(real_t)));
    cudaErrorCheck(cudaMalloc(&d_buffer, buffer_size * sizeof(real_t)));
    printf("d_buffer: %p\td_buffer_prv: %p\td_buffer_prv_prv: %p\n",
           (void *) d_buffer,
           (void *) d_buffer_prv,
           (void *) d_buffer_prv_prv);

    alloc_aux_var(&d_phi_x_prv);
    alloc_aux_var(&d_phi_y_prv);
    alloc_aux_var(&d_phi_z_prv);

    alloc_aux_var(&d_psi_x_prv);
    alloc_aux_var(&d_psi_y_prv);
    alloc_aux_var(&d_psi_z_prv);

    alloc_aux_var(&d_phi_x);
    alloc_aux_var(&d_phi_y);
    alloc_aux_var(&d_phi_z);

    alloc_aux_var(&d_psi_x);
    alloc_aux_var(&d_psi_y);
    alloc_aux_var(&d_psi_z);

    // set it all to 0 (memset only works for int!)
    cudaErrorCheck(cudaMemset(d_buffer_prv_prv, 0, buffer_size));
    cudaErrorCheck(cudaMemset(d_buffer_prv, 0, buffer_size));
    cudaErrorCheck(cudaMemset(d_buffer, 0, buffer_size));

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

    free(saved_buffer);
}

void swap_aux_var(Aux_variable *v1, Aux_variable *v2) {
    Aux_variable *temp = v1;
    v1 = v2;
    v2 = temp;
}

// Rotate the time step buffers for each dimension
void move_buffer_window() {

    real_t *temp = d_buffer_prv_prv;
    d_buffer_prv_prv = d_buffer_prv;
    d_buffer_prv = d_buffer;
    d_buffer = temp;

    // move auxiliary variables, I guess
    swap_aux_var(&d_phi_x_prv, &d_phi_x);
    swap_aux_var(&d_phi_y_prv, &d_phi_y);
    swap_aux_var(&d_phi_z_prv, &d_phi_z);
    swap_aux_var(&d_psi_x_prv, &d_psi_x);
    swap_aux_var(&d_psi_y_prv, &d_psi_y);
    swap_aux_var(&d_psi_z_prv, &d_psi_z);
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

    int_t k = Nz / 2;
    for(int j = 0; j < Ny + PADDING; j++) {
        int i;
        for(i = 0; i < Nx + PADDING - 1; i++) {
            int w = fprintf(
                out,
                "%.16lf ",
                (saved_buffer[i * ((Ny + PADDING) * (Nz + PADDING)) + j * (Nz + PADDING) + k])
                    / 1.47183118e-05);
            if(w < 0)
                printf("could not write all\n");
        }
        int w =
            fprintf(out,
                    "%.16lf\n",
                    (saved_buffer[i * ((Ny + PADDING) * (Nz + PADDING)) + j * (Nz + PADDING) + k])
                        / 1.47183118e-05);
        if(w < 0)
            printf("could not write all\n");
    }

    fclose(out);
}

// put a gaussian spike in the middle of the simulation, might be outdated
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

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort) {
    if(code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if(abort)
            exit(code);
    }
}

bool init_cuda() {
    // BEGIN: T2
    int dev_count;
    cudaErrorCheck(cudaGetDeviceCount(&dev_count));

    if(dev_count == 0) {
        fprintf(stderr, "No CUDA-compatible devices found.\n");
        return false;
    }

    cudaErrorCheck(cudaSetDevice(0));

    cudaDeviceProp prop;
    cudaErrorCheck(cudaGetDeviceProperties(&prop, 0));

    // Print the device properties
    printf("Device count: %d\n", dev_count);
    printf("Using device 0: %s\n", prop.name);
    printf("\tCompute capability: %d.%d\n", prop.major, prop.minor);
    printf("\tMultiprocessors: %d\n", prop.multiProcessorCount);
    printf("\tWarp size: %d\n", prop.warpSize);
    printf("\tGlobal memory: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
    printf("\tPer-block shared memory: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("\tPer-block registers: %d\n", prop.regsPerBlock);

    return true;

    // END: T2
}

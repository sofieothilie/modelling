#include "simulation.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define WALLTIME(t) ((double) (t).tv_sec + 1e-6 * (double) (t).tv_usec)

typedef enum { BOTTOM, TOP, LEFT, RIGHT, FRONT, BACK } Side;
#define N_SIDES (6)

__device__ __host__ Side &operator++(Side &a) {
    int n = static_cast<int>(a);
    ++n;
    a = static_cast<Side>(n);
    return a;
}

__device__ __host__ Side operator++(Side &a, int) {
    Side copy = a;
    ++a;
    return copy;
}

typedef enum { X, Y, Z } Component;
#define N_COMPONENTS (3)

// consists of 6 buffers: the whole shell around our rectangle
typedef struct {
    real_t *side[N_SIDES];
} PML_Shell;

// for each direction, one shell
typedef struct {
    PML_Shell dir[N_COMPONENTS];
} PML_Variable;

__device__ __host__ Component &operator++(Component &a) {
    int n = static_cast<int>(a);
    ++n;
    a = static_cast<Component>(n);
    return a;
}

__device__ __host__ Component operator++(Component &a, int) {
    Component copy = a;
    ++a;
    return copy;
}

typedef struct {
    int_t x;
    int_t y;
    int_t z;
} Coords;

inline void gpuAssert(const cudaError_t code, const char *file, const int line) {
    if(code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

#define cudaErrorCheck(ans)                                                                        \
    { gpuAssert((ans), __FILE__, __LINE__); }

bool init_cuda() {
    int dev_count;
    cudaErrorCheck(cudaGetDeviceCount(&dev_count));

    if(dev_count == 0) {
        fprintf(stderr, "No CUDA-compatible devices found\n");
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
}

// this function cannot be used in general because it counts the ghost cells in the memory alloc
__host__ __device__ int_t get_alloc_side_size(const Dimensions dimensions, const Side side) {
    int_t Nx = dimensions.Nx;
    int_t Ny = dimensions.Ny;
    int_t Nz = dimensions.Nz;
    int_t padding = dimensions.padding;

    switch(side) {
        case BOTTOM:
        case TOP:
            return (Nx + 2 * padding + 2) * (Ny + 2 * padding + 2)
                 * (padding + 1); // ghost cells on each side, one on top/bottom
        case LEFT:
        case RIGHT:
            return (padding + 1) * (Ny + 2 * padding + 2)
                 * Nz; // ghost cell on the side, and in the front/back part
        case FRONT:
        case BACK:
            return Nx * (padding + 1) * Nz; // ghost cells on the face
        default:
            printf("invalid side\n");
            return -1;
    }
}

PML_Shell allocate_pml_shell(const Dimensions dimensions) {
    PML_Shell shell = { 0 };

    for(Side side = BOTTOM; side < N_SIDES; side++) {
        size_t size = get_alloc_side_size(dimensions, side);

        cudaErrorCheck(cudaMalloc(&(shell.side[side]), size * sizeof(real_t)));
        cudaErrorCheck(cudaMemset(shell.side[side], 0, size));
    }

    return shell;
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

    for(Component component = X; component < N_COMPONENTS; component++) {
        variable.dir[component] = allocate_pml_shell(dimensions);
    }

    return variable;
}

void free_pml_shell(PML_Shell shell) {
    for(Side side = BOTTOM; side < N_SIDES; side++) {
        cudaErrorCheck(cudaFree(shell.side[side]));
    }
}

void free_pml_variables(PML_Variable var) {
    for(Component component = X; component < N_COMPONENTS; component++) {
        free_pml_shell(var.dir[component]);
    }
}

real_t *allocate_domain(const Dimensions dimensions) {
    real_t *result = NULL;

    int_t size = get_domain_size(dimensions);

    cudaErrorCheck(cudaMalloc(&result, size * sizeof(real_t)));
    cudaErrorCheck(cudaMemset(result, 0, size));

    return result;
}

void free_domain(real_t *buf) {
    cudaErrorCheck(cudaFree(buf));
}

// global indexing, so this is used for wave variables: nonPML
__host__ __device__ int_t gcoords_to_index(const Coords gcoords, const Dimensions dimensions) {
    const int_t i = gcoords.x;
    const int_t j = gcoords.y;
    const int_t k = gcoords.z;

    // const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;

    // if(i < 0 && j  == Ny / 2 && k == Nz /2){
    //     printf("negative i: %d\n", i);
    // }

    // coords (0,0,0) starts at PML, not in ghost cells, so shift (+1) added. ghost cells are at
    // indices [-1] and [Nx]
    return (i + 1) * (Ny + 2 * padding + 2) * (Nz + 2 * padding + 2)
         + (j + 1) * (Nz + 2 * padding + 2) + (k + 1);
}

#define U(gcoords) U[gcoords_to_index(gcoords, dimensions)]
#define U_prev(gcoords) U_prev[gcoords_to_index(gcoords, dimensions)]
#define U_prev_prev(gcoords) U_prev_prev[gcoords_to_index(gcoords, dimensions)]

__global__ void emit_source(real_t *const U, const Dimensions dimensions, const real_t value) {
    // const int_t i = blockIdx.x * blockDim.x + threadIdx.x;
    // const int_t j = blockIdx.y * blockDim.y + threadIdx.y;
    // const int_t k = blockIdx.z * blockDim.z + threadIdx.z;

    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    // const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;

    int n_source = 7;
    int spacing = 2; // spacing in terms of cells, not distance

    // const double freq = 1.0e6; // 1MHz

    // grid of sources over yz plane, at x = 5 maybe
    for(int n_i = 0; n_i < n_source; n_i++) {
        for(int n_j = 0; n_j < n_source; n_j++) {
            int idx_i = padding + Nx / 2 - (n_source / 2 * spacing) + n_i * spacing;
            int idx_j = padding + Ny / 2 - (n_source / 2 * spacing) + n_j * spacing;

            // double shift = 0;
            // (double) n_j / n_source * 2 * M_PI * 0.7;
            // double sine = sin(2 * M_PI * t * freq - shift);

            const Coords gcoords = { .x = idx_i, .y = idx_j, .z = padding + 10 };

            U(gcoords) = value;
        }
    }

    // if(t==0) {
    //     real_t delta = sqrt(((i - emit_coords.x) * (i - emit_coords.x))
    //                             / (real_t)(0.5 * (Nx + padding * 2))
    //                         + ((j - emit_coords.y) * (j - emit_coords.y))
    //                               / (real_t)(0.5 * (Ny + padding * 2))
    //                         + ((k -  emit_coords.z) * (k - emit_coords.z))
    //                               / (real_t)(0.5 * (Nz + padding * 2)));
    //     U_prev(gcoords) = U(gcoords) = exp(-t*freq*t*freq)*exp(-4.0 * delta * delta);
    // }
}

void get_recv(const real_t *d_buffer, FILE *output, Dimensions dimensions) {
    // static int_t iter = 0;

    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    // const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;

    const int_t size = get_domain_size(dimensions);

    real_t *const h_buffer = (real_t *) malloc(size * sizeof(real_t));
    cudaErrorCheck(cudaMemcpy(h_buffer, d_buffer, sizeof(real_t) * size, cudaMemcpyDeviceToHost));

    Coords dst_coords = { .x = Nx / 2 + padding, .y = Ny / 2 + padding, .z = padding + 10 };

    real_t at_dest = h_buffer[gcoords_to_index(dst_coords, dimensions)];

    fwrite(&at_dest, sizeof(real_t), 1, output);

    free(h_buffer);
}

// dim3 get_pml_grid(Dimensions dimensions, dim3 block, Side side) {
//     const int_t Nx = dimensions.Nx;
//     const int_t Ny = dimensions.Ny;
//     const int_t Nz = dimensions.Nz;
//     const int_t padding = dimensions.padding;
//     const int_t block_x = block.x;
//     const int_t block_y = block.y;
//     const int_t block_z = block.z;

//     switch(side) {
//         case BOTTOM:
//         case TOP:
//             return dim3((Nx + 2 * padding + block_x - 1) / block_x,
//                         (Ny + 2 * padding + block_y - 1) / block_y,
//                         (padding + block_z - 1) / block_z);
//         case LEFT:
//         case RIGHT:
//             return dim3((padding + block_x - 1) / block_x,
//                         (Ny + 2 * padding + block_y - 1) / block_y,
//                         (Nz + block_z - 1) / block_z);
//         case FRONT:
//         case BACK:
//             return dim3((Nx + block_x - 1) / block_x,
//                         (padding + block_y - 1) / block_y,
//                         (Nz + block_z - 1) / block_z);
//         default:
//             printf("invalid side\n");
//             exit(EXIT_FAILURE);
//     }
// }

__device__ bool in_PML(const Coords gcoords, const Dimensions dimensions) {
    const int_t i = gcoords.x;
    const int_t j = gcoords.y;
    const int_t k = gcoords.z;

    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;

    // proceeding by layer here:

    // ghost cells, and nothing
    if(i < 0 || j < 0 || k < 0)
        return false;
    if(i >= Nx + 2 * padding || j >= Ny + 2 * padding || k >= Nz + 2 * padding)
        return false;

    // physical world
    if(padding <= i && i < Nx + padding && padding <= j && j < Ny + padding && padding <= k
       && k < Nz + padding)
        return false;

    return true;
}

__device__ Coords tau_shift(const Coords gcoords,
                            const int_t shift,
                            const Component component,
                            const Dimensions dimensions) {
    const int_t i = gcoords.x;
    const int_t j = gcoords.y;
    const int_t k = gcoords.z;

    switch(component) {
        case X:
            return { i + shift, j, k };
        case Y:
            return { i, j + shift, k };
        case Z:
            return { i, j, k + shift };
        default:
            printf("error in tau shift\n");
            return gcoords;
    }
}

__device__ real_t get_K(const Coords gcoords, const Dimensions dimensions) {
    // const int_t Nx = dimensions.Nx;
    // const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;

    // const int_t i = gcoords.x;
    // const int_t j = gcoords.y;
    const int_t k = gcoords.z;

    const real_t WATER_K = 1500.0;
    const real_t PLASTIC_K = 2270.0;

    // return WATER_K;

    if(k < padding + 5 * Nz / 6)
        return WATER_K;

    return PLASTIC_K;
}

__device__ real_t get_rho(const Coords gcoords, const Dimensions dimensions) {
    // const int_t Nx = dimensions.Nx;
    // const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;

    // const int_t i = gcoords.x;
    // const int_t j = gcoords.y;
    const int_t k = gcoords.z;

    const real_t WATER_RHO = 997.0;
    const real_t PLASTIC_RHO = 1185.0;

    // return 0;

    if(k < padding + 5 * Nz / 6)
        return WATER_RHO;

    return PLASTIC_RHO;
}

__device__ real_t get_sigma(const Coords gcoords,
                            const Dimensions dimensions,
                            const Component component) {

    const real_t SIGMA = 1.0 / dimensions.dh;

    Dimensions adjusted_dim = dimensions;
    adjusted_dim.Nx += 2;
    adjusted_dim.Ny += 2;
    adjusted_dim.Nz += 2;
    adjusted_dim.padding -= 1;

    if(in_PML(gcoords, adjusted_dim))
        return SIGMA;

    return 0.0;
}


#define tau(shift) (tau_shift(gcoords, shift, component, dimensions))
#define K(gcoords) (get_K(gcoords, dimensions))
#define Rho(gcoords) (get_rho(gcoords, dimensions))
#define sigma(gcoords) (get_sigma(gcoords, dimensions, component))
#define Psi(gcoords) (get_PML_var(Psi, gcoords, component, dimensions))
#define Psi_prev(gcoords) (get_PML_var(Psi_prev, gcoords, component, dimensions))
#define Phi(gcoords) (get_PML_var(Phi, gcoords, component, dimensions))
#define Phi_prev(gcoords) (get_PML_var(Phi_prev, gcoords, component, dimensions))
#define set_Psi(value) (set_PML_var(Psi, value, gcoords, component, dimensions))
#define set_Phi(value) (set_PML_var(Phi, value, gcoords, component, dimensions))

__device__ Side get_side(const Coords gcoords, const Dimensions dimensions) {
    const int_t i = gcoords.x;
    const int_t j = gcoords.y;
    const int_t k = gcoords.z;

    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;

    if(k < padding)
        return TOP;
    if(k >= padding + Nz)
        return BOTTOM;

    if(i < padding)
        return LEFT;
    if(i >= padding + Nx)
        return RIGHT;

    if(j < padding)
        return BACK;
    if(j >= padding + Ny)
        return FRONT;

    printf("Called `get_side` outside the PML (%d %d %d)\n", i, j, k);
    return BOTTOM;
}

__device__ Coords gcoords_to_lcoords(Coords gcoords, Dimensions dimensions, Side side) {
    const int_t i = gcoords.x;
    const int_t j = gcoords.y;
    const int_t k = gcoords.z;

    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;
    if(!in_PML(gcoords, dimensions)) {
        printf("translating to lcoords, but not in a side!\n");
    }

    // this consists of translating the topleft corner of my side to the origin
    switch(side) {
        case TOP: // TOP is the only side that is already in place
            return gcoords;
        case BOTTOM:
            return Coords { .x = i, .y = j, .z = k - Nz - padding };
        case LEFT:
            return Coords { .x = i, .y = j, .z = k - padding };
        case RIGHT:
            return Coords { .x = i - Nx - padding, .y = j, .z = k - padding };
        case BACK:
            return Coords { .x = i - padding, .y = j, .z = k - padding };
        case FRONT:
            return Coords { .x = i - padding, .y = j - Ny - padding, .z = k - padding };
        default:
            printf("called on invalid side\n");
            return gcoords;
    }
}

__device__ int_t lcoords_to_index(Coords lcoords, Dimensions dimensions, Side side) {
    const int_t i = lcoords.x;
    const int_t j = lcoords.y;
    const int_t k = lcoords.z;

    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;

    // i smallest moving index, then j, then k fastest moving and contiguous
    // the ghost cells are outside the bounds in gcoords, so I need to reintegrate them now, but
    // only in sides that have them before (TOP, LEFT, BACK)
    switch(side) {
        case TOP:
            return (i + 1) * (padding + 1) * (Ny + 2 * padding + 2) + (j + 1) * (padding + 1)
                 + (k + 1);
        case BOTTOM:
            return (i + 1) * (padding + 1) * (Ny + 2 * padding + 2) + (j + 1) * (padding + 1)
                 + (k); // no ghost cell before
        case LEFT:
            return (i + 1) * (Nz) * (Ny + 2 * padding + 2) + (j + 1) * (Nz) + (k);
        case RIGHT:
            return (i) * (Nz) * (Ny + 2 * padding + 2) + (j + 1) * (Nz) + (k);
        case BACK:
            return (i) * (Nz) * (padding + 1) + (j + 1) * (Nz) + k;
        case FRONT:
            return (i) * (Nz) * (padding + 1) + (j) * (Nz) + k;

        default:
            printf("called on invalid side\n");
            return 0;
    }
}

// this interfaces with the PML weird shape
__device__ real_t get_PML_var(const PML_Variable var,
                              const Coords gcoords,
                              const Component component,
                              const Dimensions dimensions) {

    // 0. if not in PML domain, return 0
    if(!in_PML(gcoords, dimensions))
        return 0.0;

    // 1. retrieve side
    Side side = get_side(gcoords, dimensions);

    // 2. retrieve local coords in this side
    Coords lcoords = gcoords_to_lcoords(gcoords, dimensions, side);

    // 3. retrieve index of that lcoord in the side
    int_t index = lcoords_to_index(lcoords, dimensions, side);

    // 4. finally access the PML variable.
    return (var.dir[component].side[side])[index];
}

__device__ void set_PML_var(PML_Variable var,
                            const real_t value,
                            const Coords gcoords,
                            const Component component,
                            const Dimensions dimensions) {

    // 0. if not in PML domain, something is wrong
    if(!in_PML(gcoords, dimensions)) {
        printf("something is wrong lol\n");
        return;
    }

    // 1. retrieve side
    Side side = get_side(gcoords, dimensions);

    // 2. retrieve local coords in this side
    Coords lcoords = gcoords_to_lcoords(gcoords, dimensions, side);

    // 3. retrieve index of that lcoord in the side
    int_t index = lcoords_to_index(lcoords, dimensions, side);

    // 4. finally write in the PML variable.
    (var.dir[component].side[side])[index] = value;
}

void move_buffer_window(real_t **const U, real_t **const U_prev) {
    real_t *const temp = *U_prev;
    *U_prev = *U;
    *U = temp;
}

void swap_aux_variables(PML_Variable *u, PML_Variable *v) {
    PML_Variable temp = *u;
    *u = *v;
    *v = temp;
}

__device__ bool in_bounds(const Coords gcoords, const Dimensions dimensions) {
    const int_t i = gcoords.x;
    const int_t j = gcoords.y;
    const int_t k = gcoords.z;

    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;

    if(i < 0 || j < 0 || k < 0)
        return false;

    if(i >= Nx + 2 * padding || j >= Ny + 2 * padding || k >= Nz + 2 * padding)
        return false;

    return true;
}

__device__ real_t gauss_seidel(const real_t *const U,
                               const real_t *const U_prev,
                               const real_t *const U_prev_prev,
                               PML_Variable Psi,
                               const PML_Variable Psi_prev,
                               PML_Variable Phi,
                               const PML_Variable Phi_prev,
                               const Dimensions dimensions,
                               const Coords gcoords) {
    const real_t dt = dimensions.dt;

    const real_t K = K(gcoords);

    real_t result = (2.0 * U_prev(gcoords) - U_prev_prev(gcoords)) / (dt * dt);
    real_t constants = 1 / (dt * dt);

    for(Component component = X; component < N_COMPONENTS; component++) {
        const real_t dh = dimensions.dh;

        real_t tau_m1 = U(tau(-1));
        real_t tau_p1 = U(tau(+1));

        // update pml

        if(in_PML(gcoords, dimensions)) { // is it really always 0 outside this if ?
            tau_m1 += -dh * sigma(tau(-1)) * Phi(tau(-1));
            tau_p1 += dh * sigma(gcoords) * Psi(tau(+1));

            // computing next phi and psi
            const real_t psi_value =
                (-Psi(tau(+1)) * sigma(gcoords) / 2.0 - (U(tau(+1)) - U(tau(-1))) / (2.0 * dh)
                 + Psi_prev(gcoords) / (dt * K))
                / ((1.0 / (dt * K)) + (sigma(tau(-1)) / 2.0));

            const real_t phi_value =
                (-Phi(tau(-1)) * sigma(tau(-1)) / 2.0 - (U(tau(+1)) - U(tau(-1))) / (2.0 * dh)
                 + Phi_prev(gcoords) / (dt * K))
                / ((1.0 / (dt * K)) + (sigma(gcoords) / 2.0));

            set_Phi(phi_value);
            set_Psi(psi_value);
        }

        /*
                real_t result =
            (d_dt * d_dt)
                * (2 * (-K(i - 1, j, k) / (2 * d_dx) + K(i + 1, j, k) / (2 * d_dx))
                    * (-d_P(i - 1, j, k) / (2 * d_dx) + d_P(i + 1, j, k) / (2 * d_dx)) * K(i, j,
           k)
                + 2 * (-K(i, j - 1, k) / (2 * d_dy) + K(i, j + 1, k) / (2 * d_dy))
                        * (-d_P(i, j - 1, k) / (2 * d_dy) + d_P(i, j + 1, k) / (2 * d_dy)) *
           K(i, j, k)
                + 2 * (-K(i, j, k - 1) / (2 * d_dz) + K(i, j, k + 1) / (2 * d_dz))
                        * (-d_P(i, j, k - 1) / (2 * d_dz) + d_P(i, j, k + 1) / (2 * d_dz)) *
           K(i, j, k)
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

        *///working K change. its the same !

        // observations:
        // 1. using the "correct" formula, the wave invert when it shouldnt
        // 2. using the inverted formula, the pml doesnt explode at the changing boundary
        // 3. with a strong PML, using a 0 speed explodes fast

        // -> something is seriously wrong with my formula, at the PML level and normal level
        // (maybe)

        // operator
        result += 0.5 * K / (dh * dh) * (K(tau(+1)) - K(tau(-1))) * (tau_p1 - tau_m1)
                + K * K / (dh * dh) * (tau_m1 + tau_p1);

        constants += 2.0 * K * K / (dh * dh);
    }

    result /= constants;
    return result;
}

__global__ void gauss_seidel_red(real_t *const U,
                                 const real_t *const U_prev,
                                 const real_t *const U_prev_prev,
                                 PML_Variable Psi,
                                 const PML_Variable Psi_prev,
                                 PML_Variable Phi,
                                 const PML_Variable Phi_prev,
                                 const Dimensions dimensions) {
    const int_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const int_t k = blockIdx.z * blockDim.z + threadIdx.z;

    int_t m = (i + j + 1) % 2;
    int_t true_k = 2 * k + m;

    const Coords gcoords = { .x = i, .y = j, .z = true_k };

    if(!in_bounds(gcoords, dimensions))
        return;

    U(gcoords) =
        gauss_seidel(U, U_prev, U_prev_prev, Psi, Psi_prev, Phi, Phi_prev, dimensions, gcoords);
}

__global__ void gauss_seidel_black(real_t *const U,
                                   const real_t *const U_prev,
                                   const real_t *const U_prev_prev,
                                   PML_Variable Psi,
                                   const PML_Variable Psi_prev,
                                   PML_Variable Phi,
                                   const PML_Variable Phi_prev,
                                   const Dimensions dimensions) {
    const int_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const int_t k = blockIdx.z * blockDim.z + threadIdx.z;

    int_t m = (i + j) % 2;
    int_t true_k = 2 * k + m;

    const Coords gcoords = { .x = i, .y = j, .z = true_k };

    if(!in_bounds(gcoords, dimensions))
        return;

    U(gcoords) =
        gauss_seidel(U, U_prev, U_prev_prev, Psi, Psi_prev, Phi, Phi_prev, dimensions, gcoords);
}

__global__ void var_density_solver(real_t *const U,
                                   const real_t *const U_prev,
                                   PML_Variable Psi,
                                   const PML_Variable Psi_prev,
                                   PML_Variable Phi,
                                   const PML_Variable Phi_prev,
                                   const Dimensions dimensions) {
    const int_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const int_t k = blockIdx.z * blockDim.z + threadIdx.z;

    const real_t dt = dimensions.dt;

    Coords gcoords = Coords { i, j, k };

    if(!in_bounds(gcoords, dimensions)) {
        return;
    }

    // update at position  (i,j,k)
    // U contains the prev prev value!
    real_t result = -U(gcoords) + 2 * U_prev(gcoords);

    for(Component component = X; component < N_COMPONENTS; component++) {
        const real_t dh = dimensions.dh;

        real_t pml1 = 0.0, pml2 = 0.0;

        if(in_PML(gcoords, dimensions)) {
            pml1 = dh * sigma(gcoords) * Psi_prev(tau(+1));
            pml2 = -dh * sigma(tau(-1)) * Phi_prev(tau(-1));
        }

        result += dt * dt / (dh * dh) * K(gcoords) * K(gcoords) * Rho(gcoords)
                * (0.5 * (1.0 / Rho(tau(+1)) + 1.0 / Rho(gcoords))
                       * (U_prev(tau(+1)) + pml1 - U_prev(gcoords))
                   - 0.5 * (1.0 / Rho(gcoords) + 1.0 / Rho(tau(-1)))
                         * (U_prev(gcoords) - U_prev(tau(-1)) - pml2));
    }
    U(gcoords) = result;
}

__global__ void pml_var_solver(real_t *const U_prev,
                               PML_Variable Psi,
                               const PML_Variable Psi_prev,
                               PML_Variable Phi,
                               const PML_Variable Phi_prev,
                               const Dimensions dimensions) {
    const int_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const int_t k = blockIdx.z * blockDim.z + threadIdx.z;

    Coords gcoords = Coords { i, j, k };

    if(in_PML(gcoords, dimensions)) {
        const real_t dt = dimensions.dt;

        for(Component component = X; component < N_COMPONENTS; component++) {
            const real_t dh = dimensions.dh;

            // I think are wrong: dt should multiply the whole, maybe K as well.
            // time indices it also  has a s_i, but that's soo wrong
            // using U_prev instead of U, I guess it makes sense
            real_t next_phi =
                Phi_prev(gcoords)
                - 0.5 * dt * K(gcoords)
                      * (sigma(tau(-1)) * Phi_prev(tau(-1)) + sigma(gcoords) * Phi_prev(gcoords))
                - dt * K(gcoords) / (2.0 * dh) * (U_prev(tau(+1)) - U_prev(tau(-1)));

            real_t next_psi =
                Psi_prev(gcoords)
                - 0.5 * dt * K(gcoords)
                      * (sigma(tau(-1)) * Psi_prev(gcoords) + sigma(gcoords) * Psi_prev(tau(+1)))
                - dt * K(gcoords) / (2.0 * dh) * (U_prev(tau(+1)) - U_prev(tau(-1)));

            set_Phi(next_phi);
            set_Psi(next_psi);
        }
        // something is wrong. is it supposed to ripple like this ?
    }
}

void domain_save(const real_t *const d_buffer, const Dimensions dimensions) {
    static int_t iter = 0;

    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;

    const int_t size = get_domain_size(dimensions);
    real_t *const h_buffer = (real_t *) malloc(size * sizeof(real_t));
    cudaErrorCheck(cudaMemcpy(h_buffer, d_buffer, sizeof(real_t) * size, cudaMemcpyDeviceToHost));

    char filename[256];
    sprintf(filename, "wave_data/%.5d.dat", iter);
    FILE *const out = fopen(filename, "w");
    if(!out) {
        fprintf(stderr, "Could not open file '%s'!\n", filename);
        exit(EXIT_FAILURE);
    }

    const int_t j = Ny / 2 + padding;
    for(int k = 0; k < Nz + 2 * padding; k++) {
        int i;
        for(i = 0; i < Nx + 2 * padding - 1; i++) {
            const Coords gcoords = { .x = i, .y = j, .z = k };
            const int w =
                fprintf(out, "%.16lf ", (h_buffer[gcoords_to_index(gcoords, dimensions)]));
            if(w < 0)
                printf("could not write all\n");
        }
        const Coords gcoords = { .x = i, .y = j, .z = k };
        const int w = fprintf(out, "%.16lf\n", (h_buffer[gcoords_to_index(gcoords, dimensions)]));
        if(w < 0)
            printf("could not write all\n");
    }

    free(h_buffer);
    fclose(out);
    iter++;
}

int signature(real_t **sig_buf) {
    FILE *f = fopen("data/signature.dat", "rb");
    if(!f) {
        perror("Failed to open file");
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    if(fsize < 0) {
        perror("ftell failed");
        fclose(f);
        return -1;
    }
    rewind(f);

    size_t num_doubles = fsize / sizeof(double);

    double *double_buffer = (double *) calloc(num_doubles, sizeof(double));
    if(!double_buffer) {
        perror("Failed to allocate double_buffer");
        fclose(f);
        return -1;
    }

    size_t read = fread(double_buffer, sizeof(double), num_doubles, f);
    fclose(f);

    if(read != num_doubles) {
        fprintf(stderr, "fread incomplete: expected %zu, got %zu\n", num_doubles, read);
        free(double_buffer);
        return -1;
    }

    *sig_buf = (real_t *) calloc(num_doubles, sizeof(real_t));
    if(!*sig_buf) {
        perror("Failed to allocate sig_buf");
        free(double_buffer);
        return -1;
    }

    for(size_t i = 0; i < num_doubles; i++) {
        (*sig_buf)[i] = (real_t) double_buffer[i];
    }

    free(double_buffer);
    return (int) num_doubles;
}

extern "C" int simulate_wave(const simulation_parameters p) {
    const real_t dt = p.dt;
    const int_t max_iteration = p.max_iter;
    const int_t snapshot_freq = p.snapshot_freq;

    Dimensions dimensions = p.dimensions;

    // const real_t sim_Lx = p.sim_Lx;
    // const real_t sim_Ly = p.sim_Ly;
    // const real_t sim_Lz = p.sim_Lz;

    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;
    // const real_t dh = dimensions.dh;

    printf("Nx = %d, Ny = %d, Nz = %d\n", Nx, Ny, Nz);

    printf("version RK4\n");

    if(!init_cuda()) {
        fprintf(stderr, "Could not initialize CUDA\n");
        exit(EXIT_FAILURE);
    }

    PML_Variable Psi = allocate_pml_variables(dimensions);
    PML_Variable Phi = allocate_pml_variables(dimensions);
    PML_Variable Psi_prev = allocate_pml_variables(dimensions);
    PML_Variable Phi_prev = allocate_pml_variables(dimensions);

    real_t *U = allocate_domain(dimensions);
    real_t *U_prev = allocate_domain(dimensions);

    FILE *output_buffer = fopen("sensor_out/dest_wave.dat", "w");

    real_t *sig;
    int sig_len = signature(&sig);

    printf("sig_len  = %d\n", sig_len);

    struct timeval start, end;

    gettimeofday(&start, NULL);

    for(int_t iteration = 0; iteration < max_iteration; iteration++) {
        if((iteration % snapshot_freq) == 0) {
            printf("iteration %d/%d\n", iteration, max_iteration);
            cudaDeviceSynchronize();
            domain_save(U, dimensions);
        }

        get_recv(U, output_buffer, dimensions);

        int block_x = 8;
        int block_y = 8;
        int block_z = 8;
        dim3 block(block_x, block_y, block_z);
        dim3 grid((Nx + 2 * padding + block_x - 1) / block_x,
                  (Ny + 2 * padding + block_y - 1) / block_y,
                  ((Nz + 2 * padding) + block_z - 1) / block_z);

        var_density_solver<<<grid, block>>>(U, U_prev, Psi, Psi_prev, Phi, Phi_prev, dimensions);

        real_t src_freq = 1.0e6;
        real_t src_sampling_rate = 8 * src_freq;

        int signature_idx = (int) (iteration * dt * src_sampling_rate);

        if(signature_idx < sig_len) {
            real_t src_value = sig[signature_idx];
            emit_source<<<1, 1>>>(U, dimensions, src_value);
        }

        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) {
            printf("cuda kernel error: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        pml_var_solver<<<grid, block>>>(U_prev, Psi, Psi_prev, Phi, Phi_prev, dimensions);

        err = cudaGetLastError();
        if(err != cudaSuccess) {
            printf("cuda kernel error: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        // show_sigma<<<grid, block>>>(dimensions, U, U_prev);

        move_buffer_window(&U, &U_prev);
        swap_aux_variables(&Psi, &Psi_prev);
        swap_aux_variables(&Phi, &Phi_prev);
    }
    cudaDeviceSynchronize();

    fclose(output_buffer);

    gettimeofday(&end, NULL);

    double diff = WALLTIME(end) - WALLTIME(start);

    printf("time taken: %lf sec, %lf per iteration\n", diff, diff / max_iteration);

    free_domain(U);
    free_domain(U_prev);

    free_pml_variables(Psi);
    free_pml_variables(Phi);
    free_pml_variables(Psi_prev);
    free_pml_variables(Phi_prev);

    return 0;
}

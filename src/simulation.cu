#include "simulation.h"
#include <stdio.h>

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
    {                                                                                              \
        gpuAssert((ans), __FILE__, __LINE__);                                                      \
    }

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

    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;

    // coords (0,0,0) starts at PML, not in ghost cells, so shift (+1) added. ghost cells are at
    // indices [-1] and [Nx]
    return (i + 1) * (Ny + 2 * padding + 2) * (Nz + 2 * padding + 2)
         + (j + 1) * (Nz + 2 * padding + 2) + (k + 1);
}

#define U(gcoords) U[gcoords_to_index(gcoords, dimensions)]
#define U_prev(gcoords) U_prev[gcoords_to_index(gcoords, dimensions)]
#define U_prev_prev(gcoords) U_prev_prev[gcoords_to_index(gcoords, dimensions)]

__global__ void emit_source(real_t *const U, const Dimensions dimensions, const real_t t) {
    const int_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const int_t k = blockIdx.z * blockDim.z + threadIdx.z;

    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;

    const Coords gcoords = { 4 * Nx / 8 + padding, 4 * Ny / 8 + padding, Nz / 2 + padding };
    const double freq = 1.0e3; // 1MHz
    if(i == gcoords.x && j == gcoords.y && k == gcoords.z) {
        if(t * freq < 1.0) {
            U(gcoords) = sin(2 * M_PI * t * freq);
        }
    }
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
    }
}

__device__ real_t get_K(const Coords gcoords, const Dimensions dimensions) {
    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;

    const int_t i = gcoords.x;
    const int_t j = gcoords.y;
    const int_t k = gcoords.z;

    const real_t WATER_K = 1500.0;
    const real_t PLASTIC_K = 2270.0;

    return 1.0;

    return WATER_K;

    if(i < Nx / 2 && i > Nx / 6)
        return PLASTIC_K;

    return WATER_K;
}

__device__ real_t get_sigma(const Coords gcoords,
                            const Dimensions dimensions,
                            const Component component) {

    const real_t SIGMA = 1.0;

    Dimensions adjusted_dim = dimensions;
    adjusted_dim.Nx += 2;
    adjusted_dim.Ny += 2;
    adjusted_dim.Nz += 2;
    adjusted_dim.padding -= 1;

    if(in_PML(gcoords, adjusted_dim))
        return SIGMA;

    return 0.0;
}

// __host__ __device__ int_t lcoords_to_index(const Coords lcoords,
//                                            const Dimensions dimensions,
//                                            const Side side) {
//     const int_t i = lcoords.x;
//     const int_t j = lcoords.y;
//     const int_t k = lcoords.z;

//     const int_t Nx = dimensions.Nx;
//     const int_t Ny = dimensions.Ny;
//     const int_t Nz = dimensions.Nz;
//     const int_t padding = dimensions.padding;

//     switch(side) {
//         case BOTTOM:
//             return i * (Ny + padding) * padding + j * padding + k;
//         case SIDE:
//             return i * (Ny + padding) * Nz + j * Nz + k;
//         case FRONT:
//             return i * padding * Nz + j * Nz + k;
//     }
// }

#define tau(coords, shift) (tau_shift(coords, shift, component, dimensions))
#define K(gcoords) (get_K(gcoords, dimensions))
#define sigma(gcoords) (get_sigma(gcoords, dimensions, component))
#define Psi(gcoords) (get_PML_var(U, Psi, gcoords, component, dimensions))
#define Psi_prev(gcoords) (get_PML_var(U, Psi_prev, gcoords, component, dimensions))
#define Phi(gcoords) (get_PML_var(U, Phi, gcoords, component, dimensions))
#define Phi_prev(gcoords) (get_PML_var(U, Phi_prev, gcoords, component, dimensions))
#define set_Psi(gcoords, value) (set_PML_var(Psi, value, gcoords, component, dimensions))
#define set_Phi(gcoords, value) (set_PML_var(Phi, value, gcoords, component, dimensions))

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
            return Coords { .x = i, .y = j, .z = k - Nx - padding };
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

    const int_t Nx = dimensions.Nx;
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
__device__ real_t get_PML_var(const real_t *const U,
                              const PML_Variable var,
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

    if(index < 0 || index >= get_alloc_side_size(dimensions, side)) {
        printf("illegal local index at gcoords(%d %d %d)\n", gcoords.x, gcoords.y, gcoords.z);
        int_t max_idx =  get_alloc_side_size(dimensions, side);
        lcoords_to_index(lcoords, dimensions, side);
    }

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

void move_buffer_window(real_t **const U, real_t **const U_prev, real_t **const U_prev_prev) {
    real_t *const temp = *U_prev_prev;
    *U_prev_prev = *U_prev;
    *U_prev = *U;
    *U = temp;
}

void swap_aux_variables(PML_Variable *u, PML_Variable *v) {
    PML_Variable temp = *u;
    *u = *v;
    *v = temp;
}

__global__ void set_random_values(real_t *const U, const Dimensions dimensions) {
    const int_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const int_t k = blockIdx.z * blockDim.z + threadIdx.z;
    const Coords gcoords = { .x = i, .y = j, .z = k };

    U(gcoords) = (i - 2 * j + k) % (i + j + k);
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

    real_t result = (2.0 * U_prev(gcoords) - U_prev_prev(gcoords)) / (dt * dt);
    real_t constants = 1 / (dt * dt);

    for(Component component = X; component < N_COMPONENTS; component++) {
        const real_t dh = dimensions.dh[component];
        real_t PML = 0.0;

        // update pml

        // only do it in applicable part of PML: one step inside the padding
        // SOLVE: this is the problematic part. if you remove the if and solve everywhere, it
        // works fine. it should also with the if, but it doesnt
        if(in_PML(gcoords, dimensions)) {
            PML += K(gcoords) * K(gcoords)
                 * (sigma(gcoords) * Psi(tau(gcoords, +1))
                    - sigma(tau(gcoords, -1)) * Phi(tau(gcoords, -1)))
                 / (dh * dh);

            const real_t psi_value =
                (-Psi(tau(gcoords, +1)) * sigma(gcoords) * K(gcoords) / (2.0 * dh)
                 - (U(tau(gcoords, +1)) - U(tau(gcoords, -1))) * K(gcoords) / (2.0 * dh)
                 + Psi_prev(gcoords) / dt)
                / ((1.0 / dt) + (sigma(tau(gcoords, -1)) / (2.0 * dh)));

            const real_t phi_value =
                (-Phi(tau(gcoords, -1)) * sigma(tau(gcoords, -1)) * K(gcoords) / (2.0 * dh)
                 - (U(tau(gcoords, +1)) - U(tau(gcoords, -1))) * K(gcoords) / (2.0 * dh)
                 + Phi_prev(gcoords) / dt)
                / ((1.0 / dt) + (sigma(gcoords) / (2.0 * dh)));

            set_Phi(gcoords, phi_value);
            set_Psi(gcoords, psi_value);
        }

        result += 2 * (K(tau(gcoords, +1)) - K(tau(gcoords, -1)))
                    * (U(tau(gcoords, +1)) - U(tau(gcoords, -1))) * K(gcoords) / (2 * dh)
                + (U(tau(gcoords, +1)) + U(tau(gcoords, -1))) * K(gcoords) * K(gcoords) / (dh * dh)
                + PML;
        constants += K(gcoords) * K(gcoords) * 2.0 / (dh * dh);
    }

    result /= constants;
    return result;
}

__device__ bool is_red(const Coords gcoords) {
    const int_t i = gcoords.x;
    const int_t j = gcoords.y;
    const int_t k = gcoords.z;

    return (i + j + k) % 2 == 1;
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
    const Coords gcoords = { .x = i, .y = j, .z = k };

    if(!in_bounds(gcoords, dimensions))
        return;

    if(is_red(gcoords))
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
    const Coords gcoords = { .x = i, .y = j, .z = k };

    if(!in_bounds(gcoords, dimensions))
        return;

    if(!is_red(gcoords))
        U(gcoords) =
            gauss_seidel(U, U_prev, U_prev_prev, Psi, Psi_prev, Phi, Phi_prev, dimensions, gcoords);
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

    const int_t k = Nz / 2 + padding;
    for(int j = 0; j < Ny + 2 * padding; j++) {
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

__global__ void show_sigma(Dimensions dimensions, real_t *U, real_t *U_prev) {
    const int_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const int_t k = blockIdx.z * blockDim.z + threadIdx.z;
    const Coords gcoords = { .x = i, .y = j, .z = k };

    U(gcoords) = U_prev(gcoords) = get_sigma(gcoords, dimensions, X);
}

extern "C" int simulate_wave(const simulation_parameters p) {
    const real_t dt = p.dt;
    const int_t max_iteration = p.max_iter;
    const int_t snapshot_freq = p.snapshot_freq;

    Dimensions dimensions = p.dimensions;

    const real_t sim_Lx = p.sim_Lx;
    const real_t sim_Ly = p.sim_Ly;
    const real_t sim_Lz = p.sim_Lz;

    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;
    const real_t dx = dimensions.dh[X];
    const real_t dy = dimensions.dh[Y];
    const real_t dz = dimensions.dh[Z];

    printf("dx = %.4f, dy = %.4f, dz = %.4f\n", dx, dy, dz);

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
    real_t *U_prev_prev = allocate_domain(dimensions);

    for(int_t iteration = 0; iteration < max_iteration; iteration++) {
        if((iteration % snapshot_freq) == 0) {
            printf("iteration %d/%d\n", iteration, max_iteration);
            cudaDeviceSynchronize();
            domain_save(U, dimensions);
            // Phi_save(Phi, dimensions);
            // Psi_save(Psi, dimensions);
        }

        int block_x = 4;
        int block_y = 4;
        int block_z = 4;
        dim3 block(block_x, block_y, block_z);
        dim3 grid((Nx + 2 * padding + block_x - 1) / block_x,
                  (Ny + 2 * padding + block_y - 1) / block_y,
                  (Nz + 2 * padding + block_z - 1) / block_z);

        emit_source<<<grid, block>>>(U_prev, dimensions, iteration * dt);

        for(size_t iter = 0; iter < 2; iter++) {
            gauss_seidel_red<<<grid, block>>>(U,
                                              U_prev,
                                              U_prev_prev,
                                              Psi,
                                              Psi_prev,
                                              Phi,
                                              Phi_prev,
                                              dimensions);
            gauss_seidel_black<<<grid, block>>>(U,
                                                U_prev,
                                                U_prev_prev,
                                                Psi,
                                                Psi_prev,
                                                Phi,
                                                Phi_prev,
                                                dimensions);

            cudaError_t err = cudaGetLastError();
            if(err != cudaSuccess) {
                printf("cuda kernel error: %s\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
        }

        // show_sigma<<<grid, block>>>(dimensions, U, U_prev);

        move_buffer_window(&U, &U_prev, &U_prev_prev);
        swap_aux_variables(&Psi, &Psi_prev);
        swap_aux_variables(&Phi, &Phi_prev);
    }

    free_domain(U);
    free_domain(U_prev);
    free_domain(U_prev_prev);

    free_pml_variables(Psi);
    free_pml_variables(Phi);
    free_pml_variables(Psi_prev);
    free_pml_variables(Phi_prev);

    return 0;
}

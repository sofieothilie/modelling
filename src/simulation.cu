#include "simulation.h"
#include <stdio.h>

typedef enum { BOTTOM, SIDE, FRONT } Side;
#define N_SIDES (3)

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

typedef struct {
    // Indexed with Side
    real_t *buf[N_SIDES];
} PML_variable_XYZ;

typedef enum { X, Y, Z } Component;
#define N_COMPONENTS (3)

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
    // Indexed with Component
    PML_variable_XYZ var[N_COMPONENTS];
} PML_variable;

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

int_t get_side_size(const Dimensions dimensions, const Side side) {
    int_t Nx = dimensions.Nx;
    int_t Ny = dimensions.Ny;
    int_t Nz = dimensions.Nz;
    int_t padding = dimensions.padding;

    switch(side) {
        case BOTTOM:
            return (Nx + padding) * (Ny + padding) * padding;
        case SIDE:
            return padding * (Ny + padding) * Nz;
        case FRONT:
            return Nx * padding * Nz;
    }
}

PML_variable_XYZ allocate_pml_variables_XYZ(const Dimensions dimensions) {
    real_t *buf[N_SIDES] = { NULL };

    for(Side side = BOTTOM; side < N_SIDES; side++) {
        int_t size = get_side_size(dimensions, side);

        cudaErrorCheck(cudaMalloc(&(buf[side]), size * sizeof(real_t)));
        cudaErrorCheck(cudaMemset(buf[side], 0, size));
    }

    return { .buf = { buf[BOTTOM], buf[SIDE], buf[FRONT] } };
}

PML_variable allocate_pml_variables(const Dimensions dimensions) {
    PML_variable_XYZ var[N_COMPONENTS] = { NULL };

    for(Component component = X; component < N_COMPONENTS; component++) {
        var[component] = allocate_pml_variables_XYZ(dimensions);
    }

    return { .var = { var[X], var[Y], var[Z] } };
}

void free_pml_variables_XYZ(PML_variable_XYZ vars) {
    for(Side side = BOTTOM; side < N_SIDES; side++) {
        cudaErrorCheck(cudaFree(vars.buf[side]));
    }
}

void free_pml_variables(PML_variable vars) {
    for(Component component = X; component < N_COMPONENTS; component++) {
        free_pml_variables_XYZ(vars.var[component]);
    }
}

int_t get_domain_size(const Dimensions dimensions) {
    int_t Nx = dimensions.Nx;
    int_t Ny = dimensions.Ny;
    int_t Nz = dimensions.Nz;
    int_t padding = dimensions.padding;

    return (Nx + padding) * (Ny + padding) * (Nz + padding);
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

__host__ __device__ int_t per(const int_t a, const int_t m) {
    return (a + m) % m;
}

__host__ __device__ int_t gcoords_to_index(const Coords gcoords, const Dimensions dimensions) {
    const int_t i = gcoords.x;
    const int_t j = gcoords.y;
    const int_t k = gcoords.z;

    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;

    return per(i, Nx + padding) * (Ny + padding) * (Nz + padding)
         + per(j, Ny + padding) * (Nz + padding) + per(k, Nz + padding);
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

    const Coords gcoords = { 4 * Nx / 5, 4 * Ny / 5, Nz / 2 };
    const double freq = 1.0e6; // 1MHz
    if(i == gcoords.x && j == gcoords.y && k == gcoords.z) {
        if(t * freq < 1.0) {
            U(gcoords) = sin(2 * M_PI * t * freq);
        }
    }
}

dim3 get_pml_grid(Dimensions dimensions, dim3 block, Side side) {
    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;
    const int_t block_x = block.x;
    const int_t block_y = block.y;
    const int_t block_z = block.z;

    switch(side) {
        case BOTTOM:
            return dim3((Nx + padding + block_x - 1) / block_x,
                        (Ny + padding + block_y - 1) / block_y,
                        (padding + block_z - 1) / block_z);
        case SIDE:
            return dim3((padding + block_x - 1) / block_x,
                        (Ny + padding + block_y - 1) / block_y,
                        (Nz + block_z - 1) / block_z);
        case FRONT:
            return dim3((Nx + block_x - 1) / block_x,
                        (padding + block_y - 1) / block_y,
                        (Nz + block_z - 1) / block_z);
    }
}

__host__ __device__ bool in_physical_domain(const Coords gcoords, const Dimensions dimensions) {
    const int_t i = gcoords.x;
    const int_t j = gcoords.y;
    const int_t k = gcoords.z;

    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;

    if(i < 0 || j < 0 || k < 0)
        return false;

    if(i < Nx && j < Ny && k < Nz)
        return true;

    return false;
}

__device__ bool out_of_bounds(const Coords gcoords, const Dimensions dimensions) {
    const int_t i = gcoords.x;
    const int_t j = gcoords.y;
    const int_t k = gcoords.z;

    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;

    if(i < 0 || j < 0 || k < 0)
        return true;

    if(i >= Nx + padding || j >= Ny + padding || k > Nz + padding)
        return true;

    return false;
}

__device__ bool in_PML(const Coords gcoords, const Dimensions dimensions) {
    const int_t i = gcoords.x;
    const int_t j = gcoords.y;
    const int_t k = gcoords.z;

    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;

    if(i < 0 || j < 0 || k < 0)
        return false;

    if(i < Nx && j < Ny && k < Nz)
        return false;

    if(i >= Nx + padding || j >= Ny + padding || k > Nz + padding)
        return false;

    return true;
}

__host__ __device__ Side get_side(const Coords gcoords, const Dimensions dimensions) {
    const int_t i = gcoords.x;
    const int_t j = gcoords.y;
    const int_t k = gcoords.z;

    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;

    if(k >= Nz) {
        return BOTTOM;
    } else if(i >= Nx) {
        return SIDE;
    } else if(j >= Ny) {
        return FRONT;
    } else {
        printf("Called `get_side` outside the PML (%d %d %d)\n", i, j, k);
        return BOTTOM;
    }
}

__device__ Coords lcoords_to_gcoords(const Coords lcoords, const Dimensions dimensions, Side side) {
    const int_t i = lcoords.x;
    const int_t j = lcoords.y;
    const int_t k = lcoords.z;

    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;

    switch(side) {
        case BOTTOM:
            return Coords { i, j, k + Nz };
        case SIDE:
            return Coords { i + Nx, j, k };
        case FRONT:
            return Coords { i, j + Ny, k };
    }
}

__host__ __device__ Coords gcoords_to_lcoords(const Coords gcoords,
                                              const Dimensions dimensions,
                                              const Side side) {
    const int_t i = gcoords.x;
    const int_t j = gcoords.y;
    const int_t k = gcoords.z;

    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;

    if(in_physical_domain(gcoords, dimensions)) {
        printf("Called gcoords_to_lcoords from phisical domain (%d, %d, %d)\n", i, j, k);
    }

    switch(side) {
        case BOTTOM:
            return { .x = i, .y = j, .z = k - Nz };
        case SIDE:
            return { .x = i - Nx, .y = j, .z = k };
        case FRONT:
            return { .x = i, .y = j - Ny, .z = k };
    }
}

__device__ Coords tau_shift(const Coords gcoords, const int_t shift, const Component component) {
    const int_t i = gcoords.x;
    const int_t j = gcoords.y;
    const int_t k = gcoords.z;

    switch(component) {
        case X:
            return Coords { i + shift, j, k };
        case Y:
            return Coords { i, j + shift, k };
        case Z:
            return Coords { i, j, k + shift };
    }
}

__device__ real_t get_K(const Coords gcoords, const Dimensions dimensions) {
    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;

    const int_t i = per(gcoords.x, Nx + padding);
    const int_t j = per(gcoords.y, Ny + padding);
    const int_t k = per(gcoords.z, Nz + padding);

    const real_t WATER_K = 1500.0;
    const real_t PLASTIC_K = 2270.0;

    return WATER_K;

    if(i < Nx / 2 && i > Nx / 6)
        return PLASTIC_K;

    return WATER_K;
}

__device__ real_t get_sigma(const Coords gcoords,
                            const Dimensions dimensions,
                            const Component component) {
    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;
    const int_t dh = dimensions.dh[component];

    const int_t i = per(gcoords.x, Nx + padding);
    const int_t j = per(gcoords.y, Ny + padding);
    const int_t k = per(gcoords.z, Nz + padding);

    const real_t SIGMA = 2.0;

    if(in_physical_domain(gcoords, dimensions))
        return 0.0;
    return SIGMA;
}

__host__ __device__ int_t lcoords_to_index(const Coords lcoords,
                                           const Dimensions dimensions,
                                           const Side side) {
    const int_t i = lcoords.x;
    const int_t j = lcoords.y;
    const int_t k = lcoords.z;

    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;

    switch(side) {
        case BOTTOM:
            return i * (Ny + padding) * padding + j * padding + k;
        case SIDE:
            return i * (Ny + padding) * Nz + j * Nz + k;
        case FRONT:
            return i * padding * Nz + j * Nz + k;
    }
}

__device__ real_t get_PML_var(const PML_variable_XYZ var,
                              const Coords gcoords,
                              const Dimensions dimensions) {
    if(!in_PML(gcoords, dimensions))
        return 0.0;

    const Side side = get_side(gcoords, dimensions);
    const Coords lcoords = gcoords_to_lcoords(gcoords, dimensions, side);

    return var.buf[side][lcoords_to_index(lcoords, dimensions, side)];
}

__device__ void set_PML_var(const PML_variable_XYZ var,
                            const real_t value,
                            const Coords gcoords,
                            const Dimensions dimensions) {
    if(!in_PML(gcoords, dimensions))
        return;

    const Side side = get_side(gcoords, dimensions);
    const Coords lcoords = gcoords_to_lcoords(gcoords, dimensions, side);

    var.buf[side][lcoords_to_index(lcoords, dimensions, side)] = value;
}

#define tau(coords, shift) (tau_shift(coords, shift, component))
#define K(gcoords) (get_K(gcoords, dimensions))
#define sigma(gcoords) (get_sigma(gcoords, dimensions, component))
#define Psi(gcoords) (get_PML_var(Psi.var[component], gcoords, dimensions))
#define Psi_prev(gcoords) (get_PML_var(Psi_prev.var[component], gcoords, dimensions))
#define Phi(gcoords) (get_PML_var(Phi.var[component], gcoords, dimensions))
#define Phi_prev(gcoords) (get_PML_var(Phi_prev.var[component], gcoords, dimensions))
#define set_Psi(gcoords, value) (set_PML_var(Psi.var[component], value, gcoords, dimensions))
#define set_Phi(gcoords, value) (set_PML_var(Phi.var[component], value, gcoords, dimensions))

void move_buffer_window(real_t **const U, real_t **const U_prev, real_t **const U_prev_prev) {
    real_t *const temp = *U_prev_prev;
    *U_prev_prev = *U_prev;
    *U_prev = *U;
    *U = temp;
}

void swap_aux_variables(const PML_variable *u, const PML_variable *v) {
    const PML_variable *const temp = u;
    u = v;
    v = temp;
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

    if(i >= Nx + padding || j >= Ny + padding || k >= Nz + padding)
        return false;

    return true;
}

__device__ real_t gauss_seidel(const real_t *const U,
                               const real_t *const U_prev,
                               const real_t *const U_prev_prev,
                               const PML_variable Psi,
                               const PML_variable Psi_prev,
                               const PML_variable Phi,
                               const PML_variable Phi_prev,
                               const Dimensions dimensions,
                               const Coords gcoords) {
    const real_t dt = dimensions.dt;

    real_t result = (2.0 * U_prev(gcoords) - U_prev_prev(gcoords)) / (dt * dt);
    real_t constants = 1 / (dt * dt);
    for(Component component = X; component < N_COMPONENTS; component++) {
        const real_t dh = dimensions.dh[component];
        real_t PML = 0.0;
        if(!in_physical_domain(gcoords, dimensions)) {
            PML += K(gcoords) * K(gcoords)
                 * (sigma(gcoords) * Psi(tau(gcoords, +1))
                    - sigma(tau(gcoords, -1)) * Phi(tau(gcoords, -1)))
                 / (dh * dh);

            const real_t phi_value =
                (-Phi(tau(gcoords, -1)) * sigma(tau(gcoords, -1)) * K(gcoords) / (2.0 * dh)
                 - (U(tau(gcoords, +1)) - U(tau(gcoords, -1))) * K(gcoords) / (2.0 * dh)
                 + Phi_prev(gcoords) / dt)
                / ((1.0 / dt) + (sigma(gcoords) / (2.0 * dh)));

            const real_t psi_value =
                (-Psi(tau(gcoords, +1)) * sigma(gcoords) * K(gcoords) / (2.0 * dh)
                 - (U(tau(gcoords, +1)) - U(tau(gcoords, -1))) * K(gcoords) / (2.0 * dh)
                 + Psi_prev(gcoords) / dt)
                / ((1.0 / dt) + (sigma(tau(gcoords, -1)) / (2.0 * dh)));

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
                                 const PML_variable Psi,
                                 const PML_variable Psi_prev,
                                 const PML_variable Phi,
                                 const PML_variable Phi_prev,
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
                                   const PML_variable Psi,
                                   const PML_variable Psi_prev,
                                   const PML_variable Phi,
                                   const PML_variable Phi_prev,
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

    const int_t k = Nz / 2;
    for(int j = 0; j < Ny + padding; j++) {
        int i;
        for(i = 0; i < Nx + padding - 1; i++) {
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

void Phi_save(const PML_variable d_buffer, const Dimensions dimensions) {
    static int_t iter = 0;

    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;

    for(Component component = X; component < N_COMPONENTS; component++) {
        real_t *h_buffer[N_SIDES];
        for(Side side = BOTTOM; side < N_SIDES; side++) {
            const int_t size = get_side_size(dimensions, side);
            h_buffer[side] = (real_t *) malloc(sizeof(real_t) * size);
            cudaErrorCheck(cudaMemcpy(h_buffer[side],
                                      d_buffer.var[component].buf[side],
                                      sizeof(real_t) * size,
                                      cudaMemcpyDeviceToHost));
        }

        char filename[256];
        sprintf(filename, "side_data/phi_%d_%.5d.dat", component, iter);
        FILE *const out = fopen(filename, "w");
        if(!out) {
            fprintf(stderr, "Could not open file '%s'!\n", filename);
            exit(EXIT_FAILURE);
        }

        const int_t k = Nz / 2;
        for(int j = 0; j < Ny + padding; j++) {
            int i;
            for(i = 0; i < Nx + padding - 1; i++) {
                real_t value = 0.0;
                const Coords gcoords = { .x = i, .y = j, .z = k };
                if(!in_physical_domain(gcoords, dimensions)) {
                    const Side side = get_side(gcoords, dimensions);
                    const Coords lcoords = gcoords_to_lcoords(gcoords, dimensions, side);
                    value = h_buffer[side][lcoords_to_index(lcoords, dimensions, side)];
                }
                const int w = fprintf(out, "%.16lf ", value);
                if(w < 0)
                    printf("could not write all\n");
            }
            real_t value = 0.0;
            const Coords gcoords = { .x = i, .y = j, .z = k };
            if(!in_physical_domain(gcoords, dimensions)) {
                const Side side = get_side(gcoords, dimensions);
                const Coords lcoords = gcoords_to_lcoords(gcoords, dimensions, side);
                value = h_buffer[side][lcoords_to_index(lcoords, dimensions, side)];
            }
            const int w = fprintf(out, "%.16lf\n", value);
            if(w < 0)
                printf("could not write all\n");
        }

        for(Side side = BOTTOM; side < N_SIDES; side++) {
            free(h_buffer[side]);
        }
        fclose(out);
    }
    iter++;
}

void Psi_save(const PML_variable d_buffer, const Dimensions dimensions) {
    static int_t iter = 0;

    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;

    for(Component component = X; component < N_COMPONENTS; component++) {
        real_t *h_buffer[N_SIDES];
        for(Side side = BOTTOM; side < N_SIDES; side++) {
            const int_t size = get_side_size(dimensions, side);
            h_buffer[side] = (real_t *) malloc(sizeof(real_t) * size);
            cudaErrorCheck(cudaMemcpy(h_buffer[side],
                                      d_buffer.var[component].buf[side],
                                      sizeof(real_t) * size,
                                      cudaMemcpyDeviceToHost));
        }

        char filename[256];
        sprintf(filename, "side_data/psi_%d_%.5d.dat", component, iter);
        FILE *const out = fopen(filename, "w");
        if(!out) {
            fprintf(stderr, "Could not open file '%s'!\n", filename);
            exit(EXIT_FAILURE);
        }

        const int_t k = Nz / 2;
        for(int j = 0; j < Ny + padding; j++) {
            int i;
            for(i = 0; i < Nx + padding - 1; i++) {
                real_t value = 0.0;
                const Coords gcoords = { .x = i, .y = j, .z = k };
                if(!in_physical_domain(gcoords, dimensions)) {
                    const Side side = get_side(gcoords, dimensions);
                    const Coords lcoords = gcoords_to_lcoords(gcoords, dimensions, side);
                    value = h_buffer[side][lcoords_to_index(lcoords, dimensions, side)];
                }
                const int w = fprintf(out, "%.16lf ", value);
                if(w < 0)
                    printf("could not write all\n");
            }
            real_t value = 0.0;
            const Coords gcoords = { .x = i, .y = j, .z = k };
            if(!in_physical_domain(gcoords, dimensions)) {
                const Side side = get_side(gcoords, dimensions);
                const Coords lcoords = gcoords_to_lcoords(gcoords, dimensions, side);
                value = h_buffer[side][lcoords_to_index(lcoords, dimensions, side)];
            }
            const int w = fprintf(out, "%.16lf\n", value);
            if(w < 0)
                printf("could not write all\n");
        }

        for(Side side = BOTTOM; side < N_SIDES; side++) {
            free(h_buffer[side]);
        }
        fclose(out);
    }
    iter++;
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

    PML_variable Psi = allocate_pml_variables(dimensions);
    PML_variable Phi = allocate_pml_variables(dimensions);
    PML_variable Psi_prev = allocate_pml_variables(dimensions);
    PML_variable Phi_prev = allocate_pml_variables(dimensions);

    real_t *U = allocate_domain(dimensions);
    real_t *U_prev = allocate_domain(dimensions);
    real_t *U_prev_prev = allocate_domain(dimensions);

    for(int_t iteration = 0; iteration < max_iteration; iteration++) {
        if((iteration % snapshot_freq) == 0) {
            printf("iteration %d/%d\n", iteration, max_iteration);
            cudaDeviceSynchronize();
            domain_save(U, dimensions);
            Phi_save(Phi, dimensions);
            Psi_save(Psi, dimensions);
        }

        int block_x = 8;
        int block_y = 8;
        int block_z = 8;
        dim3 block(block_x, block_y, block_z);
        dim3 grid((Nx + padding + block_x - 1) / block_x,
                  (Ny + padding + block_y - 1) / block_y,
                  (Nz + padding + block_z - 1) / block_z);

        emit_source<<<grid, block>>>(U_prev, dimensions, iteration * dt);

        for(size_t iter = 0; iter < 5; iter++) {
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
        }

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

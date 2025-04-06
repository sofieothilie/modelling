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

int_t get_domain_size(const int_t Nx, const int_t Ny, const int_t Nz, const int_t padding) {
    const int_t domain_size = (Nx + padding) * (Ny + padding) * (Nz + padding);
    return domain_size;
}

int_t get_PML_size(const Dimensions d, const Side s) {
    switch(s) {
        case BOTTOM:
            return (d.Nx + d.padding) * (d.Ny + d.padding) * d.padding;
        case SIDE:
            return (d.Ny + d.padding) * d.Nz * d.padding;
        case FRONT:
            return d.Nx * d.Nz * d.padding;
    }
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

    return (Nx * padding) * (Ny + padding) * (Nz + padding);
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

void domain_save(const real_t *const d_buffer, const Dimensions dimensions) {
    static int_t iter = 0;

    int_t Nx = dimensions.Nx;
    int_t Ny = dimensions.Ny;
    int_t Nz = dimensions.Nz;
    int_t padding = dimensions.padding;

    int_t size = get_domain_size(dimensions);
    real_t *const h_buffer = (real_t *) malloc(size * sizeof(real_t));
    cudaErrorCheck(cudaMemcpy(h_buffer, d_buffer, sizeof(real_t) * size, cudaMemcpyDeviceToHost));

    char filename[256];
    sprintf(filename, "wave_data/%.5d.dat", iter);
    FILE *out = fopen(filename, "w");
    if(!out) {
        fprintf(stderr, "Could not open file '%s'!\n", filename);
        exit(EXIT_FAILURE);
    }

    int_t k = Nz / 2;
    for(int j = 0; j < Ny + padding; j++) {
        int i;
        for(i = 0; i < Nx + padding - 1; i++) {
            int w =
                fprintf(out,
                        "%.16lf ",
                        (h_buffer[i * ((Ny + padding) * (Nz + padding)) + j * (Nz + padding) + k])
                            / 1.47183118e-05);
            if(w < 0)
                printf("could not write all\n");
        }
        int w = fprintf(out,
                        "%.16lf\n",
                        (h_buffer[i * ((Ny + padding) * (Nz + padding)) + j * (Nz + padding) + k])
                            / 1.47183118e-05);
        if(w < 0)
            printf("could not write all\n");
    }

    free(h_buffer);
    fclose(out);
    iter++;
}

__device__ int_t per(const int_t a, const int_t m) {
    return (a + m) % m;
}

__device__ int_t gcoords_to_index(const Coords gcoords, const Dimensions dimensions) {
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

    int sine_x = Nx / 4;
    double freq = 1.0e6; // 1MHz

    if(i == sine_x && j == Ny / 2 && k == Nz / 2) {
        if(t * freq < 1.0) {
            Coords gcoords = { sine_x, Ny / 2, Nz / 2 };
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

    dim3 pml_bottom_gridSize((Nx + padding + block_x - 1) / block_x,
                             (Ny + padding + block_y - 1) / block_y,
                             (padding + block_z - 1) / block_z);
    dim3 pml_side_gridSize((padding + block_x - 1) / block_x,
                           (Ny + padding + block_y - 1) / block_y,
                           (Nz + block_z - 1) / block_z);
    dim3 pml_front_gridSize((Nx + block_x - 1) / block_x,
                            (padding + block_y - 1) / block_y,
                            (Nz + block_z - 1) / block_z);
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

__device__ bool in_PML(const Coords lcoords, const Dimensions dimensions, const Side side) {
    const int_t i = lcoords.x;
    const int_t j = lcoords.y;
    const int_t k = lcoords.z;

    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;

    if(i < 0 || j < 0 || k < 0)
        return false;

    switch(side) {
        case BOTTOM:
            if(i < Nx + padding && j < Ny + padding && k < padding)
                return true;
            break;
        case SIDE:
            if(i < padding && j < Ny + padding && k < Nz)
                return true;
            break;
        case FRONT:
            if(i < Nx && j < padding && k < Nz)
                return true;
            break;
    }

    return false;
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

__device__ Coords gcoords_to_lcoords(const Coords gcoords,
                                     const Dimensions dimensions,
                                     const Side side) {
    const int_t i = gcoords.x;
    const int_t j = gcoords.y;
    const int_t k = gcoords.z;

    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;

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
    const int_t i = gcoords.x;
    const int_t j = gcoords.y;
    const int_t k = gcoords.z;

    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;

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
    const int_t i = gcoords.x;
    const int_t j = gcoords.y;
    const int_t k = gcoords.z;

    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t dt = dimensions.dt;

    const real_t SIGMA = 1.0;

    if(i < Nx && j < Ny && k < Nz)
        return 0.0;
    return SIGMA;
}

// TODO: Handle borders to different PML partitions
__device__ int_t lcoords_to_index(const Coords lcoords,
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
            return per(i, Nx + padding) * (Ny + padding) * padding + per(j, Nz + padding) * padding
                 + per(k, padding);
        case SIDE:
            return per(i, padding) + per(j, Ny + padding) * padding
                 + per(k, Nz) * (Ny + padding) * padding;
        case FRONT:
            return per(i, Nx) * padding + per(j, Nz) * padding * (Nx + padding) + per(k, padding);
    }
}

__device__ real_t get_PML_var(const PML_variable_XYZ var,
                              const Coords lcoords,
                              const Dimensions dimensions,
                              const Side side) {
    return var.buf[side][lcoords_to_index(lcoords, dimensions, side)];
}

__device__ void set_PML_var(const PML_variable_XYZ var,
                            const real_t value,
                            const Coords lcoords,
                            const Dimensions dimensions,
                            const Side side) {
    var.buf[side][lcoords_to_index(lcoords, dimensions, side)] = value;
}

#define tau(coords, shift) (tau_shift(coords, shift, component))
#define K(gcoords) (get_K(gcoords, dimensions))
#define sigma(gcoords) (get_sigma(gcoords, dimensions, component))
#define Psi(lcoords) (get_PML_var(Psi.var[component], lcoords, dimensions, side))
#define Phi(lcoords) (get_PML_var(Phi.var[component], lcoords, dimensions, side))
#define set_Psi(lcoords, value) (set_PML_var(Psi.var[component], value, lcoords, dimensions, side))
#define set_Phi(lcoords, value) (set_PML_var(Phi.var[component], value, lcoords, dimensions, side))

void move_buffer_window(real_t **const U, real_t **const U_prev, real_t **const U_prev_prev) {
    real_t *const temp = *U_prev_prev;
    *U_prev_prev = *U_prev;
    *U_prev = *U;
    *U = temp;
}

__global__ void visualize_Phi(real_t *const U,
                              const PML_variable Phi,
                              const Dimensions dimensions,
                              const Side side,
                              const Component component) {
    const int_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const int_t k = blockIdx.z * blockDim.z + threadIdx.z;
    const Coords lcoords = { .x = i, .y = j, .z = k };

    U(lcoords_to_gcoords(lcoords, dimensions, side)) = Phi(lcoords);
}

__global__ void set_random_values(real_t *const U, const Dimensions dimensions) {
    const int_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const int_t k = blockIdx.z * blockDim.z + threadIdx.z;
    const Coords gcoords = { .x = i, .y = j, .z = k };

    U(gcoords) = (i - 2 * j + k) % (i + j + k);
}

__global__ void visualize_Psi(real_t *const U,
                              const PML_variable Psi,
                              const Dimensions dimensions,
                              const Side side,
                              const Component component) {
    const int_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const int_t k = blockIdx.z * blockDim.z + threadIdx.z;
    const Coords lcoords = { .x = i, .y = j, .z = k };

    U(lcoords_to_gcoords(lcoords, dimensions, side)) = Psi(lcoords);
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
                               const PML_variable Phi,
                               const Dimensions dimensions,
                               const Coords gcoords) {
    const real_t dt = dimensions.dt;

    real_t result = 2.0 * U_prev(gcoords) - U_prev_prev(gcoords);
    for(Component component = X; component < N_COMPONENTS; component++) {
        const real_t dh = dimensions.dh[component];
        real_t PML = 0.0;
        for(Side side = BOTTOM; side < N_SIDES; side++) {
            const Coords lcoords = gcoords_to_lcoords(gcoords, dimensions, side);
            if(in_PML(lcoords, dimensions, side)) {
                PML += K(gcoords) * K(gcoords)
                     * (Phi(tau(lcoords, -1)) * sigma(tau(gcoords, -1))
                        - Psi(tau(lcoords, +1)) * sigma(gcoords))
                     / (dh * dh);

                const real_t phi_value =
                    Phi(lcoords)
                    + dt
                          * (-K(gcoords) * sigma(tau(gcoords, -1)) * Phi(tau(lcoords, -1))
                                 / (2.0 * dh)
                             - K(gcoords) * sigma(gcoords) * Phi(lcoords) / (2.0 * dh)
                             - K(gcoords) * (U(tau(gcoords, +1)) - U(tau(gcoords, -1)))
                                   / (2.0 * dh));

                const real_t psi_value =
                    Psi(lcoords)
                    + dt
                          * (-K(gcoords) * sigma(tau(gcoords, -1)) * Psi(lcoords) / (2.0 * dh)
                             - K(gcoords) * sigma(gcoords) * Psi(tau(lcoords, +1)) / (2.0 * dh)
                             - K(gcoords) * (U(tau(gcoords, +1)) - U(tau(gcoords, -1)))
                                   / (2.0 * dh));

                set_Phi(lcoords, phi_value);
                set_Psi(lcoords, psi_value);
            }
        }

        result += (dt * dt)
                * (2.0 * (-K(tau(gcoords, +1)) / (2.0 * dh) + K(tau(gcoords, -1)) / (2.0 * dh))
                       * (-U(tau(gcoords, +1)) / (2.0 * dh) + U(tau(gcoords, -1)) / (2.0 * dh))
                       * K(gcoords)
                   + (-2.0 * U(gcoords) / (dh * dh) + U(tau(gcoords, -1)) / (dh * dh)
                      + U(tau(gcoords, +1)) / (dh * dh))
                         * (K(gcoords) * K(gcoords))
                   + PML);
    }
    // -(d_Phi_x(i - 1, j, k) * sigma_x(i - 1, j, k) - d_Psi_x(i + 1, j, k) * sigma_x(i, j, k))
    //     / (d_dx * d_dx)
    // - (d_Phi_y(i, j - 1, k) * sigma_y(i, j - 1, k) - d_Psi_y(i, j + 1, k) * sigma_y(i, j, k))
    //       / (d_dy * d_dy)
    // - (d_Phi_z(i, j, k - 1) * sigma_z(i, j, k - 1) - d_Psi_z(i, j, k + 1) * sigma_z(i, j, k))
    //       / (d_dz * d_dz);

    return result;
}

__global__ void gauss_seidel_red(real_t *const U,
                                 const real_t *const U_prev,
                                 const real_t *const U_prev_prev,
                                 const PML_variable Psi,
                                 const PML_variable Phi,
                                 const Dimensions dimensions) {
    const int_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const int_t k = blockIdx.z * blockDim.z + threadIdx.z;
    const Coords gcoords = { .x = i, .y = j, .z = k };

    if(!in_bounds(gcoords, dimensions))
        return;

    if((i + j + k) % 2 == 0)
        U(gcoords) = gauss_seidel(U, U_prev, U_prev_prev, Psi, Phi, dimensions, gcoords);
}

__global__ void gauss_seidel_black(real_t *const U,
                                   const real_t *const U_prev,
                                   const real_t *const U_prev_prev,
                                   const PML_variable Psi,
                                   const PML_variable Phi,
                                   const Dimensions dimensions) {
    const int_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const int_t k = blockIdx.z * blockDim.z + threadIdx.z;
    const Coords gcoords = { .x = i, .y = j, .z = k };

    if(!in_bounds(gcoords, dimensions))
        return;

    if((i + j + k) % 2 == 1)
        U(gcoords) = gauss_seidel(U, U_prev, U_prev_prev, Psi, Phi, dimensions, gcoords);
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

    real_t *U = allocate_domain(dimensions);
    real_t *U_prev = allocate_domain(dimensions);
    real_t *U_prev_prev = allocate_domain(dimensions);

    for(int_t iteration = 0; iteration < max_iteration; iteration++) {
        if((iteration % snapshot_freq) == 0) {
            printf("iteration %d/%d\n", iteration, max_iteration);
            cudaDeviceSynchronize();
            domain_save(U, dimensions);
        }

        int block_x = 8;
        int block_y = 8;
        int block_z = 8;
        dim3 block(block_x, block_y, block_z);
        dim3 grid((Nx + padding + block_x - 1) / block_x,
                  (Ny + padding + block_y - 1) / block_y,
                  (Nz + padding + block_z - 1) / block_z);

        emit_source<<<grid, block>>>(U, dimensions, iteration * dt);

        for(size_t iter = 0; iter < 5; iter++) {
            gauss_seidel_red<<<grid, block>>>(U, U_prev, U_prev_prev, Psi, Phi, dimensions);
            gauss_seidel_black<<<grid, block>>>(U, U_prev, U_prev_prev, Psi, Phi, dimensions);
        }

        move_buffer_window(&U, &U_prev, &U_prev_prev);
    }

    free_domain(U);
    free_domain(U_prev);
    free_domain(U_prev_prev);

    free_pml_variables(Psi);
    free_pml_variables(Phi);

    return 0;
}

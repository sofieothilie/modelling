#include "simulation.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

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
#define V(gcoords) V[gcoords_to_index(gcoords, dimensions)]

__global__ void emit_source(real_t *const U,
                            const Dimensions dimensions,
                            const real_t value,
                            const Coords source_coords) {
    // const int_t i = blockIdx.x * blockDim.x + threadIdx.x;
    // const int_t j = blockIdx.y * blockDim.y + threadIdx.y;
    // const int_t k = blockIdx.z * blockDim.z + threadIdx.z;

    // const int_t Nx = dimensions.Nx;
    // const int_t Ny = dimensions.Ny;
    // const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;
    const real_t dh = dimensions.dh;

    const real_t wavelength = WATER_PARAMETERS.k / SRC_FREQUENCY;

    //phased array parameters
    const int n_source = 7;
    const int spacing = (int) (0.25 * wavelength / dh); // spacing in terms of cells, not distance

    // const double freq = 1.0e6; // 1MHz

    // // grid of sources over yz plane, at x = 5 maybe
    for(int n_i = 0; n_i < n_source; n_i++) {
        for(int n_j = 0; n_j < n_source; n_j++) {
            int idx_i = padding + source_coords.x - (n_source / 2 * spacing) + n_i * spacing;
            int idx_j = padding + source_coords.y - (n_source / 2 * spacing) + n_j * spacing;

            // double shift = 0;
            // (double) n_j / n_source * 2 * M_PI * 0.7;
            // double sine = sin(2 * M_PI * t * freq - shift);

            // printf("%\n",    (int)(0.005/dh));

            const Coords gcoords = { .x = idx_i, .y = idx_j, .z = padding + source_coords.z };

            U(gcoords) = value;
        }
    }

    // Coords gcoords = { i, j, k };

    // // const Coords emit_coords = { .x = padding + Nx / 2,
    // //                              .y = padding + Ny / 2,
    // //                              .z = padding + Nz / 2 };
    // // // const double freq = 1e3;
    // real_t x = (i - padding - Nx / 2) * dh;
    // real_t y = (j - padding - Ny / 2) * dh;
    // real_t z = (k - padding - Nz / 2) * dh;

    // real_t cx = 0, cy = 0;
    // if(k >= padding && k < padding + Nz && i >= padding && i < padding + Nx && j >= padding
    //    && j < padding + Ny) {
    //     real_t delta = ((x - cx) * (x - cx) + (y - cy) * (y - cy) + (z * z));
    //     U(gcoords) = exp(-4000000.0 * delta);
    // }
}

// void get_recv(const real_t *d_buffer, FILE *output, Dimensions dimensions) {
//     // static int_t iter = 0;

//     const int_t Nx = dimensions.Nx;
//     const int_t Ny = dimensions.Ny;
//     const int_t Nz = dimensions.Nz;
//     const int_t padding = dimensions.padding;
//     const int_t dh = dimensions.dh;

//     const int_t size = get_domain_size(dimensions);

//     FILE *recv_output = fopen(sensor_output_filename, "a");

//     real_t *const h_buffer = (real_t *) malloc(size * sizeof(real_t));
//     cudaErrorCheck(cudaMemcpy(h_buffer, d_buffer, sizeof(real_t) * size,
//     cudaMemcpyDeviceToHost));

//     Coords dst_coords = { .x = Nx / 2 + padding, .y = Ny / 2 + padding, .z = padding + 10 };

//     real_t at_dest = h_buffer[gcoords_to_index(dst_coords, dimensions)];

//     fwrite(&at_dest, sizeof(real_t), 1, output);

//     free(h_buffer);
//     fclose(recv_output);
// }

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

// will tell if sigma is on, depending on the component and the position
__device__ bool border_match_component(const Coords gcoords,
                                       const Dimensions dimensions,
                                       const Component component) {
    switch(component) {
        case X:
            return gcoords.x < dimensions.padding
                || gcoords.x >= dimensions.Nx + dimensions.padding;

        case Y:
            return gcoords.y < dimensions.padding
                || gcoords.y >= dimensions.Ny + dimensions.padding;

        case Z:
            return gcoords.z < dimensions.padding
                || gcoords.z >= dimensions.Nz + dimensions.padding;
        default:
            printf("error in border match component\n");
            return 0;
    }
}

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

__device__ bool
in_PML_directional(const Coords gcoords, const Dimensions dimensions, const Component component) {
    return in_PML(gcoords, dimensions) && border_match_component(gcoords, dimensions, component);
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

__device__ __host__ MediumParameters get_params(const Coords gcoords,
                                                const Dimensions dimensions,
                                                const double *model,
                                                const Position sensor) {
    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    // const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;
    const real_t dh = dimensions.dh;

    const real_t model_y_shift = sensor.y - (Ny + 2 * padding) * dh / 2.0;
    const real_t model_x_shift = sensor.x - (Nx + 2 * padding) * dh / 2.0;
    const real_t model_z_shift = -sensor.z; // shift the model back



    // thats ugly, but wall just before source
    if(gcoords.z < padding) {
        return WALL_PARAMETERS;
    }

    // x relative to the whole domain now. now the whole model is loaded. change this when only a
    // part is loaded maybe
    const real_t x = gcoords.x * dh + model_x_shift;
    const real_t y = gcoords.y * dh + model_y_shift;
    const real_t z = gcoords.z * dh + model_z_shift;

    // check horizontal bounds
    if(x < 0 || x >= MODEL_LX || y < 0 || y >= MODEL_LY) {
        return WATER_PARAMETERS;
    }
    // obtain vertical height of model at that point

    const int_t x_idx = x * MODEL_NX / MODEL_LX;
    const int_t y_idx = y * MODEL_NY / MODEL_LY;

    // TODO this might be the wrong side,
    const real_t model_bottom = MODEL_LZ + model[x_idx * MODEL_NY + y_idx];

    // const real_t air_limit = 0.056 + 0.02; // deepness of model (from utils function RTT) + 2cm of air

    if(z < 0 || z >= model_bottom) {
        // if(z < air_limit) {
        //     return AIR_PARAMETERS;
        // }
        return WATER_PARAMETERS;
    }

    return PLASTIC_PARAMETERS;
}

void check_model_for_non_null_values(const double *model) {
    for(int i = 0; i < MODEL_NX; i++) {
        for(int j = 0; j < MODEL_NY; j++) {
            double value = model[i * MODEL_NY + j];
            if(value != 0.0) {
                printf("Non-null value found at (%d, %d): %.10lf\n", i, j, value);
            }
        }
    }
    printf("finished looping through model\n");
    // printf("illegal  access output, wtf: %lf\n", model[MODEL_NX*MODEL_NY]);
}

__device__ real_t get_sigma(const Coords gcoords,
                            const Dimensions dimensions,
                            const Component component) {

    const real_t SIGMA = 2.0;

    if(in_PML_directional(gcoords, dimensions, component)) {
        return SIGMA;
    }

    return 0.0;
}

#define tau(shift) (tau_shift(gcoords, shift, component, dimensions))
#define K(gcoords) (get_params(gcoords, dimensions, model, sensor).k)
#define Rho(gcoords) (get_params(gcoords, dimensions, model, sensor).rho)
#define sigma(gcoords) (get_sigma(gcoords, dimensions, component))
#define Psi(S, gcoords) (get_PML_var(S.Psi, gcoords, component, dimensions))
#define Phi(S, gcoords) (get_PML_var(S.Phi, gcoords, component, dimensions))
#define set_Psi(S, value) (set_PML_var(S.Psi, value, gcoords, component, dimensions))
#define set_Phi(S, value) (set_PML_var(S.Phi, value, gcoords, component, dimensions))

__device__ Side get_side(const Coords gcoords,
                         const Dimensions dimensions,
                         const Component component) {
    const int_t i = gcoords.x;
    const int_t j = gcoords.y;
    const int_t k = gcoords.z;

    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;

    switch(component) {
        case Z:
            if(k < padding)
                return TOP;
            if(k >= padding + Nz)
                return BOTTOM;
        case Y:
            if(j < padding)
                return BACK;
            if(j >= padding + Ny)
                return FRONT;
        case X:
            if(i < padding)
                return LEFT;
            if(i >= padding + Nx)
                return RIGHT;
    }

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
            return Coords { .x = i, .y = j, .z = k };
        case RIGHT:
            return Coords { .x = i - Nx - padding, .y = j, .z = k };
        case BACK:
            return Coords { .x = i, .y = j, .z = k };
        case FRONT:
            return Coords { .x = i, .y = j - Ny - padding, .z = k };
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
    switch(side) { // canbe optimized  with gc to lc, only  one operation
        case TOP:
            return (i + 1) * (padding + 1) * (Ny + 2 * padding + 2) + (j + 1) * (padding + 1)
                 + (k + 1);
        case BOTTOM:
            return (i + 1) * (padding + 1) * (Ny + 2 * padding + 2) + (j + 1) * (padding + 1)
                 + (k); // no ghost cell before
        case LEFT:
            return (i + 1) * (Nz + 2 * padding + 2) * (Ny + 2 * padding + 2)
                 + (j + 1) * (Nz + 2 * padding + 2) + (k + 1);
        case RIGHT:
            return (i) * (Nz + 2 * padding + 2) * (Ny + 2 * padding + 2)
                 + (j + 1) * (Nz + 2 * padding + 2) + (k + 1);
        case BACK:
            return (i + 1) * (Nz + 2 * padding + 2) * (padding + 1)
                 + (j + 1) * (Nz + 2 * padding + 2) + (k + 1);
        case FRONT:
            return (i + 1) * (Nz + 2 * padding + 2) * (padding + 1) + (j) * (Nz + 2 * padding + 2)
                 + (k + 1);

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

    // 0. if not in PML domain, return 0, because sigma is 0 there anyways
    if(!in_PML_directional(gcoords, dimensions, component))
        return 0.0;

    // 1. retrieve side
    Side side = get_side(gcoords, dimensions, component);

    // 2. retrieve local coords in this side
    Coords lcoords = gcoords_to_lcoords(gcoords, dimensions, side);

    // 3. retrieve index of that lcoord in the side
    int_t index = lcoords_to_index(lcoords, dimensions, side);

    // 4. finally access the PML variable.
    return (var.side[side])[index];
}

__device__ void set_PML_var(PML_Variable var,
                            const real_t value,
                            const Coords gcoords,
                            const Component component,
                            const Dimensions dimensions) {

    // 0. if not in PML domain, something is wrong
    if(!in_PML_directional(gcoords, dimensions, component)) {
        printf("something is wrong lol\n");
        return;
    }

    // 1. retrieve side
    Side side = get_side(gcoords, dimensions, component);

    // 2. retrieve local coords in this side
    Coords lcoords = gcoords_to_lcoords(gcoords, dimensions, side);

    // 3. retrieve index of that lcoord in the side
    int_t index = lcoords_to_index(lcoords, dimensions, side);

    // 4. finally write in the PML variable.
    (var.side[side])[index] = value;
}

void shift_states(SimulationState *current, SimulationState *next) {
    SimulationState temp = *current;
    *current = *next;
    *next = temp;
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

// performs Out = A + m*B for a whole simulation state
// will not be optimized in  edge cases !
__global__ void vectorized_add_mult(SimulationState Out,
                                    SimulationState A,
                                    real_t m,
                                    SimulationState B,
                                    Dimensions dimensions) {
    const int_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const int_t k = blockIdx.z * blockDim.z + threadIdx.z;

    const Coords gcoords = { .x = i, .y = j, .z = k };

    if(!in_bounds(gcoords, dimensions)) {
        return;
    }

    // add U and V
    Out.U(gcoords) = A.U(gcoords) + m * B.U(gcoords);
    Out.V(gcoords) = A.V(gcoords) + m * B.V(gcoords);

    for(Component component = X; component < N_COMPONENTS; incComp(component)) {
        if(in_PML_directional(gcoords, dimensions, component)) {
            real_t new_phi = Phi(A, gcoords) + m * Phi(B, gcoords);
            set_Phi(Out, new_phi);
            real_t new_psi = Psi(A, gcoords) + m * Psi(B, gcoords);
            set_Psi(Out, new_psi);
        }
    }
}

// this is wrong, idk where
__global__ void vectorized_add_mult2(SimulationState Out,
                                     SimulationState A,
                                     real_t m,
                                     SimulationState B,
                                     real_t n,
                                     SimulationState C,
                                     Dimensions dimensions) {
    const int_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const int_t k = blockIdx.z * blockDim.z + threadIdx.z;

    const Coords gcoords = { .x = i, .y = j, .z = k };

    if(!in_bounds(gcoords, dimensions)) {
        return;
    }

    // add U and V
    Out.U(gcoords) = A.U(gcoords) + m * B.U(gcoords) + n * C.U(gcoords);
    Out.V(gcoords) = A.V(gcoords) + m * B.V(gcoords) + n * C.U(gcoords);

    for(Component component = X; component < N_COMPONENTS; incComp(component)) {
        if(in_PML_directional(gcoords, dimensions, component)) {
            real_t new_phi = Phi(A, gcoords) + m * Phi(B, gcoords) + n * Phi(C, gcoords);
            set_Phi(Out, new_phi);
            real_t new_psi = Psi(A, gcoords) + m * Psi(B, gcoords) + n * Psi(C, gcoords);
            set_Psi(Out, new_psi);
        }
    }
}

__global__ void
vectorized_mult(SimulationState Out, real_t m, SimulationState B, Dimensions dimensions) {
    const int_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const int_t k = blockIdx.z * blockDim.z + threadIdx.z;

    const Coords gcoords = { .x = i, .y = j, .z = k };

    if(!in_bounds(gcoords, dimensions)) {
        return;
    }

    // add U and V
    Out.U(gcoords) = m * B.U(gcoords);
    Out.V(gcoords) = m * B.V(gcoords);

    for(Component component = X; component < N_COMPONENTS; incComp(component)) {
        if(in_PML_directional(gcoords, dimensions, component)) {
            real_t new_phi = m * Phi(B, gcoords);
            set_Phi(Out, new_phi);
            real_t new_psi = m * Psi(B, gcoords);
            set_Psi(Out, new_psi);
        }
    }
}

__global__ void euler_step(SimulationState deriv,
                           const SimulationState state,
                           Dimensions dimensions,
                           const double *model,
                           const Position sensor) {
    // simply input the discretized side of my half-discretized equations
    const int_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const int_t k = blockIdx.z * blockDim.z + threadIdx.z;

    // const real_t dt = dimensions.dt;
    const real_t dh = dimensions.dh;

    // int_t Nx = dimensions.Nx;
    // int_t Ny = dimensions.Ny;
    // int_t Nz = dimensions.Nz;

    const Coords gcoords = { .x = i, .y = j, .z = k };

    if(!in_bounds(gcoords, dimensions)) {
        return;
    }

    // simply transfer v to du
    deriv.U(gcoords) = state.V(gcoords);

    // main computation, this is d²u/dv². first compute pml terms, then formula
    real_t dv = 0.0;
    for(Component component = X; component < N_COMPONENTS; incComp(component)) {
        //
        real_t pml1 = 0.0, pml2 = 0.0;

        if(in_PML_directional(gcoords, dimensions, component)) {
            pml1 = sigma(gcoords) * Psi(state, tau(+1));
            pml2 = -1 * sigma(tau(-1)) * Phi(state, tau(-1));

            // also update the PML while I'm here
            real_t dphi = Phi(state, gcoords)
                        - K(gcoords) / (2.0 * dh)
                              * (sigma(tau(-1)) * Phi(state, tau(-1))
                                 + sigma(gcoords) * Phi(state, gcoords))
                        - K(gcoords) / (2.0 * dh) * (state.U(tau(+1)) - state.U(tau(-1)));

            real_t dpsi = Psi(state, gcoords)
                        - K(gcoords) / (2.0 * dh)
                              * (sigma(tau(-1)) * Psi(state, gcoords)
                                 + sigma(gcoords) * Psi(state, tau(+1)))
                        - K(gcoords) / (2.0 * dh) * (state.U(tau(+1)) - state.U(tau(-1)));

            set_Phi(deriv, dphi);
            set_Psi(deriv, dpsi);
        }

        dv += K(gcoords) * K(gcoords) * Rho(gcoords) / (2.0 * dh * dh)
            * ((1.0 / Rho(tau(+1)) + 1.0 / Rho(gcoords))
                   * (state.U(tau(+1)) + pml1 - state.U(gcoords))
               - (1.0 / Rho(gcoords) + 1.0 / Rho(tau(-1)))
                     * (state.U(gcoords) - state.U(tau(-1)) - pml2));
    }

    deriv.V(gcoords) = dv;
}

// updates the value in current
// 4  euler-steps, 7 add-mult
__host__ void RK4_step(SimulationState current,
                       SimulationState tmp,
                       SimulationState K1,
                       SimulationState K2,
                       SimulationState K3,
                       SimulationState K4,
                       Dimensions dimensions) {

    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;

    const real_t dt = dimensions.dt;

    int block_x = 8;
    int block_y = 8;
    int block_z = 8;
    dim3 block(block_x, block_y, block_z);
    dim3 grid((Nx + 2 * padding + block_x - 1) / block_x,
              (Ny + 2 * padding + block_y - 1) / block_y,
              ((Nz + 2 * padding) + block_z - 1) / block_z);

    // tmp serves as containing the parameter I'll pass to f, so that I can do some adding and
    // multiplying before passing

    // 1. K1 = f(current)
    // euler_step<<<grid, block>>>(K1, current, dimensions);
    // vectorized_add_mult<<<grid, block>>>(current, K1, dt, K1, dimensions);

    // 2. K2 = f(current + dt * K1/2)
    vectorized_add_mult<<<grid, block>>>(tmp, current, dt / 2.0, K1, dimensions);
    // euler_step<<<grid, block>>>(K2, tmp, dimensions);

    // 3. K3 = f(current + dt * K2/2)
    vectorized_add_mult<<<grid, block>>>(tmp, current, dt / 2.0, K2, dimensions);
    // euler_step<<<grid, block>>>(K3, tmp, dimensions);

    // 4. K4 = f(current + dt * K4)
    vectorized_add_mult<<<grid, block>>>(tmp, current, dt, K3, dimensions);
    // euler_step<<<grid, block>>>(K4, tmp, dimensions);

    // 5. next = Y + dt*K1/6 + dt*K2/3 + dt*K3/3 + dt*K4/6
    vectorized_add_mult<<<grid, block>>>(current, current, dt / 6.0, K1, dimensions);
    vectorized_add_mult<<<grid, block>>>(current, current, dt / 3.0, K2, dimensions);
    vectorized_add_mult<<<grid, block>>>(current, current, dt / 3.0, K3, dimensions);
    vectorized_add_mult<<<grid, block>>>(current, current, dt / 6.0, K4, dimensions);
}

// profile this !  see how much euler step and the add-mult costs

// + combine kernels  into more complex ones is quite faster !

//!! can be done  with lower storage, but more operations !!!
// 4 euler-steps, 6 add-mult,  2 mult
__host__ void RK4_step_lowstorage(SimulationState current,
                                  SimulationState inter,
                                  SimulationState next,
                                  Dimensions dimensions,
                                  const double *model,
                                  const Position sensor) {
    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;

    const real_t dt = dimensions.dt;

    int block_x = 8;
    int block_y = 8;
    int block_z = 8;
    dim3 block(block_x, block_y, block_z);
    dim3 grid((Nx + 2 * padding + block_x - 1) / block_x,
              (Ny + 2 * padding + block_y - 1) / block_y,
              ((Nz + 2 * padding) + block_z - 1) / block_z);

    // 1. next <- f(current)
    euler_step<<<grid, block>>>(next, current, dimensions, model, sensor);

    // 2. inter <- f(next)
    euler_step<<<grid, block>>>(inter, next, dimensions, model, sensor);

    // 3. inter <- 1/4*dt² * inter +  1/2*dt*next  + current
    vectorized_mult<<<grid, block>>>(inter, 0.25 * dt * dt, inter, dimensions);
    vectorized_add_mult<<<grid, block>>>(inter, inter, 0.5 * dt, next, dimensions);
    vectorized_add_mult<<<grid, block>>>(inter, inter, 1.0, current, dimensions);

    // 4. next <- 1/3*dt*next + 1/3*current + 2/3*inter
    vectorized_mult<<<grid, block>>>(next, 1.0 / 3.0 * dt, next, dimensions);
    vectorized_add_mult<<<grid, block>>>(next, next, 1.0 / 3.0, current, dimensions);
    vectorized_add_mult<<<grid, block>>>(next, next, 2.0 / 3.0, inter, dimensions);

    // vectorized_add_mult2<<<grid, block>>>(next,
    //                                       next,
    //                                       1.0 / 3.0,
    //                                       current,
    //                                       2.0 / 3.0,
    //                                       inter,
    //                                       dimensions);

    // 4. current <- f(inter)
    euler_step<<<grid, block>>>(current, inter, dimensions, model, sensor);

    // 4.5. inter <- f(current) note this is not necessary, and could directly be added to the next
    // but I was too lazy to write a new kernel that adds instead of overwrite
    euler_step<<<grid, block>>>(inter, current, dimensions, model, sensor);

    // 5. next <- next + 1/3*dt*current + 1/6dt² * inter
    vectorized_add_mult<<<grid, block>>>(next, next, 1.0 / 3.0 * dt, current, dimensions);
    vectorized_add_mult<<<grid, block>>>(next, next, 1.0 / 6.0 * dt * dt, inter, dimensions);
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
    FILE *const out = fopen(filename, "wb");
    if(!out) {
        fprintf(stderr, "Could not open file '%s'!\n", filename);
        exit(EXIT_FAILURE);
    }

    const real_t smallest_wavelength = WATER_PARAMETERS.k / SRC_FREQUENCY;
    const real_t ppw = smallest_wavelength / dimensions.dh;

    const real_t saved_ppw = 2;

    const int step = (int) (ppw / saved_ppw);
    // printf("using saving step of size %d\n", step);

    const int_t i = Nx / 2 + padding;
    for(int k = padding; k < Nz + padding; k += step) {
        int j;
        for(j = padding; j < Ny + padding; j += step) {
            const Coords gcoords = { .x = i, .y = j, .z = k };
            const int w =
                fwrite(&h_buffer[gcoords_to_index(gcoords, dimensions)], sizeof(real_t), 1, out);
            if(w != 1)
                printf("could not write all\n");
        }
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

int traces(real_t **trace_buf) { //simulation_parameters p, real_t **sig_buf) {
    FILE *f = fopen("data/extracted/150kHz_Single_source_file_traces.bin", "rb");
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

    uint32_t n_traces, n_samples;
    float dt;
    fread(&n_traces, 4, 1, f);
    fread(&n_samples, 4, 1, f);
    fread(&dt, 4, 1, f);

    printf("n_traces=%u, n_samples=%u, dt=%g\n",
          n_traces,
          n_samples,
          dt);

    size_t rec_bytes = (n_samples) * sizeof(float);
    float *buf = (float *)malloc(rec_bytes);
    *trace_buf = (real_t *) calloc(n_samples, sizeof(real_t));

    if(!buf) {
        perror("Failed to allocate double_buffer");
        fclose(f);
        return -1;
    }

    for (size_t i=0; i< n_traces; ++i) {
        float s_x, s_y, g_x, g_y;
        fread(&s_x, 4, 1, f);
        fread(&s_y, 4, 1, f);
        fread(&g_x, 4, 1, f);
        fread(&g_y, 4, 1, f);
        printf("Original source position: x=%g, y=%g, receiver position: x= %g, y=%g\n", s_x, s_y, g_x, g_y);
        // p.sensor.x = s_x;
        // p.sensor.y = s_y;
        // p.receiver.x = g_x;
        // p.receiver.y = g_y;
        size_t trace = fread(buf, 1, rec_bytes, f);
        if (trace != rec_bytes) perror("Failed to read full trace");

        for(size_t i = 0; i < n_samples; i++) {
            (*trace_buf)[i] = (real_t) buf[i];
        }
        break; // only first trace for now

    }

    free(buf);
    fclose(f);
    return n_samples;
}

__global__ void show_sigma(Dimensions dimensions,
                           real_t *U,
                           real_t *U_prev,
                           const double *model,
                           const Position sensor) {
    const int_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const int_t j = blockIdx.y * blockDim.y + threadIdx.y;
    const int_t k = blockIdx.z * blockDim.z + threadIdx.z;
    const Coords gcoords = { .x = i, .y = j, .z = k };

    U(gcoords) = U_prev[gcoords_to_index(gcoords, dimensions)] =
        get_params(gcoords, dimensions, model, sensor).k;
}

void show_model(Dimensions dimensions, double *model, const Position sensor) {
    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;

    char filename[256];
    sprintf(filename, "out_model.dat");
    FILE *out = fopen(filename, "wb");
    if(!out) {
        printf("[ERROR] File pointer is NULL!\n");
        exit(EXIT_FAILURE);
    }

    for(int k = 0; k < Nz + 2 * padding; k++) {
        for(int j = 0; j < Ny + 2 * padding; j++) {
            for(int i = 0; i < Nx + 2 * padding; i++) {
                const Coords gcoords = { i, j, k };
                const real_t k_ = K(gcoords);
                unsigned char in_model = (k_ == PLASTIC_PARAMETERS.k) ? 1 : 0;
                // printf("%d\n",(int)in_model);
                fwrite(&in_model, sizeof(unsigned char), 1, out);
            }
        }
    }
    printf("written to file\n");

    fclose(out);
}

__global__ void get_sensor_value(const SimulationState s,
                                 real_t *sensor_value,
                                 const Dimensions dimensions,
                                 Coords receiver_coords[]) {
    // const int_t Nx = dimensions.Nx;
    // const int_t Ny = dimensions.Ny;
    // const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;
    // const real_t dh = dimensions.dh;

    for(int_t idx = 0; idx < N_RECEIVERS; idx++) {

        int_t i = receiver_coords[idx].x;
        int_t j = receiver_coords[idx].y;
        int_t k = receiver_coords[idx].z;
        Coords padded_coords = { i + padding, j + padding, k + padding };
        // printf("reading at (%d,%d,%d)\n", i,j,k);
        if(in_bounds(padded_coords, dimensions))
            sensor_value[idx] = s.U(padded_coords);
    }
}

void append_value_to_files(const char *filenames[], real_t *d_value) {
    // 1. copy back value to cpu
    real_t h_value[N_RECEIVERS] = { 0 };
    cudaErrorCheck(
        cudaMemcpy(&h_value, d_value, N_RECEIVERS * sizeof(real_t), cudaMemcpyDeviceToHost));

    for(int_t i = 0; i < N_RECEIVERS; i++) {
        FILE *output_file = fopen(filenames[i], "a");
        fwrite(&(h_value[i]), sizeof(real_t), 1, output_file);
        fclose(output_file);
    }
}

extern "C" int simulate_wave(const simulation_parameters p) {
    const real_t dt = p.dt;
    const int_t max_iteration = p.max_iter;
    const int_t snapshot_freq = p.snapshot_freq;

    Dimensions dimensions = p.dimensions;

    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;
    const real_t dh = dimensions.dh;
    // const real_t dh = dimensions.dh;

    if(!init_cuda()) {
        fprintf(stderr, "Could not initialize CUDA\n");
        exit(EXIT_FAILURE);
    }

    // SimulationState currentState = allocate_simulation_state(dimensions);
    // SimulationState tmp = allocate_simulation_state(dimensions);
    // SimulationState K1 = allocate_simulation_state(dimensions);
    // SimulationState K2 = allocate_simulation_state(dimensions);
    // SimulationState K3 = allocate_simulation_state(dimensions);
    // SimulationState K4 = allocate_simulation_state(dimensions);

    // I can maybe do it with one less state but it will require some recomputations.
    // I can also reduce the space by using the redundancy of v = du
    SimulationState currentState = allocate_simulation_state(dimensions);
    SimulationState intermediateState = allocate_simulation_state(dimensions);
    SimulationState nextState = allocate_simulation_state(dimensions);

    // not counting padding, its done in source and recv already

    const Coords source_pos = { Nx / 2, Ny / 2, 0 };
    Coords recv_pos[N_RECEIVERS]; //{Nx/2,Ny/2,0.03/dimensions.dh};

    const char *sensor_filename[N_RECEIVERS];

    printf("given sensor y %lf\n", p.sensor.y);

    //this code is used to setup the sensor positions and their output files. it varies for the purpose of the simulation
    //for a simulation with many sensors, check the commit 6b164af19430ed8a79b4d12af37b1660749912f8 "setup for 26 sensors simulation"


    for(int_t i = 0; i < N_RECEIVERS; i++) {
        real_t sim_center_y = Ny / 2.0 * dh;
        recv_pos[i] = { Nx / 2, Ny/2, 0 };

        static char tmp[N_RECEIVERS][50];

        sprintf(tmp[i],
                "sensor_out/sensor_%.2lf_%.2lfd.dat",
                p.sensor.x,
                p.sensor.y);
        sensor_filename[i] = tmp[i];

        // overwrite last output file, to not append to it.
        FILE *f = fopen(sensor_filename[i], "w");
        fclose(f);
        f = NULL;
    }

    Coords *d_sensor_coords = NULL;
    cudaMalloc(&d_sensor_coords, N_RECEIVERS * sizeof(Coords));
    cudaMemcpy(d_sensor_coords, recv_pos, sizeof(Coords) * N_RECEIVERS, cudaMemcpyHostToDevice);

    // contains the value at the sensor, will be copied from the gpu every iteration, and written to
    // file.
    real_t *d_current_value_at_sensor;

    cudaMalloc(&d_current_value_at_sensor, N_RECEIVERS * sizeof(real_t));

    double *model = open_model("data/model.bin");

    // for  now we put it in global  mem, but if we can  have it small enough (especially the 3d
    // one, put it in constant memory)
    // we  can crop on the  simulation space only
    const Position sensor = p.sensor;

    double *d_model = NULL;
    cudaErrorCheck(cudaMalloc(&d_model, MODEL_NX * MODEL_NY * sizeof(double)));
    cudaErrorCheck(
        cudaMemcpy(d_model, model, MODEL_NX * MODEL_NY * sizeof(double), cudaMemcpyHostToDevice));

    real_t *sig;
    int sig_len = signature(&sig);

    struct timeval start, end;

    show_model(dimensions, model, sensor);
    // return 0;

    gettimeofday(&start, NULL);
    printf("Started simulation...\n");

    int n_stored_samples = 0;
    for(int_t iteration = 0; iteration < max_iteration; iteration++) {
        cudaDeviceSynchronize(); // is it necessary ?
        gettimeofday(&end, NULL);
        print_progress_bar(iteration, max_iteration, start, end);
        if((iteration % snapshot_freq) == 0) {
            domain_save(currentState.U, dimensions);
        }

        // when is next samples supposed to be ?
        real_t next_sample_time = (double) n_stored_samples / SAMPLE_RATE;
        if(iteration * dt >= next_sample_time) {
            n_stored_samples++;
            get_sensor_value<<<1, 1>>>(currentState,
                                       d_current_value_at_sensor,
                                       dimensions,
                                       d_sensor_coords);

            cudaDeviceSynchronize();
            append_value_to_files(sensor_filename, d_current_value_at_sensor);
        }

        // RK4_step(currentState, tmp, K1, K2, K3, K4, dimensions);
        RK4_step_lowstorage(currentState,
                            intermediateState,
                            nextState,
                            dimensions,
                            d_model,
                            sensor);
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) {
            printf("cuda kernel error: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }


        int signature_idx = (int) (iteration * dt * SAMPLE_RATE);

        int block_x = 8;
        int block_y = 8;
        int block_z = 8;
        dim3 block(block_x, block_y, block_z);
        dim3 grid((Nx + 2 * padding + block_x - 1) / block_x,
                  (Ny + 2 * padding + block_y - 1) / block_y,
                  ((Nz + 2 * padding) + block_z - 1) / block_z);

        if(signature_idx < sig_len) {
            real_t src_value = sig[signature_idx];
            emit_source<<<grid, block>>>(nextState.U, dimensions, src_value, source_pos);
        }

        shift_states(&currentState, &nextState);

    }
    cudaDeviceSynchronize();

    get_sensor_value<<<1, 1>>>(currentState,
                               d_current_value_at_sensor,
                               dimensions,
                               d_sensor_coords);
    append_value_to_files(sensor_filename, d_current_value_at_sensor);

    free_simulation_state(currentState);
    free_simulation_state(intermediateState);
    free_simulation_state(nextState);

    free_model(model);
    cudaFree(d_model);

    return 0;
}

extern "C" int simulate_rtm(const simulation_parameters p) {
    printf("Simulating RTM...\n");

    Dimensions dimensions = p.dimensions;

    const int_t Nx = dimensions.Nx;
    const int_t Ny = dimensions.Ny;
    const int_t Nz = dimensions.Nz;
    const int_t padding = dimensions.padding;
    const real_t dh = dimensions.dh;

    if(!init_cuda()) {
        fprintf(stderr, "Could not initialize CUDA\n");
        exit(EXIT_FAILURE);
    }

    SimulationState currentState = allocate_simulation_state(dimensions);
    SimulationState intermediateState = allocate_simulation_state(dimensions);
    SimulationState nextState = allocate_simulation_state(dimensions);

    // must be updated

    const real_t dt = p.dt;
    const int_t max_iteration = p.max_iter;
    const int_t snapshot_freq = p.snapshot_freq;

    const Coords source_pos ={ Nx / 2, Ny / 2, 0 }; //{1.40, 0.69, 0.023};//{ Nx / 2, Ny / 2, 0 }; // also must be updated
    Coords recv_pos[N_RECEIVERS];

    const char *sensor_filename[N_RECEIVERS];

    printf("given sensor y %lf\n", p.sensor.y);

    for(int_t i = 0; i < N_RECEIVERS; i++) {
        real_t sim_center_y = Ny / 2.0 * dh;
        recv_pos[i] = { Nx / 2, Ny/2, 0 };

        static char tmp[N_RECEIVERS][50];

        sprintf(tmp[i],
                "sensor_out/sensor_%.2lf_%.2lfd.dat",
                p.sensor.x,
                p.sensor.y);
        sensor_filename[i] = tmp[i];

        // overwrite last output file, to not append to it.
        FILE *f = fopen(sensor_filename[i], "w");
        fclose(f);
        f = NULL;
    }

    Coords *d_sensor_coords = NULL;
    cudaMalloc(&d_sensor_coords, N_RECEIVERS * sizeof(Coords));
    cudaMemcpy(d_sensor_coords, recv_pos, sizeof(Coords) * N_RECEIVERS, cudaMemcpyHostToDevice);

    // contains the value at the sensor, will be copied from the gpu every iteration, and written to
    // file.
    real_t *d_current_value_at_sensor;

    cudaMalloc(&d_current_value_at_sensor, N_RECEIVERS * sizeof(real_t));

    double *model = open_model("data/model.bin");

    // for  now we put it in global  mem, but if we can  have it small enough (especially the 3d
    // one, put it in constant memory)
    // we  can crop on the  simulation space only
    const Position sensor = p.sensor;

    double *d_model = NULL;
    cudaErrorCheck(cudaMalloc(&d_model, MODEL_NX * MODEL_NY * sizeof(double)));
    cudaErrorCheck(
        cudaMemcpy(d_model, model, MODEL_NX * MODEL_NY * sizeof(double), cudaMemcpyHostToDevice));

    real_t *trace;
    struct timeval start, end;

    int trace_len = traces(&trace);//, &start, &end); //p, &trace);


    show_model(dimensions, model, sensor);

    gettimeofday(&start, NULL);
    printf("Started simulation...\n");
    
    int n_stored_samples = trace_len;
    for(int_t iteration = max_iteration-1; iteration >= 0; iteration--) {
        cudaDeviceSynchronize(); // is it necessary ?
        gettimeofday(&end, NULL);
        print_progress_bar((max_iteration - iteration), max_iteration, start, end);
        if((iteration % snapshot_freq) == 0) {
            domain_save(currentState.U, dimensions);
        }

        // when is next samples supposed to be ?
        real_t next_sample_time = (double) n_stored_samples / SAMPLE_RATE; //updated for rtm
        if(iteration * dt <= next_sample_time) { //update for rtm, when to store samples
            n_stored_samples--;
            get_sensor_value<<<1, 1>>>(currentState,
                                       d_current_value_at_sensor,
                                       dimensions,
                                       d_sensor_coords);

            cudaDeviceSynchronize();
            append_value_to_files(sensor_filename, d_current_value_at_sensor);
        }

        RK4_step_lowstorage(currentState,
                            intermediateState,
                            nextState,
                            dimensions,
                            d_model,
                            sensor);
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess) {
            printf("cuda kernel error: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        int trace_idx = (int) (iteration * dt * SAMPLE_RATE);

        int block_x = 8;
        int block_y = 8;
        int block_z = 8;
        dim3 block(block_x, block_y, block_z);
        dim3 grid((Nx + 2 * padding + block_x - 1) / block_x,
                  (Ny + 2 * padding + block_y - 1) / block_y,
                  ((Nz + 2 * padding) + block_z - 1) / block_z);

        if(trace_idx < trace_len) {
            real_t src_value = trace[trace_idx];
            emit_source<<<grid, block>>>(nextState.U, dimensions, src_value, source_pos);
        }

        shift_states(&currentState, &nextState);

    }
    cudaDeviceSynchronize();

    get_sensor_value<<<1, 1>>>(currentState,
                               d_current_value_at_sensor,
                               dimensions,
                               d_sensor_coords);
    append_value_to_files(sensor_filename, d_current_value_at_sensor);

    free_simulation_state(currentState);
    free_simulation_state(intermediateState);
    free_simulation_state(nextState);

    free_model(model);
    free(trace);
    cudaFree(d_model);

    return 0;
}

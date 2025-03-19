#define _XOPEN_SOURCE 600
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>
#include "simulate_kernel.h"


// #define MODEL_LY (double)1//width of model in meters
// #define MODEL_LZ 0.2//depth of model in centimeters

// #define RESERVOIR_OFFSET .5//just water on each side of the reservoir

//total size of simulation
//5x1x1 cm tube of water
// #define SIM_LX 0.01
// #define SIM_LY 0.01
// #define SIM_LZ 0.01//need to add height of sensors, but thats a parameter

//source and receiver at start and end of tube
// #define SOURCE_X 0
// #define SOURCE_Y SIM_LY / 2
// #define SOURCE_Z SIM_LZ / 2

// #define RECEIVER_X SIM_LX
// #define RECEIVER_Y SOURCE_X
// #define RECEIVER_Z SOURCE_Z

#define PADDING 0
#define PML_WIDTH 20
#define PML_SHIFT 80//(1.5*d_Nx - PML_WIDTH) / 2

#define WATER_K 1500
#define PLASTIC_K 2270

// lame_parameters params_at(real_t x, real_t y, real_t z);
__device__ double K(int_t i, int_t j, int_t k);
void show_model();
static bool init_cuda();

#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)



int_t max_iteration;
int_t snapshot_freq;
double sensor_height;


#define MODEL_NX 1201
#define MODEL_NY 401

double* model;

double* signature_wave;
int signature_len;
int sampling_freq;

int_t Nx, Ny, Nz;//they must be parsed in the main before used anywhere, I guess that's a very bad way of doing it, but the template did it like this
double sim_Lx, sim_Ly, sim_Lz;

double dt;
double dx,dy,dz;

//first index is the dimension(xyz direction of vector), second is the time step
// real_t *buffers[3] = { NULL, NULL, NULL };
real_t *saved_buffer = NULL;


#define MODEL_AT(i,j) model[i + j*MODEL_NX]

//CUDA elements
real_t *d_buffer_prv, *d_buffer, *d_buffer_nxt;

//the indexing is weird, because these values only exist for the boundaries, so looks like a corner, of depth PADDING
real_t *d_phi_x, *d_phi_y, *d_phi_z;
real_t *d_psi_x, *d_psi_y, *d_psi_z;

//might be possible to avoid using this, but it would be a mess
real_t *d_phi_x_nxt, *d_phi_y_nxt, *d_phi_z_nxt;
real_t *d_psi_x_nxt, *d_psi_y_nxt, *d_psi_z_nxt;



#define PADDING_BOTTOM_INDEX 0
#define PADDING_BOTTOM_SIZE (d_Nx + PADDING) * (d_Ny + PADDING) * PADDING

#define PADDING_SIDE_INDEX PADDING_BOTTOM_SIZE // the side indexing starts right after the BOTTOM
#define PADDING_SIDE_SIZE (d_Ny + PADDING) * d_Nz * PADDING

#define PADDING_FRONT_INDEX PADDING_SIDE_INDEX + PADDING_SIDE_SIZE // FRONT starts indexing right after side
#define PADDING_FRONT_SIZE d_Nx * d_Nz * PADDING

//account for borders, (PADDING: ghost values)

#define per(i, n) ((i+n)%n)

#define padded_index(i,j,k) per(i,d_Nx) * ((d_Ny) * (d_Nz)) + (per(j,d_Ny)) * ((d_Nz)) + (per(k,d_Nz))

#define d_P_prv(i,j,k) d_buffer_prv[padded_index(i,j,k)]
#define d_P(i,j,k)     d_buffer[padded_index(i,j,k)]
#define d_P_nxt(i,j,k) d_buffer_nxt[padded_index(i,j,k)]

__constant__ int_t d_Nx, d_Ny, d_Nz;
__constant__ double d_dt, d_dx, d_dy, d_dz;
__constant__ double d_sim_Lx, d_sim_Ly, d_sim_Lz;

#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

//When calling this, consider the warp geometry to reduce divergence. more info in my notebook
// __device__ int boundary_at(real_t* buffer, int_t i, int_t j, int_t k){
//     //doing some weird indexing to retrieve the correct boundary

//     //assuming x is on the side, y axis pointing you, I guess I could have done better to have the axes in reverse order but ok

//     if(k >= d_Nz)//on the BOTTOM boundary ! axis go downwards
//         {
//             k -= d_Nz;//bring bottom layer back up
//             //dimensions of bottom layer: (Nx + PADDING, Ny + PADDING, PADDING)
//             size_t idx = i * (d_Ny + PADDING) * PADDING + j * PADDING + k;
//             return buffer[PADDING_BOTTOM_INDEX  + idx];
//         }
//     if(i >= d_Nx)//on the SIDE boundary
//         {
//             i -= d_Nx;//bring side layer to left
//             //dimensions of side layer: (PADDING, Ny + PADDING, Nz)
//             size_t idx = i * (d_Ny + PADDING) * d_Nz + j * d_Nz + k;
//             return buffer[PADDING_SIDE_INDEX + idx];
//         }
//     if(j >= d_Ny)//on the FRONT boundary!
//         {
//             j -= d_Ny;//bring front layer back
//             //dimensions of front layer: (Nx, PADDING, Nz)
//             size_t idx = i * PADDING * d_Nz + j * d_Nz + k;
//             return buffer[PADDING_FRONT_INDEX + idx];
//         }

//     //this happens when I'm not in a boundary anymore, what to do then ? I guess that can happen, but will never be processed further, so just ignore this
//     return 0;
// }


__device__ real_t buf_at(real_t* buffer, int_t i, int_t j, int_t k){
    i = i - PML_SHIFT;

    if (i < 0 || i >= PML_WIDTH) return 0;//out of bounds, can happen

    // if(i == PML_WIDTH/2 && j == d_Ny/2 && k == d_Nz/2)
    //     printf("then, reading %lf at index %d\n",  buffer[i * (d_Ny * d_Nz) + j * d_Nz + k]);

    return buffer[i * (d_Ny * d_Nz) + per(j,d_Ny) * d_Nz + per(k,d_Nz)];//access might need to be periodic
}

__device__ void set_buf(real_t* buffer, real_t val, int_t i, int_t j, int_t k){
    i = i - PML_SHIFT;

    if (i < 0 || i >= PML_WIDTH || j < 0 || j >= d_Ny || k < 0 || k >= d_Nz) {
        printf("wrong indexing %d, %d, %d!\n",i,j,k);
        return;
    }//out of bounds, canNOT happen

    // printf("writing in memory at %d %d %d\n",i,j,k);
    // if(i == PML_WIDTH/2 && j == d_Ny/2 && k == d_Nz/2)
    //     printf("writing %lf at index %d\n", val, i * (d_Ny * d_Nz) + j * d_Nz + k);
    buffer[i * (d_Ny * d_Nz) + j * d_Nz + k] = val;
    // if(i == PML_WIDTH/2 && j == d_Ny/2 && k == d_Nz/2)
    //     printf("reading %lf at index %d\n",  buffer[i * (d_Ny * d_Nz) + j * d_Nz + k]);

}

__device__ real_t sigma(int_t i, int_t j, int_t k){
    i = i - PML_SHIFT;//find local coord back; here it only depends on i


    if (i < 0 || i >= PML_WIDTH) return 0;//try to reduce branching
    return   1;
}

__global__ void show_sigma(real_t * d_buffer_prv, real_t * d_buffer){
    int i = blockIdx.x * blockDim.x + threadIdx.x ;
    int j = blockIdx.y * blockDim.y + threadIdx.y ;
    int k = blockIdx.z * blockDim.z + threadIdx.z ;

    if(i >= d_Nx || j >= d_Ny || k >= d_Nz) return;//out of bounds. maybe try better way to deal with this, that induce less waste

    d_P(i,j,k) = d_P_prv(i,j,k) = sigma(i,j,k);
}


// Rotate the time step buffers for each dimension
void move_buffer_window ()
{

    real_t *temp = d_buffer_prv;
    d_buffer_prv = d_buffer;
    d_buffer = d_buffer_nxt;
    d_buffer_nxt = temp;

    //move auxiliary variables, I guess
    temp = d_phi_x;
    d_phi_x = d_phi_x_nxt;
    d_phi_x_nxt = temp;

    temp = d_phi_y;
    d_phi_y = d_phi_y_nxt;
    d_phi_y_nxt = temp;

    temp = d_phi_z;
    d_phi_z = d_phi_z_nxt;
    d_phi_z_nxt = temp;

    temp = d_psi_x;
    d_psi_x = d_psi_x_nxt;
    d_psi_x_nxt = temp;

    temp = d_psi_y;
    d_psi_y = d_psi_y_nxt;
    d_psi_y_nxt = temp;

    temp = d_psi_z;
    d_psi_z = d_psi_z_nxt;
    d_psi_z_nxt = temp;


}


// // Save the present time step in a numbered file under 'data/'
void domain_save ( int_t step )
{
    char filename[256];
    sprintf ( filename, "wave_data/%.5d.dat", step );
    FILE *out = fopen ( filename, "wb" );
    if (!out) {
        printf("[ERROR] File pointer is NULL!\n");
        exit(EXIT_FAILURE);
    }


    //cudaMemcpy2D() I can use that if I take other axis than YZ maybe ? not sure...

    // for (size_t i = 0; i < Ny; i++)
    // {
    //     // cudaErrorCheck(cudaMemcpy(&saved_buffer[Nx*i], &(d_buffer[(Nx/2+PADDING) * (Ny+2*PADDING) * (Nz+2*PADDING) + (i+PADDING) * ((Nz + 2*PADDING)) + (0+PADDING)]), Nz*sizeof(real_t), cudaMemcpyDeviceToHost));
    // }
    cudaErrorCheck(cudaMemcpy(saved_buffer, d_buffer, sizeof(real_t) * (Nx + 2*PADDING)*(Ny + 2*PADDING)*(Nz + 2*PADDING), cudaMemcpyDeviceToHost));


    // for (int i = 0; i < d_Ny; i++)
    // {
    //     for (int j = 0; j < d_Nz; j++)
    //     {
    //         printf("(%d,%d) => %lf\n", i,j,saved_buffer[i]);
    //     }

    // }

    for(int j =0; j<Ny; j++)//take some slice
    {
        for ( int_t i=0; i<Nx; i++ ){
        int w = fwrite (&saved_buffer[(i+PADDING) * ((Ny + 2*PADDING) * (Nz + 2*PADDING)) + (j+PADDING) * ((Nz + 2*PADDING)) + (Nz/2+PADDING)],
         sizeof(real_t), 1, out);//take horizontal slice from middle, around yz axis
        if(w!=1) printf("could write all\n");
        }
    }


    cudaErrorCheck(cudaDeviceSynchronize());//necessary ?


    fclose ( out );
}

// __global__ void fill_buffers(real_t *d_buffer_prv, real_t *d_buffer, real_t *d_buffer_nxt){
//     int f= 0;
//     for(int j =0; j<d_Ny; j++)//take some slice
//     {
//         for ( int_t k=0; k<d_Nz; k++ ){
//             for ( int_t i=0; i<d_Nx; i++ ){

//             d_P(i,j,k) = 1;
//             d_P_nxt(i,j,k) =.5;
//             d_P_prv(i,j,k)=0;
//         }}}
// }

__global__ void init_buffers(real_t * d_buffer_prv, real_t * d_buffer){
    int x_center = 3*d_Nx / 4;
    int y_center = d_Ny / 2;
    int n = 10;
    for(int i = x_center -n; i <=x_center+n;i++){
        for(int j = y_center - n; j <= y_center+n;j++){
            for(int k = d_Nz/2-n; k <= d_Nz/2 +1;k++){
                //dst to center
                real_t delta = ((i - x_center) * (i-x_center) / (double)d_Nx + (j - y_center)*(j - y_center) / (double)d_Ny + (k - d_Nz/2)*(k - d_Nz/2)/(double)d_Nz);
                // printf("%d\n", delta);
                d_P_prv(i,j,k) = d_P(i,j,k) = exp(-4.0*delta);
            }
        }
    }
}

// Set up our three buffers, and fill two with an initial perturbation
void domain_initialize ()//at this point I can load an optional starting state. (I would need two steps actually)
{
    //alloc cpu memory for saving image
    //I take a YZ portion of the plane
    saved_buffer = (real_t*)calloc((Nx+2*PADDING)*(Ny+2*PADDING)*(Nz+2*PADDING), sizeof(real_t));
    if(!saved_buffer){
        fprintf(stderr, "[ERROR] could not allocate cpu memory\n");
        exit(EXIT_FAILURE);
    }

    // //alloc memory in GPU
    size_t mem_size =  sizeof(real_t) * (Nx + 2*PADDING)*(Ny + 2*PADDING)*(Nz + 2*PADDING);
    cudaErrorCheck(cudaMalloc(&d_buffer_prv, mem_size));
    cudaErrorCheck(cudaMalloc(&d_buffer, mem_size));
    cudaErrorCheck(cudaMalloc(&d_buffer_nxt, mem_size));
    // printf("alloced %zu for P\n", mem_size);
        size_t border_size = PML_WIDTH * Ny * Nz * sizeof(real_t);
        cudaErrorCheck(cudaMalloc(&d_phi_x, border_size));
        cudaErrorCheck(cudaMalloc(&d_phi_y, border_size));
        cudaErrorCheck(cudaMalloc(&d_phi_z, border_size));

        cudaErrorCheck(cudaMalloc(&d_psi_x, border_size));
        cudaErrorCheck(cudaMalloc(&d_psi_y, border_size));
        cudaErrorCheck(cudaMalloc(&d_psi_z, border_size));

        cudaErrorCheck(cudaMalloc(&d_phi_x_nxt, border_size));
        cudaErrorCheck(cudaMalloc(&d_phi_y_nxt, border_size));
        cudaErrorCheck(cudaMalloc(&d_phi_z_nxt, border_size));

        cudaErrorCheck(cudaMalloc(&d_psi_x_nxt, border_size));
        cudaErrorCheck(cudaMalloc(&d_psi_y_nxt, border_size));
        cudaErrorCheck(cudaMalloc(&d_psi_z_nxt, border_size));

    //set it all to 0 (memset only works for int!)
    cudaErrorCheck(cudaMemset(d_buffer_prv, 0, mem_size));
    cudaErrorCheck(cudaMemset(d_buffer,     0, mem_size));
    cudaErrorCheck(cudaMemset(d_buffer_nxt, 0, mem_size));

        cudaErrorCheck(cudaMemset(d_phi_x_nxt, 0, border_size));
        cudaErrorCheck(cudaMemset(d_phi_y_nxt, 0, border_size));
        cudaErrorCheck(cudaMemset(d_phi_z_nxt, 0, border_size));

        cudaErrorCheck(cudaMemset(d_psi_x_nxt, 0, border_size));
        cudaErrorCheck(cudaMemset(d_psi_y_nxt, 0, border_size));
        cudaErrorCheck(cudaMemset(d_psi_z_nxt, 0, border_size));

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

    // cudaErrorCheck(cudaMemcpyToSymbol(d_model_Nx, &model_Nx, sizeof(int_t)));
    // cudaErrorCheck(cudaMemcpyToSymbol(d_model_Ny, &model_Ny, sizeof(int_t)));

    //set a gaussian bump
    // init_buffers<<<1,1>>>(d_buffer_prv, d_buffer);


}


// Get rid of all the memory allocations
void domain_finalize ( void )
{
    cudaFree(d_buffer_prv);
    cudaFree(d_buffer);
    cudaFree(d_buffer_nxt);

    cudaFree(d_phi_x);
    cudaFree(d_phi_y);
    cudaFree(d_phi_z);

    cudaFree(d_psi_x);
    cudaFree(d_psi_y);
    cudaFree(d_psi_z);


    free(saved_buffer);
}

// __global__ void emit_sine(double t, real_t *d_buffer_prv, real_t *d_buffer, real_t *d_buffer_nxt){
//     //emit sin from center, at each direction
//     double freq = 1e6;//1MHz
//     int n = 1;__global__ void fill_buffers(real_t *d_buffer_prv, real_t *d_buffer, real_t *d_buffer_nxt){

//     if(t < 1./freq){
//         double sine = sin(2*M_PI*t*freq);
//         // for (int x = d_Nx/2 - n; x <= d_Nx/2+n; x++) {
//         // for (int y = d_Ny/2 - n; y <= d_Ny/2+n; y++) {
//         // for (int z = d_Nz/2 - n; z <= d_Nz/2+n; z++) {

//         //     d_P(x,y,z) = sine;
//         // }}}
//         d_P(d_Nx/2,d_Ny/2,d_Nz/2) += sine;

//     }
// }

//I could split phi and psi because they dont touch exactly the same indices (at border), so less warping
__global__ void aux_variable_step(const real_t* d_buffer, real_t* d_phi_x, real_t* d_phi_y, real_t* d_phi_z, real_t* d_psi_x, real_t* d_psi_y, real_t* d_psi_z, real_t* d_phi_x_nxt, real_t* d_phi_y_nxt, real_t* d_phi_z_nxt, real_t* d_psi_x_nxt, real_t* d_psi_y_nxt, real_t* d_psi_z_nxt)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x + PML_SHIFT;//local coord passed, turned to global coordinates
    int j = blockIdx.y * blockDim.y + threadIdx.y ;
    int k = blockIdx.z * blockDim.z + threadIdx.z ;



    if(i >= PML_SHIFT + PML_WIDTH || j >= d_Ny || k >= d_Nz) {
        // printf("pre exit aux var func: %d %d %d\nmax &%d %d %d\n",i,j,k, PML_SHIFT + PML_WIDTH, d_Nz, d_Ny);
        return;
    }

    // printf("aux var func\n");


    //next phi

    //notice the parameters I pass to sigma, the factor is the same for all directions, depending only on depth in layer
    real_t next_phi_x = (- 0.5 / d_dx * (sigma(i-1,j,k) * buf_at(d_phi_x, i-1,j,k) + sigma(i,j,k) * buf_at(d_phi_x, i,j,k))
                        - 0.5 / d_dx * (d_P(i+1,j,k) - d_P(i-1,j,k))) * d_dt + buf_at(d_phi_x, i,j,k);
    real_t next_phi_y = (- 0.5 / d_dy * (sigma(i,j-1,k) * buf_at(d_phi_y, i,j-1,k) + sigma(i,j,k) * buf_at(d_phi_y, i,j,k))
                        - 0.5 / d_dy * (d_P(i,j+1,k) - d_P(i,j-1,k))) * d_dt + buf_at(d_phi_y, i,j,k);
    real_t next_phi_z = ( - 0.5 / d_dz * (sigma(i,j,k-1) * buf_at(d_phi_z, i,j,k-1) + sigma(i,j,k) * buf_at(d_phi_z, i,j,k))
                        - 0.5 / d_dz * (d_P(i,j,k+1) - d_P(i,j,k-1))) * d_dt + buf_at(d_phi_z, i,j,k);

    real_t next_psi_x = (- 0.5 / d_dx * (sigma(i-1,j,k) * buf_at(d_psi_x, i,j,k) + sigma(i,j,k) * buf_at(d_psi_x,i+1,j,k))
                        - 0.5 / d_dx * (d_P(i+1,j,k) - d_P(i-1,j,k))) * d_dt + buf_at(d_psi_x,i,j,k);
    real_t next_psi_y = (- 0.5 / d_dy * (sigma(i,j-1,k) * buf_at(d_psi_y, i,j,k) + sigma(i,j,k) * buf_at(d_psi_y,i,j+1,k))
                        - 0.5 / d_dy * (d_P(i,j+1,k) - d_P(i,j-1,k))) * d_dt + buf_at(d_psi_y,i,j,k);
    real_t next_psi_z = (- 0.5 / d_dz * (sigma(i,j,k-1) * buf_at(d_psi_z, i,j,k) + sigma(i,j,k) * buf_at(d_psi_z,i,j,k+1))
                        - 0.5 / d_dz * (d_P(i,j,k+1) - d_P(i,j,k-1))) * d_dt + buf_at(d_psi_z,i,j,k);


    //write at correct place in memory
    set_buf(d_phi_x_nxt, next_phi_x,i,j,k);
    set_buf(d_phi_y_nxt, next_phi_y,i,j,k);
    set_buf(d_phi_z_nxt, next_phi_z,i,j,k);

    set_buf(d_psi_x_nxt, next_psi_x,i,j,k);
    set_buf(d_psi_y_nxt, next_psi_y,i,j,k);
    set_buf(d_psi_z_nxt, next_psi_z,i,j,k);




    // if(i == d_Nx/2 && j == d_Ny/2 && k == d_Nz/2){
    //     printf("-> %lf\n", buf_at(d_phi_x_nxt, i,j,k));
    //     printf("phix: %lf -> %lf\n", buf_at(d_phi_x, i,j,k), next_phi_x);
    // }

    // printf("calling from %d %d %d\n",i,j,k);



}

__global__ void time_step (real_t * d_buffer_prv, real_t * d_buffer, real_t *d_buffer_nxt, real_t* d_phi_x, real_t* d_phi_y, real_t* d_phi_z, real_t* d_psi_x, real_t* d_psi_y, real_t* d_psi_z, double t)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x ;
    int j = blockIdx.y * blockDim.y + threadIdx.y ;
    int k = blockIdx.z * blockDim.z + threadIdx.z ;

    if(i >= d_Nx || j >= d_Ny || k >= d_Nz) return;//out of bounds. maybe try better way to deal with this, that induce less waste

    int sine_x = d_Nx / 4;

    if(i == sine_x && j == d_Ny/2 && k == d_Nz/2){
        //emit sin from center, at each direction
        double freq = 1e3;//1MHz
        int n = 1;

        if(t < 1./freq){
            double sine = sin(2*M_PI*t*freq);
            // for (int x = d_Nx/2 - n; x <= d_Nx/2+n; x++) {
            // for (int y = d_Ny/2 - n; y <= d_Ny/2+n; y++) {
            // for (int z = d_Nz/2 - n; z <= d_Nz/2+n; z++) {

            //     d_P(x,y,z) = sine;
            // }}}
            d_P_nxt(sine_x,d_Ny/2,d_Nz/2) = sine;
            return;
        }
    }
    // sine_x = 3 * d_Nx / 4;

    // if(i == sine_x && j == d_Ny/2 && k == d_Nz/2){
    //     //emit sin from center, at each direction
    //     double freq = 1e3;//1MHz
    //     int n = 1;

    //     if(t < 1./freq){
    //         double sine = sin(2*M_PI*t*freq);
    //         // for (int x = d_Nx/2 - n; x <= d_Nx/2+n; x++) {
    //         // for (int y = d_Ny/2 - n; y <= d_Ny/2+n; y++) {
    //         // for (int z = d_Nz/2 - n; z <= d_Nz/2+n; z++) {

    //         //     d_P(x,y,z) = sine;
    //         // }}}
    //         d_P_nxt(sine_x,d_Ny/2,d_Nz/2) = sine;
    //         return;
    //     }
    // }

    // if(j == d_Ny/2 && k == d_Nz/2 && i == d_Nx/2){
    //     for(int i = 0; i < PML_WIDTH; i++){
    //         printf("%.2lf ", buf_at(d_psi_x, i,d_Ny/2, d_Nz/2));
    //     }
    //     printf("\n");
    // }


    // Debug prints for buf_at if they are non-zero
    // if(j == d_Ny/2 && k == d_Nz/2){

    //     double psi_x_val = buf_at(d_psi_x, i, j, k);
    //     if (psi_x_val != 0) printf("t: %.2lf, buf_at(d_psi_x, %d, %d, %d) = %f\n", t, i, j, k, psi_x_val);

    //     // double phi_x_val = buf_at(d_phi_x, i-1, j, k);
    //     // if (phi_x_val != 0) printf("buf_at(d_phi_x-1, %d, %d, %d) = %f\n", i-1, j, k, phi_x_val);

    //     // double psi_y_val = buf_at(d_psi_y, i, j+1, k);
    //     // if (psi_y_val != 0) printf("buf_at(d_psi_y, %d, %d, %d) = %f\n", i, j+1, k, psi_y_val);

    //     // double phi_y_val = buf_at(d_phi_y, i, j-1, k);
    //     // if (phi_y_val != 0) printf("buf_at(d_phi_y, %d, %d, %d) = %f\n", i, j-1, k, phi_y_val);

    //     // double psi_z_val = buf_at(d_psi_z, i, j, k+1);
    //     // if (psi_z_val != 0) printf("buf_at(d_psi_z, %d, %d, %d) = %f\n", i, j, k+1, psi_z_val);

    //     // double phi_z_val = buf_at(d_phi_z, i, j, k-1);
    //     // if (phi_z_val != 0) printf("buf_at(d_phi_z, %d, %d, %d) = %f\n", i, j, k-1, phi_z_val);
    //     }


    real_t tmp      = 1. / (d_dx*d_dx) * (-2*d_P(i,j,k) + d_P(i-1,j,k) + d_P(i+1,j,k))
                    + 1. / (d_dy*d_dy) * (-2*d_P(i,j,k) + d_P(i,j-1,k) + d_P(i,j+1,k))
                    + 1. / (d_dz*d_dz) * (-2*d_P(i,j,k) + d_P(i,j,k-1) + d_P(i,j,k+1))
                    + 1. / (d_dx*d_dx) * (sigma(i,j,k) * buf_at(d_psi_x, i+1,j,k) - sigma(i-1,j,k) * buf_at(d_phi_x, i-1,j,k))
                    + 1. / (d_dy*d_dy) * (sigma(i,j,k) * buf_at(d_psi_y, i,j+1,k) - sigma(i,j-1,k) * buf_at(d_phi_y, i,j-1,k))
                    + 1. / (d_dz*d_dz) * (sigma(i,j,k) * buf_at(d_psi_z, i,j,k+1) - sigma(i,j,k-1) * buf_at(d_phi_z, i,j,k-1));


    d_P_nxt(i,j,k) = tmp * d_dt * d_dt + 2 * d_P(i,j,k) - d_P_prv(i,j,k);


    // if(i == d_Nx/2 && j == d_Ny/2 && k == d_Nz/2){
    //     //printf("-> %lf\n", buf_at(d_phi_x_nxt, i,j,k));
    //     printf("phix: %lf\n", buf_at(d_phi_x, i,j,k));
    // }


}


__global__ void boundary_condition (real_t *d_buffer_prv, real_t *d_buffer, real_t *d_buffer_nxt){
    // X boundaries (left and right)

    // for (int y = -PADDING; y < d_Ny + PADDING; y++) {
    //     for (int z = -PADDING; z < d_Nz + PADDING; z++) {
    //         // Extrapolate for left boundary (x = 0)
    //         d_P_nxt(-1, y, z) = (d_P_nxt(0,y,z) - d_P(0,y,z)) * (1) + d_P_nxt(-1,y,z);
    //         // if(y==d_Ny/2 && z == d_Nz/2)
    //         //     printf("pnxt: %lf, p: %lf, pnxt(-1):\n", d_P_nxt(0,y,z), d_P(0,y,z));

    //         // Extrapolate for right boundary (x = Nx - 1)
    //         // d_P_nxt(d_Nx, y, z) = d_P_nxt(d_Nx - 1,y,z);

    //     }
    // }

    // Y boundaries (top and bottom)

    // for (int x = -(PADDING + PMLAYE; x < d_Nx + (PADDING + PMLAYER; x++) {
    //     for (int z = -(PADDING + PMLAYER; z < d_Nz + (PADDING + PMLAYER; z++) {
    //         for (int p = 1; p <= (PADDING + PMLAYER; p++) {
    //             // Extrapolate for bottom boundary (y = 0)
    //             d_P_nxt(x, -p, z) = d_P_nxt(x, 0, z) ;

    //             // Extdition<<<1,1>>>(d_buffer_prv, d_buffer, d_buffer_nxt);//for now
    //             d_P_nxt(x, d_Ny + p - 1, z) = d_P_nxt(x, d_Ny - 1, z);
    //         }
    //     }
    // }

    // // Z boundaries (front and FRONT)

    // for (int x = -(PADDING + PMLAYER; x < d_Nx + (PADDING + PMLAYER; x++) {
    //     for (int y = -(PADDING + PMLAYER; y < d_Ny + (PADDING + PMLAYER; y++) {
    //         for (int p = 1; p <= (PADDING + PMLAYER; p++) {
    //             // Extrapolate for front boundary (z = 0)
    //             d_P_nxt(x, y, -p) = d_P_nxt(x, y, 0);

    //             // Extrapolate for back boundary (z = d_Nz - 1)
    //             d_P_nxt(x, y, d_Nz + p - 1) = d_P_nxt(x, y, d_Nz - 1);
    //         }
    //     }
    // }

    // for (int i = 0; i < d_Nx; i++)
    // {
    //     for (int j = 0; j < d_Ny; j++)
    //     {
    //         for (int k = 0; k < d_Nz; k++)
    //         {
    //             printf("(%d,%d,%d) => %lf,%lf,%lf\n", i,j,k,d_P_prv(i,j,k),d_P(i,j,k),d_P_nxt(i,j,k));
    //         }

    //     }

    // }


}



// Main time integration.
void simulation_loop( void )
{
    // Go through each time step
    // I think we should not think in terms of iteration but in term of time
    // fill_buffers<<<1,1>>>(d_buffer_prv, d_buffer, d_buffer_nxt);

    for ( int_t iteration=0; iteration<max_iteration; iteration++ )
    {

        if ( (iteration % snapshot_freq)==0 )
        {
            printf("iteration %d/%d\n", iteration, max_iteration);
            cudaDeviceSynchronize();
            domain_save ( iteration / snapshot_freq );
        }

        // Derive step t+1 from steps t and t-1
        cudaDeviceSynchronize();
        // emit_sine<<<1,1>>>(iteration*dt, d_buffer_prv, d_buffer, d_buffer_nxt);
        // cudaDeviceSynchronize();

        int block_x = 8;
        int block_y = 8;
        int block_z = 8 ;
        dim3 blockSize(block_x,block_y,block_z);
        dim3 gridSize((Nx + block_x - 1) / block_x, (Ny + block_y - 1) / block_y, (Nz + block_z - 1) / block_z);
        dim3 pml_gridSize((PML_WIDTH + block_x - 1) / block_x, (Ny + block_y - 1) / block_y, (Nz + block_z - 1) / block_z);

        // printf("grid size: %u %u %u\n", gridSize.x, gridSize.y, gridSize.z);
        // printf("block size: %u %u %u\n", blockSize.x, blockSize.y, blockSize.z);

        // show_sigma<<<gridSize, blockSize>>>(d_buffer_prv, d_buffer)

        time_step<<<gridSize, blockSize>>>(d_buffer_prv, d_buffer, d_buffer_nxt, d_phi_x, d_phi_y, d_phi_z, d_psi_x, d_psi_y, d_psi_z, iteration*dt);
        // cudaDeviceSynchronize();
        aux_variable_step<<<pml_gridSize, blockSize>>>(d_buffer, d_phi_x, d_phi_y, d_phi_z, d_psi_x, d_psi_y, d_psi_z, d_phi_x_nxt, d_phi_y_nxt, d_phi_z_nxt, d_psi_x_nxt, d_psi_y_nxt, d_psi_z_nxt);
        // cudaDeviceSynchronize();

        boundary_condition<<<1,1>>>(d_buffer_prv, d_buffer, d_buffer_nxt);//for now
        cudaDeviceSynchronize();


        // printf("prv phix %p, cur phix %p\n", d_phi_x, d_phi_x_nxt);


        // Rotate the time step buffers
        move_buffer_window();
        cudaDeviceSynchronize();
    }
}


extern "C" int simulate_wave(simulation_parameters p)
{
    dt = p.dt;
    max_iteration = p.max_iter;
    snapshot_freq= p.snapshot_freq;
    sensor_height = p.sensor_height;
    // SIM_LZ = MODEL_LZ + RESERVOIR_OFFSET + sensor_height;//need to add height of sensors, but thats a parameter


    // model_Nx = r_model_nx;
    // model_Ny = r_model_ny;

    Nx = p.Nx;
    Ny = p.Ny;
    Nz = p.Nz;

    sim_Lx = p.sim_Lx;
    sim_Ly = p.sim_Ly;
    sim_Lz = p.sim_Lz;


    //the simulation size is fixed, and resolution is a parameter. the resolution should make sense I guess
    dx = sim_Lx / Nx;
    dy = sim_Ly / Ny;
    dz = sim_Lz / Nz;
    // dx = 0.0001;//I'll need to make sure these are always small enough.
    // dy = 0.0001;
    // dz = 0.0001;
    printf("dx = %.4f, dy = %.4f, dz = %.4f\n", dx, dy, dz);


    //I need to create dx, dy, dz from the resolution given, knowing the shape of the reservoir (which is fixed) and adjust to that

    //FIRST PARSE AND SETUP SIMULATION PARAMETERS (done in domain_initialize)
    model = p.model_data;

    init_cuda();

    // Set up the initial state of the domain
    domain_initialize();

    struct timeval t_start, t_end;

    //show_model();
    gettimeofday(&t_start, NULL);
    simulation_loop();
    gettimeofday(&t_end, NULL);

    printf("Total elapsed time: %lf seconds\n", WALLTIME(t_end) - WALLTIME(t_start));




    // Clean up and shut down
    domain_finalize();
    printf("dx = %.4f, dy = %.4f, dz = %.4f\n", dx, dy, dz);


    exit ( EXIT_SUCCESS );
}



__device__ double K(int_t i, int_t j, int_t k){

    double x = i*d_dx, y=j*d_dy, z = k*d_dz;
    printf("K is called, thats not right\n");
    // if(j < 60){
    //     return WATER_K;
    // }else if(j > 65){
    //     return PLASTIC_K;
    // }else{
    //     double close_to_plastic = ((double)j - 60.)/5.;
    //     return close_to_plastic * PLASTIC_K + (1-close_to_plastic)*WATER_K;
    // }

    //to test in smaller space
    // if(j > 300){
    //     return PLASTIC_K;
    // }
    return WATER_K;

    //printf("x = %.4f, y = %.4f, z = %.4f\n", x, y, z);

    //1. am I (xy) on the model ?
    // if(RESERVOIR_OFFSET < x && x < MODEL_LX + RESERVOIR_OFFSET &&
    //     RESERVOIR_OFFSET < y && y < MODEL_LY + RESERVOIR_OFFSET){
    //     //yes!
    //     //printf("on the model, z = %lf\n", z);

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
    //     //     // printf("x = %lf, y = %lf, RESERVOIR_OFFSET = %lf, MODEL_LX = %lf, MODEL_LY = %lf, model_Nx = %d, model_Ny = %d\n", x, y, RESERVOIR_OFFSET, MODEL_LX, MODEL_LY, model_Nx, model_Ny);
    //     //     //printf("x_idx = %d, y_idx = %d\n", x_idx, y_idx);

    //     //     //I am in the model !
    //     //     //printf("in the model !\n");
    //     //     return PLASTIC_K;
    //     // }

    // }

    return WATER_K;
}

static bool init_cuda()
{
// BEGIN: T2
    int dev_count;
    cudaErrorCheck(cudaGetDeviceCount(&dev_count));


     if (dev_count == 0) {
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

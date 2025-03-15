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

#define RESERVOIR_OFFSET .5//just water on each side of the reservoir

//total size of simulation
//5x1x1 cm tube of water
#define SIM_LX 0.01
#define SIM_LY 0.01 
#define SIM_LZ 0.01//need to add height of sensors, but thats a parameter

//source and receiver at start and end of tube
#define SOURCE_X 0
#define SOURCE_Y SIM_LY / 2
#define SOURCE_Z SIM_LZ / 2

#define RECEIVER_X SIM_LX
#define RECEIVER_Y SOURCE_X
#define RECEIVER_Z SOURCE_Z

#define PMLAYER 30
#define PADDING 1

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


int_t model_Nx, model_Ny;

real_t* model;

double* signature_wave;
int signature_len;
int sampling_freq;

int_t Nx, Ny, Nz;//they must be parsed in the main before used anywhere, I guess that's a very bad way of doing it, but the template did it like this

real_t dt;
real_t dx,dy,dz;

//first index is the dimension(xyz direction of vector), second is the time step
// real_t *buffers[3] = { NULL, NULL, NULL };
real_t *saved_buffer = NULL;


#define MODEL_AT(i,j) model[i + j*model_Nx]

//CUDA elements
real_t *d_buffer_prv, *d_buffer, *d_buffer_nxt;

//account for borders, (PADDING: ghost values)
#define padded_index(i,j,k) (i+(PADDING + PMLAYER)) * ((d_Ny + 2*(PADDING + PMLAYER)) * (d_Nz + 2*(PADDING + PMLAYER))) + (j+(PADDING + PMLAYER)) * ((d_Nz + 2*(PADDING + PMLAYER))) + (k+(PADDING + PMLAYER))

#define d_P_prv(i,j,k) d_buffer_prv[padded_index(i,j,k)]
#define d_P(i,j,k)     d_buffer[padded_index(i,j,k)]
#define d_P_nxt(i,j,k) d_buffer_nxt[padded_index(i,j,k)]

__constant__ int_t d_Nx, d_Ny, d_Nz;
__constant__ real_t d_dt, d_dx, d_dy, d_dz;
__constant__ int_t d_model_Nx, d_model_Ny;

#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}



// Rotate the time step buffers for each dimension
void move_buffer_window ()
{

    real_t *temp = d_buffer_prv;
    d_buffer_prv = d_buffer;
    d_buffer = d_buffer_nxt;
    d_buffer_nxt = temp;

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
    //     // cudaErrorCheck(cudaMemcpy(&saved_buffer[Nx*i], &(d_buffer[(Nx/2+(PADDING + PMLAYER)) * (Ny+2*(PADDING + PMLAYER)) * (Nz+2*(PADDING + PMLAYER)) + (i+(PADDING + PMLAYER)) * ((Nz + 2*(PADDING + PMLAYER))) + (0+(PADDING + PMLAYER))]), Nz*sizeof(real_t), cudaMemcpyDeviceToHost));
    // }
    cudaErrorCheck(cudaMemcpy(saved_buffer, d_buffer, sizeof(real_t) * (Nx + 2*(PADDING + PMLAYER))*(Ny + 2*(PADDING + PMLAYER))*(Nz + 2*(PADDING + PMLAYER)), cudaMemcpyDeviceToHost));
    
    
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
        int w = fwrite (&saved_buffer[(i+(PADDING + PMLAYER)) * ((Ny + 2*(PADDING + PMLAYER)) * (Nz + 2*(PADDING + PMLAYER))) + (j+(PADDING + PMLAYER)) * ((Nz + 2*(PADDING + PMLAYER))) + (Nz/2+(PADDING + PMLAYER))],
         sizeof(real_t), 1, out);//take horizontal slice from middle, around yz axis
        if(w!=1) printf("could write all\n");
        }
    }
    

    cudaErrorCheck(cudaDeviceSynchronize());//necessary ?

    
    fclose ( out );
}

__global__ void fill_buffers(real_t *d_buffer_prv, real_t *d_buffer, real_t *d_buffer_nxt){
    int f= 0;
    for(int j =0; j<d_Ny; j++)//take some slice 
    {
        for ( int_t k=0; k<d_Nz; k++ ){
            for ( int_t i=0; i<d_Nx; i++ ){

            d_P(i,j,k) = 1;
            d_P_nxt(i,j,k) =.5;
            d_P_prv(i,j,k)=0;
        }}}
}


// Set up our three buffers, and fill two with an initial perturbation
void domain_initialize ()//at this point I can load an optional starting state. (I would need two steps actually)
{
    //alloc cpu memory for saving image
    //I take a YZ portion of the plane
    saved_buffer = (real_t*)calloc((Nx+2*(PADDING + PMLAYER))*(Ny+2*(PADDING + PMLAYER))*(Nz+2*(PADDING + PMLAYER)), sizeof(real_t));
    if(!saved_buffer){
        fprintf(stderr, "[ERROR] could not allocate cpu memory\n");
        exit(EXIT_FAILURE);
    }

    // //alloc memory in GPU
    size_t mem_size =  sizeof(real_t) * (Nx + 2*(PADDING + PMLAYER))*(Ny + 2*(PADDING + PMLAYER))*(Nz + 2*(PADDING + PMLAYER));
    cudaErrorCheck(cudaMalloc(&d_buffer_prv, mem_size));
    cudaErrorCheck(cudaMalloc(&d_buffer, mem_size));
    cudaErrorCheck(cudaMalloc(&d_buffer_nxt, mem_size));
    printf("alloced %zu for P\n", mem_size);

    printf("all fine\n");
    //set it all to 0
    cudaErrorCheck(cudaMemset(d_buffer_prv, 0, mem_size));
    cudaErrorCheck(cudaMemset(d_buffer, 0, mem_size));
    cudaErrorCheck(cudaMemset(d_buffer_nxt, 0, mem_size));

    cudaErrorCheck(cudaMemcpyToSymbol(d_Nx, &Nx, sizeof(int_t)));
    cudaErrorCheck(cudaMemcpyToSymbol(d_Ny, &Ny, sizeof(int_t)));
    cudaErrorCheck(cudaMemcpyToSymbol(d_Nz, &Nz, sizeof(int_t)));

    cudaErrorCheck(cudaMemcpyToSymbol(d_dx, &dx, sizeof(real_t)));
    cudaErrorCheck(cudaMemcpyToSymbol(d_dy, &dy, sizeof(real_t)));
    cudaErrorCheck(cudaMemcpyToSymbol(d_dz, &dz, sizeof(real_t)));
    cudaErrorCheck(cudaMemcpyToSymbol(d_dt, &dt, sizeof(real_t)));

    cudaErrorCheck(cudaMemcpyToSymbol(d_model_Nx, &model_Nx, sizeof(int_t)));
    cudaErrorCheck(cudaMemcpyToSymbol(d_model_Ny, &model_Ny, sizeof(int_t)));
    // fill_buffers<<<1,1>>>(d_buffer_prv, d_buffer, d_buffer_nxt);

}


// Get rid of all the memory allocations
void domain_finalize ( void )
{
    cudaFree(d_buffer_prv);
    cudaFree(d_buffer);
    cudaFree(d_buffer_nxt);

    

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

__global__ void time_step (const real_t * const d_buffer_prv, const real_t * const d_buffer, real_t *d_buffer_nxt, double t)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x - PMLAYER;
    int j = blockIdx.y * blockDim.y + threadIdx.y - PMLAYER;
    int k = blockIdx.z * blockDim.z + threadIdx.z - PMLAYER;

    if(i >= d_Nx + PMLAYER || j >= d_Ny + PMLAYER || k >= d_Nz + PMLAYER) return;//out of bounds. maybe try better way to deal with this, that induce less waste


    if(i == d_Nx/2 && j == d_Ny/2 && k == d_Nz/2){
        //emit sin from center, at each direction
        double freq = 1e6;//1MHz
        int n = 1;

        // if(t < 1./freq){
            double sine = sin(2*M_PI*t*freq);
            // for (int x = d_Nx/2 - n; x <= d_Nx/2+n; x++) {
            // for (int y = d_Ny/2 - n; y <= d_Ny/2+n; y++) {
            // for (int z = d_Nz/2 - n; z <= d_Nz/2+n; z++) {

            //     d_P(x,y,z) = sine;
            // }}}
            d_P_nxt(d_Nx/2,d_Ny/2,d_Nz/2) = sine;
            return;
        }
    // }

    // printf("time step at cell %d %d %d\n", i,j,k);
    //I am using ijk instead of xyz since it is the index, not the position anymore. position can be computed x = i*dx, etc.

    // //1st
    // P_nxt(i, j, k) = (dt*dt*(dx*dx*dy*dy*((K(i, j, k - 1) - K(i, j, k + 1))*(K(i, j, k - 1) - K(i, j, k + 1))*P(i, j, k) + 2*(K(i, j, k - 1) - K(i, j, k + 1))*(P(i, j, k - 1) - P(i, j, k + 1))*K(i, j, k) + 4*(-2*K(i, j, k) + K(i, j, k - 1) + K(i, j, k + 1))*K(i, j, k)*P(i, j, k) + 2*(-2*P(i, j, k) + P(i, j, k - 1) + P(i, j, k + 1))*K(i, j, k)*K(i, j, k))
    //  + dx*dx*dz*dz*((K(i, j - 1, k) - K(i, j + 1, k))*(K(i, j - 1, k) - K(i, j + 1, k))*P(i, j, k) + 2*(K(i, j - 1, k) - K(i, j + 1, k))*(P(i, j - 1, k) - P(i, j + 1, k))*K(i, j, k) + 4*(-2*K(i, j, k) + K(i, j - 1, k) + K(i, j + 1, k))*K(i, j, k)*P(i, j, k) + 2*(-2*P(i, j, k) + P(i, j - 1, k) + P(i, j + 1, k))*K(i, j, k)*K(i, j, k)) 
    //  + dy*dy*dz*dz*((K(i - 1, j, k) - K(i + 1, j, k))*(K(i - 1, j, k) - K(i + 1, j, k))*P(i, j, k) + 2*(K(i - 1, j, k) - K(i + 1, j, k))*(P(i - 1, j, k) - P(i + 1, j, k))*K(i, j, k) + 4*(-2*K(i, j, k) + K(i - 1, j, k) + K(i + 1, j, k))*K(i, j, k)*P(i, j, k) + 2*(-2*P(i, j, k) + P(i - 1, j, k) + P(i + 1, j, k))*K(i, j, k)*K(i, j, k))) 
    //  + 2*dx*dx*dy*dy*dz*dz*(2*P(i, j, k) - P_prv(i, j, k)))/(2*dx*dx*dy*dy*dz*dz);

    //2nd: smaller stencil
    
    double dampen = 0 <= i && i < d_Nx 
                &&  0 <= j && j < d_Ny
                &&  0 <= k && k < d_Nz ? 1 : 0.66 ;

    d_P_nxt(i, j, k) = dampen * (d_dt*d_dt*(d_dx*d_dx*d_dy*d_dy*((K(i, j, k - 1) - K(i, j, k + 1))*(d_P(i, j, k - 1) - d_P(i, j, k + 1)) + 2*(-2*d_P(i, j, k) + d_P(i, j, k - 1) + d_P(i, j, k + 1))*K(i, j, k)) 
    + d_dx*d_dx*d_dz*d_dz*((K(i, j - 1, k) - K(i, j + 1, k))*(d_P(i, j - 1, k) - d_P(i, j + 1, k)) + 2*(-2*d_P(i, j, k) + d_P(i, j - 1, k) + d_P(i, j + 1, k))*K(i, j, k)) 
    + d_dy*d_dy*d_dz*d_dz*((K(i - 1, j, k) - K(i + 1, j, k))*(d_P(i - 1, j, k) - d_P(i + 1, j, k)) + 2*(-2*d_P(i, j, k) + d_P(i - 1, j, k) + d_P(i + 1, j, k))*K(i, j, k)))*K(i, j, k) 
    + 2*d_dx*d_dx*d_dy*d_dy*d_dz*d_dz*(2*d_P(i,j,k) - d_P_prv(i,j,k)))/(2*d_dx*d_dx*d_dy*d_dy*d_dz*d_dz);
    // d_P_prv(i, j, k) = 1;
    // d_P(i, j, k) = 1;
    // d_P_nxt(i, j, k) = 1;//(double)(i+j+k)/(d_Nx+d_Ny+d_Nz);
    //3nd: larger stencil
    // P_nxt(i, j, k) = (dt*dt*(dx*dx*dy*dy*(((K(i, j, k - 3) - 9*K(i, j, k - 2) + 45*K(i, j, k - 1) - 45*K(i, j, k + 1) + 9*K(i, j, k + 2) - K(i, j, k + 3))*P(i, j, k) + (P(i, j, k - 3) - 9*P(i, j, k - 2) + 45*P(i, j, k - 1) - 45*P(i, j, k + 1) + 9*P(i, j, k + 2) - P(i, j, k + 3))*K(i, j, k))*(K(i, j, k - 3) - 9*K(i, j, k - 2) + 45*K(i, j, k - 1) - 45*K(i, j, k + 1) + 9*K(i, j, k + 2) - K(i, j, k + 3)) + 2*((K(i, j, k - 3) - 9*K(i, j, k - 2) + 45*K(i, j, k - 1) - 45*K(i, j, k + 1) + 9*K(i, j, k + 2) - K(i, j, k + 3))*(P(i, j, k - 3) - 9*P(i, j, k - 2) + 45*P(i, j, k - 1) - 45*P(i, j, k + 1) + 9*P(i, j, k + 2) - P(i, j, k + 3)) + 10*(-490*K(i, j, k) + 2*K(i, j, k - 3) - 27*K(i, j, k - 2) + 270*K(i, j, k - 1) + 270*K(i, j, k + 1) - 27*K(i, j, k + 2) + 2*K(i, j, k + 3))*P(i, j, k) + 10*(-490*P(i, j, k) + 2*P(i, j, k - 3) - 27*P(i, j, k - 2) + 270*P(i, j, k - 1) + 270*P(i, j, k + 1) - 27*P(i, j, k + 2) + 2*P(i, j, k + 3))*K(i, j, k))*K(i, j, k)) + dx*dx*dz*dz*(((K(i, j - 3, k) - 9*K(i, j - 2, k) + 45*K(i, j - 1, k) - 45*K(i, j + 1, k) + 9*K(i, j + 2, k) - K(i, j + 3, k))*P(i, j, k) + (P(i, j - 3, k) - 9*P(i, j - 2, k) + 45*P(i, j - 1, k) - 45*P(i, j + 1, k) + 9*P(i, j + 2, k) - P(i, j + 3, k))*K(i, j, k))*(K(i, j - 3, k) - 9*K(i, j - 2, k) + 45*K(i, j - 1, k) - 45*K(i, j + 1, k) + 9*K(i, j + 2, k) - K(i, j + 3, k)) + 2*((K(i, j - 3, k) - 9*K(i, j - 2, k) + 45*K(i, j - 1, k) - 45*K(i, j + 1, k) + 9*K(i, j + 2, k) - K(i, j + 3, k))*(P(i, j - 3, k) - 9*P(i, j - 2, k) + 45*P(i, j - 1, k) - 45*P(i, j + 1, k) + 9*P(i, j + 2, k) - P(i, j + 3, k)) + 10*(-490*K(i, j, k) + 2*K(i, j - 3, k) - 27*K(i, j - 2, k) + 270*K(i, j - 1, k) + 270*K(i, j + 1, k) - 27*K(i, j + 2, k) + 2*K(i, j + 3, k))*P(i, j, k) + 10*(-490*P(i, j, k) + 2*P(i, j - 3, k) - 27*P(i, j - 2, k) + 270*P(i, j - 1, k) + 270*P(i, j + 1, k) - 27*P(i, j + 2, k) + 2*P(i, j + 3, k))*K(i, j, k))*K(i, j, k)) + dy*dy*dz*dz*(((K(i - 3, j, k) - 9*K(i - 2, j, k) + 45*K(i - 1, j, k) - 45*K(i + 1, j, k) + 9*K(i + 2, j, k) - K(i + 3, j, k))*P(i, j, k) + (P(i - 3, j, k) - 9*P(i - 2, j, k) + 45*P(i - 1, j, k) - 45*P(i + 1, j, k) + 9*P(i + 2, j, k) - P(i + 3, j, k))*K(i, j, k))*(K(i - 3, j, k) - 9*K(i - 2, j, k) + 45*K(i - 1, j, k) - 45*K(i + 1, j, k) + 9*K(i + 2, j, k) - K(i + 3, j, k)) + 2*((K(i - 3, j, k) - 9*K(i - 2, j, k) + 45*K(i - 1, j, k) - 45*K(i + 1, j, k) + 9*K(i + 2, j, k) - K(i + 3, j, k))*(P(i - 3, j, k) - 9*P(i - 2, j, k) + 45*P(i - 1, j, k) - 45*P(i + 1, j, k) + 9*P(i + 2, j, k) - P(i + 3, j, k)) + 10*(-490*K(i, j, k) + 2*K(i - 3, j, k) - 27*K(i - 2, j, k) + 270*K(i - 1, j, k) + 270*K(i + 1, j, k) - 27*K(i + 2, j, k) + 2*K(i + 3, j, k))*P(i, j, k) + 10*(-490*P(i, j, k) + 2*P(i - 3, j, k) - 27*P(i - 2, j, k) + 270*P(i - 1, j, k) + 270*P(i + 1, j, k) - 27*P(i + 2, j, k) + 2*P(i + 3, j, k))*K(i, j, k))*K(i, j, k))) + 3600*dx*dx*dy*dy*dz*dz*(2*P(i, j, k) - P_prv(i, j, k)))/(3600*dx*dx*dy*dy*dz*dz);
    
}


__global__ void boundary_condition (const real_t *d_buffer_prv, const real_t *d_buffer, real_t *d_buffer_nxt)//TODO add rest of (PADDING + PMLAYER)
{
// X boundaries (left and right)

// for (int y = -(PADDING + PMLAYER); y < d_Ny + (PADDING + PMLAYER); y++) {
//     for (int z = -(PADDING + PMLAYER); z < d_Nz + (PADDING + PMLAYER); z++) {
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

// // Z boundaries (front and back)

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
        int block_z = 16;
        dim3 blockSize(block_x,block_y,block_z);
        dim3 gridSize(((Nx + PMLAYER*2) + block_x - 1) / block_x, ((Ny + PMLAYER*2) + block_y - 1) / block_y, ((Nz + PMLAYER*2) + block_z - 1) / block_z);

        // printf("grid size: %u %u %u\n", gridSize.x, gridSize.y, gridSize.z);
        // printf("block size: %u %u %u\n", blockSize.x, blockSize.y, blockSize.z);

        time_step<<<gridSize, blockSize>>>(d_buffer_prv, d_buffer, d_buffer_nxt, iteration*dt);
        cudaDeviceSynchronize();

        boundary_condition<<<1,1>>>(d_buffer_prv, d_buffer, d_buffer_nxt);//for now
        cudaDeviceSynchronize();


        


        // Rotate the time step buffers
        move_buffer_window();
        cudaDeviceSynchronize();
    }
}


extern "C" int simulate_wave(real_t* model_data, int_t n_x, int_t n_y, int_t n_z, double r_dt, int r_max_iter, int r_snapshot_freq, double r_sensor_height, int_t r_model_nx, int_t r_model_ny)
{
    dt =r_dt;
    max_iteration = r_max_iter;
    snapshot_freq=r_snapshot_freq;
    sensor_height = r_sensor_height;
    // SIM_LZ = MODEL_LZ + RESERVOIR_OFFSET + sensor_height;//need to add height of sensors, but thats a parameter


    model_Nx = r_model_nx;
    model_Ny = r_model_ny;
    
    Nx = n_x;
    Ny = n_y;
    Nz = n_z;

    

    //the simulation size is fixed, and resolution is a parameter. the resolution should make sense I guess
    dx = (double)SIM_LX / Nx; 
    dy = (double)SIM_LY / Ny;
    dz = (double)SIM_LZ / Nz;
    // dx = 0.0001;//I'll need to make sure these are always small enough.
    // dy = 0.0001;
    // dz = 0.0001;
    printf("dx = %.4f, dy = %.4f, dz = %.4f\n", dx, dy, dz);


    //I need to create dx, dy, dz from the resolution given, knowing the shape of the reservoir (which is fixed) and adjust to that

    //FIRST PARSE AND SETUP SIMULATION PARAMETERS (done in domain_initialize)
    model = model_data;

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

    //just water
    // return WATER_K;

    real_t x = i*d_dx, y=j*d_dy, z = k*d_dz;
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
    //     real_t model_bottom = RESERVOIR_OFFSET - MODEL_AT(x_idx, y_idx);
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

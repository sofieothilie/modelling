#define _XOPEN_SOURCE 600
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>

#include "modeling.h"


// #define MODEL_LX (double)3//length of model in meters
// #define MODEL_LY (double)1//width of model in meters
// #define MODEL_LZ 0.2//depth of model in centimeters

#define RESERVOIR_OFFSET .5//just water on each side of the reservoir

//total size of simulation
//5x1x1 cm tube of water
#define SIM_LX 0.02
#define SIM_LY 0.001 
#define SIM_LZ 0.001//need to add height of sensors, but thats a parameter

//source and receiver at start and end of tube
#define SOURCE_X 0
#define SOURCE_Y SIM_LY / 2
#define SOURCE_Z SIM_LZ / 2

#define RECEIVER_X SIM_LX
#define RECEIVER_Y SOURCE_X
#define RECEIVER_Z SOURCE_Z

#define PADDING 3

#define WATER_K 1500
#define PLASTIC_K 2270

// lame_parameters params_at(real_t x, real_t y, real_t z);
double K(int_t i, int_t j, int_t k);
void show_model();


// Convert 'struct timeval' into seconds in double prec. floating point
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
real_t
    *buffers[3] = { NULL, NULL, NULL };

//account for borders, (PADDING: ghost values)
#define P_prv(i,j,k) buffers[0][(i+PADDING) * (Ny * Nz) + (j+PADDING) * (Nz) + (k+PADDING)]
#define P(i,j,k)     buffers[1][(i+PADDING) * (Ny * Nz) + (j+PADDING) * (Nz) + (k+PADDING)]
#define P_nxt(i,j,k) buffers[2][(i+PADDING) * (Ny * Nz) + (j+PADDING) * (Nz) + (k+PADDING)]

#define MODEL_AT(i,j) model[i + j*model_Nx]

FILE* recv_file;

// Rotate the time step buffers for each dimension
void move_buffer_window ( void )
{
    real_t *temp = buffers[0];
    buffers[0] = buffers[1];
    buffers[1] = buffers[2];
    buffers[2] = temp;
}




// // Save the present time step in a numbered file under 'data/'
void domain_save ( int_t step )
{
    char filename[256];
    sprintf ( filename, "data/%.5d.dat", step );
    FILE *out = fopen ( filename, "wb" );
    if (!out) {
        printf("[ERROR] File pointer is NULL!\n");
        exit(EXIT_FAILURE);
    }
    for ( int_t i=0; i<Nx; i++ )//take some slice 
    {
        for(int j =0; j<Ny; j++)
        fwrite (&P(i,j,Nz/2), sizeof(real_t), 1, out );//take horizontal slice from middle, around yz axis
    }
    
    fclose ( out );
}


// Set up our three buffers, and fill two with an initial perturbation
void domain_initialize ()//at this point I can load an optional starting state. (I would need two steps actually)
{




    //alloc memory
    for (int t = 0; t < 3; t++) {//for all time steps (prev, cur, next)
        real_t *temp = calloc((Nx + 2*PADDING) * (Ny + 2*PADDING) * (Nz + 2*PADDING), sizeof(real_t));
        // printf("alloced %d\n", (Nx + 2*PADDING) * (Ny + 2*PADDING) * (Nz + 2*PADDING));
        if(temp == NULL){
            fprintf(stderr, "[ERROR] could not allocate enough memory for all buffers\n");
            exit(EXIT_FAILURE);
        }
        buffers[t] = temp;
    }
    

    // for ( int_t i=0; i<M; i++ )
    // {
    //     for ( int_t j=0; j<N; j++ )
    //     {
    //         // Calculate delta (radial distance) adjusted for M x N grid
    //         real_t delta = sqrt ( ((i - M/2.0) * (i - M/2.0)) / (real_t)M +
    //                             ((j - N/2.0) * (j - N/2.0)) / (real_t)N );
    //         U_prv(i,j) = U(i,j) = exp ( -4.0*delta*delta );
    //     }
    // }
    //get some value in the center
    // for (int x = Nx/2 - n; x < Nx/2+n; x++) {
    // for (int y = Ny/2 - n; y < Ny/2+n; y++) {
    // for (int z = Nz/2 - n; z < Nz/2+n; z++) {
    //     Uy(x,y,z) = Uz(x,y,z) = 1;
    //     Uy_prv(x,y,z) = Uz_prv(x,y,z) = 1;
    // }}}

    // Ux(Nx/2, Ny/2, Nz/2) =  Ux_prv(Nx/2, Ny/2, Nz/2) = 1;

    // Set the time step for 2D case
    // dt = dx*dy*dz / (c * sqrt (dx*dx+dy*dy));
}


// Get rid of all the memory allocations
void domain_finalize ( void )
{
    for (int t = 0; t < 3; t++) {//for all time steps (prev, cur, next)
        free(buffers[t]);
    }
}

void emit_sine(double t){
    //emit sin from center, at each direction
    double freq = 1e6;//1MHz
    int n = 1;
    if(t < 1./freq){
        double center_value = sin(2*M_PI*t*freq);
        for (int x = Nx/2 - n; x <= Nx/2+n; x++) {
        for (int y = Ny/2 - n; y <= Ny/2+n; y++) {
        for (int z = Nz/2 - n; z <= Nz/2+n; z++) {
            P(x,y,z) = center_value;
        }}}
    }
}

void emit_signature(double t){
    //get current index in signature
    int sig_idx = (int)(t*sampling_freq);
    if(sig_idx >= signature_len)
        return;//don't emit anymore

    //compute emission indices
    int emit_i = (int)(SOURCE_X / dx),
    emit_j = (int)(SOURCE_Y / dy),
    emit_k = (int)(SOURCE_Z / dz);

    //clamp them if they're outside
    if(emit_i >= Nx) emit_i = Nx-1;
    if(emit_j >= Ny) emit_j = Ny-1;
    if(emit_k >= Nz) emit_k = Nz-1;

    //printf("emitting from %d,%d,%d\n", emit_i,emit_j,emit_k);
    //printf("siglen: %d\n", signature_len);
    //emit
    P(emit_i+1,emit_j,emit_k) = signature_wave[sig_idx];
    // printf("emitting %lf\n", signature_wave[sig_idx]);
    //P(Nx/2,Ny/2,Nz/2) = signature_wave[sig_idx];

    //P(emit_i,emit_j,emit_k) = 1;
    //P(Nx/2,Ny/2,Nz/2) = 1;
    // int n = 2;
    // for (int x = Nx/2 - n; x <= Nx/2+n; x++) {
    // for (int y = Ny/2 - n; y <= Ny/2+n; y++) {
    // for (int z = 0 ; z < Nz; z++) {
    //     P(x,y,z) = 1;
    // }}}
    //maybe I should scale it down ? idk
}

void recv(){//ADJUST SAMPLING FREQUENCY

    //compute receiver indices
    int recv_i = (int)(RECEIVER_X / dx),
    recv_j = (int)(RECEIVER_Y / dy),
    recv_k = (int)(RECEIVER_Z / dz);

    //clamp them if they're outside
    if(recv_i >= Nx) recv_i = Nx-1;
    if(recv_j >= Ny) recv_j = Ny-1;
    if(recv_k >= Nz) recv_k = Nz-1;

    fwrite(&P(recv_i, recv_j, recv_k), sizeof(real_t), 1, recv_file);
}

// Integration formula (Eq. 9 from the pdf document)
void time_step ( double t )
{
    recv();
    emit_signature(t);
    //emit_sine(t);


    #pragma omp parallel for collapse(3)
    for (int i = 0; i < Nx; i++) {
    for (int j = 0; j < Ny; j++) {
    for (int k = 0; k < Nz; k++) {
        //I am using ijk instead of xyz since it is the index, not the position anymore. position can be computed x = i*dx, etc.

        // //1st
        // P_nxt(i, j, k) = (dt*dt*(dx*dx*dy*dy*((K(i, j, k - 1) - K(i, j, k + 1))*(K(i, j, k - 1) - K(i, j, k + 1))*P(i, j, k) + 2*(K(i, j, k - 1) - K(i, j, k + 1))*(P(i, j, k - 1) - P(i, j, k + 1))*K(i, j, k) + 4*(-2*K(i, j, k) + K(i, j, k - 1) + K(i, j, k + 1))*K(i, j, k)*P(i, j, k) + 2*(-2*P(i, j, k) + P(i, j, k - 1) + P(i, j, k + 1))*K(i, j, k)*K(i, j, k))
        //  + dx*dx*dz*dz*((K(i, j - 1, k) - K(i, j + 1, k))*(K(i, j - 1, k) - K(i, j + 1, k))*P(i, j, k) + 2*(K(i, j - 1, k) - K(i, j + 1, k))*(P(i, j - 1, k) - P(i, j + 1, k))*K(i, j, k) + 4*(-2*K(i, j, k) + K(i, j - 1, k) + K(i, j + 1, k))*K(i, j, k)*P(i, j, k) + 2*(-2*P(i, j, k) + P(i, j - 1, k) + P(i, j + 1, k))*K(i, j, k)*K(i, j, k)) 
        //  + dy*dy*dz*dz*((K(i - 1, j, k) - K(i + 1, j, k))*(K(i - 1, j, k) - K(i + 1, j, k))*P(i, j, k) + 2*(K(i - 1, j, k) - K(i + 1, j, k))*(P(i - 1, j, k) - P(i + 1, j, k))*K(i, j, k) + 4*(-2*K(i, j, k) + K(i - 1, j, k) + K(i + 1, j, k))*K(i, j, k)*P(i, j, k) + 2*(-2*P(i, j, k) + P(i - 1, j, k) + P(i + 1, j, k))*K(i, j, k)*K(i, j, k))) 
        //  + 2*dx*dx*dy*dy*dz*dz*(2*P(i, j, k) - P_prv(i, j, k)))/(2*dx*dx*dy*dy*dz*dz);

        //2nd
        
        // P_nxt(x, y, z) = (dt*dt*(dx*dx*dy*dy*((K(x, y, z - 1) - K(x, y, z + 1))*(P(x, y, z - 1) - P(x, y, z + 1)) + 2*(-2*P(x, y, z) + P(x, y, z - 1) + P(x, y, z + 1))*K(x, y, z)) 
        // + dx*dx*dz*dz*((K(x, y - 1, z) - K(x, y + 1, z))*(P(x, y - 1, z) - P(x, y + 1, z)) + 2*(-2*P(x, y, z) + P(x, y - 1, z) + P(x, y + 1, z))*K(x, y, z)) 
        // + dy*dy*dz*dz*((K(x - 1, y, z) - K(x + 1, y, z))*(P(x - 1, y, z) - P(x + 1, y, z)) + 2*(-2*P(x, y, z) + P(x - 1, y, z) + P(x + 1, y, z))*K(x, y, z)))*K(x, y, z) 
        // + 2*dx*dx*dy*dy*dz*dz*(2*P(x, y, z) - P_prv(x, y, z)))/(2*dx*dx*dy*dy*dz*dz);

        //3nd
        P_nxt(i, j, k) = (dt*dt*(dx*dx*dy*dy*(((K(i, j, k - 3) - 9*K(i, j, k - 2) + 45*K(i, j, k - 1) - 45*K(i, j, k + 1) + 9*K(i, j, k + 2) - K(i, j, k + 3))*P(i, j, k) + (P(i, j, k - 3) - 9*P(i, j, k - 2) + 45*P(i, j, k - 1) - 45*P(i, j, k + 1) + 9*P(i, j, k + 2) - P(i, j, k + 3))*K(i, j, k))*(K(i, j, k - 3) - 9*K(i, j, k - 2) + 45*K(i, j, k - 1) - 45*K(i, j, k + 1) + 9*K(i, j, k + 2) - K(i, j, k + 3)) + 2*((K(i, j, k - 3) - 9*K(i, j, k - 2) + 45*K(i, j, k - 1) - 45*K(i, j, k + 1) + 9*K(i, j, k + 2) - K(i, j, k + 3))*(P(i, j, k - 3) - 9*P(i, j, k - 2) + 45*P(i, j, k - 1) - 45*P(i, j, k + 1) + 9*P(i, j, k + 2) - P(i, j, k + 3)) + 10*(-490*K(i, j, k) + 2*K(i, j, k - 3) - 27*K(i, j, k - 2) + 270*K(i, j, k - 1) + 270*K(i, j, k + 1) - 27*K(i, j, k + 2) + 2*K(i, j, k + 3))*P(i, j, k) + 10*(-490*P(i, j, k) + 2*P(i, j, k - 3) - 27*P(i, j, k - 2) + 270*P(i, j, k - 1) + 270*P(i, j, k + 1) - 27*P(i, j, k + 2) + 2*P(i, j, k + 3))*K(i, j, k))*K(i, j, k)) + dx*dx*dz*dz*(((K(i, j - 3, k) - 9*K(i, j - 2, k) + 45*K(i, j - 1, k) - 45*K(i, j + 1, k) + 9*K(i, j + 2, k) - K(i, j + 3, k))*P(i, j, k) + (P(i, j - 3, k) - 9*P(i, j - 2, k) + 45*P(i, j - 1, k) - 45*P(i, j + 1, k) + 9*P(i, j + 2, k) - P(i, j + 3, k))*K(i, j, k))*(K(i, j - 3, k) - 9*K(i, j - 2, k) + 45*K(i, j - 1, k) - 45*K(i, j + 1, k) + 9*K(i, j + 2, k) - K(i, j + 3, k)) + 2*((K(i, j - 3, k) - 9*K(i, j - 2, k) + 45*K(i, j - 1, k) - 45*K(i, j + 1, k) + 9*K(i, j + 2, k) - K(i, j + 3, k))*(P(i, j - 3, k) - 9*P(i, j - 2, k) + 45*P(i, j - 1, k) - 45*P(i, j + 1, k) + 9*P(i, j + 2, k) - P(i, j + 3, k)) + 10*(-490*K(i, j, k) + 2*K(i, j - 3, k) - 27*K(i, j - 2, k) + 270*K(i, j - 1, k) + 270*K(i, j + 1, k) - 27*K(i, j + 2, k) + 2*K(i, j + 3, k))*P(i, j, k) + 10*(-490*P(i, j, k) + 2*P(i, j - 3, k) - 27*P(i, j - 2, k) + 270*P(i, j - 1, k) + 270*P(i, j + 1, k) - 27*P(i, j + 2, k) + 2*P(i, j + 3, k))*K(i, j, k))*K(i, j, k)) + dy*dy*dz*dz*(((K(i - 3, j, k) - 9*K(i - 2, j, k) + 45*K(i - 1, j, k) - 45*K(i + 1, j, k) + 9*K(i + 2, j, k) - K(i + 3, j, k))*P(i, j, k) + (P(i - 3, j, k) - 9*P(i - 2, j, k) + 45*P(i - 1, j, k) - 45*P(i + 1, j, k) + 9*P(i + 2, j, k) - P(i + 3, j, k))*K(i, j, k))*(K(i - 3, j, k) - 9*K(i - 2, j, k) + 45*K(i - 1, j, k) - 45*K(i + 1, j, k) + 9*K(i + 2, j, k) - K(i + 3, j, k)) + 2*((K(i - 3, j, k) - 9*K(i - 2, j, k) + 45*K(i - 1, j, k) - 45*K(i + 1, j, k) + 9*K(i + 2, j, k) - K(i + 3, j, k))*(P(i - 3, j, k) - 9*P(i - 2, j, k) + 45*P(i - 1, j, k) - 45*P(i + 1, j, k) + 9*P(i + 2, j, k) - P(i + 3, j, k)) + 10*(-490*K(i, j, k) + 2*K(i - 3, j, k) - 27*K(i - 2, j, k) + 270*K(i - 1, j, k) + 270*K(i + 1, j, k) - 27*K(i + 2, j, k) + 2*K(i + 3, j, k))*P(i, j, k) + 10*(-490*P(i, j, k) + 2*P(i - 3, j, k) - 27*P(i - 2, j, k) + 270*P(i - 1, j, k) + 270*P(i + 1, j, k) - 27*P(i + 2, j, k) + 2*P(i + 3, j, k))*K(i, j, k))*K(i, j, k))) + 3600*dx*dx*dy*dy*dz*dz*(2*P(i, j, k) - P_prv(i, j, k)))/(3600*dx*dx*dy*dy*dz*dz);

    }
    }
    }
}


void boundary_condition ( void )//mirroring boundary condition, try with inverse mirror (*-1)
{
// X boundaries (left and right)
#pragma omp parallel for collapse(2)
for (int y = -PADDING; y < Ny + PADDING; y++) {
    for (int z = -PADDING; z < Nz + PADDING; z++) {
        for (int p = 1; p <= PADDING; p++) {
            // Extrapolate for left boundary (x = 0)
            P_nxt(-p, y, z) = 3 * P_nxt(p - 1, y, z) - 3 * P_nxt(p, y, z) + P_nxt(p + 1, y, z);
            
            // Extrapolate for right boundary (x = Nx - 1)
            P_nxt(Nx + p - 1, y, z) = 3 * P_nxt(Nx - p, y, z) - 3 * P_nxt(Nx - p - 1, y, z) + P_nxt(Nx - p - 2, y, z);
        }
    }
}

// Y boundaries (top and bottom)
#pragma omp parallel for collapse(2)
for (int x = -PADDING; x < Nx + PADDING; x++) {
    for (int z = -PADDING; z < Nz + PADDING; z++) {
        for (int p = 1; p <= PADDING; p++) {
            // Extrapolate for bottom boundary (y = 0)
            P_nxt(x, -p, z) = 3 * P_nxt(x, p - 1, z) - 3 * P_nxt(x, p, z) + P_nxt(x, p + 1, z);
            
            // Extrapolate for top boundary (y = Ny - 1)
            P_nxt(x, Ny + p - 1, z) = 3 * P_nxt(x, Ny - p, z) - 3 * P_nxt(x, Ny - p - 1, z) + P_nxt(x, Ny - p - 2, z);
        }
    }
}

// Z boundaries (front and back)
#pragma omp parallel for collapse(2)
for (int x = -PADDING; x < Nx + PADDING; x++) {
    for (int y = -PADDING; y < Ny + PADDING; y++) {
        for (int p = 1; p <= PADDING; p++) {
            // Extrapolate for front boundary (z = 0)
            P_nxt(x, y, -p) = 3 * P_nxt(x, y, p - 1) - 3 * P_nxt(x, y, p) + P_nxt(x, y, p + 1);
            
            // Extrapolate for back boundary (z = Nz - 1)
            P_nxt(x, y, Nz + p - 1) = 3 * P_nxt(x, y, Nz - p) - 3 * P_nxt(x, y, Nz - p - 1) + P_nxt(x, y, Nz - p - 2);
        }
    }
}

}


// Main time integration.
void simulation_loop( void )
{
    // Go through each time step
    // I think we should not think in terms of iteration but in term of time
    for ( int_t iteration=0; iteration<=max_iteration; iteration++ )
    {
        if ( (iteration % snapshot_freq)==0 )
        {
            printf("iteration %d\n",iteration);
            domain_save ( iteration / snapshot_freq );
        }

        // Derive step t+1 from steps t and t-1
        time_step(iteration*dt);
        //boundary_condition();

        // Rotate the time step buffers
        move_buffer_window();
    }
}


int simulate(real_t* model_data, int_t n_x, int_t n_y, int_t n_z, double r_dt, int r_max_iter, int r_snapshot_freq, double r_sensor_height, int_t r_model_nx, int_t r_model_ny, double* sign, int sign_len, int fs)
{
    dt =r_dt;
    max_iteration = r_max_iter;
    snapshot_freq=r_snapshot_freq;
    sensor_height = r_sensor_height;
    // SIM_LZ = MODEL_LZ + RESERVOIR_OFFSET + sensor_height;//need to add height of sensors, but thats a parameter

    signature_wave = sign;
    signature_len = sign_len;
    sampling_freq = fs;

    model_Nx = r_model_nx;
    model_Ny = r_model_ny;
    
    Nx = n_x;
    Ny = n_y;
    Nz = n_z;
    printf("opening recv file\n");
    recv_file = fopen("receiver.dat", "wb");
    if (!recv_file) {
        printf("[ERROR] File pointer is NULL!\n");
        exit(EXIT_FAILURE);
    }

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

    // Set up the initial state of the domain
    domain_initialize();

    struct timeval t_start, t_end;

    gettimeofday ( &t_start, NULL );
    //show_model();
    simulation_loop();
    gettimeofday ( &t_end, NULL );

    printf ( "Total elapsed time: %lf seconds\n",
        WALLTIME(t_end) - WALLTIME(t_start)
    );

    // Clean up and shut down
    domain_finalize();
    fclose(recv_file);
    printf("dx = %.4f, dy = %.4f, dz = %.4f\n", dx, dy, dz);

    exit ( EXIT_SUCCESS );
}


void show_model(){

    char filename[256];
    sprintf ( filename, "model.dat" );
    FILE *out = fopen ( filename, "wb" );
    if (!out) {
        printf("[ERROR] File pointer is NULL!\n");
        exit(EXIT_FAILURE);
    }


    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Ny; j++) {
            for (int i = 0; i < Nx; i++) {
                double K_ = K(i, j, k);
                unsigned char in_model = (K_ == PLASTIC_K) ? 1 : 0;
                fwrite(&in_model, sizeof(unsigned char), 1, out);
            }
        }
    }
    printf("written to file\n");

    fclose ( out );

}

double K(int_t i, int_t j, int_t k){

    //just water
    return WATER_K;

    real_t x = i*dx, y=j*dy, z = k*dz;
    // if(j < 60){
    //     return WATER_K;
    // }else if(j > 65){
    //     return PLASTIC_K;
    // }else{
    //     double close_to_plastic = ((double)j - 60.)/5.;
    //     return close_to_plastic * PLASTIC_K + (1-close_to_plastic)*WATER_K;
    // }

    //to test in smaller space
    // if(j > 60){
    //     return PLASTIC_K;
    // }
    // return WATER_K;

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

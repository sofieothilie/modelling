#define _XOPEN_SOURCE 600
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>


#include "modeling.h"


#define MODEL_LX (double)3//length of model in meters
#define MODEL_LY (double)1//width of model in meters
#define MODEL_LZ 0.2//depth of model in centimeters

#define RESERVOIR_OFFSET .5//just water on each side of the reservoir

//total size of simulation
double SIM_LX = MODEL_LX + 2*RESERVOIR_OFFSET;
double SIM_LY = MODEL_LY + 2*RESERVOIR_OFFSET; 
double SIM_LZ = MODEL_LZ + RESERVOIR_OFFSET;//need to add height of sensors, but thats a parameter

typedef struct {
    real_t lambda, mu, rho;
} lame_parameters;


lame_parameters WATER_LAME_PARAMETERS = {
    .lambda = 2200,
    .mu = 0,
    .rho = 0.5
};

lame_parameters PLASTIC_LAME_PARAMETERS = {
    .lambda = 1555.5555555556,
    .mu = 666.6,
    .rho = 0.5
};

lame_parameters params_at(real_t x, real_t y, real_t z);
void show_model();



// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

int_t max_iteration = 5000;
int_t snapshot_freq = 10;
double sensor_height;

// Simulation parameters: size, step count, and how often to save the state
// int_t
//     N = 256,
//     M = 256,
//     max_iteration = 4000,
//     snapshot_freq = 20;

int_t model_Nx, model_Ny;

real_t* model;

int_t Nx, Ny, Nz;//they must be parsed in the main before used anywhere, I guess that's a very bad way of doing it, but the template did it like this

real_t dt = 0.005;
real_t dx = 1.,
        dy = 1.,
        dz = 1.;

// Wave equation parameters, time step is derived from the space step
// const real_t
//     c  = 1.0,
//     dx = 1.0,
//     dy = 1.0;
// real_t
//     dt;

//first index is the dimension(xyz direction of vector), second is the time step
real_t
    *buffers[3][3] = { NULL, NULL, NULL };

//account for borders, (ghost values)
#define Ux_prv(i,j,k) buffers[0][0][(i+1) * (Ny * Nz) + (j+1) * (Nz) + (k+1)]
#define Ux(i,j,k)     buffers[0][1][(i+1) * (Ny * Nz) + (j+1) * (Nz) + (k+1)]
#define Ux_nxt(i,j,k) buffers[0][2][(i+1) * (Ny * Nz) + (j+1) * (Nz) + (k+1)]

#define Uy_prv(i,j,k) buffers[1][0][(i+1) * (Ny * Nz) + (j+1) * (Nz) + (k+1)]
#define Uy(i,j,k)     buffers[1][1][(i+1) * (Ny * Nz) + (j+1) * (Nz) + (k+1)]
#define Uy_nxt(i,j,k) buffers[1][2][(i+1) * (Ny * Nz) + (j+1) * (Nz) + (k+1)]

#define Uz_prv(i,j,k) buffers[2][0][(i+1) * (Ny * Nz) + (j+1) * (Nz) + (k+1)]
#define Uz(i,j,k)     buffers[2][1][(i+1) * (Ny * Nz) + (j+1) * (Nz) + (k+1)]
#define Uz_nxt(i,j,k) buffers[2][2][(i+1) * (Ny * Nz) + (j+1) * (Nz) + (k+1)]

#define MODEL_AT(i,j) model[i * model_Ny + j];


// Rotate the time step buffers for each dimension
void move_buffer_window ( void )
{
    for (int d = 0; d < 3; d++) {
        real_t *temp = buffers[d][0];
        buffers[d][0] = buffers[d][1];
        buffers[d][1] = buffers[d][2];
        buffers[d][2] = temp;
        }
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
    // for ( int_t i=0; i<Ny; i++ )//take some slice 
    // {
    //     fwrite ( &Ux(Nx/2,i,0), sizeof(real_t), Nz, out );//take horizontal slice from middle, around yz axis
    // }

    //save norm of vector
    for (int_t i = 0; i < Ny; i++) {
    for (int_t j = 0; j < Nz; j++) {
        real_t norm = sqrt(Ux(Nx/2, i, j) * Ux(Nx/2, i, j) +
                           Uy(Nx/2, i, j) * Uy(Nx/2, i, j) +
                           Uz(Nx/2, i, j) * Uz(Nx/2, i, j));
        fwrite(&norm, sizeof(real_t), 1, out);
    }
    }   
    fclose ( out );
}


// Set up our three buffers, and fill two with an initial perturbation
void domain_initialize (int_t n_x, int_t n_y, int_t n_z)//at this point I can load an optional starting state. (I would need two steps actually)
{
    Nx = n_x;
    Ny = n_y;
    Nz = n_z;

    //the simulation size is fixed, and resolution is a parameter. the resolution should make sense I guess
    // dx = (double)SIM_LX / Nx; 
    // dy = (double)SIM_LY / Ny;
    // dz = (double)SIM_LZ / Nz;
    dx = 0.001;
    dy = 0.001;
    dz = 0.001;

    printf("dx = %.4f, dy = %.4f, dz = %.4f\n", dx, dy, dz);


    //alloc memory
    for (int d = 0; d < 3; d++) {//for all dimensions
        for (int t = 0; t < 3; t++) {//for all time steps (prev, cur, next)
            real_t *temp = calloc((Nx + 2) * (Ny + 2) * (Nz + 2), sizeof(real_t));
            printf("alloced %d\n", (Nx + 2) * (Ny + 2) * (Nz + 2));
            if(temp == NULL){
                fprintf(stderr, "[ERROR] could not allocate enough memory for all buffers\n");
                exit(EXIT_FAILURE);
            }
            buffers[d][t] = temp;
        }
    }
    printf("initialized memory\n");
    //howto INIT ???

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
    for (int d = 0; d < 3; d++) {//for all dimensions
        for (int t = 0; t < 3; t++) {//for all time steps (prev, cur, next)
            free(buffers[d][t]);
        }
    }
}


// Integration formula (Eq. 9 from the pdf document)
void time_step ( double t )
{

    //emit sin from center, at each direction
    double freq = 10;

    if(t < 1./freq){
        double center_value = sin(2*M_PI*t*freq);
        Ux(Nx/2,Ny/2,Nz/2) = center_value;
        Uy(Nx/2,Ny/2,Nz/2) = center_value;
        Uz(Nx/2,Ny/2,Nz/2) = center_value;
    }

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < Nx; i++) {
    for (int j = 0; j < Ny; j++) {
    for (int k = 0; k < Nz; k++) {
        //I am using ijk instead of xyz since it is the index, not the position anymore. position can be computed x = i*dx, etc.

        lame_parameters p = WATER_LAME_PARAMETERS;//params_at(i*dx,j*dy,k*dz);

        double lambda = p.lambda;
        double mu = p.mu;
        double rho = p.rho;


        Ux_nxt(i, j, k) = (dt*dt*(4*dx*dx*dy*dy*(-2*Ux(i, j, k) + Ux(i, j, -1 + k) + Ux(i, j, 1 + k))*mu 
                        + 4*dx*dx*dz*dz*(-2*Ux(i, j, k) + Ux(i, -1 + j, k) + Ux(i, 1 + j, k))*mu 
                        + dx*dy*dy*dz*(lambda + mu)*(Uz(-1 + i, j, -1 + k) - Uz(-1 + i, j, 1 + k) - Uz(1 + i, j, -1 + k) + Uz(1 + i, j, 1 + k)) 
                        + dx*dy*dz*dz*(lambda + mu)*(Uy(-1 + i, -1 + j, k) - Uy(-1 + i, 1 + j, k) - Uy(1 + i, -1 + j, k) + Uy(1 + i, 1 + j, k)) 
                        + 4*dy*dy*dz*dz*(lambda + 2*mu)*(-2*Ux(i, j, k) + Ux(-1 + i, j, k) + Ux(1 + i, j, k)))/4 
                        + dx*dx*dy*dy*dz*dz*(2*Ux(i, j, k) - Ux_prv(i, j, k))*rho)/(dx*dx*dy*dy*dz*dz*rho);

        Uy_nxt(i, j, k) = (dt*dt*(4*dx*dx*dy*dy*(-2*Uy(i, j, k) + Uy(i, j, -1 + k) + Uy(i, j, 1 + k))*mu 
                        + dx*dx*dy*dz*(lambda + mu)*(Uz(i, -1 + j, -1 + k) - Uz(i, -1 + j, 1 + k) - Uz(i, 1 + j, -1 + k) + Uz(i, 1 + j, 1 + k)) 
                        + 4*dx*dx*dz*dz*(lambda + 2*mu)*(-2*Uy(i, j, k) + Uy(i, -1 + j, k) + Uy(i, 1 + j, k)) 
                        + dx*dy*dz*dz*(lambda + mu)*(Ux(-1 + i, -1 + j, k) - Ux(-1 + i, 1 + j, k) - Ux(1 + i, -1 + j, k) + Ux(1 + i, 1 + j, k)) 
                        + 4*dy*dy*dz*dz*(-2*Uy(i, j, k) + Uy(-1 + i, j, k) + Uy(1 + i, j, k))*mu)/4 
                        + dx*dx*dy*dy*dz*dz*(2*Uy(i, j, k) - Uy_prv(i, j, k))*rho)/(dx*dx*dy*dy*dz*dz*rho);

        Uz_nxt(i, j, k) = (dt*dt*(4*dx*dx*dy*dy*(lambda + 2*mu)*(-2*Uz(i, j, k) + Uz(i, j, -1 + k) + Uz(i, j, 1 + k)) 
                        + dx*dx*dy*dz*(lambda + mu)*(Uy(i, -1 + j, -1 + k) - Uy(i, -1 + j, 1 + k) - Uy(i, 1 + j, -1 + k) + Uy(i, 1 + j, 1 + k)) 
                        + 4*dx*dx*dz*dz*(-2*Uz(i, j, k) + Uz(i, -1 + j, k) + Uz(i, 1 + j, k))*mu 
                        + dx*dy*dy*dz*(lambda + mu)*(Ux(-1 + i, j, -1 + k) - Ux(-1 + i, j, 1 + k) - Ux(1 + i, j, -1 + k) + Ux(1 + i, j, 1 + k)) 
                        + 4*dy*dy*dz*dz*(-2*Uz(i, j, k) + Uz(-1 + i, j, k) + Uz(1 + i, j, k))*mu)/4 
                        + dx*dx*dy*dy*dz*dz*(2*Uz(i, j, k) - Uz_prv(i, j, k))*rho)/(dx*dx*dy*dy*dz*dz*rho);

    }
    }
    }
}


// What shall we do ? nothing for now
void boundary_condition ( void )
{
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < Ny; y++) {
        for (int z = 0; z < Nz; z++) {
            Ux_nxt(-1, y, z) = Ux_nxt(Nx, y, z) = 0.0;
        }
    }

    #pragma omp parallel for collapse(2)
    for (int x = 0; x < Nx; x++) {
        for (int z = 0; z < Nz; z++) {
            Ux_nxt(x, -1, z) = Ux_nxt(x, Ny, z) = 0.0;
        }
    }

    #pragma omp parallel for collapse(2)
    for (int x = 0; x < Nx; x++) {
        for (int y = 0; y < Ny; y++) {
            Ux_nxt(x, y, -1) = Ux_nxt(x, y, Nz) = 0.0;
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
        boundary_condition();
        time_step(iteration*dt);

        // Rotate the time step buffers
        move_buffer_window();
    }
}


int simulate(real_t* model_data, int_t n_x, int_t n_y, int_t n_z, double r_dt, int r_max_iter, int r_snapshot_freq, double r_sensor_height, int_t r_model_nx, int_t r_model_ny)
{
    dt =r_dt;
    max_iteration = r_max_iter;
    snapshot_freq=r_snapshot_freq;
    sensor_height = r_sensor_height;
    SIM_LZ = MODEL_LZ + RESERVOIR_OFFSET + sensor_height;//need to add height of sensors, but thats a parameter

    model_Nx = r_model_nx;
    model_Ny = r_model_ny;

    //I need to create dx, dy, dz from the resolution given, knowing the shape of the reservoir (which is fixed) and adjust to that

    //FIRST PARSE AND SETUP SIMULATION PARAMETERS (done in domain_initialize)
    model = model_data;

    // Set up the initial state of the domain
    domain_initialize(n_x, n_y, n_z);

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
    exit ( EXIT_SUCCESS );
}


lame_parameters params_at(real_t x, real_t y, real_t z){

    //printf("x = %.4f, y = %.4f, z = %.4f\n", x, y, z);


    //1. am I (xy) on the model ? 
    if(RESERVOIR_OFFSET < x && x < MODEL_LX + RESERVOIR_OFFSET &&
        RESERVOIR_OFFSET < y && y < MODEL_LY + RESERVOIR_OFFSET){
        //yes!
        //printf("on the model, z = %lf\n", z);

        //2. am I IN the model ?

        //figure out closest indices (approximated for now)
        int_t x_idx = (int_t)((x - RESERVOIR_OFFSET) * (double)model_Nx / MODEL_LX);
        int_t y_idx = (int_t)((y - RESERVOIR_OFFSET) * (double)model_Ny / MODEL_LY);


        //model height at this point (assume RESERVOIR_OFFSET below model)
        //model stores negative value of depth, so I invert it
        real_t model_bottom = RESERVOIR_OFFSET - MODEL_AT(x_idx, y_idx);
        //printf("min: %lf, max: %lf\n", model_bottom, RESERVOIR_OFFSET + MODEL_LZ);


        if(model_bottom <= z && z < RESERVOIR_OFFSET + MODEL_LZ){
            // printf("x = %lf, y = %lf, RESERVOIR_OFFSET = %lf, MODEL_LX = %lf, MODEL_LY = %lf, model_Nx = %d, model_Ny = %d\n", x, y, RESERVOIR_OFFSET, MODEL_LX, MODEL_LY, model_Nx, model_Ny);
            printf("x_idx = %d, y_idx = %d\n", x_idx, y_idx);

            //I am in the model !
            //printf("in the model !\n");
            return PLASTIC_LAME_PARAMETERS;
        }
    }

    return WATER_LAME_PARAMETERS;
}

void show_model(){

    char filename[256];
    sprintf ( filename, "model.dat" );
    FILE *out = fopen ( filename, "wb" );
    if (!out) {
        printf("[ERROR] File pointer is NULL!\n");
        exit(EXIT_FAILURE);
    }
    // for ( int_t i=0; i<Ny; i++ )//take some slice 
    // {
    //     fwrite ( &Ux(Nx/2,i,0), sizeof(real_t), Nz, out );//take horizontal slice from middle, around yz axis
    // }

    //save norm of vector
//     for (int_t i = 0; i < Ny; i++) {
//     for (int_t j = 0; j < Nz; j++) {
//         real_t norm = sqrt(Ux(Nx/2, i, j) * Ux(Nx/2, i, j) +
//                            Uy(Nx/2, i, j) * Uy(Nx/2, i, j) +
//                            Uz(Nx/2, i, j) * Uz(Nx/2, i, j));
//         fwrite(&norm, sizeof(real_t), 1, out);
//     }
// }
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Ny; j++) {
            for (int i = 0; i < Nx; i++) {
                lame_parameters params = params_at(i*dx, j*dy, k*dz);
                unsigned char in_model = (params.lambda == PLASTIC_LAME_PARAMETERS.lambda) ? 1 : 0;
                fwrite(&in_model, sizeof(unsigned char), 1, out);
            }
        }
    }
    printf("written to file\n");

    fclose ( out );

}
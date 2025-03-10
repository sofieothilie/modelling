#define _XOPEN_SOURCE 600
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/time.h>

#include "modeling.h"


typedef struct {
    real_t lambda, mu, rho;
} lame_parameters;

lame_parameters WATER_LAME_PARAMETERS = {
    .lambda = 1555.5555555556,
    .mu = 666.6,
    .rho = 0.5
};

lame_parameters PLASTIC_LAME_PARAMETERS = {
    .lambda = 0,
    .mu = 0,
    .rho = 0
};

lame_parameters params_at(int_t x, int_t y, int_t z, real_t* model);


// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

int_t max_iteration = 100;
int_t snapshot_freq = 1;

// Simulation parameters: size, step count, and how often to save the state
// int_t
//     N = 256,
//     M = 256,
//     max_iteration = 4000,
//     snapshot_freq = 20;

int_t model_Nx, model_Ny;

real_t* model;

int_t Nx, Ny, Nz;//they must be parsed in the main before used anywhere, I guess that's a very bad way of doing it, but the template did it like this

real_t dt = 0.01;
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
    for ( int_t i=0; i<Ny; i++ )//take some slice 
    {
        fwrite ( &Uy(Nx/2,i,0), sizeof(real_t), Nz, out );//take horizontal slice from middle, around yz axis
    }
    fclose ( out );
}


// Set up our three buffers, and fill two with an initial perturbation
void domain_initialize (int_t n_x, int_t n_y, int_t n_z)//at this point I can load an optional starting state. (I would need two steps actually)
{
    Nx = n_x;
    Ny = n_y;
    Nz = n_z;

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
    int n = 1;
    printf("init values\n");
    //get some value in the center
    for (int x = Nx/2 - n; x < Nx/2+n; x++) {
    for (int y = Ny/2 - n; y < Ny/2+n; y++) {
    for (int z = Nz/2 - n; z < Nz/2+n; z++) {
        Uy(x,y,z) = Uz(x,y,z) = .5;
        Uy_prv(x,y,z) = Uz_prv(x,y,z) = 1;
    }}}

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
void time_step ( void )
{
    for (int x = 0; x < Nx; x++) {
    for (int y = 0; y < Ny; y++) {
    for (int z = 0; z < Nz; z++) {

        lame_parameters p = params_at(x,y,z, model);

        double lambda = p.lambda;
        double mu = p.mu;
        double rho = p.rho;


        Ux_nxt(x, y, z) = (dt*dt*(4*dx*dx*dy*dy*(-2*Ux(x, y, z) + Ux(x, y, -1 + z) + Ux(x, y, 1 + z))*mu 
                        + 4*dx*dx*dz*dz*(-2*Ux(x, y, z) + Ux(x, -1 + y, z) + Ux(x, 1 + y, z))*mu 
                        + dx*dy*dy*dz*(lambda + mu)*(Uz(-1 + x, y, -1 + z) - Uz(-1 + x, y, 1 + z) - Uz(1 + x, y, -1 + z) + Uz(1 + x, y, 1 + z)) 
                        + dx*dy*dz*dz*(lambda + mu)*(Uy(-1 + x, -1 + y, z) - Uy(-1 + x, 1 + y, z) - Uy(1 + x, -1 + y, z) + Uy(1 + x, 1 + y, z)) 
                        + 4*dy*dy*dz*dz*(lambda + 2*mu)*(-2*Ux(x, y, z) + Ux(-1 + x, y, z) + Ux(1 + x, y, z)))/4 
                        + dx*dx*dy*dy*dz*dz*(2*Ux(x, y, z) - Ux_prv(x, y, z))*rho)/(dx*dx*dy*dy*dz*dz*rho);

        Uy_nxt(x, y, z) = (dt*dt*(4*dx*dx*dy*dy*(-2*Uy(x, y, z) + Uy(x, y, -1 + z) + Uy(x, y, 1 + z))*mu 
                        + dx*dx*dy*dz*(lambda + mu)*(Uz(x, -1 + y, -1 + z) - Uz(x, -1 + y, 1 + z) - Uz(x, 1 + y, -1 + z) + Uz(x, 1 + y, 1 + z)) 
                        + 4*dx*dx*dz*dz*(lambda + 2*mu)*(-2*Uy(x, y, z) + Uy(x, -1 + y, z) + Uy(x, 1 + y, z)) 
                        + dx*dy*dz*dz*(lambda + mu)*(Ux(-1 + x, -1 + y, z) - Ux(-1 + x, 1 + y, z) - Ux(1 + x, -1 + y, z) + Ux(1 + x, 1 + y, z)) 
                        + 4*dy*dy*dz*dz*(-2*Uy(x, y, z) + Uy(-1 + x, y, z) + Uy(1 + x, y, z))*mu)/4 
                        + dx*dx*dy*dy*dz*dz*(2*Uy(x, y, z) - Uy_prv(x, y, z))*rho)/(dx*dx*dy*dy*dz*dz*rho);

        Uz_nxt(x, y, z) = (dt*dt*(4*dx*dx*dy*dy*(lambda + 2*mu)*(-2*Uz(x, y, z) + Uz(x, y, -1 + z) + Uz(x, y, 1 + z)) 
                        + dx*dx*dy*dz*(lambda + mu)*(Uy(x, -1 + y, -1 + z) - Uy(x, -1 + y, 1 + z) - Uy(x, 1 + y, -1 + z) + Uy(x, 1 + y, 1 + z)) 
                        + 4*dx*dx*dz*dz*(-2*Uz(x, y, z) + Uz(x, -1 + y, z) + Uz(x, 1 + y, z))*mu 
                        + dx*dy*dy*dz*(lambda + mu)*(Ux(-1 + x, y, -1 + z) - Ux(-1 + x, y, 1 + z) - Ux(1 + x, y, -1 + z) + Ux(1 + x, y, 1 + z)) 
                        + 4*dy*dy*dz*dz*(-2*Uz(x, y, z) + Uz(-1 + x, y, z) + Uz(1 + x, y, z))*mu)/4 
                        + dx*dx*dy*dy*dz*dz*(2*Uz(x, y, z) - Uz_prv(x, y, z))*rho)/(dx*dx*dy*dy*dz*dz*rho);

    }
    }
    }
}


// What shall we do ? nothing for now
void boundary_condition ( void )
{
    // for ( int_t i=0; i<N; i++ )
    // {
    //     U(i,-1) = U(i,1);
    //     U(i,N)  = U(i,N-2);
    // }
    // for ( int_t j=0; j<N; j++ )
    // {
    //     U(-1,j) = U(1,j);
    //     U(M,j)  = U(M-2,j);
    // }

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
            domain_save ( iteration / snapshot_freq );
        }

        // Derive step t+1 from steps t and t-1
        boundary_condition();
        time_step();

        // Rotate the time step buffers
        move_buffer_window();
    }
}


int simulate(real_t* model_data, int_t n_x, int_t n_y, int_t n_z)
{

    //FIRST PARSE AND SETUP SIMULATION PARAMETERS (done in domain_initialize)
    model = model_data;

    // Set up the initial state of the domain
    domain_initialize(n_x, n_y, n_z);

    struct timeval t_start, t_end;

    gettimeofday ( &t_start, NULL );
    simulation_loop();
    gettimeofday ( &t_end, NULL );

    printf ( "Total elapsed time: %lf seconds\n",
        WALLTIME(t_end) - WALLTIME(t_start)
    );

    // Clean up and shut down
    domain_finalize();
    exit ( EXIT_SUCCESS );
}


lame_parameters params_at(int_t x, int_t y, int_t z, real_t* model){
    //figure out which x,y indices it is pointing to on the model
    int_t x_idx, y_idx;
    //figure out what the model height means and where I am relative to it (I guess I decide this)
    real_t model_height = model[x_idx * model_Ny + y_idx];//which index ordering ? should be x_idx * model_Ny + y_idx since row_major

    //

    //if higher, then water, if lower, then container, if lower than container then smth else 
    return WATER_LAME_PARAMETERS;
}

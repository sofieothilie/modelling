#include "argument_utils.h"
#include "simulation.h"
#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_RECORD_SIZE (10 * 1024 * 1024) // 10 MB guard

/*
 * Read the binary "meta" file format described in the repo notes.
 * This populates options->sensors with all receiver positions found in the file
 * and sets options->n_sensors accordingly.
 * Returns uint32_t number of sources found in the universe. On error, returns -1.
 */
uint32_t read_universe(FILE *f) {
    /* meta header layout */
    /* As of now, these are not used for the simulation. Hehe, buuut idk. Maybe the future holds for different things.
    */
    uint32_t n_samples = 0;
    uint32_t n_traces = 0;
    float dt = 0.0f;
    char data_path[256] = {0};
    uint32_t n_sources = 0;

    if(fread(&n_samples, sizeof(n_samples), 1, f) != 1) return -1;
    if(fread(&n_traces, sizeof(n_traces), 1, f) != 1) return -1;
    if(fread(&dt, sizeof(dt), 1, f) != 1) return -1;
    if(fread(&data_path, sizeof(data_path), 1, f) != 1) return -1;
    if(fread(&n_sources, sizeof(n_sources), 1, f) != 1) return -1;

    printf("n_samples: %i\n", n_samples);
    printf("n_traces: %i\n", n_traces);
    printf("dt: %f\n", dt);
    printf("data_path: %s\n", data_path);
    printf("n_sources: %i\n", n_sources);
    // Return number of sources to know how many times to call read_world
    return n_sources;
}


/*
* Read a world record from the meta file.
* Populates p->source and p->sensors accordingly.
* Returns 0 on success, -1 on failure.
*/

int read_world(FILE *f, simulation_parameters *p){
    /* start with zero sensors */
    float sx = 0.0f, sy = 0.0f;

    uint32_t n_receivers = 0;

    if(fread(&sx, sizeof(sx), 1, f) != 1) goto fail;
    if(fread(&sy, sizeof(sy), 1, f) != 1) goto fail;
    if(fread(&n_receivers, sizeof(n_receivers), 1, f) != 1) goto fail;

    /* store source position */
    p->source.x = sx;
    p->source.y = sy;
    p->source.z = p->transducer_height;
    printf("Source position: (%.3f, %.3f, %.3f)\n", sx, sy, p->transducer_height);

    /* read receiver X and Y arrays */
    float *gx = (float *) malloc(n_receivers * sizeof(float));
    float *gy = (float *) malloc(n_receivers * sizeof(float));
    if(!gx || !gy) {
        free(gx);
        free(gy);
        goto fail;
    }

    if(fread(gx, sizeof(float), n_receivers, f) != n_receivers) {
        free(gx);
        free(gy);
        goto fail;
    }
    if(fread(gy, sizeof(float), n_receivers, f) != n_receivers) {
        free(gx);
        free(gy);
        goto fail;
    }

    Position *tmp = (Position *) realloc(p->sensors, n_receivers * sizeof(Position));
    if(!tmp) {
        free(gx);
        free(gy);
        goto fail;
    }

    p->n_sensors = n_receivers;
    p->sensors = tmp;

    for(uint32_t i = 0; i < n_receivers; ++i) {
        p->sensors[i].x = gx[i];
        p->sensors[i].y = gy[i];
        /* place receiver at configured transducer height */
        p->sensors[i].z = p->transducer_height;
    }

    free(gx);
    free(gy);

    return 0;

fail:
    if(p->sensors) free(p->sensors);
    p->sensors = NULL;
    p->n_sensors = 0;
    return -1;
}

int main(int argc, char **argv) {
    OPTIONS *options = parse_args(argc, argv);

    // first compute dh.
    real_t smallest_wavelength = WATER_PARAMETERS.k / SRC_FREQUENCY;
    real_t dh = smallest_wavelength / options->ppw;

    real_t dt = 0.9 * dh / (PLASTIC_PARAMETERS.k * sqrt(3));
    options->dt = dt;

    int_t Nx = (int) ceil(options->sim_Lx / dh);
    int_t Ny = (int) ceil(options->sim_Ly / dh);
    int_t Nz = (int) ceil(options->sim_Lz / dh);

    Dimensions d = {
        .Nx = Nx,
        .Ny = Ny,
        .Nz = Nz,
        .padding = options->padding,
        .dh = dh,
        .dt = options->dt,
    };

    print_start_info(d);

    if(options->print_info) {
        // do not run, only print launch informations
        return 0;
    }

    simulation_parameters p = {
        .dimensions = d,
        .sim_Lx = options->sim_Lx,
        .sim_Ly = options->sim_Ly,
        .sim_Lz = options->sim_Lz,
        .snapshot_freq = options->snapshot_frequency,
        .dt = options->dt,
        .RTM = options->RTM,
        .transducer_height = options->transducer_height,
    };

    FILE *f = fopen(options->meta_path, "rb");
    if(!f) {
        printf("Failed to open meta file.\n");
        return -1;
    }

    uint32_t n_sources = read_universe(f);
    if (n_sources == (uint32_t)-1) {
        printf("Failed to read universe from meta file.\n");
        fclose(f);
        return -1;
    }

    double *model = open_model("data/model.bin");

    for (int i = 1; i <= n_sources; i++) {

        read_world(f, &p);

        real_t RTT_n_iteration = RTT(model, options, &p) / dt;
        p.max_iter = (int_t)RTT_n_iteration;
        printf("using %lf iterations\n", RTT_n_iteration);

        if (options->RTM) {
            // int err = simulate_rtm(&p);
            printf("RTM not implemented yet.\n");
        }
        else {
            int err = simulate_wave(&p);
            // printf("Wave simulation not implemented yet.\n");
        }
        break; // only simulate first source for now
    } 
    free_model(model);

    fclose(f);

    return 0;
}

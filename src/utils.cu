#include "argument_utils.h"
#include "utils.h"
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <unistd.h>

int init_cuda() {
    int dev_count;
    cudaErrorCheck(cudaGetDeviceCount(&dev_count));

    if(dev_count == 0) {
        fprintf(stderr, "No CUDA-compatible devices found\n");
        return 0;
    }

    cudaErrorCheck(cudaSetDevice(0));

    cudaDeviceProp prop;
    cudaErrorCheck(cudaGetDeviceProperties(&prop, 0));

    // Print the device properties
    printf("----------------------------------------\n");
    printf("Device count: %d\n", dev_count);
    printf("Using device 0: %s\n", prop.name);
    printf("\tCompute capability: %d.%d\n", prop.major, prop.minor);
    printf("\tMultiprocessors: %d\n", prop.multiProcessorCount);
    printf("\tWarp size: %d\n", prop.warpSize);
    printf("\tGlobal memory: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
    printf("\tPer-block shared memory: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("\tPer-block registers: %d\n", prop.regsPerBlock);
    printf("----------------------------------------\n\n");

    return 1;
}

int get_terminal_width() {
    struct winsize w;
    if(ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == -1) {
        return 80; // fallback default
    }
    return w.ws_col;
}

void clear_three_lines() {
    printf("\33[2K\r"); // clear current line (progress bar)
    printf("\33[A");    // move cursor up
    printf("\33[2K\r"); // clear previous line (info)
    printf("\33[A");    // move cursor up
    printf("\33[2K\r");
    fflush(stdout);
}

void print_progress_bar(int current_iteration,
                        int total_iterations,
                        struct timeval start,
                        struct timeval now) {
    int width = get_terminal_width() - 10; // total width of the bar
    double progress = (double) current_iteration / total_iterations;
    int pos = (int) (progress * width);

    // Calculate elapsed time
    double elapsed_time = WALLTIME(now) - WALLTIME(start);

    // Estimate time per iteration
    double time_per_iteration = elapsed_time / current_iteration;

    // Time remaining
    int total_time = (int) (time_per_iteration * total_iterations);
    // int elapsed_time_int = (int)((total_iterations - current_iteration) * time_per_iteration);
    if(current_iteration > 0) {
        clear_three_lines();
    }

    printf("[");
    for(int i = 0; i < width; ++i) {
        if(i < pos)
            printf("#");
        else
            printf(" ");
    }
    printf("] %.1f%%\n", progress * 100.0);

    // Print the percentage below the progress bar
    printf("Elapsed Time: %02d:%02d:%02d",
           (int) elapsed_time / 3600,
           ((int) elapsed_time % 3600) / 60,
           (int) elapsed_time % 60);
    printf(" / Total time of simulation: %02d:%02d:%02d",
           total_time / 3600,
           (total_time % 3600) / 60,
           total_time % 60);
    printf(" - Time per iteration: %.4lfs\n", time_per_iteration);

    fflush(stdout);
}

void print_start_info(Dimensions dimensions) {

    int_t Nx = dimensions.Nx;
    int_t Ny = dimensions.Ny;
    int_t Nz = dimensions.Nz;
    real_t dh = dimensions.dh;
    real_t dt = dimensions.dt;
    int_t padding = dimensions.padding;

    printf("-----------------------------------\n");
    printf("------ Simulation Parameters ------\n");
    printf("-----------------------------------\n");
    printf("dh: %lf\n", dh);
    printf("dt: %e\n", dt);
    printf("Grid dimensions: Nx = %d, Ny = %d, Nz = %d\n", Nx, Ny, Nz);
    printf("Added padding cells: %d\n", padding);
    printf("\n-----------------------------------\n");

    printf("---- Memory Allocation Details ----\n");
    printf("-----------------------------------\n");
    size_t tot_x = Nx + 2 * padding + 2;
    size_t tot_y = Ny + 2 * padding + 2;
    size_t tot_z = Nz + 2 * padding + 2;

    size_t full_buffer_size = tot_x * tot_y * tot_z * sizeof(real_t);
    size_t PML_shell_size = 2 * (Nx + 2 * padding + 2) * (Ny + 2 * padding + 2) * (padding + 1)
                          + 2 * (padding + 1) * (Ny + 2 * padding + 2) * (Nz + 2 * padding + 2)
                          + 2 * (Nx + 2 * padding + 2) * (padding + 1) * (Nz + 2 * padding + 2);

    PML_shell_size *= sizeof(real_t);

    cudaDeviceProp prop;
    cudaErrorCheck(cudaGetDeviceProperties(&prop, 0));
    double mem_capacity = prop.totalGlobalMem / (1024. * 1024.);

    printf("3 sim state x (2 full buffers + 1 PML shell)\n");
    printf("1 buffer size: (%zux%zux%zu) x %zu bytes = %.2lf MB\n",
           tot_x,
           tot_y,
           tot_z,
           sizeof(real_t),
           full_buffer_size / (1024. * 1024.));
    printf("PML shell size: %.2lf MB\n", PML_shell_size / (1024. * 1024.));
    printf(" => total allocated = %.2lf MB / %.2lf MB\n",
           3. * (2. * full_buffer_size + PML_shell_size) / (1024. * 1024.),
           mem_capacity);
    printf("-----------------------------------\n\n");
}

Coords PositionToCoords(Position p, Dimensions d, Position source) {
    real_t dh = d.dh;
    const int_t Nx = d.Nx;
    const int_t Ny = d.Ny;
    const int_t padding = d.padding;

    // Calculate the simulation domain center in world coordinates (centered around source)
    const real_t sim_center_x = source.x;
    const real_t sim_center_y = source.y;
    
    // Convert absolute world position to grid-relative position
    // Grid origin (0,0,0) corresponds to (source - half_domain_size) in world space
    // So position source.x maps to grid coordinate Nx/2
    const real_t relative_x = p.x - sim_center_x;
    const real_t relative_y = p.y - sim_center_y;
    const real_t relative_z = p.z;

    // Convert to grid coordinates (grid center is at Nx/2 + padding)
    Coords coords;
    coords.x = (int_t)(relative_x / dh + Nx / 2.0 + padding);
    coords.y = (int_t)(relative_y / dh + Ny / 2.0 + padding);
    coords.z = (int_t)(relative_z / dh);
    
    if(coords.x < 0 || coords.x >= 2 * d.padding + d.Nx || coords.y < 0
       || coords.y >= 2 * d.padding + d.Ny || coords.z < 0 || coords.z >= 2 * d.padding + d.Nz) {
        fprintf(stderr, "Error: Coordinates (%f, %f, %f) out of bounds.\n", p.x, p.y, p.z);
        coords.x = coords.y = coords.z = -1; // Set to invalid values
    }
    return coords;
}

// computes the number of iterations for a round trip time, assuming worst case model traversal
real_t RTT(double *model, OPTIONS *options, simulation_parameters *p) {
    // need to get how much is plastic, how much is water.
    real_t z_total = options->sim_Lz;

    real_t model_start = options->transducer_height;
    real_t model_shallowest_end = model_start + MODEL_LZ;
    real_t model_deepest_end = model_start;

    // iterate on whole simulated part of model and find shallowest part
    // wave travels faster in the plastic, so the worst case is when it goes through the shallowest
    // part
    real_t increment = 0.001; // 1mm increment
    for(real_t x = (p->sensors[0].x) - options->sim_Lx / 2.0;
        x <= (p->sensors[0].x) + options->sim_Lx / 2.0;
        x += increment) {
        for(real_t y = p->sensors[0].y - options->sim_Ly / 2.0;
            y <= p->sensors[0].y + options->sim_Ly / 2.0;
            y += increment) {
            // Clamp x and y to valid model bounds
            if(x < 0 || x >= MODEL_LX || y < 0 || y >= MODEL_LY) {
                continue; // Skip out-of-bounds positions
            }
            
            const int_t x_idx = x * MODEL_NX / MODEL_LX;
            const int_t y_idx = y * MODEL_NY / MODEL_LY;
            
            // Safety check: ensure indices are within bounds
            if(x_idx < 0 || x_idx >= MODEL_NX || y_idx < 0 || y_idx >= MODEL_NY) {
                continue;
            }

            const real_t model_depth =
                model_start + MODEL_LZ + (double) model[x_idx * MODEL_NY + y_idx];

            // printf("model value %lf\n", model[x_idx * MODEL_NY + y_idx]);
            // capture smallest value
            if(model_depth < model_shallowest_end)
                model_shallowest_end = model_depth;

            if(model_depth > model_deepest_end)
                model_deepest_end = model_depth;
        }
    }

    printf("model bottom spans {%lf - %lf}\n", model_shallowest_end, model_deepest_end);

    if(model_shallowest_end > z_total) {
        printf("[Note] Simulation not deep enough, the plastic goes %lf lower than that!\n",
               model_shallowest_end - z_total);
        model_shallowest_end = z_total; // do not simulate too much
    }
    if(model_deepest_end < z_total) {
        printf("[Note] Simulating %lf m too much\n", z_total - model_deepest_end);
    }

    real_t z_after_model = z_total - model_shallowest_end;

    real_t distance_in_plastic = model_shallowest_end - model_start;
    real_t distance_in_water = model_deepest_end - distance_in_plastic;

    real_t vertical_travel_time =
        distance_in_plastic / PLASTIC_PARAMETERS.k + distance_in_water / WATER_PARAMETERS.k;
    // horoizontal travel approx, supposed proportional to vertical
    real_t horizontal_travel_time =
        0.5 * vertical_travel_time * max(options->sim_Lx, options->sim_Ly) / options->sim_Lz;

    real_t emissions_latency = 3.125e-5; // 250 samples at 8Mhz
    real_t safety_latency = 5 * emissions_latency;

    real_t total_sim_time = 2
                              * sqrt(vertical_travel_time * vertical_travel_time
                                     + horizontal_travel_time * horizontal_travel_time)
                          + emissions_latency + safety_latency;

    return total_sim_time;
}
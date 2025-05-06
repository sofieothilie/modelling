#include  "utils.h"
#include <stdio.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <sys/time.h>


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
    // printf("----------------------------------------\n");
    // printf("Device count: %d\n", dev_count);
    // printf("Using device 0: %s\n", prop.name);
    // printf("\tCompute capability: %d.%d\n", prop.major, prop.minor);
    // printf("\tMultiprocessors: %d\n", prop.multiProcessorCount);
    // printf("\tWarp size: %d\n", prop.warpSize);
    // printf("\tGlobal memory: %zu MB\n", prop.totalGlobalMem / (1024 * 1024));
    // printf("\tPer-block shared memory: %zu KB\n", prop.sharedMemPerBlock / 1024);
    // printf("\tPer-block registers: %d\n", prop.regsPerBlock);
    // printf("----------------------------------------\n\n");

    

    return 1;
}

int get_terminal_width() {
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == -1) {
        return 80; // fallback default
    }
    return w.ws_col;
}


void clear_three_lines() {
    printf("\33[2K\r");   // clear current line (progress bar)
    printf("\33[A");      // move cursor up
    printf("\33[2K\r");   // clear previous line (info)
    printf("\33[A");      // move cursor up
    printf("\33[2K\r");
    fflush(stdout);
}

void print_progress_bar(int current_iteration, int total_iterations, struct timeval start, struct timeval now) {
    int width = get_terminal_width() - 10; // total width of the bar
    double progress = (double)current_iteration / total_iterations;
    int pos = (int)(progress * width);
    
    // Calculate elapsed time
    double elapsed_time = WALLTIME(now) - WALLTIME(start);
    
    // Estimate time per iteration
    double time_per_iteration = elapsed_time / current_iteration;
    
    // Time remaining
    int total_time  = (int)(time_per_iteration * total_iterations);
    // int elapsed_time_int = (int)((total_iterations - current_iteration) * time_per_iteration);
    if(current_iteration > 0){
        clear_three_lines();
    }

    printf("[");
    for (int i = 0; i < width; ++i) {
        if (i < pos) printf("#");
        else printf(" ");
    }
    printf("] %.1f%%\n", progress * 100.0);
    
    // Print the percentage below the progress bar
    printf("Elapsed Time: %02d:%02d:%02d", (int)elapsed_time / 3600, ((int)elapsed_time % 3600) / 60, (int)elapsed_time % 60);
    printf(" / Total time of simulation: %02d:%02d:%02d", total_time / 3600, (total_time % 3600) / 60, total_time % 60);
    printf(" - Time per iteration: %.4lfs\n", time_per_iteration);

    fflush(stdout);

}

void print_start_info(Dimensions dimensions){

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
           3. * (2. * full_buffer_size + PML_shell_size) / (1024. * 1024.), mem_capacity);
    printf("-----------------------------------\n\n");
}
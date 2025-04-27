#include  "utils.h"
#include <stdio.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <sys/time.h>


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

    

    return true;
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
    printf(" / Total time of simulation: %02d:%02d:%02d\n", total_time / 3600, (total_time % 3600) / 60, total_time % 60);

    fflush(stdout);

}

#include "argument_utils.h"
#include "simulation.h"

#include <getopt.h>
#include <memory.h>
#include <string.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

OPTIONS
*parse_args(int argc, char **argv) {
    /*
     * Argument parsing: default parameters
     */

    real_t sim_Lx = 0.01, sim_Ly = 0.01, sim_Lz = 0.01;
    real_t transducer_height = 0.023;
    int n_sensors = 0; /* start empty; read_universe will fill if provided */
    real_t dt = 1e-8;
    real_t ppw = 10;
    int_t max_iteration = 50;
    int_t padding = 5;
    int_t snapshot_frequency = 10;
    int_t print_info = 0;
    int_t RTM = 0;
    Position *sensors = NULL;


    static struct option const long_options[] = {
        { "help", no_argument, 0, '?' },
        { "simul_x", required_argument, 0, 'x' },
        { "simul_y", required_argument, 0, 'y' },
        { "simul_z", required_argument, 0, 'z' },
        { "meta-file", required_argument, 0, 'm' },
        { "ppw", required_argument, 0, 'p' },
        { "padding", required_argument, 0, 'P' },
        { "dt", required_argument, 0, 't' },
        { "max_iteration", required_argument, 0, 'i' },
        { "snapshot_frequency", required_argument, 0, 's' },
        { "print-info", no_argument, 0, 'I' },
        { 0, 0, 0, 0 },
        { "RTM", no_argument, 0, 'R' },
    };

    static char const *short_options = "?x:y:z:p:P:t:i:s:I:m:R:";
        /* fixed-size meta path in binary meta format */
        #define META_PATH_FIELD_SIZE 256
    char *meta_path_opt = NULL; /* parsed by getopt if -m/--meta-file is used */
    {
        char *endptr;
        int c;
        int option_index = 0;

        while((c = getopt_long(argc, argv, short_options, long_options, &option_index)) != -1) {
            switch(c) {
                case '?':
                    help(argv[0], 0, NULL);
                    exit(0);
                    break;
                case 'x':
                    sim_Lx = strtod(optarg, &endptr);
                    if(endptr == optarg || sim_Lx < 0) {
                        help(argv[0], c, optarg);
                        return NULL;
                    }
                    break;
                case 'y':
                    sim_Ly = strtod(optarg, &endptr);
                    if(endptr == optarg || sim_Ly < 0) {
                        help(argv[0], c, optarg);
                        return NULL;
                    }
                    break;
                case 'z':
                    sim_Lz = strtod(optarg, &endptr);
                    if(endptr == optarg || sim_Lz < 0) {
                        help(argv[0], c, optarg);
                        return NULL;
                    }
                    break;
                case 'p':
                    ppw = strtod(optarg, &endptr);
                    if(endptr == optarg || ppw < 0) {
                        if(ppw <= 2) {
                            printf("%lf points per wavelength is way too low, that has no chance "
                                   "of working man\n",
                                   ppw);
                        }
                        help(argv[0], c, optarg);
                        return NULL;
                    }
                    break;
                case 'P':
                    padding = strtol(optarg, &endptr, 10);
                    if(endptr == optarg || padding < 0) {
                        help(argv[0], c, optarg);
                        return NULL;
                    }
                    break;

                case 't':
                    dt = strtod(optarg, &endptr);
                    if(endptr == optarg || dt <= 0) {
                        help(argv[0], c, optarg);
                        return NULL;
                    }
                    break;
                case 'i':
                    max_iteration = strtol(optarg, &endptr, 10);
                    if(endptr == optarg || max_iteration < 0) {
                        help(argv[0], c, optarg);
                        return NULL;
                    }
                    break;
                case 's':
                    snapshot_frequency = strtol(optarg, &endptr, 10);
                    if(endptr == optarg || snapshot_frequency < 0) {
                        help(argv[0], c, optarg);
                        return NULL;
                    }
                    break;
                case 'I':
                    printf("caught an I\n");
                    print_info = 1;
                    break;
                case 'm':
                    /* copy -m/--meta-file argument for later */
                    if(optarg) {
                        /* validate length immediately at parse time */
                        if(strlen(optarg) >= META_PATH_FIELD_SIZE) {
                            fprintf(stderr,
                                    "error: meta-path '%s' is too long (%zu bytes) — max %d bytes allowed\n",
                                    optarg, strlen(optarg), META_PATH_FIELD_SIZE - 1);
                            return NULL;
                        }

                        /* replace existing value if present */
                        if(meta_path_opt) free(meta_path_opt);
                        meta_path_opt = strdup(optarg);
                        if(!meta_path_opt) {
                            perror("strdup");
                            return NULL;
                        }
                    }
                    break;
                case 'R':
                    RTM = strtol(optarg, &endptr, 10);
                    break;
                default:
                    abort();
            }
        }
    }

    /* determine final meta_path (option takes precedence over positional) */
    char *meta_path = NULL;
    if(meta_path_opt) {
        meta_path = meta_path_opt; /* use provided -m value */
    } else if(optind < argc) {
        /* positional argument provided; copy then validate below */
        meta_path = strdup(argv[optind]);
        if(!meta_path) {
            perror("strdup");
            if(meta_path_opt) free(meta_path_opt);
            return NULL;
        }
    }

    OPTIONS *args_parsed = malloc(sizeof(OPTIONS));
    args_parsed->sim_Lx = sim_Lx;
    args_parsed->sim_Ly = sim_Ly;
    args_parsed->sim_Lz = sim_Lz;
    args_parsed->dt = dt;
    args_parsed->print_info = print_info;
    args_parsed->ppw = ppw;
    args_parsed->padding = padding;
    args_parsed->max_iteration = max_iteration;
    args_parsed->snapshot_frequency = snapshot_frequency;
    args_parsed->sensors = sensors;
    args_parsed->RTM = RTM;
    args_parsed->n_sensors = n_sensors;
    args_parsed->transducer_height = transducer_height;
    args_parsed->meta_path = meta_path;

    return args_parsed;
}

void help(char const *exec, char const opt, char const *optarg) {
    FILE *out = stdout;

    if(opt != 0) {
        out = stderr;
        if(optarg) {
            fprintf(out, "Invalid parameter - %c %s\n", opt, optarg);
        } else {
            fprintf(out, "Invalid parameter - %c\n", opt);
        }
    }

    // fprintf(out, "%s [options]\n", exec);
    // fprintf(out, "\n");
    // fprintf(out, "Options                   Description                     Restriction
    // Default\n"); fprintf(out, "  -m, --y_size            size of the y dimension         n>0
    // 256\n"    ); fprintf(out, "  -n, --x_size            size of the x dimension         n>0
    // 256\n"    ); fprintf(out, "  -i, --max_iteration     number of iterations            i>0
    // 100000\n" ); fprintf(out, "  -s, --snapshot_freq     snapshot frequency              s>0
    // 1000\n"  );

    // fprintf(out, "\n");
    // fprintf(out, "Example: %s -m 256 -n 256 -i 100000 -s 1000\n", exec);
}

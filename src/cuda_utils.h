

#define cudaErrorCheck(ans)                                                                        \
    {                                                                                              \
        gpuAssert((ans), __FILE__, __LINE__);                                                      \
    }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true);


bool init_cuda();
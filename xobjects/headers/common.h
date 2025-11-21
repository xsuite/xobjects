// copyright ################################# //
// This file is part of the Xtrack Package.    //
// Copyright (c) CERN, 2025.                   //
// ########################################### //


#ifndef XOBJECTS_COMMON_H
#define XOBJECTS_COMMON_H

/*
    Common macros for vectorization and parallelization, as well as common
    arithmetic operations.
*/

#ifdef XO_CONTEXT_CPU_SERIAL
    // We are on CPU, without OpenMP

    #define VECTORIZE_OVER(INDEX_NAME, COUNT) \
        for (int64_t INDEX_NAME = 0; INDEX_NAME < (COUNT); INDEX_NAME++) {

    #define END_VECTORIZE \
        }
#endif  // XO_CONTEXT_CPU_SERIAL

#ifdef XO_CONTEXT_CPU_OPENMP
    // We are on CPU with the OpenMP context switched on

    #define VECTORIZE_OVER(INDEX_NAME, COUNT) \
        _Pragma("omp parallel for") \
        for (int64_t INDEX_NAME = 0; INDEX_NAME < (COUNT); INDEX_NAME++) {

    #define END_VECTORIZE \
        }
#endif  // XO_CONTEXT_CPU_OPENMP


#ifdef XO_CONTEXT_CUDA
    // We are on a CUDA GPU

    #define VECTORIZE_OVER(INDEX_NAME, COUNT) { \
            int64_t INDEX_NAME = blockDim.x * blockIdx.x + threadIdx.x; \
            if (INDEX_NAME < (COUNT)) {

    #define END_VECTORIZE \
            } \
        }
#endif  // XO_CONTEXT_CUDA


#ifdef XO_CONTEXT_CL
    // We are on an OpenCL GPU

    #define VECTORIZE_OVER(INDEX_NAME, COUNT) \
        { \
            int64_t INDEX_NAME = get_global_id(0); \
            if (INDEX_NAME < (COUNT)) { \

    #define END_VECTORIZE \
            } \
        }
#endif  // XO_CONTEXT_CL


/*
    Qualifier keywords for GPU and optimisation
*/

#ifdef XO_CONTEXT_CPU // for both serial and OpenMP
    #define GPUKERN
    #define GPUFUN      static inline
    #define GPUGLMEM
    #define RESTRICT    restrict
#endif


#ifdef XO_CONTEXT_CUDA
    #define GPUKERN     __global__
    #define GPUFUN      __device__
    #define GPUGLMEM
    #define RESTRICT
#endif // XO_CONTEXT_CUDA


#ifdef XO_CONTEXT_CL
    #define GPUKERN     __kernel
    #define GPUFUN
    #define GPUGLMEM    __global
    #define RESTRICT
#endif // XO_CONTEXT_CL


/*
    Common maths-related macros
*/

#define POW2(X) ((X)*(X))
#define POW3(X) ((X)*(X)*(X))
#define POW4(X) ((X)*(X)*(X)*(X))
#define NONZERO(X) ((X) != 0.0)
#define NONZERO_TOL(X, TOL) (fabs((X)) > (TOL))


#ifndef VECTORIZE_OVER
#error "Unknown context, or the expected context (XO_CONTEXT_*) flag undefined. Try updating Xobjects?"
#endif

#endif  // XOBJECTS_COMMON_H

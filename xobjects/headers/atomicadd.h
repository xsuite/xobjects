// copyright ################################# //
// This file is part of the Xfields Package.   //
// Copyright (c) CERN, 2021.                   //
// ########################################### //

#ifndef _ATOMICADD_H_
#define _ATOMICADD_H_

/*
    Atomic add function (double type) for different contexts.
    Following the blueprint of CUDA's atomicAdd function, the return
    value is the old value of the address before the addition.
*/

#if defined(XO_CONTEXT_CPU_SERIAL)
    inline double atomicAdd(double *addr, double val)
    {
        double old_val = *addr;
        *addr = *addr + val;
        return old_val;
    }
#elif defined(XO_CONTEXT_CPU_OPENMP)
    inline double atomicAdd(double *addr, double val)
    {
        double old_val = *addr;
        #pragma omp atomic
        *addr += val;
        return old_val;
    }
#elif defined(XO_CONTEXT_CL)
    #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
    inline double atomicAdd(volatile __global double *addr, double val)
    {
        union {
            long u64;
            double f64;
        } next, expected, current;
        current.f64 = *addr;
        do {
            expected.f64 = current.f64;
            next.f64 = expected.f64 + val;
            current.u64 = atom_cmpxchg(
                (volatile __global long *)addr,
                (long) expected.u64,
                (long) next.u64);
        } while( current.u64 != expected.u64 );
        return current.f64;
    }
#elif defined(XO_CONTEXT_CUDA)
    // CUDA already provides this
#else
    #error "Atomic add not implemented for this context"
#endif

#endif // _ATOMICADD_H_

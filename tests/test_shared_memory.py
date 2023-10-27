import xobjects as xo
import numpy as np


def test_shared_memory():
    for test_context in xo.context.get_test_contexts():
        print(repr(test_context))
        if isinstance(test_context, xo.ContextCupy):
            test_context.default_block_size = 2
            test_context.default_shared_mem_size_bytes = (
                32  # 4 * 8 bytes = memory for 4 doubles
            )
            print(
                f"[test_shared_memory.py] default_block_size: {test_context.default_block_size}, default_shared_mem_size_bytes: {test_context.default_shared_mem_size_bytes}"
            )
        else:
            print(
                "[test_shared_memory.py] skipping test for cpu and pyopencl context"
            )
            continue

        _test_shared_memory_kernel = xo.Kernel(
            c_name="test_shared_memory",
            args=[
                xo.Arg(xo.Float64, const=True, pointer=True, name="input_arr"),
                xo.Arg(xo.Float64, pointer=True, name="result"),
                xo.Arg(xo.Float64, const=True, name="n"),
            ],
            n_threads="n",
        )

        _test_shared_memory_kernels = {
            "test_shared_memory": _test_shared_memory_kernel,
        }

        class TestElement(xo.HybridClass):
            _xofields = {}
            _extra_c_sources = [
                """
                __global__ void test_shared_memory(const double* input_arr, double* result, const int n) {
                  // simple kernel to test shared memory
                  // reduction with an array of 4 doubles using 2 blocks each 2 threads
                  // use reduction with interleaved addressing: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
                  // all threads within a block have access to shared memory

                  unsigned int tid = threadIdx.x;  // thread ID within the block: 0,1
                  unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;  // global thread ID: 0,1,2,3

                  // init shared memory with chunk of input array
                  extern __shared__ double sdata[2];
                  sdata[tid] = input_arr[gid];
                  __syncthreads();

                  // sum s[0] += s[1]
                  if (tid == 0){
                    sdata[tid] += sdata[tid + 1]; 

                    // write sum from shared to global mem
                    atomicAdd(&result[tid], sdata[tid]);
                  }
                }
                """
            ]
            _kernels = _test_shared_memory_kernels

            def __init__(
                self, _context=None, _buffer=None, _offset=None, _xobject=None
            ):
                if _xobject is not None:
                    self.xoinitialize(
                        _xobject=_xobject,
                        _context=_context,
                        _buffer=_buffer,
                        _offset=_offset,
                    )
                    return

                self.xoinitialize(
                    _context=_context, _buffer=_buffer, _offset=_offset
                )

                self.compile_kernels(only_if_needed=True)

        # end class def

        telem = TestElement(_context=test_context)

        n = 4
        input_arr = test_context.nplike_lib.array(
            [1.3, -5.6, 0.01, 90], dtype=np.float64
        )
        result = test_context.zeros(1, dtype=np.float64)  # init result buffer

        # call the dummy CUDA kernel using the shared memory
        print(f"input array: {input_arr}, result buffer: {result}")
        telem._context.kernels.test_shared_memory(
            input_arr=input_arr, result=result, n=np.int64(n)
        )

        # test if sum is correct
        assert result[0] == np.sum(input_arr)
        print(f"result GPU: {result[0]}, result CPU: {np.sum(input_arr)}")

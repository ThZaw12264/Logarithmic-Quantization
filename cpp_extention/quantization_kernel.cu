#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include "headers.h"
#include "helper_func.cu"


template <typename scalar_t, int block_size>
__global__ void quant_bfp_kernel(optim_state, output, size, max_entry, wl)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    unsigned int max_entry_bits = FLOAT_TO_BITS(&max_entry[index]);
    unsigned int max_exp = max_entry_bits << 1 >> 24 << 23;
    float base_float = 6*BITS_TO_FLOAT(&max_exp);

    float target_rebase = a[index]+base_float;
    unsigned int target_bits = FLOAT_TO_BITS(&target_rebase);
    unsigned int quantized = round_bitwise_nearest(target_bits, man_bits);
    float quantize_float = BITS_TO_FLOAT(&quantized)-base_float;

    unsigned int quantize_bits = FLOAT_TO_BITS(&quantize_float); 
    unsigned int clip_quantize = clip_max_exponent(man_bits-2, max_exp, quantize_bits); // sign bit, virtual bit
    quantize_float = BITS_TO_FLOAT(&clip_quantize);

    o[index] = quantize_float;
  }
}

Tensor quant_bfp_cuda(
    Tensor optim_state, int wl, int dim
)
{
    cudaSetDevice(optim_state.get_device());
    auto o = at::zeros_like(optim_state);
    int64_t size = optim_state.numel();

    Tensor max_entry = get_max_entry(optim_state, dim);
    int blockSize = 1024;
    int blockNums = (size + blockSize - 1) / blockSize;

    quant_bfp_kernel<<<blockNums, blockSize>>>(optim_state.data_ptr<float>(),
                                                    o.data_ptr<float>(),
                                                    size,
                                                    max_entry.data_ptr<float>(),
                                                    wl);
    return {o, max_entry};
}


template <typename scalar_t, int block_size>
__global__ void dequant_bfp_kernel(q_optim_state, max_entry, output, size, max_entry, wl)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    // Extract mantissa, sign

    // recalculate exponent/mantissa from block to bits

    // Combine bits to float


  }
}


Tensor dequantize_bfp_cuda(Tensor q_optim_state, Tensor max_entry, int wl, int dim)
{

}

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include "headers.h"
#include "helper_func.cu"


template <typename scalar_t>
__global__ void block_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> optim_state,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> output,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> max_entry,
    int size, int man_bits) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        unsigned int max_entry_bits = FLOAT_TO_BITS(&max_entry[index]);
        unsigned int max_exp = max_entry_bits << 1 >> 24 << 23;  // Remove sign bit and isolate exp
        float base_float = 6 * BITS_TO_FLOAT(&max_exp); //

        float target_rebase = optim_state[index] + base_float;
        unsigned int target_bits = FLOAT_TO_BITS(&target_rebase);
        unsigned int quantized = round_bitwise_nearest(target_bits, man_bits);
        float quantize_float = BITS_TO_FLOAT(&quantized) - base_float;

        unsigned int quantize_bits = FLOAT_TO_BITS(&quantize_float); 
        unsigned int clip_quantize = clip_max_exponent(man_bits - 2, max_exp, quantize_bits);
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

    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "block_kernel", ([&] {
        block_kernel<float><<<blockNums, blockSize>>>(optim_state.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                                    o.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                    max_entry.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                    size,
                                                    wl);
    return {o, max_entry};
}


// template <typename scalar_t, int block_size>
// __global__ void dequant_bfp_kernel(q_optim_state, max_entry, output, size, max_entry, wl)
// {
//   int index = blockIdx.x * blockDim.x + threadIdx.x;
//   if (index < size) {
//     // Extract mantissa, sign

//     // recalculate exponent/mantissa from block to bits

//     // Combine bits to float


//   }
// }


// Tensor dequantize_bfp_cuda(Tensor q_optim_state, Tensor max_entry, int wl, int dim)
// {

// }

#include <torch/extension.h>
#include <torch/torch.h>
#include "headers.h"

using torch::Tensor;
using torch::IntArrayRef;

// CUDA declarations
Tensor quant_bfp_cuda(Tensor optim_state, int wl, int dim);


Tensor quantize_bfp(Tensor optim_state, int wl, int dim)
{
    CHECK_INPUT(optim_state);
    return quant_bfp_cuda(optim_state, wl, dim);
}
#include <torch/extension.h>
#include <torch/torch.h>
#include "quant_cuda.h"

using torch::Tensor;
using torch::IntArrayRef;

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                             \
    CHECK_CUDA(x);                                                                                 \
    CHECK_CONTIGUOUS(x)


#define SIG_MAGIC_NUM 0x80000000
#define EXP_MAGIC_NUM 0x7f800000
#define MAN_MAGIC_NUM 0x007fffff
#define ROUND_MAGIC_NUM 0x00400000
#define LEADING_1 0x00800000


Tensor quantize_bfp(Tensor optim_state, int wl, int dim)
{
    CHECK_INPUT(optim_state);
    return quant_bfp_cuda(optim_state, wl, dim);
}

PYBIND11_MODULE(quantization, m){
    m.def('quantize_bfp', &quantize_bfp, "Block Floating Point (CUDA)")
}
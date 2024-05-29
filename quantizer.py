
import torch
import os
from torch.utils.cpp_extension import load



def block_quantize(x, wl, dim=-1):
    assert isinstance(x, torch.Tensor), "x is not a single precision Floating Point Tensor"
    assert x.is_cuda, "tensor needs to be on CUDA"
    current_path = os.path.dirname(os.path.realpath(__file__))
    quant_cuda = load(
        name="quant_cuda",
        sources=[
            os.path.join(current_path, "cpp_extention/quantization.cpp"),
            os.path.join(current_path, "cpp_extention/quantization_kernel.cu"),
            os.path.join(current_path, "cpp_extention/helper_funct.cu"),
        ],
    )

    out = quant_cuda.block_quantize_nearest(x.contiguous(), wl, dim)
    return out
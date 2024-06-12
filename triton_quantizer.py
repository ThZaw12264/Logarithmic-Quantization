import torch
import torch._inductor.config
torch._inductor.config.trace.enabled = True
torch._inductor.config.max_autotune = True


import os
os.environ["TORCH_LOGS"] = "output_code"


def bfp_quantize(x, mantissa_bits=8):
    '''
    Quantizes input tensor to block floating point format
    inputs  x:              original tensor
            mantissa_bits:  width of the mantissa in bits
    output  xbfp:           quantized tensor
    '''
    maxvalue = torch.max(torch.abs(x))
    max_exp = torch.frexp(maxvalue)[1]
    x_mantissa, x_exp = torch.frexp(x)

    shift_exp = max_exp - x_exp

    bfp_mantissa = x_mantissa.view(torch.int32) >> (shift_exp + (24 - mantissa_bits))
    bfp_mantissa <<= (shift_exp + (24 - mantissa_bits))
    bfp_mantissa = bfp_mantissa.view(torch.float32)

    bfp_exp = max_exp - shift_exp

    x_bfp = torch.ldexp(bfp_mantissa, bfp_exp)

    return x_bfp



bfp_fn = torch.compile(bfp_quantize, backend="inductor")
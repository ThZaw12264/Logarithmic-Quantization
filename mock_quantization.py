import torch
import copy
import random
import numpy as np

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def bfp_quantize(x, width, size):
    '''
    Quantizes input tensor to block floating point format
    inputs  x:        original tensor
            width:    width of each integer element in bits
            size:     block size = number of elements per block
    output  x_bfp:   quantized tensor
    '''
    max_value = torch.max(torch.abs(x))
    max_exp = torch.frexp(max_value)[1]
    x_mantissa, x_exp = torch.frexp(x)

    shift_exp = max_exp - x_exp

    bfp_mantissa = x_mantissa.view(torch.int32) >> (shift_exp + 17)
    bfp_mantissa <<= (shift_exp + 17)
    bfp_mantissa = bfp_mantissa.view(torch.float32)

    bfp_exp = max_exp - shift_exp

    x_bfp = torch.ldexp(bfp_mantissa, bfp_exp)

    return x_bfp


if __name__ == "__main__":
    x = torch.randn(4,4)
    x_bfp = bfp_quantize(x,8,16)

    print("Original tensor", x)
    print("BFP 8-bit elements", x_bfp)


# original tensor tensor([[-1.1258, -1.1524, -0.2506, -0.4339],
#         [ 0.8487,  0.6920, -0.3160, -2.1152],
#         [ 0.3223, -1.2633,  0.3500,  0.3081],
#         [ 0.1198,  1.2377,  1.1168, -0.2473]])

# BFP 8-bit elements tensor([[-1.1250, -1.1250, -0.2500, -0.4062],
#         [ 0.8438,  0.6875, -0.3125, -2.0938],
#         [ 0.3125, -1.2500,  0.3438,  0.2812],
#         [ 0.0938,  1.2188,  1.0938, -0.2188]])
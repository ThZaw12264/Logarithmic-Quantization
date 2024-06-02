import torch
import copy
import random
import numpy as np

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def round_fp16(x):
  '''
  Quantizes input tensor to FP8 data format
  inputs  x:      original tensor
          exp:    number of bits used for exponent field
                  e.g. E5M2 has 5 exp bits, E4M3 has 4 exp bits
  output  x_16:   quantized tensor
  '''
  exp = 5
  # TODO: implement the quantization code
  x_fp16 = x.to(torch.float16)
  x_int = x_fp16.view(torch.int16)
  x_int = x_int.bitwise_and(0xFF00)
  x_fp8 = x_int.view(torch.float16)
  x_fp8 = x_fp8.to(torch.float32)
  return x_fp8

def bfp_quantize(x):
    '''
    Quantizes input tensor to block floating point format
    inputs  x:        original tensor
            width:    width of each integer element in bits
            size:     block size = number of elements per block
    output  xbfp:   quantized tensor
    '''
    width = 8
    size = 16
    maxvalue = torch.max(torch.abs(x))
    max_exp = torch.frexp(maxvalue)[1]
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
    x_bfp = bfp_quantize(x)

    print("Original tensor", x)
    print("BFP 8-bit elements", x_bfp)
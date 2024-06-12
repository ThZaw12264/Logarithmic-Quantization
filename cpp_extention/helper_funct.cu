#define FLOAT_TO_BITS(x) (*reinterpret_cast<unsigned int*>(x))
#define BITS_TO_FLOAT(x) (*reinterpret_cast<float*>(x))

__device__ __forceinline__ unsigned int extract_exponent(float *a) {
  unsigned int temp = *(reinterpret_cast<unsigned int*>(a));
  temp = (temp << 1 >> 24); // single preciision, 1 sign bit, 23 mantissa bits
  return temp-127+1; // exponent offset and virtual bit
}

__device__ __forceinline__ unsigned int clip_exponent(int exp_bits, int man_bits,
                                                      unsigned int old_num,
                                                      unsigned int quantized_num) {
  if (quantized_num == 0)
    return quantized_num;

  int quantized_exponent_store = quantized_num << 1 >> 1 >> 23; // 1 sign bit, 23 mantissa bits
  int max_exponent_store = (1 << (exp_bits - 1)) + 127; // we are not reserving an exponent bit for infinity, nan, etc
  // Clippping Value Up
  if (quantized_exponent_store > max_exponent_store)
  {
    unsigned int max_man = (unsigned int)-1 << 9 >> 9 >> (23 - man_bits) << (23 - man_bits); // 1 sign bit, 8 exponent bits, 1 virtual bit
    unsigned int max_num = ((unsigned int)max_exponent_store << 23) | max_man;
    unsigned int old_sign = old_num >> 31 << 31;
    quantized_num = old_sign | max_num;
  }
  return quantized_num;
}


__device__ __forceinline__ unsigned int clip_max_exponent(int man_bits,
                                                          unsigned int max_exponent,
                                                          unsigned int quantized_num) {
  unsigned int quantized_exponent = quantized_num << 1 >> 24 << 23; // 1 sign bit, 23 mantissa bits
  if (quantized_exponent > max_exponent) {
    unsigned int max_man = (unsigned int ) -1 << 9 >> 9 >> (23-man_bits) << (23-man_bits); // 1 sign bit, 8 exponent bits
    unsigned int max_num = max_exponent | max_man;
    unsigned int old_sign = quantized_num >> 31 << 31;
    quantized_num = old_sign | max_num;
  }
  return quantized_num;
}

Tensor get_max_entry(Tensor a, int dim) {
  Tensor max_entry;
  if (dim == -1) {
    max_entry = at::max(at::abs(a)).expand_as(a).contiguous();
  } else if (dim == 0) {
    Tensor input_view = a.view({a.size(0), -1});
    max_entry = std::get<0>(input_view.abs().max(1, true)).expand_as(input_view).view_as(a).contiguous();
  } else {
    Tensor input_transpose = a.transpose(0, dim);
    Tensor input_view = input_transpose.contiguous().view({input_transpose.size(0), -1});
    Tensor max_transpose = std::get<0>(input_view.abs().max(1, true)).expand_as(input_view).view_as(input_transpose);
    max_entry = max_transpose.transpose(dim, 0).contiguous();
  }
  return max_entry;
}
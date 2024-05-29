from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='quantization',
    ext_modules=[
        CUDAExtension('quantization_cuda', [
            'quantization.cpp',
            'quantization_kernel.cu',
            'helper_funct.cu'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
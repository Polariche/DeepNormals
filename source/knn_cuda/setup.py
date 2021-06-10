from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='knn_cuda',
    ext_modules=[
        CUDAExtension('knn_cuda', [
            'knn_cuda.cpp',
            'knn_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
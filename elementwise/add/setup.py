from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="add_ops",
    version="0.1",
    ext_modules=[CUDAExtension(
        name="add_native_ops",
        sources=["add_kernels.cpp", "add_native_kernel.cu", "add_float4_kernel.cu"],
        extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '--use_fast_math']},
    )],
    cmdclass={'build_ext': BuildExtension},
)

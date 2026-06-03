from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# setup.py 示例
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

ext_modules = [
    CUDAExtension(
        name="custom_layernorm._C",  # 编译后的模块名
        sources=[
            "csrc/layernorm.cpp",     # ← 从 csrc/ 读取
            "csrc/layernorm_cuda.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3", "--use_fast_math"],
        }
    )
]

setup(
    name="custom_layernorm",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    packages=["custom_layernorm"],
)
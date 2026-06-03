from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = [
    CUDAExtension(
        name="layernorm_ops._C",  # 编译后的模块名
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
    name="layernorm_ops",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    packages=["layernorm_ops"],
)
import pybind11
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="focal-w4a16",
    version="0.0.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="focal_w4a16._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/w4a16_gemv.cu",
            ],
            include_dirs=[pybind11.get_include()],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-lineinfo",
                    "-Xptxas=-v",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

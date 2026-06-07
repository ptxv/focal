from pathlib import Path

from setuptools import find_packages, setup


PROJECT_ROOT = Path(__file__).parent.resolve()


def cuda_extension_build_config():
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
    except ImportError as exc:
        raise RuntimeError(
            "Building focal requires torch in the build environment. "
            "Install torch first; if pip build isolation hides it, use --no-build-isolation."
        ) from exc

    if CUDA_HOME is None:
        raise RuntimeError(
            "Building focal requires a CUDA toolkit with nvcc; CUDA_HOME is not set"
        )

    extension_modules = [
        CUDAExtension(
            name="focal.contiguous_gqa_decode_attention_cuda",
            sources=[
                str(PROJECT_ROOT / "focal" / "kernels" / "contiguous_gqa_decode_attention.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo", "-Xptxas=-v"],
            },
        )
    ]
    return extension_modules, {"build_ext": BuildExtension}


extension_modules, build_commands = cuda_extension_build_config()


setup(
    name="focal",
    version="0.1.0",
    description="Fused CUDA Kernel Optimization Library",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["torch"],
    ext_modules=extension_modules,
    cmdclass=build_commands,
)

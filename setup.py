from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent.resolve()


def cuda_extensions():
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
    except Exception as exc:
        raise RuntimeError(
            "Building focal requires torch in the build environment. "
            "Install torch first; if pip build isolation hides it, use --no-build-isolation."
        ) from exc

    if CUDA_HOME is None:
        raise RuntimeError(
            "Building focal requires a CUDA toolkit with nvcc; CUDA_HOME is not set"
        )

    ext_modules = [
        CUDAExtension(
            name="focal.cuda_native",
            sources=[
                str(ROOT / "csrc" / "bindings.cpp"),
                str(ROOT / "csrc" / "decode_attn_contig.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo", "-Xptxas=-v"],
            },
        )
    ]
    return ext_modules, {"build_ext": BuildExtension}


ext_modules, cmdclass = cuda_extensions()


setup(
    name="focal",
    version="0.1.0",
    description="Fused CUDA Kernel Optimization Library",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["torch"],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)

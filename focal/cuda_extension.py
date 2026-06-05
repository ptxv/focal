CUDA_EXTENSION_MODULE = None
CUDA_EXTENSION_ERROR = None


def load_cuda_extension():
    global CUDA_EXTENSION_MODULE, CUDA_EXTENSION_ERROR

    if CUDA_EXTENSION_MODULE is not None:
        return CUDA_EXTENSION_MODULE
    if CUDA_EXTENSION_ERROR is not None:
        # Avoid repeated import attempts once the optional extension is known missing.
        return None

    try:
        from . import cuda_native
    except ImportError as exc:
        CUDA_EXTENSION_ERROR = exc
        return None

    # Cache the loaded module so dispatch pays the import cost once.
    CUDA_EXTENSION_MODULE = cuda_native
    return CUDA_EXTENSION_MODULE


def cuda_extension_available():
    return load_cuda_extension() is not None


def require_cuda_extension():
    ext = load_cuda_extension()
    if ext is not None:
        return ext

    detail = f" ({CUDA_EXTENSION_ERROR})" if CUDA_EXTENSION_ERROR is not None else ""
    raise RuntimeError(
        "focal CUDA extension is unavailable. Build it with: "
        "python -m pip install -e . --no-build-isolation"
        f"{detail}"
    )

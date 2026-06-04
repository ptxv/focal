_CUDA_EXT = None
_CUDA_EXT_ERROR = None


def load_cuda_extension():
    global _CUDA_EXT, _CUDA_EXT_ERROR

    if _CUDA_EXT is not None:
        return _CUDA_EXT
    if _CUDA_EXT_ERROR is not None:
        # Avoid repeated import attempts once the optional extension is known missing.
        return None

    try:
        from . import _C
    except ImportError as exc:
        _CUDA_EXT_ERROR = exc
        return None

    # Cache the loaded module so dispatch pays the import cost once.
    _CUDA_EXT = _C
    return _CUDA_EXT


def cuda_extension_available():
    return load_cuda_extension() is not None


def require_cuda_extension():
    ext = load_cuda_extension()
    if ext is not None:
        return ext

    detail = f" ({_CUDA_EXT_ERROR})" if _CUDA_EXT_ERROR is not None else ""
    raise RuntimeError(
        "focal CUDA extension is unavailable. Build it with: "
        "FOCAL_BUILD_CUDA=1 python -m pip install -e . --no-build-isolation"
        f"{detail}"
    )

import numpy as np

try:
    import cupy as cp
    from .gpu_registration import register
    _cupy_available = True
except Exception:  # pragma: no cover - cupy may be missing
    cp = None
    _cupy_available = False
    register = None


# ----------------------- utils -----------------------

def _get_xp():
    return cp if _cupy_available else np


# ----------------------- API -----------------------

def registration(fixed: np.ndarray, moving: np.ndarray, alpha: float = 1.6, levels: int = 5, verbose: bool = True) -> np.ndarray:
    assert fixed.dtype == np.float32 and moving.dtype == np.float32
    if register is None:
        raise RuntimeError("cupy is required for GPU registration")
    moved, _, _, _ = register(fixed, moving)
    return moved


def registration_fields(fixed: np.ndarray, moving: np.ndarray, alpha: float = 1.6, levels: int = 5, verbose: bool = True):
    assert fixed.dtype == np.float32 and moving.dtype == np.float32
    if register is None:
        raise RuntimeError("cupy is required for GPU registration")
    _, vz, vy, vx = register(fixed, moving)
    return vz, vy, vx


def registration_imwarp_fields(fixed: np.ndarray, moving: np.ndarray, alpha: float = 1.6, levels: int = 5, verbose: bool = True):
    assert fixed.dtype == np.float32 and moving.dtype == np.float32
    if register is None:
        raise RuntimeError("cupy is required for GPU registration")
    moved, vz, vy, vx = register(fixed, moving)
    return moved, vz, vy, vx


def deeds(fixed: np.ndarray, moving: np.ndarray, alpha: float = 1.6, levels: int = 5, verbose: bool = True) -> np.ndarray:
    """Wrapper around :func:`registration` for backward compatibility."""
    return registration(fixed, moving, alpha=alpha, levels=levels, verbose=verbose)


def deeds_fields(
    fixed: np.ndarray,
    moving: np.ndarray,
    alpha: float = 1.6,
    levels: int = 5,
    verbose: bool = True,
) -> tuple:
    """Wrapper around :func:`registration_fields` returning the flow fields."""
    return registration_fields(fixed, moving, alpha=alpha, levels=levels, verbose=verbose)


def deeds_imwarp_fields(
    fixed: np.ndarray,
    moving: np.ndarray,
    alpha: float = 1.6,
    levels: int = 5,
    verbose: bool = True,
) -> tuple:
    """Wrapper around :func:`registration_imwarp_fields` returning image and flow fields."""
    return registration_imwarp_fields(fixed, moving, alpha=alpha, levels=levels, verbose=verbose)

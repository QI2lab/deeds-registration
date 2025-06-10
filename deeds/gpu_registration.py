import numpy as np
import cupy as cp
from .gpu_kernels import popcount_kernel

try:
    from cupyx.scipy.ndimage import uniform_filter as gpu_filter
    cp.cuda.runtime.getDeviceCount()
    _gpu_ok = True
except Exception:  # pragma: no cover - GPU not available
    _gpu_ok = False
    gpu_filter = None

try:
    from scipy.ndimage import uniform_filter as cpu_filter
except Exception:  # pragma: no cover - scipy may be missing
    cpu_filter = None


# ----------------------------------------------------------------------
# Basic utilities
# ----------------------------------------------------------------------

def _imshift_cpu(img: np.ndarray, dy: int, dx: int, dz: int) -> np.ndarray:
    m, n, o = img.shape
    out = np.empty_like(img)
    for k in range(o):
        kz = min(max(k + dz, 0), o - 1)
        for j in range(n):
            jx = min(max(j + dx, 0), n - 1)
            for i in range(m):
                iy = min(max(i + dy, 0), m - 1)
                out[i, j, k] = img[iy, jx, kz]
    return out


_imshift_kernel = cp.ElementwiseKernel(
    "raw float32 img, int32 m, int32 n, int32 o, int32 dy, int32 dx, int32 dz",
    "float32 out",
    r"""
    int z = i / (m * n);
    int tmp = i - z * m * n;
    int y = tmp / m;
    int x = tmp - y * m;
    int yy = min(max(y + dy, 0), m - 1);
    int xx = min(max(x + dx, 0), n - 1);
    int zz = min(max(z + dz, 0), o - 1);
    out = img[yy + xx * m + zz * m * n];
    """,
    "imshift_kernel",
)


def _imshift_gpu(img: cp.ndarray, dy: int, dx: int, dz: int) -> cp.ndarray:
    m, n, o = img.shape
    out = cp.empty_like(img)
    _imshift_kernel(img.ravel(), m, n, o, dy, dx, dz, out.ravel())
    return out


def _boxfilter_cpu(arr: np.ndarray, hw: int) -> np.ndarray:
    return cpu_filter(arr, size=hw * 2 + 1, mode="nearest")


def _boxfilter_gpu(arr: cp.ndarray, hw: int) -> cp.ndarray:
    return gpu_filter(arr, size=hw * 2 + 1, mode="nearest")


def _filter1_cpu(image: np.ndarray, filt: np.ndarray, dim: int) -> np.ndarray:
    from scipy.ndimage import convolve1d

    axis = dim - 1
    return convolve1d(image, filt.astype(image.dtype), axis=axis, mode="nearest")


def _filter1_gpu(image: cp.ndarray, filt: cp.ndarray, dim: int) -> cp.ndarray:
    from cupyx.scipy.ndimage import convolve1d

    axis = dim - 1
    return convolve1d(image, filt.astype(image.dtype), axis=axis, mode="nearest")


def _volfilter_cpu(image: np.ndarray, length: int, sigma: float) -> np.ndarray:
    hw = (length - 1) // 2
    f = np.exp(-((np.arange(length) - hw) ** 2) / (2 * sigma * sigma))
    f /= f.sum()
    out = _filter1_cpu(image, f, 1)
    out = _filter1_cpu(out, f, 2)
    out = _filter1_cpu(out, f, 3)
    return out


def _volfilter_gpu(image: cp.ndarray, length: int, sigma: float) -> cp.ndarray:
    hw = (length - 1) // 2
    f = cp.exp(-((cp.arange(length) - hw) ** 2) / (2 * sigma * sigma))
    f /= f.sum()
    out = _filter1_gpu(image, f, 1)
    out = _filter1_gpu(out, f, 2)
    out = _filter1_gpu(out, f, 3)
    return out


# ----------------------------------------------------------------------
# Descriptor computation (simplified MIND)
# ----------------------------------------------------------------------

def _descriptor_cpu(image: np.ndarray, qs: int) -> np.ndarray:
    dx = [qs, qs, -qs, 0, qs, 0]
    dy = [qs, -qs, 0, -qs, 0, qs]
    dz = [0, 0, qs, qs, qs, qs]
    patches = []
    for d in range(6):
        shifted = _imshift_cpu(image, dy[d], dx[d], dz[d])
        diff2 = (shifted - image) ** 2
        patches.append(_boxfilter_cpu(diff2, qs))
    d1 = np.stack(patches, axis=0)

    sx = [-qs, 0, -qs, 0, 0, qs, 0, 0, 0, -qs, 0, 0]
    sy = [0, -qs, 0, qs, 0, 0, 0, qs, 0, 0, 0, -qs]
    sz = [0, 0, 0, 0, -qs, 0, -qs, 0, -qs, 0, -qs, 0]
    index = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

    mind = np.empty((12,) + image.shape, dtype=np.float32)
    for l in range(12):
        mind[l] = _imshift_cpu(d1[index[l]], sy[l], sx[l], sz[l])

    minval = mind.min(axis=0)
    mind -= minval
    noise = mind.mean(axis=0)
    noise[noise < 1e-6] = 1e-6
    mind /= noise

    compare = -np.log((np.arange(1, 6) + 0.5) / 6.0)
    tablei = np.array([0, 1, 3, 7, 15, 31], dtype=np.uint64)

    result = np.zeros(image.shape, dtype=np.uint64)
    tabled = np.uint64(1)
    for l in range(12):
        quant = (mind[l][..., None] < compare).sum(axis=-1)
        vals = tablei[quant].astype(np.uint64)
        result += vals * tabled
        tabled = np.uint64(tabled * 32)
    return result


def _descriptor_gpu(image: cp.ndarray, qs: int) -> cp.ndarray:
    dx = [qs, qs, -qs, 0, qs, 0]
    dy = [qs, -qs, 0, -qs, 0, qs]
    dz = [0, 0, qs, qs, qs, qs]
    patches = []
    for d in range(6):
        shifted = _imshift_gpu(image, dy[d], dx[d], dz[d])
        diff2 = (shifted - image) ** 2
        patches.append(_boxfilter_gpu(diff2, qs))
    d1 = cp.stack(patches, axis=0)

    sx = [-qs, 0, -qs, 0, 0, qs, 0, 0, 0, -qs, 0, 0]
    sy = [0, -qs, 0, qs, 0, 0, 0, qs, 0, 0, 0, -qs]
    sz = [0, 0, 0, 0, -qs, 0, -qs, 0, -qs, 0, -qs, 0]
    index = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

    mind = cp.empty((12,) + image.shape, dtype=cp.float32)
    for l in range(12):
        mind[l] = _imshift_gpu(d1[index[l]], sy[l], sx[l], sz[l])

    minval = mind.min(axis=0)
    mind -= minval
    noise = mind.mean(axis=0)
    mind /= cp.maximum(noise, 1e-6)

    compare = cp.asarray(-np.log((np.arange(1, 6) + 0.5) / 6.0), dtype=cp.float32)
    tablei = cp.asarray([0, 1, 3, 7, 15, 31], dtype=cp.uint64)

    result = cp.zeros(image.shape, dtype=cp.uint64)
    tabled = cp.uint64(1)
    for l in range(12):
        quant = (mind[l][..., None] < compare).sum(axis=-1)
        vals = tablei[quant].astype(cp.uint64)
        result += vals * tabled
        tabled = cp.uint64(tabled * 32)
    return result


# ----------------------------------------------------------------------
# Interpolation and warping helpers
# ----------------------------------------------------------------------

def _interp3_cpu(image: np.ndarray, coords: tuple) -> np.ndarray:
    """Trilinear interpolation on CPU using scipy."""
    from scipy.ndimage import map_coordinates

    return map_coordinates(image, coords, order=1, mode="nearest")


def _interp3_gpu(image: cp.ndarray, coords: tuple) -> cp.ndarray:
    """Trilinear interpolation on GPU using cupy."""
    from cupyx.scipy.ndimage import map_coordinates

    return map_coordinates(image, coords, order=1, mode="nearest")


def _warp_image_cpu(image: np.ndarray, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Warp image on CPU using displacement fields."""
    m, n, o = image.shape
    yy, xx, zz = np.meshgrid(np.arange(m), np.arange(n), np.arange(o), indexing="ij")
    coords = (yy + v, xx + u, zz + w)
    return _interp3_cpu(image, coords)


def _warp_image_gpu(image: cp.ndarray, u: cp.ndarray, v: cp.ndarray, w: cp.ndarray) -> cp.ndarray:
    """Warp image on GPU using displacement fields."""
    m, n, o = image.shape
    yy, xx, zz = cp.meshgrid(cp.arange(m), cp.arange(n), cp.arange(o), indexing="ij")
    coords = (yy + v, xx + u, zz + w)
    return _interp3_gpu(image, coords)


def _warp_image_cl_cpu(
    im1: np.ndarray,
    im1b: np.ndarray,
    u1: np.ndarray,
    v1: np.ndarray,
    w1: np.ndarray,
):
    """Warp image and compute SSD metrics on CPU."""
    warped = _warp_image_cpu(im1, u1, v1, w1)
    ssd = float(((im1b - warped) ** 2).mean())
    ssd0 = float(((im1b - im1) ** 2).mean())
    return warped, ssd, ssd0


def _warp_image_cl_gpu(
    im1: cp.ndarray,
    im1b: cp.ndarray,
    u1: cp.ndarray,
    v1: cp.ndarray,
    w1: cp.ndarray,
):
    """Warp image and compute SSD metrics on GPU."""
    warped = _warp_image_gpu(im1, u1, v1, w1)
    ssd = float(cp.mean((im1b - warped) ** 2).get())
    ssd0 = float(cp.mean((im1b - im1) ** 2).get())
    return warped, ssd, ssd0


def _interp3xyz_cpu(data: np.ndarray, out_shape: tuple) -> np.ndarray:
    """Trilinear resize of a 3-D array using scipy."""
    from scipy.ndimage import zoom

    zoom_factors = [o / i for o, i in zip(out_shape, data.shape)]
    return zoom(data, zoom_factors, order=1)


def _interp3xyz_gpu(data: cp.ndarray, out_shape: tuple) -> cp.ndarray:
    """Trilinear resize of a 3-D array on GPU."""
    from cupyx.scipy.ndimage import zoom

    zoom_factors = [o / i for o, i in zip(out_shape, data.shape)]
    return zoom(data, zoom_factors, order=1)


def _interp3xyzB_cpu(data: np.ndarray, out_shape: tuple) -> np.ndarray:
    """Variant B of trilinear resize using scipy."""
    return _interp3xyz_cpu(data, out_shape)


def _interp3xyzB_gpu(data: cp.ndarray, out_shape: tuple) -> cp.ndarray:
    """Variant B of trilinear resize on GPU."""
    return _interp3xyz_gpu(data, out_shape)


def _data_cost_cpu(
    data: np.ndarray,
    data2: np.ndarray,
    step1: int,
    hw: int,
    alpha: float,
) -> np.ndarray:
    """Simplified data cost computation on CPU."""
    offsets = range(-hw, hw + 1)
    m, n, o = data.shape
    m1, n1, o1 = m // step1, n // step1, o // step1
    len2 = (2 * hw + 1) ** 3
    result = np.zeros((m1, n1, o1, len2), dtype=np.float32)
    idx = 0
    for dz in offsets:
        for dx in offsets:
            for dy in offsets:
                for z in range(o1):
                    z1 = z * step1
                    z2 = min(max(z1 + dz * step1, 0), o - 1)
                    for x in range(n1):
                        x1 = x * step1
                        x2 = min(max(x1 + dx * step1, 0), n - 1)
                        for y in range(m1):
                            y1 = y * step1
                            y2 = min(max(y1 + dy * step1, 0), m - 1)
                            t = data[y1, x1, z1] ^ data2[y2, x2, z2]
                            result[y, x, z, idx] = bin(int(t)).count("1") * alpha
                idx += 1
    return result


def _data_cost_gpu(
    data: cp.ndarray,
    data2: cp.ndarray,
    step1: int,
    hw: int,
    alpha: float,
) -> cp.ndarray:
    """Simplified data cost computation on GPU."""
    offsets = range(-hw, hw + 1)
    m, n, o = data.shape
    m1, n1, o1 = m // step1, n // step1, o // step1
    len2 = (2 * hw + 1) ** 3
    result = cp.zeros((m1, n1, o1, len2), dtype=cp.float32)
    idx = 0
    for dz in offsets:
        for dx in offsets:
            for dy in offsets:
                z1 = cp.arange(o1) * step1
                x1 = cp.arange(n1) * step1
                y1 = cp.arange(m1) * step1
                Z1, X1, Y1 = cp.meshgrid(z1, x1, y1, indexing="ij")
                Z2 = cp.clip(Z1 + dz * step1, 0, o - 1)
                X2 = cp.clip(X1 + dx * step1, 0, n - 1)
                Y2 = cp.clip(Y1 + dy * step1, 0, m - 1)
                a = data[Y1, X1, Z1]
                b = data2[Y2, X2, Z2]
                diff = cp.bitwise_xor(a, b)
                ham = cp.unpackbits(diff.view(cp.uint8), axis=-1).sum(axis=-1)
                result[..., idx] = ham.astype(cp.float32) * alpha
                idx += 1
    return result


def _data_cost_cl_cpu(
    data: np.ndarray,
    data2: np.ndarray,
    step1: int,
    hw: int,
    quant: float,
    alpha: float,
) -> np.ndarray:
    """Wrapper around ``_data_cost_cpu`` returning a flattened array."""
    res4d = _data_cost_cpu(data, data2, step1, hw, alpha)
    return res4d.reshape(-1, res4d.shape[-1])


def _data_cost_cl_gpu(
    data: cp.ndarray,
    data2: cp.ndarray,
    step1: int,
    hw: int,
    quant: float,
    alpha: float,
) -> cp.ndarray:
    """GPU wrapper around ``_data_cost_gpu`` returning a flattened array."""
    res4d = _data_cost_gpu(data, data2, step1, hw, alpha)
    return res4d.reshape(-1, res4d.shape[-1])


def _upsample_deformations_cpu(u0: np.ndarray, v0: np.ndarray, w0: np.ndarray, new_shape: tuple):
    m, n, o = new_shape
    scale_m = m / u0.shape[0]
    scale_n = n / u0.shape[1]
    scale_o = o / u0.shape[2]
    yy, xx, zz = np.meshgrid(
        np.arange(m) / scale_m,
        np.arange(n) / scale_n,
        np.arange(o) / scale_o,
        indexing="ij",
    )
    u1 = _interp3_cpu(u0, (yy, xx, zz))
    v1 = _interp3_cpu(v0, (yy, xx, zz))
    w1 = _interp3_cpu(w0, (yy, xx, zz))
    return u1, v1, w1


def _upsample_deformations_gpu(u0: cp.ndarray, v0: cp.ndarray, w0: cp.ndarray, new_shape: tuple):
    m, n, o = new_shape
    scale_m = m / u0.shape[0]
    scale_n = n / u0.shape[1]
    scale_o = o / u0.shape[2]
    yy, xx, zz = cp.meshgrid(
        cp.arange(m) / scale_m,
        cp.arange(n) / scale_n,
        cp.arange(o) / scale_o,
        indexing="ij",
    )
    u1 = _interp3_gpu(u0, (yy, xx, zz))
    v1 = _interp3_gpu(v0, (yy, xx, zz))
    w1 = _interp3_gpu(w0, (yy, xx, zz))
    return u1, v1, w1


def _jacobian_cpu(u: np.ndarray, v: np.ndarray, w: np.ndarray, factor: int) -> float:
    """Return standard deviation of the Jacobian determinant."""
    factor1 = 1.0 / float(factor)
    grad = np.array([-0.5, 0.0, 0.5], dtype=np.float32)

    j11 = _filter1_cpu(u, grad, 2) * factor1
    j12 = _filter1_cpu(u, grad, 1) * factor1
    j13 = _filter1_cpu(u, grad, 3) * factor1

    j21 = _filter1_cpu(v, grad, 2) * factor1
    j22 = _filter1_cpu(v, grad, 1) * factor1
    j23 = _filter1_cpu(v, grad, 3) * factor1

    j31 = _filter1_cpu(w, grad, 2) * factor1
    j32 = _filter1_cpu(w, grad, 1) * factor1
    j33 = _filter1_cpu(w, grad, 3) * factor1

    j11 += 1.0
    j22 += 1.0
    j33 += 1.0

    det = (
        j11 * (j22 * j33 - j23 * j32)
        - j21 * (j12 * j33 - j13 * j32)
        + j31 * (j12 * j23 - j13 * j22)
    )

    jmean = det.mean()
    jstd = np.sqrt(((det - jmean) ** 2).mean())
    return float(jstd)


def _jacobian_gpu(u: cp.ndarray, v: cp.ndarray, w: cp.ndarray, factor: int) -> float:
    factor1 = 1.0 / float(factor)
    grad = cp.asarray([-0.5, 0.0, 0.5], dtype=cp.float32)

    j11 = _filter1_gpu(u, grad, 2) * factor1
    j12 = _filter1_gpu(u, grad, 1) * factor1
    j13 = _filter1_gpu(u, grad, 3) * factor1

    j21 = _filter1_gpu(v, grad, 2) * factor1
    j22 = _filter1_gpu(v, grad, 1) * factor1
    j23 = _filter1_gpu(v, grad, 3) * factor1

    j31 = _filter1_gpu(w, grad, 2) * factor1
    j32 = _filter1_gpu(w, grad, 1) * factor1
    j33 = _filter1_gpu(w, grad, 3) * factor1

    j11 += 1.0
    j22 += 1.0
    j33 += 1.0

    det = (
        j11 * (j22 * j33 - j23 * j32)
        - j21 * (j12 * j33 - j13 * j32)
        + j31 * (j12 * j23 - j13 * j22)
    )

    jmean = det.mean()
    jstd = cp.sqrt(((det - jmean) ** 2).mean())
    return float(cp.asnumpy(jstd))


def _warp_affine_cpu(
    image: np.ndarray,
    X: np.ndarray,
    u1: np.ndarray,
    v1: np.ndarray,
    w1: np.ndarray,
):
    """Affine warp on CPU using displacements and 3x4 matrix."""
    m, n, o = image.shape
    yy, xx, zz = np.meshgrid(np.arange(m), np.arange(n), np.arange(o), indexing="ij")
    y1 = yy * X[0] + xx * X[1] + zz * X[2] + X[3] + v1
    x1 = yy * X[4] + xx * X[5] + zz * X[6] + X[7] + u1
    z1 = yy * X[8] + xx * X[9] + zz * X[10] + X[11] + w1
    return _interp3_cpu(image, (y1, x1, z1))


def _warp_affine_gpu(
    image: cp.ndarray,
    X: cp.ndarray,
    u1: cp.ndarray,
    v1: cp.ndarray,
    w1: cp.ndarray,
):
    """Affine warp on GPU using displacements and 3x4 matrix."""
    m, n, o = image.shape
    yy, xx, zz = cp.meshgrid(cp.arange(m), cp.arange(n), cp.arange(o), indexing="ij")
    y1 = yy * X[0] + xx * X[1] + zz * X[2] + X[3] + v1
    x1 = yy * X[4] + xx * X[5] + zz * X[6] + X[7] + u1
    z1 = yy * X[8] + xx * X[9] + zz * X[10] + X[11] + w1
    return _interp3_gpu(image, (y1, x1, z1))


def _warp_affine_s_cpu(
    image: np.ndarray,
    X: np.ndarray,
    u1: np.ndarray,
    v1: np.ndarray,
    w1: np.ndarray,
):
    """Nearest-neighbour affine warp on CPU."""
    m, n, o = image.shape
    yy, xx, zz = np.meshgrid(np.arange(m), np.arange(n), np.arange(o), indexing="ij")
    y1 = yy * X[0] + xx * X[1] + zz * X[2] + X[3] + v1
    x1 = yy * X[4] + xx * X[5] + zz * X[6] + X[7] + u1
    z1 = yy * X[8] + xx * X[9] + zz * X[10] + X[11] + w1
    y1 = np.clip(np.rint(y1), 0, m - 1).astype(np.int32)
    x1 = np.clip(np.rint(x1), 0, n - 1).astype(np.int32)
    z1 = np.clip(np.rint(z1), 0, o - 1).astype(np.int32)
    idx = y1 + x1 * m + z1 * m * n
    return image.reshape(-1)[idx.ravel()].reshape(image.shape)


def _warp_affine_s_gpu(
    image: cp.ndarray,
    X: cp.ndarray,
    u1: cp.ndarray,
    v1: cp.ndarray,
    w1: cp.ndarray,
):
    """Nearest-neighbour affine warp on GPU."""
    m, n, o = image.shape
    yy, xx, zz = cp.meshgrid(cp.arange(m), cp.arange(n), cp.arange(o), indexing="ij")
    y1 = yy * X[0] + xx * X[1] + zz * X[2] + X[3] + v1
    x1 = yy * X[4] + xx * X[5] + zz * X[6] + X[7] + u1
    z1 = yy * X[8] + xx * X[9] + zz * X[10] + X[11] + w1
    y1 = cp.clip(cp.rint(y1), 0, m - 1).astype(cp.int32)
    x1 = cp.clip(cp.rint(x1), 0, n - 1).astype(cp.int32)
    z1 = cp.clip(cp.rint(z1), 0, o - 1).astype(cp.int32)
    idx = y1 + x1 * m + z1 * m * n
    return image.reshape(-1)[idx.ravel()].reshape(image.shape)


def _consistent_mapping_cpu(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    u2: np.ndarray,
    v2: np.ndarray,
    w2: np.ndarray,
    factor: int,
):
    """Symmetrise flow fields on CPU."""
    u = u.copy().astype(np.float32)
    v = v.copy().astype(np.float32)
    w = w.copy().astype(np.float32)
    u2 = u2.copy().astype(np.float32)
    v2 = v2.copy().astype(np.float32)
    w2 = w2.copy().astype(np.float32)
    factor1 = 1.0 / float(factor)
    u *= factor1
    v *= factor1
    w *= factor1
    u2 *= factor1
    v2 *= factor1
    w2 *= factor1
    for _ in range(10):
        tmp_u = _warp_image_cpu(u2, u, v, w)
        tmp_v = _warp_image_cpu(v2, u, v, w)
        tmp_w = _warp_image_cpu(w2, u, v, w)
        u = 0.5 * u - 0.5 * tmp_u
        v = 0.5 * v - 0.5 * tmp_v
        w = 0.5 * w - 0.5 * tmp_w
        tmp_u2 = _warp_image_cpu(u, u2, v2, w2)
        tmp_v2 = _warp_image_cpu(v, u2, v2, w2)
        tmp_w2 = _warp_image_cpu(w, u2, v2, w2)
        u2 = 0.5 * u2 - 0.5 * tmp_u2
        v2 = 0.5 * v2 - 0.5 * tmp_v2
        w2 = 0.5 * w2 - 0.5 * tmp_w2
    u *= factor
    v *= factor
    w *= factor
    u2 *= factor
    v2 *= factor
    w2 *= factor
    return u, v, w, u2, v2, w2


def _consistent_mapping_gpu(
    u: cp.ndarray,
    v: cp.ndarray,
    w: cp.ndarray,
    u2: cp.ndarray,
    v2: cp.ndarray,
    w2: cp.ndarray,
    factor: int,
):
    """Symmetrise flow fields on GPU."""
    factor1 = 1.0 / float(factor)
    u = u.astype(cp.float32) * factor1
    v = v.astype(cp.float32) * factor1
    w = w.astype(cp.float32) * factor1
    u2 = u2.astype(cp.float32) * factor1
    v2 = v2.astype(cp.float32) * factor1
    w2 = w2.astype(cp.float32) * factor1
    for _ in range(10):
        tmp_u = _warp_image_gpu(u2, u, v, w)
        tmp_v = _warp_image_gpu(v2, u, v, w)
        tmp_w = _warp_image_gpu(w2, u, v, w)
        u = 0.5 * u - 0.5 * tmp_u
        v = 0.5 * v - 0.5 * tmp_v
        w = 0.5 * w - 0.5 * tmp_w
        tmp_u2 = _warp_image_gpu(u, u2, v2, w2)
        tmp_v2 = _warp_image_gpu(v, u2, v2, w2)
        tmp_w2 = _warp_image_gpu(w, u2, v2, w2)
        u2 = 0.5 * u2 - 0.5 * tmp_u2
        v2 = 0.5 * v2 - 0.5 * tmp_v2
        w2 = 0.5 * w2 - 0.5 * tmp_w2
    u *= factor
    v *= factor
    w *= factor
    u2 *= factor
    v2 *= factor
    w2 *= factor
    return u, v, w, u2, v2, w2


# ----------------------------------------------------------------------
# Message passing and regularisation helpers
# ----------------------------------------------------------------------

def _message_dt_cpu(cost: np.ndarray, offsetx: float, offsety: float, offsetz: float):
    """Simplified dynamic programming message computation on CPU."""
    len1 = cost.shape[0]
    z = ((np.arange(len1 * 2 + 1) - len1 + offsety) ** 2).astype(np.float32)
    buffer = np.empty_like(cost)
    inds = np.empty_like(cost, dtype=np.int32)
    for k in range(len1):
        for j in range(len1):
            val = cost[:, j, k]
            for i in range(len1):
                arr = val + z[i - np.arange(len1) + len1]
                idx = int(arr.argmin())
                buffer[i, j, k] = arr[idx]
                inds[i, j, k] = idx + j * len1 + k * len1 * len1
    z = ((np.arange(len1 * 2 + 1) - len1 + offsetx) ** 2).astype(np.float32)
    buffer2 = np.empty_like(cost)
    inds2 = np.empty_like(cost, dtype=np.int32)
    for k in range(len1):
        for i in range(len1):
            val = buffer[i, :, k]
            indb = inds[i, :, k]
            for j in range(len1):
                arr = val + z[j - np.arange(len1) + len1]
                idx = int(arr.argmin())
                buffer2[i, j, k] = arr[idx]
                inds2[i, j, k] = indb[idx]
    z = ((np.arange(len1 * 2 + 1) - len1 + offsetz) ** 2).astype(np.float32)
    out = np.empty_like(cost)
    indout = np.empty_like(cost, dtype=np.int32)
    for j in range(len1):
        for i in range(len1):
            val = buffer2[i, j, :]
            indb = inds2[i, j, :]
            for k in range(len1):
                arr = val + z[k - np.arange(len1) + len1]
                idx = int(arr.argmin())
                out[i, j, k] = arr[idx]
                indout[i, j, k] = indb[idx]
    return out, indout


def _message_dt_gpu(cost: cp.ndarray, offsetx: float, offsety: float, offsetz: float):
    """Simplified dynamic programming message computation on GPU."""
    len1 = cost.shape[0]
    z = ((cp.arange(len1 * 2 + 1) - len1 + offsety) ** 2).astype(cp.float32)
    buffer = cp.empty_like(cost)
    inds = cp.empty_like(cost, dtype=cp.int32)
    for k in range(len1):
        for j in range(len1):
            val = cost[:, j, k]
            for i in range(len1):
                arr = val + z[i - cp.arange(len1) + len1]
                idx = int(cp.argmin(arr))
                buffer[i, j, k] = arr[idx]
                inds[i, j, k] = idx + j * len1 + k * len1 * len1
    z = ((cp.arange(len1 * 2 + 1) - len1 + offsetx) ** 2).astype(cp.float32)
    buffer2 = cp.empty_like(cost)
    inds2 = cp.empty_like(cost, dtype=cp.int32)
    for k in range(len1):
        for i in range(len1):
            val = buffer[i, :, k]
            indb = inds[i, :, k]
            for j in range(len1):
                arr = val + z[j - cp.arange(len1) + len1]
                idx = int(cp.argmin(arr))
                buffer2[i, j, k] = arr[idx]
                inds2[i, j, k] = indb[idx]
    z = ((cp.arange(len1 * 2 + 1) - len1 + offsetz) ** 2).astype(cp.float32)
    out = cp.empty_like(cost)
    indout = cp.empty_like(cost, dtype=cp.int32)
    for j in range(len1):
        for i in range(len1):
            val = buffer2[i, j, :]
            indb = inds2[i, j, :]
            for k in range(len1):
                arr = val + z[k - cp.arange(len1) + len1]
                idx = int(cp.argmin(arr))
                out[i, j, k] = arr[idx]
                indout[i, j, k] = indb[idx]
    return out, indout


def _regularisation_cpu(costall: np.ndarray, u0: np.ndarray, v0: np.ndarray, w0: np.ndarray, hw: int, step1: int, quant: float, ordered: np.ndarray, parents: np.ndarray, edgemst: np.ndarray):
    """Very simplified regularisation selecting minimal costs on CPU."""
    len1 = hw * 2 + 1
    len2 = len1 ** 3
    xs = np.zeros(len2, dtype=np.float32)
    ys = np.zeros(len2, dtype=np.float32)
    zs = np.zeros(len2, dtype=np.float32)
    idx = 0
    for k in range(len1):
        for j in range(len1):
            for i in range(len1):
                xs[idx] = (j - hw) * quant
                ys[idx] = (i - hw) * quant
                zs[idx] = (k - hw) * quant
                idx += 1
    sz = costall.shape[0]
    u1 = np.zeros(sz, dtype=np.float32)
    v1 = np.zeros(sz, dtype=np.float32)
    w1 = np.zeros(sz, dtype=np.float32)
    for ii in range(sz):
        costs = costall[ii].reshape(len1, len1, len1)
        msg, _ = _message_dt_cpu(costs, 0.0, 0.0, 0.0)
        ind = int(msg.argmin())
        u1[ii] = u0[ii] + xs[ind]
        v1[ii] = v0[ii] + ys[ind]
        w1[ii] = w0[ii] + zs[ind]
    return u1, v1, w1


def _regularisation_gpu(costall: cp.ndarray, u0: cp.ndarray, v0: cp.ndarray, w0: cp.ndarray, hw: int, step1: int, quant: float, ordered: cp.ndarray, parents: cp.ndarray, edgemst: cp.ndarray):
    """Very simplified regularisation selecting minimal costs on GPU."""
    len1 = hw * 2 + 1
    len2 = len1 ** 3
    xs = cp.zeros(len2, dtype=cp.float32)
    ys = cp.zeros(len2, dtype=cp.float32)
    zs = cp.zeros(len2, dtype=cp.float32)
    idx = 0
    for k in range(len1):
        for j in range(len1):
            for i in range(len1):
                xs[idx] = (j - hw) * quant
                ys[idx] = (i - hw) * quant
                zs[idx] = (k - hw) * quant
                idx += 1
    sz = costall.shape[0]
    u1 = cp.zeros(sz, dtype=cp.float32)
    v1 = cp.zeros(sz, dtype=cp.float32)
    w1 = cp.zeros(sz, dtype=cp.float32)
    for ii in range(sz):
        costs = costall[ii].reshape(len1, len1, len1)
        msg, _ = _message_dt_gpu(costs, 0.0, 0.0, 0.0)
        ind = int(cp.argmin(msg))
        u1[ii] = u0[ii] + xs[ind]
        v1[ii] = v0[ii] + ys[ind]
        w1[ii] = w0[ii] + zs[ind]
    return u1, v1, w1


# ----------------------------------------------------------------------
# Registration entry point using phase correlation for translation
# ----------------------------------------------------------------------

def _phase_corr(fixed, moving, xp):
    F = xp.fft.fftn(fixed)
    M = xp.fft.fftn(moving)
    R = F * xp.conj(M)
    R /= xp.maximum(xp.abs(R), 1e-8)
    corr = xp.fft.ifftn(R)
    max_idx = xp.unravel_index(xp.argmax(xp.abs(corr)), corr.shape)
    shifts = []
    for idx, dim in zip(max_idx, corr.shape):
        shift = int(idx)
        if shift > dim // 2:
            shift -= dim
        shifts.append(shift)
    return tuple(shifts)


def register(fixed: np.ndarray, moving: np.ndarray):
    if _gpu_ok:
        cp_fixed = cp.asarray(fixed, dtype=cp.float32)
        cp_moving = cp.asarray(moving, dtype=cp.float32)
        dy, dx, dz = _phase_corr(cp_fixed, cp_moving, cp)
        shifted = cp.roll(cp_moving, shift=(dy, dx, dz), axis=(0, 1, 2))
        vz = cp.full_like(cp_fixed, dy, dtype=cp.float32)
        vy = cp.full_like(cp_fixed, dx, dtype=cp.float32)
        vx = cp.full_like(cp_fixed, dz, dtype=cp.float32)
        return cp.asnumpy(shifted), cp.asnumpy(vz), cp.asnumpy(vy), cp.asnumpy(vx)
    else:
        if cpu_filter is None:
            raise RuntimeError("scipy is required for CPU fallback")
        dy, dx, dz = _phase_corr(fixed, moving, np)
        shifted = np.roll(moving, shift=(dy, dx, dz), axis=(0, 1, 2))
        vz = np.full_like(fixed, dy, dtype=np.float32)
        vy = np.full_like(fixed, dx, dtype=np.float32)
        vx = np.full_like(fixed, dz, dtype=np.float32)
        return shifted, vz, vy, vx

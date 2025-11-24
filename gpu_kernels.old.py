# gpu_kernels.py
import numpy as np
import cv2

# Intentar importar PyCUDA (entorno legacy GTX 550 + CUDA 8)
try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # crea contexto al importar
    from pycuda.compiler import SourceModule

    HAS_PYCUDA = True
except ImportError:
    HAS_PYCUDA = False


# =========================
#   KERNELS GPU (PyCUDA)
# =========================

if HAS_PYCUDA:
    KERNEL_CODE = r"""
    __global__ void invert(unsigned char* img, int width, int height, int channels) {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x >= width || y >= height) return;

        int idx = (y * width + x) * channels;
        for (int c = 0; c < channels; ++c) {
            img[idx + c] = 255 - img[idx + c];
        }
    }

    __global__ void adjust_brightness(unsigned char* img, int width, int height, int channels,
                                      float alpha, float beta) {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x >= width || y >= height) return;

        int idx = (y * width + x) * channels;
        for (int c = 0; c < channels; ++c) {
            float val = img[idx + c];
            val = alpha * val + beta;
            if (val < 0.0f) val = 0.0f;
            if (val > 255.0f) val = 255.0f;
            img[idx + c] = (unsigned char)(val);
        }
    }

    __global__ void adjust_contrast(unsigned char* img, int width, int height, int channels,
                                    float alpha, float midpoint) {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x >= width || y >= height) return;

        int idx = (y * width + x) * channels;
        for (int c = 0; c < channels; ++c) {
            float val = img[idx + c];
            val = (val - midpoint) * alpha + midpoint;
            if (val < 0.0f) val = 0.0f;
            if (val > 255.0f) val = 255.0f;
            img[idx + c] = (unsigned char)(val);
        }
    }
    """

    _mod = SourceModule(KERNEL_CODE)
    _invert_kernel = _mod.get_function("invert")
    _brightness_kernel = _mod.get_function("adjust_brightness")
    _contrast_kernel = _mod.get_function("adjust_contrast")


def _check_image(image: np.ndarray):
    if image.dtype != np.uint8:
        raise ValueError("Image must be uint8")
    if image.ndim != 3:
        raise ValueError("Expected HxWxC image")


# =========================
#   API PÚBLICA
# =========================

def gpu_invert(image: np.ndarray) -> np.ndarray:
    """
    Invierte colores. Usa GPU si PyCUDA está disponible, si no CPU (OpenCV).
    """
    _check_image(image)

    # Fallback CPU si no hay PyCUDA (portátil Kubuntu 24.04, etc.)
    if not HAS_PYCUDA:
        return cv2.bitwise_not(image)

    height, width, channels = image.shape
    img_flat = image.copy()

    img_gpu = cuda.mem_alloc(img_flat.nbytes)
    cuda.memcpy_htod(img_gpu, img_flat)

    block = (16, 16, 1)
    grid = ((width + block[0] - 1) // block[0],
            (height + block[1] - 1) // block[1],
            1)

    _invert_kernel(
        img_gpu,
        np.int32(width),
        np.int32(height),
        np.int32(channels),
        block=block,
        grid=grid,
    )

    result = np.empty_like(img_flat)
    cuda.memcpy_dtoh(result, img_gpu)
    img_gpu.free()
    return result


def gpu_brightness(image: np.ndarray, alpha: float = 1.2, beta: float = 10.0) -> np.ndarray:
    """
    Ajusta brillo/contraste simple. GPU si hay PyCUDA, CPU si no.
    """
    _check_image(image)

    if not HAS_PYCUDA:
        # CPU: alpha = factor contraste, beta = brillo
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    height, width, channels = image.shape
    img_flat = image.copy()

    img_gpu = cuda.mem_alloc(img_flat.nbytes)
    cuda.memcpy_htod(img_gpu, img_flat)

    block = (16, 16, 1)
    grid = ((width + block[0] - 1) // block[0],
            (height + block[1] - 1) // block[1],
            1)

    _brightness_kernel(
        img_gpu,
        np.int32(width),
        np.int32(height),
        np.int32(channels),
        np.float32(alpha),
        np.float32(beta),
        block=block,
        grid=grid,
    )

    result = np.empty_like(img_flat)
    cuda.memcpy_dtoh(result, img_gpu)
    img_gpu.free()
    return result


def gpu_contrast(image: np.ndarray, alpha: float = 1.3, midpoint: float = 127.0) -> np.ndarray:
    """
    Ajusta contraste alrededor de un punto medio. GPU si hay PyCUDA, CPU si no.
    """
    _check_image(image)

    if not HAS_PYCUDA:
        # CPU: aproximación usando convertScaleAbs
        beta = midpoint * (1 - alpha)
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    height, width, channels = image.shape
    img_flat = image.copy()

    img_gpu = cuda.mem_alloc(img_flat.nbytes)
    cuda.memcpy_htod(img_gpu, img_flat)

    block = (16, 16, 1)
    grid = ((width + block[0] - 1) // block[0],
            (height + block[1] - 1) // block[1],
            1)

    _contrast_kernel(
        img_gpu,
        np.int32(width),
        np.int32(height),
        np.int32(channels),
        np.float32(alpha),
        np.float32(midpoint),
        block=block,
        grid=grid,
    )

    result = np.empty_like(img_flat)
    cuda.memcpy_dtoh(result, img_gpu)
    img_gpu.free()
    return result

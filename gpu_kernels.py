import numpy as np

import pycuda.autoinit  # crea el contexto CUDA al importar
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# Kernels CUDA sencillos para imágenes uint8 (0–255), BGR o RGB de 3 canales

KERNEL_CODE = r"""
extern "C" {

__global__ void invert_kernel(unsigned char *img, int width, int height, int channels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height * channels;

    if (idx < total)
    {
        img[idx] = 255 - img[idx];
    }
}

__global__ void brightness_kernel(unsigned char *img, int width, int height, int channels, int delta)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height * channels;

    if (idx < total)
    {
        int val = (int)img[idx] + delta;
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        img[idx] = (unsigned char)val;
    }
}

__global__ void contrast_kernel(unsigned char *img, int width, int height, int channels, float alpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height * channels;

    if (idx < total)
    {
        // Ajuste simple: nuevo = 128 + alpha * (valor - 128)
        float val = (float)img[idx];
        val = 128.0f + alpha * (val - 128.0f);

        if (val < 0.0f) val = 0.0f;
        if (val > 255.0f) val = 255.0f;
        img[idx] = (unsigned char)val;
    }
}

}
"""

# Compilar los kernels con nvcc usando g++-5
# IMPORTANTE: asegúrate de tener instalado /usr/bin/g++-5
#   sudo apt install gcc-5 g++-5

_mod = SourceModule(
    KERNEL_CODE,
    options=["-ccbin", "/usr/bin/g++-5"]
)

_invert_kernel = _mod.get_function("invert_kernel")
_brightness_kernel = _mod.get_function("brightness_kernel")
_contrast_kernel = _mod.get_function("contrast_kernel")


def _ensure_uint8_3ch(image: np.ndarray) -> np.ndarray:
    """Asegura que la imagen es uint8 y 3 canales (H, W, 3)."""
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.ndim == 2:
        # gris -> 3 canales replicados
        image = np.stack([image] * 3, axis=-1)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Se esperaba imagen (H, W, 3). Shape recibida: {image.shape}")

    return image


def gpu_invert(image: np.ndarray) -> np.ndarray:
    """Invierte los colores de la imagen en la GPU."""
    img = _ensure_uint8_3ch(image).copy()

    height, width, channels = img.shape
    total = width * height * channels

    # Reservar memoria en GPU
    img_gpu = cuda.mem_alloc(img.nbytes)

    # Copiar a GPU
    cuda.memcpy_htod(img_gpu, img)

    # Configurar grid/bloques
    threads_per_block = 256
    blocks = (total + threads_per_block - 1) // threads_per_block

    # Lanzar kernel
    _invert_kernel(
        img_gpu,
        np.int32(width),
        np.int32(height),
        np.int32(channels),
        block=(threads_per_block, 1, 1),
        grid=(blocks, 1, 1),
    )

    # Copiar resultado de vuelta
    result = np.empty_like(img)
    cuda.memcpy_dtoh(result, img_gpu)

    img_gpu.free()
    return result


def gpu_brightness(image: np.ndarray, delta: int) -> np.ndarray:
    """Ajusta brillo en la GPU. delta puede ser positivo o negativo."""
    img = _ensure_uint8_3ch(image).copy()

    height, width, channels = img.shape
    total = width * height * channels

    img_gpu = cuda.mem_alloc(img.nbytes)
    cuda.memcpy_htod(img_gpu, img)

    threads_per_block = 256
    blocks = (total + threads_per_block - 1) // threads_per_block

    _brightness_kernel(
        img_gpu,
        np.int32(width),
        np.int32(height),
        np.int32(channels),
        np.int32(delta),
        block=(threads_per_block, 1, 1),
        grid=(blocks, 1, 1),
    )

    result = np.empty_like(img)
    cuda.memcpy_dtoh(result, img_gpu)

    img_gpu.free()
    return result


def gpu_contrast(image: np.ndarray, alpha: float) -> np.ndarray:
    """Ajusta contraste en la GPU. alpha > 1 aumenta, 0<alpha<1 reduce."""
    img = _ensure_uint8_3ch(image).copy()

    height, width, channels = img.shape
    total = width * height * channels

    img_gpu = cuda.mem_alloc(img.nbytes)
    cuda.memcpy_htod(img_gpu, img)

    threads_per_block = 256
    blocks = (total + threads_per_block - 1) // threads_per_block

    _contrast_kernel(
        img_gpu,
        np.int32(width),
        np.int32(height),
        np.int32(channels),
        np.float32(alpha),
        block=(threads_per_block, 1, 1),
        grid=(blocks, 1, 1),
    )

    result = np.empty_like(img)
    cuda.memcpy_dtoh(result, img_gpu)

    img_gpu.free()
    return result


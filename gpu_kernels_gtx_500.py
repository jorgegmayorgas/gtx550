# gpu_kernels.py
import numpy as np
import pycuda.autoinit  # Creates a CUDA context on import
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# -----------------------------------------------------------------------------
# CUDA Kernels (simple enough for Fermi / CUDA 8)
# -----------------------------------------------------------------------------

kernel_code = r"""
extern "C" {

__global__ void invert(unsigned char *img, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        img[idx] = 255 - img[idx];
    }
}

__global__ void brightness(unsigned char *img, int size, int shift)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        int val = (int)img[idx] + shift;
        if (val < 0)   val = 0;
        if (val > 255) val = 255;
        img[idx] = (unsigned char)val;
    }
}

__global__ void contrast(unsigned char *img, int size, float factor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        // Center around 128, scale, clamp
        float val = ((float)img[idx] - 128.0f) * factor + 128.0f;
        if (val < 0.0f)   val = 0.0f;
        if (val > 255.0f) val = 255.0f;
        img[idx] = (unsigned char)(val);
    }
}

}
"""

# Compile kernels once at import
mod = SourceModule(kernel_code)

invert_kernel = mod.get_function("invert")
brightness_kernel = mod.get_function("brightness")
contrast_kernel = mod.get_function("contrast")


# -----------------------------------------------------------------------------
# Helper to run a 1D kernel over a flat uint8 array
# -----------------------------------------------------------------------------

def _run_1d_kernel(kernel, flat_arr, *kernel_args):
    """
    kernel: a pycuda function
    flat_arr: np.ndarray (uint8, 1D)
    kernel_args: extra CUDA kernel args
    """
    assert flat_arr.dtype == np.uint8
    size = flat_arr.size

    # Allocate device buffer
    d_img = cuda.mem_alloc(flat_arr.nbytes)
    cuda.memcpy_htod(d_img, flat_arr)

    # Configure grid and block
    threads_per_block = 256
    blocks = (size + threads_per_block - 1) // threads_per_block

    # Launch kernel
    kernel(
        d_img,
        np.int32(size),
        *kernel_args,
        block=(threads_per_block, 1, 1),
        grid=(blocks, 1, 1)
    )

    # Copy back
    result = np.empty_like(flat_arr)
    cuda.memcpy_dtoh(result, d_img)
    d_img.free()
    return result


# -----------------------------------------------------------------------------
# Public functions: expect HxWxC uint8 images (OpenCV format)
# -----------------------------------------------------------------------------

def gpu_invert(img_bgr: np.ndarray) -> np.ndarray:
    """
    Invert all channels (B, G, R).
    """
    h, w, c = img_bgr.shape
    flat = img_bgr.reshape(-1)
    res_flat = _run_1d_kernel(invert_kernel, flat)
    return res_flat.reshape((h, w, c))


def gpu_brightness(img_bgr: np.ndarray, shift: int) -> np.ndarray:
    """
    Adjust brightness by integer shift [-255, 255].
    Positive -> brighter, negative -> darker.
    """
    h, w, c = img_bgr.shape
    flat = img_bgr.reshape(-1)
    res_flat = _run_1d_kernel(brightness_kernel, flat, np.int32(shift))
    return res_flat.reshape((h, w, c))


def gpu_contrast(img_bgr: np.ndarray, factor: float) -> np.ndarray:
    """
    Adjust contrast (e.g. 0.8 = less contrast, 1.2 = more contrast).
    """
    h, w, c = img_bgr.shape
    flat = img_bgr.reshape(-1)
    res_flat = _run_1d_kernel(contrast_kernel, flat, np.float32(factor))
    return res_flat.reshape((h, w, c))


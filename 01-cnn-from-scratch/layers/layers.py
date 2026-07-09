import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from activations.activations import sigmoid, tanh_func, relu, prelu, softmax
from pooling_ops.pooling_ops import max_pool_func, avg_pool_func
from utils.helpers import (
    save_plot,
    display_and_save_single,
    display_and_save_multi_channel,
    pad_image,
    stack_kernels_for_display,
)


# Core convolution function
# Accepts: image (HWC_in), kernel (kHkWC_in), stride, padding, activation function.
# Performs correlation (no kernel flip) over the padded image at the given stride.
# Applies the specified non-linear activation (sigmoid, tanh, ReLU, or PReLU).
# Displays and saves: input image, filter kernel, output activation map.
# Returns: single 2D feature map of shape (H_out  W_out).
def convolution(
    image,
    kernel,
    stride,
    padding,
    activation_func,
    activation_name="",
    display=True,
    subfolder=None,
):
    H_in, W_in = image.shape[0], image.shape[1]
    C_in = image.shape[2] if image.ndim == 3 else 1
    kH, kW = kernel.shape[0], kernel.shape[1]

    if image.ndim == 2:
        image_3d = image[:, :, np.newaxis]
    else:
        image_3d = image

    if kernel.ndim == 2:
        kernel_3d = kernel[:, :, np.newaxis]
    else:
        kernel_3d = kernel

    padded = pad_image(image_3d, padding)

    H_out = (H_in + 2 * padding - kH) // stride + 1
    W_out = (W_in + 2 * padding - kW) // stride + 1

    feature_map = np.zeros((H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            h_end = h_start + kH
            w_start = j * stride
            w_end = w_start + kW
            region = padded[h_start:h_end, w_start:w_end, :]
            feature_map[i, j] = np.sum(region * kernel_3d)

    if activation_func is not None:
        feature_map = activation_func(feature_map)

    if display:
        display_and_save_single(
            image_3d, "Input Image (Convolution)", "conv_input_image.png", subfolder
        )
        display_and_save_single(
            kernel_3d, "Filter Kernel", "conv_kernel.png", subfolder
        )
        display_and_save_single(
            feature_map,
            f"Output Activation Map ({activation_name})",
            f"conv_output_{activation_name}.png",
            subfolder,
        )

    print(f"[Convolution] Output shape: {feature_map.shape}")
    return feature_map


# Core pooling function
# Accepts: single 2D activation map, pooling function (max_pool / avg_pool), stride.
# Slides a window of pool_sizepool_size over the activation map at the given stride.
# Applies the pooling function on each window to produce the pooled output.
# Displays and saves: input activation map and pooled output.
# Returns: pooled 2D array of shape (H_out  W_out).
def pooling(
    activation_map,
    pool_func,
    stride,
    pool_size=2,
    pool_name="",
    display=True,
    subfolder=None,
):
    H_in, W_in = activation_map.shape

    H_out = (H_in - pool_size) // stride + 1
    W_out = (W_in - pool_size) // stride + 1

    pooled = np.zeros((H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            h_end = h_start + pool_size
            w_start = j * stride
            w_end = w_start + pool_size
            window = activation_map[h_start:h_end, w_start:w_end]
            pooled[i, j] = pool_func(window)

    if display:
        display_and_save_single(
            activation_map,
            "Input Activation Map (Pooling)",
            "pool_input.png",
            subfolder,
        )
        display_and_save_single(
            pooled,
            f"Pooled Output ({pool_name})",
            f"pool_output_{pool_name}.png",
            subfolder,
        )

    print(f"[Pooling] Output shape: {pooled.shape}")
    return pooled


# Multi-channel convolution layer function
# Accepts: 3D volume (HWC_in), list of kernels (each kHkWC_in), stride, padding, activation function.
# Convolves the input volume with each kernel individually using the convolution() function.
# Stacks the resulting feature maps into an output volume of shape (W  H  C) where C = number of kernels.
# Displays and saves: input volume, filter kernels (as a grid), output activation maps (per channel).
# Verifies the output shape matches the expected (W  H  C).
# Returns: output activation volume of shape (H_out  W_out  num_kernels).
def convolution_layer(
    volume,
    kernels,
    stride,
    padding,
    activation_func,
    activation_name="",
    subfolder=None,
):
    H_in, W_in, C_in = volume.shape
    num_kernels = len(kernels)
    kH, kW = kernels[0].shape[0], kernels[0].shape[1]

    H_out = (H_in + 2 * padding - kH) // stride + 1
    W_out = (W_in + 2 * padding - kW) // stride + 1

    output_volume = np.zeros((H_out, W_out, num_kernels))

    for k in range(num_kernels):
        output_volume[:, :, k] = convolution(
            volume,
            kernels[k],
            stride,
            padding,
            activation_func,
            activation_name,
            display=False,
        )

    display_and_save_multi_channel(
        volume, "Input Volume", "conv_layer_input_volume", subfolder
    )

    kernel_grid = stack_kernels_for_display(kernels)
    display_and_save_single(
        kernel_grid, "Filter Kernels", "conv_layer_kernels.png", subfolder
    )

    display_and_save_multi_channel(
        output_volume, "Output Activation Map", "conv_layer_output_volume", subfolder
    )

    print(f"[Convolution Layer] Output shape: {output_volume.shape}")
    print(f"  Expected: W={W_out}, H={H_out}, C={num_kernels}")
    assert output_volume.shape == (H_out, W_out, num_kernels), "Output size mismatch!"

    return output_volume


# Multi-channel pooling layer function
# Accepts: 3D activation volume (HWC), pooling function (max_pool / avg_pool), stride.
# Applies the pooling operation independently on each channel of the volume.
# Displays and saves: input volume and output volume (per channel).
# Returns: pooled output volume of shape (H_out  W_out  C).
def pooling_layer(volume, pool_func, stride, pool_size=2, pool_name="", subfolder=None):
    H_in, W_in, C_in = volume.shape

    H_out = (H_in - pool_size) // stride + 1
    W_out = (W_in - pool_size) // stride + 1

    output_volume = np.zeros((H_out, W_out, C_in))

    for c in range(C_in):
        output_volume[:, :, c] = pooling(
            volume[:, :, c], pool_func, stride, pool_size, pool_name, display=False
        )

    display_and_save_multi_channel(
        volume, "Input Volume", "pool_layer_input_volume", subfolder
    )
    display_and_save_multi_channel(
        output_volume,
        "Output Volume",
        f"pool_layer_output_volume_{pool_name}",
        subfolder,
    )

    print(f"[Pooling Layer] Output shape: {output_volume.shape}")
    return output_volume


# Flattening function
# Accepts: 3D activation volume (HWC), desired output vector size, optional identity flag.
# Computes the flattened input size = H * W * C.
# If identity=True: uses identity matrix I(output_size, flat_input_size).
# If identity=False: uses random weight matrix W of shape (output_size, H*W*C).
# Computes: output_vector = W @ flattened_input_vector.
# Verifies the output vector has the correct shape.
# Returns: 1D output vector of shape (output_size,).
def flattening(activation_volume, output_size, identity=False):
    H, W, C = activation_volume.shape
    flat_input_size = H * W * C

    if identity:
        assert output_size == flat_input_size, (
            "Identity mapping requires output_size == H*W*C"
        )
        W_mat = np.eye(output_size, flat_input_size)
    else:
        W_mat = np.random.randn(output_size, flat_input_size) * 0.01

    flat_vector = activation_volume.flatten()

    output_vector = W_mat @ flat_vector

    mode = "identity" if identity else "random"
    print(f"[Flattening] Input volume shape: {activation_volume.shape}")
    print(f"  Flattened size: {flat_input_size}")
    print(f"  Weight matrix shape: {W_mat.shape} ({mode})")
    print(f"  Output vector shape: {output_vector.shape}")
    assert output_vector.shape == (output_size,), "Output size mismatch!"

    return output_vector

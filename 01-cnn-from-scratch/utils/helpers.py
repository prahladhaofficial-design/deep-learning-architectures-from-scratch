import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Helper  saves a matplotlib figure to the outputs/ directory or a subfolder
def save_plot(fig, filename, subfolder=None):
    if subfolder:
        folder = os.path.join(OUTPUT_DIR, subfolder)
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
    else:
        filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, dpi=100, bbox_inches="tight")
    plt.close(fig)


# Helper  displays a single 2D array or single-channel image as a heatmap and saves it
def display_and_save_single(arr, title, filename, subfolder=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            im = ax.imshow(arr[:, :, 0], cmap="viridis")
        elif arr.shape[2] == 3:
            arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            im = ax.imshow(arr_norm)
        else:
            im = ax.imshow(arr[:, :, 0], cmap="viridis")
    else:
        im = ax.imshow(arr, cmap="viridis")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.axis("off")
    save_plot(fig, filename, subfolder)


# Helper  displays a multi-channel 3D array as a grid of heatmaps (one per channel) and saves it
def display_and_save_multi_channel(
    arr_3d, title_prefix, filename_prefix, subfolder=None
):
    C = arr_3d.shape[2]
    cols = min(C, 4)
    rows = int(np.ceil(C / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)
    for c in range(C):
        r, cl = divmod(c, cols)
        im = axes[r, cl].imshow(arr_3d[:, :, c], cmap="viridis")
        axes[r, cl].set_title(f"{title_prefix} Channel {c}")
        axes[r, cl].axis("off")
        fig.colorbar(im, ax=axes[r, cl], fraction=0.046)
    for c in range(C, rows * cols):
        r, cl = divmod(c, cols)
        axes[r, cl].axis("off")
    save_plot(fig, f"{filename_prefix}.png", subfolder)


# Helper  applies zero-padding around the spatial dimensions of a 2D or 3D array
# Implemented manually without using np.pad() as per module requirements
def pad_image(image, padding):
    if padding == 0:
        return image
    if image.ndim == 2:
        H, W = image.shape
        padded = np.zeros((H + 2 * padding, W + 2 * padding), dtype=image.dtype)
        padded[padding : H + padding, padding : W + padding] = image
        return padded
    H, W, C = image.shape
    padded = np.zeros(
        (H + 2 * padding, W + 2 * padding, C),
        dtype=image.dtype,
    )
    padded[padding : H + padding, padding : W + padding, :] = image
    return padded


# Helper  stacks multiple 2D/3D kernels into a single 2D grid for visualization
def stack_kernels_for_display(kernels):
    num = len(kernels)
    kH, kW = kernels[0].shape[0], kernels[0].shape[1]
    cols = min(num, 4)
    rows = int(np.ceil(num / cols))
    grid = np.zeros((rows * kH, cols * kW))
    for k in range(num):
        r, c = divmod(k, cols)
        kr = kernels[k]
        if kr.ndim == 3:
            kr = np.mean(kr, axis=2)
        grid[r * kH : (r + 1) * kH, c * kW : (c + 1) * kW] = kr
    return grid


# Helper  generates a random image of specified dimensions with values in [0, 1]
def generate_random_image(height=32, width=32, channels=3):
    return np.random.rand(height, width, channels)


# Helper  generates a list of random filter kernels, each of shape (kernel_height  kernel_width  input_channels)
def generate_random_kernels(num_kernels, kernel_height, kernel_width, input_channels):
    return [
        np.random.randn(kernel_height, kernel_width, input_channels)
        for _ in range(num_kernels)
    ]


# Helper  generates a random weight matrix of shape (output_size, input_size) using He initialization
def generate_random_weights(input_size, output_size):
    std = np.sqrt(2.0 / input_size)
    return np.random.randn(output_size, input_size) * std


# Helper  generates a random bias vector of given size initialized to zeros
def generate_random_bias(size):
    return np.zeros(size)

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Activation functions and their derivatives
from activations.activations import (
    sigmoid,
    sigmoid_derivative,
    tanh_func,
    tanh_func_derivative,
    relu,
    relu_derivative,
    prelu,
    prelu_derivative,
    softmax,
)

# Pooling functions
from pooling_ops.pooling_ops import max_pool_func, avg_pool_func

# Core CNN tensor operations
from layers.layers import (
    convolution,
    pooling,
    convolution_layer,
    pooling_layer,
    flattening,
)

# MLP classifier head
from network.mlp import mlp

# Full CNN architecture
from network.cnn_architecture import feed_forward_cnn

# Helper utilities
from utils.helpers import (
    save_plot,
    display_and_save_single,
    display_and_save_multi_channel,
    pad_image,
    stack_kernels_for_display,
    generate_random_image,
    generate_random_kernels,
    generate_random_weights,
    generate_random_bias,
)


def load_cifar10_image(filename="0017.jpg"):
    """
    Load a CIFAR-10 image from the local cifar10_images folder.
    Returns the image as a float32 array of shape (32, 32, 3) with values in [0, 1].
    """
    import matplotlib.image as mpimg

    image_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "cifar10_images"
    )
    filepath = os.path.join(image_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CIFAR-10 image not found at {filepath}")

    image = mpimg.imread(filepath)

    if image.shape[2] == 4:
        image = image[:, :, :3]

    if image.dtype != np.float32:
        image = image.astype(np.float32) / 255.0

    assert image.shape == (32, 32, 3), (
        f"Expected image shape (32, 32, 3), got {image.shape}"
    )

    return image


def main():
    np.random.seed(42)

    # Load CIFAR-10 image once  used throughout all tests
    print("Loading CIFAR-10 image (0017.jpg)...")
    cifar_image = load_cifar10_image(filename="0017.jpg")
    print(f"CIFAR-10 image shape: {cifar_image.shape}")
    print(f"CIFAR-10 image dtype: {cifar_image.dtype}")
    print(f"Value range: [{cifar_image.min():.4f}, {cifar_image.max():.4f}]")

    # ============================================================
    # (i) Test Convolution function with all activations
    # ============================================================
    print("\n========== (i) Convolution Function ==========")
    kernel = generate_random_kernels(1, 5, 5, 3)[0]

    for act_func, act_name in [
        (sigmoid, "sigmoid"),
        (tanh_func, "tanh"),
        (relu, "relu"),
        (lambda x: prelu(x, 0.01), "prelu"),
    ]:
        print(f"\n--- Activation: {act_name} ---")
        out = convolution(
            cifar_image,
            kernel,
            stride=1,
            padding=2,
            activation_func=act_func,
            activation_name=act_name,
            subfolder="part_i",
        )
        print(f"Output shape: {out.shape}")
        print(f"Output range: [{out.min():.4f}, {out.max():.4f}]")

    # ============================================================
    # (ii) Test Pooling function
    # ============================================================
    print("\n========== (ii) Pooling Function ==========")
    conv_result = convolution(
        cifar_image,
        kernel,
        stride=1,
        padding=2,
        activation_func=relu,
        activation_name="relu",
        display=False,
    )
    act_map = conv_result

    for pool_func, pool_name in [(max_pool_func, "max"), (avg_pool_func, "avg")]:
        print(f"\n--- Pooling: {pool_name} ---")
        out = pooling(
            act_map, pool_func, stride=2, pool_name=pool_name, subfolder="part_ii"
        )
        print(f"Output shape: {out.shape}")

    # ============================================================
    # (iii) Test Convolution layer function
    # ============================================================
    print("\n========== (iii) Convolution Layer Function ==========")
    kernels = generate_random_kernels(4, 5, 5, 3)
    conv_out = convolution_layer(
        cifar_image,
        kernels,
        stride=1,
        padding=2,
        activation_func=relu,
        activation_name="relu",
        subfolder="part_iii",
    )
    print(f"Output volume shape: {conv_out.shape}")
    print(f"Expected: W=32, H=32, C=4")

    # ============================================================
    # (iv) Test Pooling layer function
    # ============================================================
    print("\n========== (iv) Pooling Layer Function ==========")
    pool_out = pooling_layer(
        conv_out, max_pool_func, stride=2, pool_name="max", subfolder="part_iv"
    )
    print(f"Output volume shape: {pool_out.shape}")

    # ============================================================
    # (v) Test Flattening function
    # ============================================================
    print("\n========== (v) Flattening Function ==========")
    flat_out = flattening(pool_out, output_size=256)
    print(f"Output vector shape: {flat_out.shape}")

    # ============================================================
    # (vi) Test MLP
    # ============================================================
    print("\n========== (vi) MLP ==========")
    output_raw, output_softmax = mlp(
        vector=flat_out,
        num_hidden_layers=1,
        hidden_layer_sizes=[49],
        activation_func=relu,
        output_size=10,
        activation_name="relu",
    )
    print(f"Raw output: {output_raw}")
    print(f"Softmax output: {output_softmax}")
    print(f"Sum of softmax: {np.sum(output_softmax):.6f}")

    # ============================================================
    # (vii) Feed-forward CNN with CIFAR-10 image
    # ============================================================
    print("\n========== (vii) Feed-Forward CNN (CIFAR-10) ==========")
    cnn_raw, cnn_softmax = feed_forward_cnn(cifar_image)
    print(f"Final CNN output (softmax): {cnn_softmax}")
    print(f"Predicted class index: {np.argmax(cnn_softmax)}")

    print("\n===== ALL TESTS COMPLETED =====")


if __name__ == "__main__":
    main()

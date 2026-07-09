import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from activations.activations import relu, softmax
from pooling_ops.pooling_ops import max_pool_func
from layers.layers import convolution_layer, pooling_layer, flattening
from network.mlp import mlp


# the NumPy CNN architecture  Feed-forward CNN architecture
# Chains: Input -> Conv1 -> MaxPool1 -> Conv2 -> MaxPool2 -> Flatten -> MLP
# Architecture:
#   Input: 32323
#   Conv1: 4 kernels of 553, ReLU, stride=1, pad=2  32324
#   MaxPool1: 22, stride=2  16164
#   Conv2: 4 kernels of 554, ReLU, stride=1, pad=2  16164
#   MaxPool2: 22, stride=2  884
#   Flatten (identity mapping): 8*8*4 = 256  256
#   MLP: 256  49 (ReLU)  10 (softmax)
# Verifies output dimension is 10.
def feed_forward_cnn(image):
    assert image.shape == (32, 32, 3), (
        f"Expected input shape (32, 32, 3), got {image.shape}"
    )

    np.random.seed(42)

    # ---- Convolution Layer 1 ----
    # 4 kernels of size 553 (3 input channels)
    kernels1 = [np.random.randn(5, 5, 3) * 0.1 for _ in range(4)]

    conv1_out = convolution_layer(
        volume=image,
        kernels=kernels1,
        stride=1,
        padding=2,
        activation_func=relu,
        activation_name="relu",
        subfolder="part_vii",
    )
    print(f"Conv1 output shape: {conv1_out.shape}")
    assert conv1_out.shape == (32, 32, 4), f"Conv1 shape mismatch: {conv1_out.shape}"

    # ---- Max Pooling Layer 1 ----
    pool1_out = pooling_layer(
        volume=conv1_out,
        pool_func=max_pool_func,
        stride=2,
        pool_size=2,
        pool_name="max",
        subfolder="part_vii",
    )
    print(f"Pool1 output shape: {pool1_out.shape}")
    assert pool1_out.shape == (16, 16, 4), f"Pool1 shape mismatch: {pool1_out.shape}"

    # ---- Convolution Layer 2 ----
    # 4 kernels of size 554 (4 input channels from previous pooling)
    kernels2 = [np.random.randn(5, 5, 4) * 0.1 for _ in range(4)]

    conv2_out = convolution_layer(
        volume=pool1_out,
        kernels=kernels2,
        stride=1,
        padding=2,
        activation_func=relu,
        activation_name="relu",
        subfolder="part_vii",
    )
    print(f"Conv2 output shape: {conv2_out.shape}")
    assert conv2_out.shape == (16, 16, 4), f"Conv2 shape mismatch: {conv2_out.shape}"

    # ---- Max Pooling Layer 2 ----
    pool2_out = pooling_layer(
        volume=conv2_out,
        pool_func=max_pool_func,
        stride=2,
        pool_size=2,
        pool_name="max",
        subfolder="part_vii",
    )
    print(f"Pool2 output shape: {pool2_out.shape}")
    assert pool2_out.shape == (8, 8, 4), f"Pool2 shape mismatch: {pool2_out.shape}"

    # ---- Flattening Layer (identity mapping) ----
    # 884 = 256, identity mapping means W = I(256, 256)
    flatten_out = flattening(pool2_out, output_size=256, identity=True)
    print(f"Flatten output shape: {flatten_out.shape}")
    assert flatten_out.shape == (256,), f"Flatten shape mismatch: {flatten_out.shape}"

    # ---- MLP ----
    # One hidden layer of 49 nodes, output layer of 10 nodes, ReLU activation, softmax output
    output_raw, output_softmax = mlp(
        vector=flatten_out,
        num_hidden_layers=1,
        hidden_layer_sizes=[49],
        activation_func=relu,
        output_size=10,
        activation_name="relu",
    )

    print(f"MLP raw output shape: {output_raw.shape}")
    print(f"MLP softmax output shape: {output_softmax.shape}")
    assert output_softmax.shape == (10,), (
        f"MLP output shape mismatch: {output_softmax.shape}"
    )

    # ---- Verification ----
    print("\n===== Feed-Forward CNN Verification =====")
    print(f"Input image shape: {image.shape}")
    print(f"Output vector shape (softmax): {output_softmax.shape}")
    print(f"Output vector (softmax): {output_softmax}")
    print(f"Sum of softmax output: {np.sum(output_softmax):.6f}")
    print(
        f"Output is valid probability distribution: {np.isclose(np.sum(output_softmax), 1.0)}"
    )
    print("=========================================\n")

    return output_raw, output_softmax

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from activations.activations import softmax
from utils.helpers import generate_random_weights, generate_random_bias


# Multilayer Perceptron classifier head
# Accepts: input vector, number of hidden layers, list of hidden layer sizes,
#          activation function for hidden layers, output layer size.
# Performs forward pass through all hidden layers (linear + activation).
# Computes output layer (linear only, no activation).
# Returns: (output_without_softmax, output_with_softmax).
def mlp(
    vector,
    num_hidden_layers,
    hidden_layer_sizes,
    activation_func,
    output_size,
    activation_name="",
):
    assert len(hidden_layer_sizes) == num_hidden_layers, (
        f"hidden_layer_sizes length ({len(hidden_layer_sizes)}) must match num_hidden_layers ({num_hidden_layers})"
    )

    current = vector.copy()
    input_size = len(current)

    weights = []
    biases = []

    for i in range(num_hidden_layers):
        W = generate_random_weights(input_size, hidden_layer_sizes[i])
        b = generate_random_bias(hidden_layer_sizes[i])
        weights.append(W)
        biases.append(b)
        z = W @ current + b
        current = activation_func(z)
        input_size = hidden_layer_sizes[i]

    # Output layer: linear transformation only (no activation)
    W_out = generate_random_weights(input_size, output_size)
    b_out = generate_random_bias(output_size)
    weights.append(W_out)
    biases.append(b_out)

    output_raw = W_out @ current + b_out

    output_softmax = softmax(output_raw)

    print(f"[MLP] Number of hidden layers: {num_hidden_layers}")
    print(f"  Hidden layer sizes: {hidden_layer_sizes}")
    print(f"  Output size: {output_size}")
    print(f"  Activation: {activation_name}")
    print(f"  Raw output shape: {output_raw.shape}")
    print(f"  Softmax output shape: {output_softmax.shape}")
    print(f"  Sum of softmax output: {np.sum(output_softmax):.6f}")

    return output_raw, output_softmax

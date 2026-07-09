# NumPy CNN From Scratch

This component implements the forward pass of a convolutional neural network using NumPy. It focuses on the mechanics behind image convolution, pooling, activation functions, flattening, and dense classification without relying on deep learning framework layers.

## What Is Included

- Convolution over single-channel and multi-channel tensors.
- Max pooling and average pooling primitives.
- Convolution and pooling layers that operate on image volumes.
- Activation functions: sigmoid, tanh, ReLU, PReLU, and softmax.
- MLP classifier head and a complete CNN feed-forward pipeline.
- Visual outputs showing intermediate feature maps and tensor transformations.

## Structure

```text
activations/       Activation functions and derivatives
layers/            Convolution, pooling, layer, and flattening operations
network/           MLP head and complete CNN architecture
pooling_ops/       Pooling helpers
utils/             Plotting, image loading, padding, and random tensor utilities
outputs/           Saved operation and feature-map visualizations
main.py            End-to-end demo runner
```

## How To Run

```bash
python main.py
```

The script loads a local CIFAR-style image, applies the implemented operations, verifies tensor shapes, and writes visualizations into `outputs/`.

# Report: NumPy CNN From Scratch

## 1. Overview

This project implements a complete Convolutional Neural Network (CNN) feed-forward pipeline entirely from scratch using only NumPy. No built-in functions for convolution, pooling, non-linearity, padding, or striding are used. Every component is implemented manually with explicit loops and mathematical operations.

## 2. Architecture Implemented

The CNN follows the exact architecture specified in the project (the NumPy CNN architecture):

```
Input: CIFAR-10 image (32  32  3)  loaded from local file (cifar10_images/0017.jpg)

Conv Layer 1: 4 kernels of 553, ReLU, stride=1, padding=2
      Output: 32  32  4
Max Pooling 1: 22 window, stride=2
      Output: 16  16  4
Conv Layer 2: 4 kernels of 554, ReLU, stride=1, padding=2
      Output: 16  16  4
Max Pooling 2: 22 window, stride=2
      Output: 8  8  4
Flattening Layer (identity mapping, W = I_256)
      Output: 256-dimensional vector
MLP: 256  49 (ReLU)  10 (softmax)
      Output: 10-dimensional probability vector
```

## 3. Key Observations

### 3.1 Activation Functions (Part i)

| Activation | Formula | Output Range | Observed Range (3232 CIFAR-10) |
|---|---|---|---|
| **Sigmoid** | `1 / (1 + e^(-x))` | (0, 1) | [0.0002, 0.6490] |
| **Tanh** | `(e^x - e^(-x)) / (e^x + e^(-x))` | (-1, 1) | [-1.0000, 0.5475] |
| **ReLU** | `max(0, x)` | [0, ) | [0.0000, 0.6148] |
| **PReLU** | `x if x>0 else x` (=0.01) | (-, ) | [-0.0870, 0.6148] |

**Observations:**
- Sigmoid compresses all values into (0, 1), with the minimum near 0.0002 indicating strong negative pre-activation values from the CIFAR-10 image.
- Tanh reaches the full -1.0 boundary, confirming strong negative correlations in the CIFAR-10 image features.
- ReLU zeroes out all negative values (true sparsity observed at 0.0000 minimum).
- PReLU preserves negative leakage (=0.01), with minimum at -0.0870, confirming the parametric behavior.
- All activations were implemented from scratch without using `scipy`, `torch`, or `tensorflow`. Even `np.tanh` was replaced with the exponential formula.

### 3.2 Pooling Functions (Part ii)

| Pooling Type | Input (3232) | Output (1616) | Stride |
|---|---|---|---|
| Max Pooling | 3232 (CIFAR-10 derived) | 1616 | 2 |
| Average Pooling | 3232 (CIFAR-10 derived) | 1616 | 2 |

**Observations:**
- The input activation map is derived from convolving the CIFAR-10 image with a random 553 kernel and applying ReLU.
- Max pooling selects the highest value in each 22 window, preserving the most prominent features.
- Average pooling computes the mean, providing a smoother representation.
- Output dimension formula: `(H - pool_size) / stride + 1 = (32 - 2) / 2 + 1 = 16`

### 3.3 Convolution Layer (Part iii)

- Input volume: 32323 (CIFAR-10 image), 4 kernels of 553, stride=1, padding=2
- Output volume: 32324 (verified: W=32, H=32, C=4)
- Each kernel operates on all 3 input channels (RGB) simultaneously, producing one 2D feature map.
- The 4 feature maps are stacked along the channel dimension, showing how different random kernels extract different features from the same CIFAR-10 image.

### 3.4 Pooling Layer (Part iv)

- Input volume: 32324 (CIFAR-10 derived from conv layer), pool_size=2, stride=2
- Output volume: 16164
- Pooling is applied independently per channel, preserving the channel depth.

### 3.5 Flattening Function (Part v)

| Mode | Weight Matrix | Input | Output |
|---|---|---|---|
| Random | W ~ N(0, 2), shape (256, 1024) | 16164 = 1024 | (256,) |
| Identity | W = I(256, 256) | 884 = 256 | (256,) |

**Observations:**
- The flattening function supports two modes: random weight matrix (general use) and identity mapping (for the CNN architecture).
- Identity mapping ensures the flattened vector passes through unchanged, as required by the CNN specification.

### 3.6 Multilayer Perceptron (Part vi)

- Input: 256-dimensional vector (from flattening the CIFAR-10 derived pooling output)
- Architecture: 256  49 (ReLU)  10 (linear)
- Raw output: `[-2.776, 1.948, 1.877, -2.922, -1.730, -0.316, -0.412, 3.335, 0.596, -3.580]`
- Softmax output: `[0.001, 0.155, 0.145, 0.001, 0.004, 0.016, 0.015, 0.622, 0.040, 0.001]`
- Sum of softmax: **1.000000**

**Observations:**
- The MLP returns both raw (logits) and softmax (probabilities) outputs.
- Softmax correctly normalizes the output into a valid probability distribution.
- He initialization (`sqrt(2/n)`) was used for weights to prevent vanishing/exploding gradients.
- The softmax output shows strong class discrimination (range: 0.001 to 0.622), with class 7 having the highest probability.

### 3.7 Full Feed-Forward CNN (Part vii)  CIFAR-10 Image

| Stage | Input Shape | Operation | Output Shape | Verified |
|---|---|---|---|---|
| Input |  | CIFAR-10 image (0017.jpg) | 32  32  3 |  |
| Conv1 | 32323 | 4(553), ReLU, s=1, p=2 | 32  32  4 |  |
| Pool1 | 32324 | Max 22, s=2 | 16  16  4 |  |
| Conv2 | 16164 | 4(554), ReLU, s=1, p=2 | 16  16  4 |  |
| Pool2 | 16164 | Max 22, s=2 | 8  8  4 |  |
| Flatten | 884 | Identity (W=I) | 256 |  |
| MLP | 256 | 49 (ReLU)  10 (softmax) | 10 |  |

**Final Output (CIFAR-10 image 0017.jpg):**
- Softmax vector: `[0.135, 0.114, 0.079, 0.071, 0.111, 0.091, 0.057, 0.122, 0.069, 0.152]`
- Sum: **1.000000**
- Valid probability distribution: **True**
- Predicted class index: **9** (highest probability: 0.152)

**Observations:**
- The network successfully processes a real CIFAR-10 image (32323, float32, range [0.0627, 0.8902]) through all layers and produces a 10-dimensional output vector.
- The softmax output is a valid probability distribution (all values positive, sum = 1.0).
- Since weights are randomly initialized (not trained), the output represents a random classification. With training, the network would learn to discriminate between CIFAR-10 classes.
- The spatial dimension reduction from 3232 to 88 through two pooling stages demonstrates effective feature compression.
- The identity flattening preserves all 256 features before the MLP stage.

## 4. Implementation Details

### 4.1 No Built-in Functions Used
- **Convolution**: Manual nested loops for correlation (no `scipy.signal.convolve2d`, no `cv2.filter2D`)
- **Pooling**: Manual sliding window (no `np.max` on the full array, only on individual windows)
- **Non-linearity**: All activations computed from scratch (even `tanh` implemented via exponentials, not `np.tanh`)
- **Padding**: Manual zero-padding array construction (no `np.pad`)
- **Striding**: Manual index stepping in loops

### 4.2 Numerical Stability
- Sigmoid: Input clipped to [-500, 500] to prevent overflow in `exp()`
- Tanh: Computed as `(e^x - e^(-x)) / (e^x + e^(-x))` with clipped inputs
- Softmax: Max-subtraction trick (`x - max(x)`) to prevent overflow

### 4.3 Weight Initialization
- Convolution kernels: `N(0, 0.12)`
- MLP weights: He initialization `N(0, sqrt(2/n_in))`
- Flattening (identity): `I(n, n)`
- Biases: Initialized to zeros

### 4.4 Generated Visualizations (outputs/ folder)

All images are saved in the `outputs/` directory, organized into subfolders by module part. Each image is a heatmap visualization with a colorbar indicating the value range.

**Note:** All tests (i) through (vii) use the **same real CIFAR-10 image** (`cifar10_images/0017.jpg`) as input. Only the filter kernels and MLP weights are randomly initialized.

#### Part (i)  Convolution Function (`outputs/part_i/`)  6 images

| Image | Data Source | Description |
|---|---|---|
| `conv_input_image.png` | **CIFAR-10** (32323) | The real CIFAR-10 image (0017.jpg) used as input to the convolution function. Displayed as a normalized RGB image. |
| `conv_kernel.png` | **Random** (553) | A single randomly initialized 553 filter kernel. Displayed as a grayscale heatmap averaged across the 3 input channels. |
| `conv_output_sigmoid.png` | **CIFAR-10**  **Sigmoid** | The 3232 output activation map from correlating the CIFAR-10 image with the random kernel, then applying sigmoid. Values compressed into (0, 1). Observed range: [0.0002, 0.6490]. |
| `conv_output_tanh.png` | **CIFAR-10**  **Tanh** | The 3232 output activation map after applying tanh. Values span approximately [-1, 1]. Observed range: [-1.0000, 0.5475]. |
| `conv_output_relu.png` | **CIFAR-10**  **ReLU** | The 3232 output activation map after applying ReLU. All negative values clipped to 0. Observed range: [0.0000, 0.6148]. |
| `conv_output_prelu.png` | **CIFAR-10**  **PReLU** | The 3232 output activation map after applying PReLU (=0.01). Small negative values preserved. Observed range: [-0.0870, 0.6148]. |

#### Part (ii)  Pooling Function (`outputs/part_ii/`)  3 images

| Image | Data Source | Description |
|---|---|---|
| `pool_input.png` | **CIFAR-10** derived (3232) | The 3232 single-channel activation map obtained by convolving the CIFAR-10 image with a random 553 kernel and applying ReLU. Used as input for both max and average pooling tests. |
| `pool_output_max.png` | **CIFAR-10**  **Max** | The 1616 output after max pooling (22 window, stride 2). Each cell = max of corresponding 22 region. Preserves the most prominent features. |
| `pool_output_avg.png` | **CIFAR-10**  **Avg** | The 1616 output after average pooling (22 window, stride 2). Each cell = mean of corresponding 22 region. Produces a smoother representation. |

#### Part (iii)  Convolution Layer Function (`outputs/part_iii/`)  3 images

| Image | Data Source | Description |
|---|---|---|
| `conv_layer_input_volume.png` | **CIFAR-10** (32323) | The CIFAR-10 image used as the input volume, displayed as 3 separate channel heatmaps (R, G, B channels). |
| `conv_layer_kernels.png` | **Random** (4 kernels, 553 each) | A grid of the 4 randomly initialized 553 filter kernels. Each kernel displayed as a grayscale heatmap. |
| `conv_layer_output_volume.png` | **CIFAR-10**  **ReLU** | The 32324 output activation volume, displayed as 4 separate channel heatmaps. Each channel = feature map from one kernel after ReLU. |

#### Part (iv)  Pooling Layer Function (`outputs/part_iv/`)  2 images

| Image | Data Source | Description |
|---|---|---|
| `pool_layer_input_volume.png` | **CIFAR-10** derived (32324) | The 32324 output volume from the convolution layer, displayed as 4 separate channel heatmaps before pooling. |
| `pool_layer_output_volume_max.png` | **CIFAR-10**  **Max** | The 16164 output volume after max pooling (22 window, stride 2) across all 4 channels. Spatial dimensions reduced from 3232 to 1616. |

#### Part (vii)  Feed-Forward CNN (`outputs/part_vii/`)  5 images

| Image | Data Source | Description |
|---|---|---|
| `conv_layer_input_volume.png` | **CIFAR-10** (32323) | The CIFAR-10 image as input to Conv1, displayed as 3 channel heatmaps. |
| `conv_layer_kernels.png` | **Random** (4 kernels, 553 each) | The 4 randomly initialized 553 kernels for Conv1, displayed as a grid. |
| `conv_layer_output_volume.png` | **CIFAR-10**  **Conv1**  **ReLU** | The 32324 output from Conv1, displayed as 4 channel heatmaps. |
| `pool_layer_input_volume.png` | **CIFAR-10** derived (16164) | The 16164 output from MaxPool1, displayed as 4 channel heatmaps. |
| `pool_layer_output_volume_max.png` | **CIFAR-10**  **Pool1**  **Max** | The 884 output from MaxPool2, displayed as 4 channel heatmaps. Spatial dimensions reduced from 1616 to 88. |

## 5. Project Structure

```
Q1/
 main.py                          # Entry point, tests all functions
 activations/
    activations.py               # sigmoid, tanh, relu, prelu, softmax + derivatives
 pooling_ops/
    pooling_ops.py               # max_pool_func, avg_pool_func
 layers/
    layers.py                    # convolution, pooling, conv_layer, pool_layer, flattening
 network/
    mlp.py                       # Multilayer Perceptron
    cnn_architecture.py          # Full feed-forward CNN
 utils/
    helpers.py                   # Display, padding, random generation utilities
 cifar10_images/
    0017.jpg                     # CIFAR-10 input image
 outputs/                         # Generated visualization images (19 images total)
    part_i/                      # Part (i)  Convolution function (6 images)
    part_ii/                     # Part (ii)  Pooling function (3 images)
    part_iii/                    # Part (iii)  Convolution layer function (3 images)
    part_iv/                     # Part (iv)  Pooling layer function (2 images)
    part_vii/                    # Part (vii)  Feed-forward CNN (5 images)
 report.md                        # This file
```

## 6. Conclusion

All CNN building blocks have been implemented and verified. The CNN architecture correctly processes a real CIFAR-10 image (32323) through two convolution-pooling stages, flattens the result, and passes it through an MLP to produce a 10-dimensional classification output. Every component was implemented from scratch without using any built-in deep learning library functions, and all output dimensions match the expected values. The softmax output forms a valid probability distribution with sum = 1.0.

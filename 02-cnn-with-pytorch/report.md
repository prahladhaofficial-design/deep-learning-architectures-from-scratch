# PyTorch CNN Experiment Report

## 2a) CNN Architecture and Training Setup

The CNN architecture from the NumPy CNN architecture was implemented in PyTorch with the following specification:

| Layer | Configuration | Output Shape |
|-------|--------------|--------------|
| Input | 32x32x3 RGB image | (N, 3, 32, 32) |
| Conv1 | 4 kernels, 5x5, stride=1, pad=2, ReLU | (N, 4, 32, 32) |
| MaxPool1 | 2x2, stride=2 | (N, 4, 16, 16) |
| Conv2 | 4 kernels, 5x5, stride=1, pad=2, ReLU | (N, 4, 16, 16) |
| MaxPool2 | 2x2, stride=2 | (N, 4, 8, 8) |
| Flatten | - | (N, 256) |
| FC1 (Hidden) | 256 -> 49, ReLU | (N, 49) |
| FC2 (Output) | 49 -> 10 | (N, 10) |
| Softmax | Applied at inference | (N, 10) |

All network parameters were initialized randomly using PyTorch's default initialization (Kaiming uniform for Conv2d and Linear layers).

**Training Configuration:**
- Training set: 100 images per class (1000 total)
- Test set: 10 images per class (100 total)
- Epochs: 15
- Learning rate: 0.001
- Cross-entropy loss with one-hot encoded labels (dimension 10)

---

## 2b) Training with Different Optimizers and Batch Sizes

### 2b-i) Vanilla SGD (eta = 0.001)

| Epoch | Train Loss | Train Error | Test Error | Train Acc | Test Acc |
|-------|-----------|-------------|------------|-----------|----------|
| 1 | 2.3050 | 2.3052 | 2.2946 | 0.1000 | 0.1000 |
| 5 | 2.3051 | 2.3047 | 2.2948 | 0.1020 | 0.1000 |
| 10 | 2.3042 | 2.3059 | 2.2949 | 0.1010 | 0.1000 |
| 15 | 2.3041 | 2.3050 | 2.2950 | 0.1000 | 0.1000 |

**Observation:** Vanilla SGD with eta = 0.001 shows virtually no learning. The loss remains at approximately 2.305 (close to ln(10) = 2.3026, the theoretical loss for random guessing on 10 classes). Accuracy stays at 10% (random chance) throughout all 15 epochs. The gradient norms are small (~0.02), and with lr=0.001, the weight updates are on the order of 0.00002 per step, which is insufficient to make meaningful progress in 15 epochs on this architecture.

### 2b-ii) Momentum SGD (eta = 0.001, alpha = 0.9)

| Epoch | Train Loss | Train Error | Test Error | Train Acc | Test Acc |
|-------|-----------|-------------|------------|-----------|----------|
| 1 | 2.3050 | 2.3050 | 2.2946 | 0.1010 | 0.1000 |
| 5 | 2.3043 | 2.3038 | 2.2965 | 0.1000 | 0.1000 |
| 10 | 2.3028 | 2.3038 | 2.2972 | 0.0990 | 0.1000 |
| 15 | 2.3022 | 2.3024 | 2.2973 | 0.1000 | 0.0900 |

**Observation:** Momentum SGD also fails to learn with eta = 0.001. Despite the momentum coefficient (alpha = 0.9) accumulating velocity from past gradients, the base learning rate is too small for the accumulated velocity to overcome the flat loss landscape. The loss decreases marginally from 2.3050 to 2.3022 over 15 epochs, but accuracy remains at random-chance levels.

### 2b-iii) RMSProp (eta = 0.001, rho = 0.9)

| Epoch | Train Loss | Train Error | Test Error | Train Acc | Test Acc |
|-------|-----------|-------------|------------|-----------|----------|
| 1 | 2.3023 | 2.2960 | 2.2863 | 0.1000 | 0.1000 |
| 3 | 2.2439 | 2.2462 | 2.2408 | 0.2010 | 0.2400 |
| 5 | 2.0850 | 2.0706 | 2.0923 | 0.2660 | 0.2400 |
| 7 | 1.9297 | 1.8370 | 1.8445 | 0.3610 | 0.3600 |
| 10 | 1.7847 | 1.7315 | 1.7724 | 0.3770 | 0.3500 |
| 13 | 1.6946 | 1.7086 | 1.9303 | 0.4060 | 0.3500 |
| 15 | 1.6560 | 1.8611 | 1.9415 | 0.3440 | 0.3100 |

**Observation:** RMSProp is the only optimizer that shows meaningful learning with eta = 0.001. The per-parameter adaptive learning rates allow RMSProp to make effective progress even with a small base learning rate. Training error decreases from 2.2960 to 1.8611, and training accuracy reaches 34.4% (with peak of 40.6% at epoch 13). However, there is noticeable fluctuation in the error curve, particularly a spike at epoch 12 (train error jumps to 2.0544), indicating instability in the optimization landscape.

### 2b) Batch Size Experiment

Training with RMSProp (eta = 0.001, rho = 0.9) using different batch sizes:

| Batch Size | Final Train Acc | Final Test Acc | Final Train Error | Final Test Error |
|-----------|----------------|---------------|------------------|-----------------|
| 16 | 0.3770 | 0.3000 | 1.7673 | 1.8683 |
| 32 | 0.3440 | 0.3100 | 1.8611 | 1.9415 |
| 64 | 0.3350 | 0.3600 | 1.8978 | 1.9088 |
| 128 | 0.3220 | 0.3100 | 1.9212 | 2.0256 |

**Observation:** Smaller batch sizes (16, 32) achieve better training accuracy, likely due to the noise in gradient estimates acting as implicit regularization. Batch size 16 achieves the highest training accuracy (37.7%) but batch size 64 achieves the best test accuracy (36.0%), suggesting a better generalization trade-off. Batch size 128 shows the worst performance across all metrics, likely because the larger batch size provides less stochasticity and the model converges to a sharper minimum.

### Key Comparison: Optimizer Performance

| Optimizer | Final Train Acc | Final Test Acc | Final Train Error | Final Test Error |
|-----------|----------------|---------------|------------------|-----------------|
| SGD (eta=0.001) | 0.1000 | 0.1000 | 2.3050 | 2.2950 |
| Momentum (eta=0.001, alpha=0.9) | 0.1000 | 0.0900 | 2.3024 | 2.2973 |
| RMSProp (eta=0.001, rho=0.9) | 0.3440 | 0.3100 | 1.8611 | 1.9415 |

**Observation:** RMSProp significantly outperforms both SGD and Momentum at the same learning rate. This is because RMSProp maintains per-parameter adaptive learning rates by dividing the gradient by a running average of its magnitude, which is particularly effective when different parameters have gradients of vastly different scales (as is common in CNNs with both convolutional and fully-connected layers).

---

## 2c) Evaluation Results

### 2c-i) Training and Test Error Curves

Error curves (cross-entropy loss) were plotted for each optimizer as a function of epochs. The plots are saved in `outputs/part_c/`.

**SGD:** Flat line at ~2.305 for both train and test error, indicating no learning.

**Momentum:** Nearly flat line with very slight decrease, still at ~2.30, indicating negligible learning.

**RMSProp:** Clear downward trend in training error from 2.30 to 1.66. Test error follows a similar trend but with more variance, reaching 1.94 at epoch 15. The gap between train and test error widens in later epochs, suggesting some overfitting.

### 2c-ii) Activation Map Visualization

Activation maps were visualized for both convolution layers using the RMSProp-trained model:

**Conv1 Activation Maps (32x32x4):**
- 4 channels, each producing a 32x32 activation map
- Early-layer features capture edges, corners, and simple textures
- Different channels respond to different visual patterns in the input images
- Spatial resolution preserved (same as input due to pad=2)

**Conv2 Activation Maps (16x16x4):**
- 4 channels, each producing a 16x16 activation map
- After max pooling, spatial resolution is halved
- Features are more abstract, combining patterns from Conv1
- Some channels show strong activations while others are sparse

### 2c-iii) Classification Accuracy Summary

| Optimizer | Train Accuracy | Test Accuracy |
|-----------|---------------|--------------|
| SGD | 10.0% | 10.0% |
| Momentum | 10.0% | 9.0% |
| RMSProp | 34.4% | 31.0% |

RMSProp achieves 34.4% training accuracy and 31.0% test accuracy, which is significantly above the random baseline of 10%. The 3.4% gap between train and test accuracy indicates mild overfitting.

---

## 2d) t-SNE Visualization of Bottleneck Features

The bottleneck features (256-dimensional flattened output after the second max pooling layer, before the MLP) were visualized using t-SNE for the 100 test images (10 per class).

### After Epoch 1

At the beginning of training, the bottleneck features are essentially random projections of the input images. The t-SNE plot shows:
- All 10 classes are heavily intermixed with no clear cluster structure
- Points are scattered uniformly across the 2D embedding space
- This is expected behavior since the network weights are near-random after only one epoch of training

### After Epoch 15 (Last Epoch)

After 15 epochs of RMSProp training, the t-SNE plot shows:
- Some degree of class clustering begins to emerge
- Certain classes (e.g., ship, truck) start to form loose clusters
- However, significant overlap remains between visually similar classes (e.g., cat/dog, bird/frog)
- The clustering is not perfect, consistent with the 31% test accuracy

### Comparison

The side-by-side t-SNE comparison shows a subtle but noticeable improvement in class separation from epoch 1 to epoch 15. The features have started to encode class-discriminative information, but 15 epochs of training on 1000 images is insufficient for strong feature learning.

---

## 2e) Dropout Experiments

Dropout was applied at the MLP hidden layer (49 nodes) with rates p = 0.0, 0.2, 0.5, 0.8. All models were trained using RMSProp (eta = 0.001, rho = 0.9) for 15 epochs.

### Results

| Dropout Rate (p) | Train Acc | Test Acc | Train Error | Test Error | Train-Test Gap |
|-----------------|-----------|----------|------------|-----------|---------------|
| 0.0 (No Dropout) | 34.4% | 31.0% | 1.8611 | 1.9415 | +3.4% |
| 0.2 | 36.9% | 38.0% | 1.7986 | 1.8409 | -1.1% |
| 0.5 | 30.1% | 32.0% | 1.9147 | 1.9733 | -1.9% |
| 0.8 | 26.9% | 22.0% | 2.0753 | 2.0726 | +4.9% |

### Observations

**p = 0.0 (No Dropout):**
- Baseline model without regularization
- Train accuracy (34.4%) > Test accuracy (31.0%), showing mild overfitting
- The model memorizes training patterns that don't generalize well

**p = 0.2:**
- **Best performing configuration** - achieves the highest test accuracy (38.0%)
- Test accuracy exceeds train accuracy (38.0% vs 36.9%), indicating no overfitting
- The moderate dropout rate acts as an effective regularizer, preventing the model from co-adapting features while still allowing sufficient learning capacity
- This is the sweet spot for this small dataset

**p = 0.5:**
- Both train and test accuracy drop compared to p=0.2
- Test accuracy (32.0%) still slightly better than train accuracy (30.1%)
- The higher dropout rate starts to impair the model's learning capacity
- The model is under-regularized at p=0.2 but over-regularized at p=0.5

**p = 0.8:**
- Worst performance among all dropout rates
- Only 26.9% train accuracy and 22.0% test accuracy
- Such aggressive dropout (80% of neurons dropped) severely limits the model's capacity
- The train-test gap returns to positive (+4.9%), suggesting the model is too weak to learn meaningful patterns

### Key Finding

Dropout at p=0.2 improves test accuracy from 31.0% (no dropout) to 38.0%, a 7 percentage point improvement. This demonstrates that moderate dropout is highly effective for small datasets where overfitting is a concern. However, excessive dropout (p=0.8) is detrimental as it prevents the network from learning useful representations.

---

## 2f) Batch Normalization Experiments

Batch normalization was applied at the output of the MLP hidden layer (before ReLU activation). Three configurations were compared, all trained with RMSProp:

### Results

| Model | Train Acc | Test Acc | Train Error | Test Error |
|-------|-----------|----------|------------|-----------|
| Vanilla (no BN, no dropout) | 34.4% | 31.0% | 1.8611 | 1.9415 |
| Dropout only (p=0.2) | 36.9% | 38.0% | 1.7986 | 1.8409 |
| BN + Dropout (p=0.2) | 60.1% | 30.0% | 1.1884 | 2.1373 |

### Observations

**Vanilla Model:**
- Baseline: 34.4% train accuracy, 31.0% test accuracy
- Mild overfitting (3.4% gap)

**Dropout Only (p=0.2):**
- Best generalization: 36.9% train, 38.0% test accuracy
- Negative train-test gap (-1.1%), meaning the model generalizes better than it fits training data
- Dropout effectively prevents co-adaptation of features

**BN + Dropout (p=0.2):**
- **Severe overfitting observed:** 60.1% train accuracy vs only 30.0% test accuracy
- The train-test gap is massive: 30.1 percentage points
- Training error drops rapidly to 1.09 (much lower than any other model)
- Test error increases to 2.14 (worse than random baseline of ~2.30 in later epochs)

### Why BN + Dropout Overfits Severely

The combination of batch normalization and dropout creates a problematic interaction:

1. **Batch Normalization** normalizes the hidden layer outputs, which accelerates training and allows the model to fit the training data very quickly (60.1% train accuracy).

2. **Dropout** randomly zeroes 20% of neurons during training, but during evaluation (test mode), all neurons are active and their outputs are scaled.

3. **The mismatch:** During training, BN computes running mean/variance statistics that are influenced by the dropout pattern. During testing, these running statistics are used without dropout, creating a distribution shift between training and test time.

4. **Small dataset amplification:** With only 1000 training images, the BN statistics are estimated from very few samples, making them unreliable. The model memorizes training patterns (60.1% train accuracy) but fails completely on test data (30.0%).

### Comparison Summary

| Aspect | Vanilla | Dropout (p=0.2) | BN + Dropout (p=0.2) |
|--------|---------|-----------------|---------------------|
| Train Accuracy | 34.4% | 36.9% | 60.1% |
| Test Accuracy | 31.0% | 38.0% | 30.0% |
| Overfitting | Mild | None (negative gap) | Severe |
| Best for | - | Generalization | Not recommended |

**Conclusion:** For this small dataset (1000 training images), dropout alone (p=0.2) is the best regularization strategy. Batch normalization, while accelerating training, combined with dropout leads to severe overfitting due to the interaction between BN statistics and dropout patterns, amplified by the small dataset size.

---

## Overall Summary

1. **Optimizer choice matters critically:** RMSProp is the only optimizer that learns effectively at eta=0.001. SGD and Momentum require higher learning rates.

2. **RMSProp achieves 31-38% accuracy:** On a 10-class problem with only 1000 training images, this is reasonable but far from state-of-the-art.

3. **Dropout at p=0.2 is optimal:** Improves test accuracy by 7 percentage points over the baseline and eliminates overfitting.

4. **Batch normalization + dropout is harmful:** The combination causes severe overfitting (60% train vs 30% test) on this small dataset.

5. **Batch size 64 generalizes best:** While batch size 16 achieves higher training accuracy, batch size 64 achieves the best test accuracy among the batch sizes tested.

6. **t-SNE shows gradual feature learning:** Features become somewhat more class-separable after 15 epochs, but strong clustering requires more training.

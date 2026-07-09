# PyTorch CIFAR-10 CNN Experiments

This component trains and analyzes a convolutional neural network for CIFAR-10 image classification using PyTorch. It extends the earlier NumPy CNN architecture into a full experiment pipeline with data loading, training loops, evaluation, optimizer comparison, regularization, and representation visualization.

## What Is Included

- CIFAR-10 dataset wrapper with per-class subset creation.
- CNN model with convolution, pooling, dense layers, optional dropout, and optional batch normalization.
- Training utilities for SGD, momentum, RMSProp, and related comparisons.
- Evaluation utilities for loss, error, and accuracy tracking.
- Activation-map visualization for convolutional layers.
- t-SNE plots comparing learned feature representations across training.
- Dropout and batch-normalization experiments with saved plots.

## Structure

```text
data/              CIFAR-10 dataset loader
model/             CNN architecture
training/          Trainer and evaluator utilities
experiments/       Dropout and batch-normalization studies
visualization/     Error curves, t-SNE, and activation maps
utils/             Logging, output paths, metrics, and helpers
outputs/           Generated plots and analysis artifacts
main.py            Full experiment runner
```

## How To Run

Place CIFAR-10 data under `CIFAR-10-data/` if it is not already available, then run:

```bash
python main.py
```

The runner trains on a compact balanced subset, evaluates on a balanced test subset, and writes plots into `outputs/`.

## Experiment Highlights

- Batch-size comparison over 16, 32, 64, and 128.
- Optimizer comparison across SGD, momentum, and RMSProp-style training.
- Feature-map inspection after convolutional layers.
- Representation comparison with t-SNE between early and later epochs.
- Dropout and batch-normalization effects on train-test behavior.

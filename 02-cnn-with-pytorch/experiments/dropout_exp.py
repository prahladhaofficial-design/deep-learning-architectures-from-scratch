import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.cnn_model import CNNModel
from training.trainer import train_model, set_seed
from training.evaluators import Evaluator
from visualization.plots import plot_train_test_error


def run_dropout_experiment(
    train_images,
    train_labels,
    test_images,
    test_labels,
    dropout_p,
    lr=0.001,
    rho=0.9,
    num_epochs=15,
    batch_size=32,
    device="cpu",
    label="dropout",
):
    """
    Train the CNN with a specific dropout rate at the MLP hidden layer.
    Uses RMSProp optimizer.

    Args:
        train_images: (N, 3, 32, 32) numpy array
        train_labels: (N,) numpy array of integer labels
        test_images: (M, 3, 32, 32) numpy array
        test_labels: (M,) numpy array of integer labels
        dropout_p: dropout probability (0.2, 0.5, 0.8)
        lr: learning rate
        rho: RMSProp decay rate
        num_epochs: number of epochs
        batch_size: mini-batch size
        device: torch device
        label: identifier for output filenames

    Returns:
        history dict from training
    """
    set_seed(42)

    # Convert numpy arrays to torch tensors
    train_tensor_x = torch.tensor(train_images, dtype=torch.float32)
    train_tensor_y = torch.tensor(train_labels, dtype=torch.long)
    test_tensor_x = torch.tensor(test_images, dtype=torch.float32)
    test_tensor_y = torch.tensor(test_labels, dtype=torch.long)

    # Create DataLoaders
    train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
    test_dataset = TensorDataset(test_tensor_x, test_tensor_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model with specified dropout rate
    model = CNNModel(dropout_p=dropout_p, use_batchnorm=False)
    evaluator = Evaluator()

    print(f"\n{'=' * 60}")
    print(f"[Dropout Experiment] p={dropout_p}")
    print(f"{'=' * 60}")

    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer_name="rmsprop",
        lr=lr,
        rho=rho,
        num_epochs=num_epochs,
        device=device,
        evaluator=evaluator,
    )

    # Plot error curves
    plot_train_test_error(
        history["train_errors"],
        history["test_errors"],
        optimizer_name=f"Dropout_p{dropout_p}",
        subfolder="part_e",
    )

    final_train_acc = history["train_accs"][-1]
    final_test_acc = history["test_accs"][-1]
    print(
        f"\n[Dropout p={dropout_p}] Final Train Acc: {final_train_acc:.4f}, Test Acc: {final_test_acc:.4f}"
    )

    return history


def run_all_dropout_experiments(
    train_images,
    train_labels,
    test_images,
    test_labels,
    dropout_rates=None,
    lr=0.001,
    num_epochs=15,
    batch_size=32,
    device="cpu",
):
    """
    Run dropout experiments for multiple dropout rates.
    Compares p=0.0 (no dropout), 0.2, 0.5, 0.8 and reports differences
    between training and testing performance.

    Args:
        train_images: (N, 3, 32, 32) numpy array
        train_labels: (N,) numpy array
        test_images: (M, 3, 32, 32) numpy array
        test_labels: (M,) numpy array
        dropout_rates: list of dropout probabilities to test
        lr: learning rate
        num_epochs: number of epochs
        batch_size: mini-batch size
        device: torch device

    Returns:
        results: dict mapping dropout_rate -> history dict
    """
    if dropout_rates is None:
        dropout_rates = [0.0, 0.2, 0.5, 0.8]

    results = {}

    for p in dropout_rates:
        history = run_dropout_experiment(
            train_images=train_images,
            train_labels=train_labels,
            test_images=test_images,
            test_labels=test_labels,
            dropout_p=p,
            lr=lr,
            num_epochs=num_epochs,
            batch_size=batch_size,
            device=device,
            label=f"dropout_p{p}",
        )
        results[p] = history

    # Print comparison summary including no-dropout baseline
    print(f"\n{'=' * 60}")
    print("[Dropout Experiment] Summary Comparison (with no-dropout baseline)")
    print(f"{'=' * 60}")
    print(
        f"{'Dropout p':<12} {'Train Acc':<12} {'Test Acc':<12} {'Train Err':<12} {'Test Err':<12}"
    )
    print("-" * 60)
    for p, hist in results.items():
        train_acc = hist["train_accs"][-1]
        test_acc = hist["test_accs"][-1]
        train_err = hist["train_errors"][-1]
        test_err = hist["test_errors"][-1]
        print(
            f"{p:<12.1f} {train_acc:<12.4f} {test_acc:<12.4f} {train_err:<12.4f} {test_err:<12.4f}"
        )

    # Print gap analysis between train and test
    print(f"\n[Dropout Experiment] Train-Test Gap Analysis")
    print("-" * 60)
    print(f"{'Dropout p':<12} {'Train-Test Acc Gap':<20} {'Overfitting Indicator'}")
    print("-" * 60)
    for p, hist in results.items():
        train_acc = hist["train_accs"][-1]
        test_acc = hist["test_accs"][-1]
        gap = train_acc - test_acc
        indicator = "High" if gap > 0.15 else "Moderate" if gap > 0.05 else "Low"
        print(f"{p:<12.1f} {gap:<20.4f} {indicator}")

    return results

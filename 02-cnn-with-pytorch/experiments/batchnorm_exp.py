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


def run_batchnorm_experiment(
    train_images,
    train_labels,
    test_images,
    test_labels,
    dropout_p=0.2,
    lr=0.001,
    rho=0.9,
    num_epochs=15,
    batch_size=32,
    device="cpu",
):
    """
    Train the CNN with batch normalization at the MLP hidden layer
    and optional dropout.

    Batch norm is applied at the output of the hidden layer BEFORE ReLU.
    Dropout is applied after ReLU.

    Args:
        train_images: (N, 3, 32, 32) numpy array
        train_labels: (N,) numpy array
        test_images: (M, 3, 32, 32) numpy array
        test_labels: (M,) numpy array
        dropout_p: dropout probability (default 0.2)
        lr: learning rate
        rho: RMSProp decay rate
        num_epochs: number of epochs
        batch_size: mini-batch size
        device: torch device

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

    # Create model with batch norm and dropout
    model = CNNModel(dropout_p=dropout_p, use_batchnorm=True)
    evaluator = Evaluator()

    print(f"\n{'=' * 60}")
    print(f"[BatchNorm + Dropout Experiment] BN=True, dropout_p={dropout_p}")
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
        optimizer_name=f"BN_Dropout_p{dropout_p}",
        subfolder="part_f",
    )

    final_train_acc = history["train_accs"][-1]
    final_test_acc = history["test_accs"][-1]
    print(
        f"\n[BN+Dropout p={dropout_p}] Final Train Acc: {final_train_acc:.4f}, Test Acc: {final_test_acc:.4f}"
    )

    return history


def run_all_batchnorm_experiments(
    train_images,
    train_labels,
    test_images,
    test_labels,
    lr=0.001,
    num_epochs=15,
    batch_size=32,
    device="cpu",
):
    """
    Run batch normalization experiments and compare with vanilla and dropout-only models.

    Trains three models:
        1. Vanilla (no BN, no dropout)
        2. Dropout only (p=0.2)
        3. BN + Dropout (p=0.2)

    Args:
        train_images: (N, 3, 32, 32) numpy array
        train_labels: (N,) numpy array
        test_images: (M, 3, 32, 32) numpy array
        test_labels: (M,) numpy array
        lr: learning rate
        num_epochs: number of epochs
        batch_size: mini-batch size
        device: torch device

    Returns:
        results: dict with keys "vanilla", "dropout", "bn_dropout"
    """
    results = {}

    # 1. Vanilla model (no BN, no dropout)
    set_seed(42)
    train_tensor_x = torch.tensor(train_images, dtype=torch.float32)
    train_tensor_y = torch.tensor(train_labels, dtype=torch.long)
    test_tensor_x = torch.tensor(test_images, dtype=torch.float32)
    test_tensor_y = torch.tensor(test_labels, dtype=torch.long)

    train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
    test_dataset = TensorDataset(test_tensor_x, test_tensor_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"\n{'=' * 60}")
    print("[BatchNorm Experiment] Vanilla model (no BN, no dropout)")
    print(f"{'=' * 60}")

    vanilla_model = CNNModel(dropout_p=0.0, use_batchnorm=False)
    vanilla_evaluator = Evaluator()
    results["vanilla"] = train_model(
        model=vanilla_model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer_name="rmsprop",
        lr=lr,
        rho=0.9,
        num_epochs=num_epochs,
        device=device,
        evaluator=vanilla_evaluator,
    )
    plot_train_test_error(
        results["vanilla"]["train_errors"],
        results["vanilla"]["test_errors"],
        optimizer_name="Vanilla",
        subfolder="part_f",
    )

    # 2. Dropout only (p=0.2)
    print(f"\n{'=' * 60}")
    print("[BatchNorm Experiment] Dropout only (p=0.2)")
    print(f"{'=' * 60}")

    set_seed(42)
    dropout_model = CNNModel(dropout_p=0.2, use_batchnorm=False)
    dropout_evaluator = Evaluator()
    results["dropout"] = train_model(
        model=dropout_model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer_name="rmsprop",
        lr=lr,
        rho=0.9,
        num_epochs=num_epochs,
        device=device,
        evaluator=dropout_evaluator,
    )
    plot_train_test_error(
        results["dropout"]["train_errors"],
        results["dropout"]["test_errors"],
        optimizer_name="Dropout_p0.2",
        subfolder="part_f",
    )

    # 3. BN + Dropout (p=0.2)
    print(f"\n{'=' * 60}")
    print("[BatchNorm Experiment] BN + Dropout (p=0.2)")
    print(f"{'=' * 60}")

    set_seed(42)
    bn_dropout_model = CNNModel(dropout_p=0.2, use_batchnorm=True)
    bn_dropout_evaluator = Evaluator()
    results["bn_dropout"] = train_model(
        model=bn_dropout_model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer_name="rmsprop",
        lr=lr,
        rho=0.9,
        num_epochs=num_epochs,
        device=device,
        evaluator=bn_dropout_evaluator,
    )
    plot_train_test_error(
        results["bn_dropout"]["train_errors"],
        results["bn_dropout"]["test_errors"],
        optimizer_name="BN_Dropout_p0.2",
        subfolder="part_f",
    )

    # Print comparison summary
    print(f"\n{'=' * 60}")
    print("[BatchNorm Experiment] Summary Comparison")
    print(f"{'=' * 60}")
    print(
        f"{'Model':<20} {'Train Acc':<12} {'Test Acc':<12} {'Train Err':<12} {'Test Err':<12}"
    )
    print("-" * 68)
    for name, hist in results.items():
        train_acc = hist["train_accs"][-1]
        test_acc = hist["test_accs"][-1]
        train_err = hist["train_errors"][-1]
        test_err = hist["test_errors"][-1]
        print(
            f"{name:<20} {train_acc:<12.4f} {test_acc:<12.4f} {train_err:<12.4f} {test_err:<12.4f}"
        )

    return results

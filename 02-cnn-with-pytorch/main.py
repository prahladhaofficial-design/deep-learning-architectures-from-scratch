import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------------------------------------------
# Add project root to path so all local packages are importable
# -------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# -------------------------------------------------------------------
# Local imports
# -------------------------------------------------------------------
from data.dataset import CIFAR10Dataset, CIFAR10_CLASSES
from model.cnn_model import CNNModel
from training.trainer import train_model, set_seed
from training.evaluators import Evaluator
from visualization.plots import (
    plot_train_test_error,
    plot_all_optimizers_comparison,
    plot_tsne,
    plot_tsne_comparison,
)
from visualization.activation_viz import visualize_all_conv_activations
from experiments.dropout_exp import run_all_dropout_experiments
from experiments.batchnorm_exp import run_all_batchnorm_experiments
from utils.helpers import (
    get_output_dir,
    save_plot,
    compute_accuracy,
    compute_error,
    cross_entropy_loss,
    log_training_info,
)

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CIFAR-10-data")
NUM_EPOCHS = 15
BATCH_SIZES = [16, 32, 64, 128]
BEST_BATCH_SIZE = 32
LR = 0.001
MOMENTUM = 0.9
RHO = 0.9
TRAIN_PER_CLASS = 100
TEST_PER_CLASS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data():
    """
    Load CIFAR-10 dataset and create training/test subsets.

    Returns:
        train_loader: DataLoader for 100 imgs/class (1000 total)
        test_loader: DataLoader for 10 imgs/class (100 total)
        train_images: (1000, 3, 32, 32) numpy array
        train_labels: (1000,) numpy array
        test_images: (100, 3, 32, 32) numpy array
        test_labels: (100,) numpy array
    """
    print("\n========== Loading CIFAR-10 Data ==========")

    # Load full datasets
    full_train = CIFAR10Dataset(DATA_DIR, split="train")
    full_test = CIFAR10Dataset(DATA_DIR, split="test")

    # Create subsets: 100 per class for train, 10 per class for test
    train_subset = full_train.get_subset(TRAIN_PER_CLASS)
    test_subset = full_test.get_subset(TEST_PER_CLASS)

    # Convert to numpy arrays (CHW format for PyTorch)
    train_images = []
    train_labels = []
    for i in range(len(train_subset)):
        img, lbl = train_subset[i]
        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        train_images.append(img)
        train_labels.append(lbl)
    train_images = np.array(train_images, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int64)

    test_images = []
    test_labels = []
    for i in range(len(test_subset)):
        img, lbl = test_subset[i]
        img = np.transpose(img, (2, 0, 1))
        test_images.append(img)
        test_labels.append(lbl)
    test_images = np.array(test_images, dtype=np.float32)
    test_labels = np.array(test_labels, dtype=np.int64)

    print(f"Train images shape: {train_images.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Test images shape: {test_images.shape}")
    print(f"Test labels shape: {test_labels.shape}")

    # Create DataLoaders
    train_tensor_x = torch.tensor(train_images, dtype=torch.float32)
    train_tensor_y = torch.tensor(train_labels, dtype=torch.long)
    test_tensor_x = torch.tensor(test_images, dtype=torch.float32)
    test_tensor_y = torch.tensor(test_labels, dtype=torch.long)

    train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
    test_dataset = TensorDataset(test_tensor_x, test_tensor_y)

    train_loader = DataLoader(train_dataset, batch_size=BEST_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BEST_BATCH_SIZE, shuffle=False)

    return (
        train_loader,
        test_loader,
        train_images,
        train_labels,
        test_images,
        test_labels,
    )


def part_batch_size_experiment(train_images, train_labels, test_images, test_labels):
    """
    Batch-size experiment with different mini-batch sizes.
    Trains with batch sizes 16, 32, 64, 128 using RMSProp and compares.
    """
    print("\n========== Batch Size Experiment ==========")

    batch_results = {}

    for bs in BATCH_SIZES:
        set_seed(42)

        # Create DataLoaders with current batch size
        train_tensor_x = torch.tensor(train_images, dtype=torch.float32)
        train_tensor_y = torch.tensor(train_labels, dtype=torch.long)
        test_tensor_x = torch.tensor(test_images, dtype=torch.float32)
        test_tensor_y = torch.tensor(test_labels, dtype=torch.long)

        train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
        test_dataset = TensorDataset(test_tensor_x, test_tensor_y)

        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

        model = CNNModel(dropout_p=0.0, use_batchnorm=False)
        evaluator = Evaluator()

        print(f"\n--- Batch Size: {bs} ---")
        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer_name="rmsprop",
            lr=LR,
            rho=RHO,
            num_epochs=NUM_EPOCHS,
            device=DEVICE,
            evaluator=evaluator,
        )

        batch_results[bs] = {
            "train_errors": history["train_errors"],
            "test_errors": history["test_errors"],
            "train_accs": history["train_accs"],
            "test_accs": history["test_accs"],
        }

        # Plot error curves for this batch size
        plot_train_test_error(
            history["train_errors"],
            history["test_errors"],
            optimizer_name=f"RMSProp_bs{bs}",
            subfolder="part_b",
        )

    # Print comparison summary
    print(f"\n{'=' * 60}")
    print("[Batch Size Experiment] Summary Comparison")
    print(f"{'=' * 60}")
    print(
        f"{'Batch Size':<12} {'Train Acc':<12} {'Test Acc':<12} {'Train Err':<12} {'Test Err':<12}"
    )
    print("-" * 60)
    for bs, res in batch_results.items():
        train_acc = res["train_accs"][-1]
        test_acc = res["test_accs"][-1]
        train_err = res["train_errors"][-1]
        test_err = res["test_errors"][-1]
        print(
            f"{bs:<12} {train_acc:<12.4f} {test_acc:<12.4f} "
            f"{train_err:<12.4f} {test_err:<12.4f}"
        )

    return batch_results


def part_a_b_c(
    train_loader, test_loader, train_images, train_labels, test_images, test_labels
):
    """
    Train the CNN with random initialization, compare three optimizers, compute errors, visualize activations, and report accuracy
    """
    print("\n========== Training with 3 Optimizers ==========")

    all_results = {}

    optimizers_config = [
        ("SGD", {"lr": LR}),
        ("Momentum", {"lr": LR, "momentum": MOMENTUM}),
        ("RMSProp", {"lr": LR, "rho": RHO}),
    ]

    for opt_name, opt_kwargs in optimizers_config:
        set_seed(42)

        model = CNNModel(dropout_p=0.0, use_batchnorm=False)
        evaluator = Evaluator()

        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer_name=opt_name,
            num_epochs=NUM_EPOCHS,
            device=DEVICE,
            evaluator=evaluator,
            **opt_kwargs,
        )

        # Plot train/test error curves.
        plot_train_test_error(
            history["train_errors"],
            history["test_errors"],
            optimizer_name=opt_name,
            subfolder="part_c",
        )

        all_results[opt_name] = {
            "train_errors": history["train_errors"],
            "test_errors": history["test_errors"],
        }

        # Visualize activation maps using the RMSProp model's collected activations.
        if opt_name == "RMSProp":
            if history.get("conv1_epoch1") is not None:
                visualize_all_conv_activations(
                    history["conv1_last"],
                    history["conv2_last"],
                    sample_indices=[0, 1, 2, 3],
                    subfolder="part_c",
                )

        # Report classification accuracy.
        final_train_acc = history["train_accs"][-1]
        final_test_acc = history["test_accs"][-1]
        print(f"\n[{opt_name}] Final Train Accuracy: {final_train_acc:.4f}")
        print(f"[{opt_name}] Final Test Accuracy: {final_test_acc:.4f}")

    # Plot all optimizers comparison
    plot_all_optimizers_comparison(all_results, subfolder="part_c")

    return all_results


def part_d(train_loader, test_loader, test_labels):
    """
    t-SNE visualization of bottleneck features after epoch 1 and the last epoch.
    Uses the RMSProp-trained model's collected features.
    """
    print("\n========== t-SNE Visualization ==========")

    set_seed(42)

    model = CNNModel(dropout_p=0.0, use_batchnorm=False)
    evaluator = Evaluator()

    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer_name="rmsprop",
        lr=LR,
        rho=RHO,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        evaluator=evaluator,
    )

    # Extract bottleneck features from epoch 1 and last epoch
    bottleneck_epoch1 = history["bottleneck_epoch1"]
    bottleneck_last = history["bottleneck_last"]
    labels = history["test_labels"]

    if bottleneck_epoch1 is not None and bottleneck_last is not None:
        # Convert to numpy for t-SNE
        features_epoch1 = bottleneck_epoch1.numpy()
        features_last = bottleneck_last.numpy()

        # Individual t-SNE plots
        plot_tsne(
            features_epoch1,
            labels,
            CIFAR10_CLASSES,
            epoch_label="Epoch_1",
            subfolder="part_d",
        )
        plot_tsne(
            features_last,
            labels,
            CIFAR10_CLASSES,
            epoch_label=f"Epoch_{NUM_EPOCHS}",
            subfolder="part_d",
        )

        # Side-by-side comparison
        plot_tsne_comparison(
            features_epoch1,
            features_last,
            labels,
            CIFAR10_CLASSES,
            subfolder="part_d",
        )
    else:
        print("[t-SNE] Warning: Bottleneck features not collected. Skipping.")


def part_e(train_images, train_labels, test_images, test_labels):
    """
    Dropout experiments with p = 0.2, 0.5, 0.8 using RMSProp.
    """
    print("\n========== Dropout Experiments ==========")

    results = run_all_dropout_experiments(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        lr=LR,
        num_epochs=NUM_EPOCHS,
        batch_size=BEST_BATCH_SIZE,
        device=DEVICE,
    )

    return results


def part_f(train_images, train_labels, test_images, test_labels):
    """
    Batch normalization plus dropout experiments.
    Compares vanilla, dropout-only, and BN+dropout models.
    """
    print("\n========== Batch Normalization Experiments ==========")

    results = run_all_batchnorm_experiments(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        lr=LR,
        num_epochs=NUM_EPOCHS,
        batch_size=BEST_BATCH_SIZE,
        device=DEVICE,
    )

    return results


def main():
    """
    Main entry point for the PyTorch CNN experiment pipeline.
    Orchestrates all parts (a) through (f).
    """
    print("=" * 60)
    print("PyTorch CNN Experiment Pipeline")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch sizes to experiment with: {BATCH_SIZES}")
    print(f"Best batch size used: {BEST_BATCH_SIZE}")
    print(
        f"Train subset: {TRAIN_PER_CLASS} images/class ({TRAIN_PER_CLASS * 10} total)"
    )
    print(f"Test subset: {TEST_PER_CLASS} images/class ({TEST_PER_CLASS * 10} total)")

    # Load data
    (
        train_loader,
        test_loader,
        train_images,
        train_labels,
        test_images,
        test_labels,
    ) = load_data()

    # Experiment with different batch sizes (16, 32, 64, 128).
    part_batch_size_experiment(train_images, train_labels, test_images, test_labels)

    # Train with three optimizers, evaluate, and visualize.
    part_a_b_c(
        train_loader,
        test_loader,
        train_images,
        train_labels,
        test_images,
        test_labels,
    )

    # t-SNE visualization.
    part_d(train_loader, test_loader, test_labels)

    # Dropout experiments.
    part_e(train_images, train_labels, test_images, test_labels)

    # Batch normalization experiments.
    part_f(train_images, train_labels, test_images, test_labels)

    print("\n" + "=" * 60)
    print("ALL CNN EXPERIMENTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()

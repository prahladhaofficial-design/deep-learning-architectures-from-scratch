import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import save_plot


def plot_train_test_error(
    train_errors, test_errors, optimizer_name, subfolder="part_c"
):
    """
    Plot training and test error as a function of epochs.
    Saves the figure to outputs/<subfolder>/.
    """
    epochs = range(1, len(train_errors) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    # Blue circles for training error, red squares for test error
    ax.plot(
        epochs, train_errors, "b-o", label="Training Error", linewidth=2, markersize=5
    )
    ax.plot(epochs, test_errors, "r-s", label="Test Error", linewidth=2, markersize=5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title(f"Training and Test Error vs Epochs ({optimizer_name})", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(list(epochs))

    filename = f"error_curve_{optimizer_name.lower()}.png"
    save_plot(fig, filename, subfolder)


def plot_all_optimizers_comparison(results_dict, subfolder="part_c"):
    """
    Plot error curves for all optimizers side by side for easy comparison.
    Left panel: training errors. Right panel: test errors.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for opt_name, res in results_dict.items():
        epochs = range(1, len(res["train_errors"]) + 1)
        axes[0].plot(
            epochs, res["train_errors"], "-o", label=opt_name, linewidth=2, markersize=4
        )
        axes[1].plot(
            epochs, res["test_errors"], "-s", label=opt_name, linewidth=2, markersize=4
        )

    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Training Error", fontsize=12)
    axes[0].set_title("Training Error Comparison", fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Test Error", fontsize=12)
    axes[1].set_title("Test Error Comparison", fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    save_plot(fig, "error_comparison_all_optimizers.png", subfolder)


def plot_tsne(features, labels, class_names, epoch_label, subfolder="part_d"):
    """
    Plot t-SNE visualization of bottleneck features.
    Reduces high-dimensional features to 2D and colors points by class.
    """
    print(
        f"[t-SNE] Computing t-SNE for {features.shape[0]} samples with {features.shape[1]} features..."
    )

    # Reduce to 2D; adjust perplexity for small datasets
    tsne = TSNE(
        n_components=2, random_state=42, perplexity=min(30, features.shape[0] - 1)
    )
    features_2d = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Assign a distinct color from tab10 palette to each class
    unique_classes = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))

    for cls_idx, color in zip(unique_classes, colors):
        mask = labels == cls_idx
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[color],
            label=class_names[cls_idx],
            s=60,
            alpha=0.8,
            edgecolors="w",
            linewidth=0.5,
        )

    ax.set_title(
        f"t-SNE Visualization of Bottleneck Features ({epoch_label})", fontsize=14
    )
    ax.legend(fontsize=9, loc="best", scatterpoints=1)
    ax.grid(True, alpha=0.2)

    filename = f"tsne_{epoch_label.lower()}.png"
    save_plot(fig, filename, subfolder)


def plot_tsne_comparison(
    features_epoch1, features_epoch_last, labels, class_names, subfolder="part_d"
):
    """
    Plot side-by-side t-SNE for epoch 1 and last epoch.
    Shows how feature clustering improves after training.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for idx, (features, title) in enumerate(
        [
            (features_epoch1, "After Epoch 1"),
            (features_epoch_last, "After Last Epoch"),
        ]
    ):
        print(f"[t-SNE] Computing t-SNE for {title}...")
        tsne = TSNE(
            n_components=2, random_state=42, perplexity=min(30, features.shape[0] - 1)
        )
        features_2d = tsne.fit_transform(features)

        unique_classes = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))

        for cls_idx, color in zip(unique_classes, colors):
            mask = labels == cls_idx
            axes[idx].scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=[color],
                label=class_names[cls_idx],
                s=60,
                alpha=0.8,
                edgecolors="w",
                linewidth=0.5,
            )

        axes[idx].set_title(f"t-SNE: {title}", fontsize=13)
        axes[idx].legend(fontsize=8, loc="best", scatterpoints=1)
        axes[idx].grid(True, alpha=0.2)

    save_plot(fig, "tsne_comparison_epoch1_vs_last.png", subfolder)

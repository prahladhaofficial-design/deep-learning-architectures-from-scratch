import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_output_dir(subfolder, base_dir=None):
    """
    Create and return the path to an output subfolder inside outputs/.
    Ensures the directory exists before returning.
    """
    if base_dir is None:
        # Default to this project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "outputs", subfolder)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_plot(fig, filename, subfolder, base_dir=None, dpi=150):
    """
    Save a matplotlib figure to the outputs/<subfolder>/ directory.
    Closes the figure after saving to free memory.
    """
    out_dir = get_output_dir(subfolder, base_dir)
    filepath = os.path.join(out_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Utils] Saved plot: {filepath}")


def one_hot_encode(label, num_classes=10):
    """
    Convert an integer label to a one-hot encoded numpy array.
    """
    one_hot = np.zeros(num_classes, dtype=np.float32)
    one_hot[label] = 1.0
    return one_hot


def one_hot_batch(labels, num_classes=10):
    """
    Convert a list/array of integer labels to a 2D one-hot encoded array.
    """
    one_hots = np.zeros((len(labels), num_classes), dtype=np.float32)
    for i, lbl in enumerate(labels):
        one_hots[i, lbl] = 1.0
    return one_hots


def cross_entropy_loss(predictions, one_hot_targets):
    """
    Compute cross-entropy loss given softmax predictions and one-hot targets.
    Uses clipping for numerical stability to avoid log(0).

    Args:
        predictions: (N, C) array of softmax probabilities
        one_hot_targets: (N, C) array of one-hot encoded labels

    Returns:
        Scalar loss value (mean over batch)
    """
    eps = 1e-12
    # Clip predictions to prevent log(0) which causes -inf
    predictions = np.clip(predictions, eps, 1.0 - eps)
    # Cross-entropy: -sum(y * log(y_hat)) / N
    loss = -np.sum(one_hot_targets * np.log(predictions)) / predictions.shape[0]
    return loss


def compute_accuracy(predictions, labels):
    """
    Compute classification accuracy.

    Args:
        predictions: (N, C) array of softmax probabilities or logits
        labels: (N,) array of integer class labels

    Returns:
        Accuracy as a float (0.0 to 1.0)
    """
    # Predicted class is the index of the highest score
    predicted_classes = np.argmax(predictions, axis=1)
    correct = np.sum(predicted_classes == labels)
    return correct / len(labels)


def compute_error(predictions, labels):
    """
    Compute classification error rate (1 - accuracy).

    Args:
        predictions: (N, C) array of softmax probabilities or logits
        labels: (N,) array of integer class labels

    Returns:
        Error rate as a float (0.0 to 1.0)
    """
    return 1.0 - compute_accuracy(predictions, labels)


def log_training_info(
    epoch, train_loss, train_error, test_error, train_acc, test_acc, optimizer_name
):
    """
    Print formatted training progress for a single epoch.
    """
    print(
        f"[{optimizer_name}] Epoch {epoch + 1:3d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Error: {train_error:.4f} | "
        f"Test Error: {test_error:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Test Acc: {test_acc:.4f}"
    )

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def cross_entropy_one_hot(logits, one_hot_targets):
    """
    Compute cross-entropy loss with one-hot encoded targets.

    The module specifies that ground-truth labels are one-hot encoded
    vectors of dimension 10. This function computes:
        loss = -mean(sum(y_one_hot * log_softmax(logits), dim=1))

    Args:
        logits: (N, C) raw output from the network
        one_hot_targets: (N, C) one-hot encoded labels

    Returns:
        Scalar loss value
    """
    log_probs = F.log_softmax(logits, dim=1)
    loss = -torch.sum(one_hot_targets * log_probs) / logits.size(0)
    return loss


def create_optimizer(model, optimizer_name, lr=0.001, momentum=0.9, rho=0.9):
    """
    Create an optimizer based on the specified name.

    Args:
        model: PyTorch model
        optimizer_name: "sgd", "momentum", or "rmsprop"
        lr: learning rate
        momentum: momentum coefficient (for momentum optimizer)
        rho: decay rate (for RMSProp)

    Returns:
        PyTorch optimizer instance
    """
    if optimizer_name.lower() == "sgd":
        # Vanilla SGD without momentum
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    elif optimizer_name.lower() == "momentum":
        # SGD with momentum
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name.lower() == "rmsprop":
        # RMSProp optimizer
        return torch.optim.RMSprop(model.parameters(), lr=lr, alpha=rho)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def train_epoch(model, train_loader, optimizer, device, use_one_hot=True):
    """
    Train the model for one epoch over the training data.

    Uses custom cross-entropy with one-hot encoded labels for this experiment.

    Args:
        model: PyTorch model
        train_loader: DataLoader for training data (yields images, integer_labels)
        optimizer: PyTorch optimizer
        device: torch device
        use_one_hot: if True, convert integer labels to one-hot before loss

    Returns:
        avg_loss: average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, labels in train_loader:
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(images)

        # Convert integer labels to one-hot (labels are one-hot encoded with dimension 10)
        if use_one_hot:
            num_classes = logits.size(1)
            one_hot_labels = F.one_hot(labels, num_classes=num_classes).float()
            loss = cross_entropy_one_hot(logits, one_hot_labels)
        else:
            loss = F.cross_entropy(logits, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def train_model(
    model,
    train_loader,
    test_loader,
    optimizer_name,
    lr=0.001,
    momentum=0.9,
    rho=0.9,
    num_epochs=15,
    device="cpu",
    evaluator=None,
):
    """
    Full training loop for the CNN.

    Trains for num_epochs, evaluates on train and test sets after each epoch,
    and collects metrics for plotting.

    Args:
        model: PyTorch CNN model
        train_loader: DataLoader for training subset (100 imgs/class)
        test_loader: DataLoader for test subset (10 imgs/class)
        optimizer_name: "sgd", "momentum", or "rmsprop"
        lr: learning rate
        momentum: momentum coefficient
        rho: RMSProp decay rate
        num_epochs: number of training epochs
        device: "cpu" or "cuda"
        evaluator: optional evaluator object for collecting activations/features

    Returns:
        history: dict with train_losses, train_errors, test_errors,
                 train_accs, test_accs, and optional activation/feature lists
    """
    # Create the optimizer
    optimizer = create_optimizer(
        model, optimizer_name, lr=lr, momentum=momentum, rho=rho
    )

    model.to(device)

    history = {
        "train_losses": [],
        "train_errors": [],
        "test_errors": [],
        "train_accs": [],
        "test_accs": [],
    }

    # Optional: collect activations after epoch 1 and last epoch
    if evaluator is not None:
        history["conv1_epoch1"] = None
        history["conv2_epoch1"] = None
        history["conv1_last"] = None
        history["conv2_last"] = None
        history["bottleneck_epoch1"] = None
        history["bottleneck_last"] = None
        history["test_labels"] = None

    for epoch in range(num_epochs):
        # Train for one epoch (uses one-hot cross-entropy for this experiment)
        train_loss = train_epoch(
            model, train_loader, optimizer, device, use_one_hot=True
        )

        # Evaluate on training and test sets
        if evaluator is not None:
            eval_results = evaluator.evaluate(model, train_loader, test_loader, device)

            train_error = eval_results["train_error"]
            test_error = eval_results["test_error"]
            train_acc = eval_results["train_acc"]
            test_acc = eval_results["test_acc"]

            history["train_losses"].append(train_loss)
            history["train_errors"].append(train_error)
            history["test_errors"].append(test_error)
            history["train_accs"].append(train_acc)
            history["test_accs"].append(test_acc)

            # Collect activations and bottleneck features after epoch 1 and last epoch
            if epoch == 0:
                history["conv1_epoch1"] = eval_results["conv1_activations"]
                history["conv2_epoch1"] = eval_results["conv2_activations"]
                history["bottleneck_epoch1"] = eval_results["bottleneck_features"]
                history["test_labels"] = eval_results["test_labels"]

            if epoch == num_epochs - 1:
                history["conv1_last"] = eval_results["conv1_activations"]
                history["conv2_last"] = eval_results["conv2_activations"]
                history["bottleneck_last"] = eval_results["bottleneck_features"]
        else:
            # Fallback: compute basic metrics without evaluator
            train_error, train_acc = _evaluate_simple(model, train_loader, device)
            test_error, test_acc = _evaluate_simple(model, test_loader, device)
            history["train_losses"].append(train_loss)
            history["train_errors"].append(train_error)
            history["test_errors"].append(test_error)
            history["train_accs"].append(train_acc)
            history["test_accs"].append(test_acc)

        # Print progress
        print(
            f"[{optimizer_name.upper()}] Epoch {epoch + 1:3d}/{num_epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Err: {train_error:.4f} | "
            f"Test Err: {test_error:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Acc: {test_acc:.4f}"
        )

    return history


def _evaluate_simple(model, data_loader, device):
    """
    Simple evaluation returning cross-entropy loss (error) and accuracy.
    Used as fallback when no evaluator is provided.
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            # Compute cross-entropy loss with one-hot labels
            num_classes = logits.size(1)
            one_hot_labels = F.one_hot(labels, num_classes=num_classes).float()
            log_probs = F.log_softmax(logits, dim=1)
            loss = -torch.sum(one_hot_labels * log_probs) / logits.size(0)
            total_loss += loss.item()
            num_batches += 1

            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / num_batches
    acc = correct / total
    error = avg_loss  # error = cross-entropy loss
    return error, acc

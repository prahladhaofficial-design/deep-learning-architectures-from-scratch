import torch
import torch.nn.functional as F
import numpy as np


class Evaluator:
    """
    Evaluates the CNN model on train and test sets.
    Computes cross-entropy loss (error), classification accuracy,
    and collects activation maps and bottleneck features.
    """

    def __init__(self):
        pass

    def evaluate(self, model, train_loader, test_loader, device):
        """
        Evaluate model on both train and test sets.
        Error = cross-entropy loss (for this experiment).
        Also collects intermediate activations and bottleneck features from the test set.

        Returns:
            dict with train_error, test_error, train_acc, test_acc,
                  conv1_activations, conv2_activations, bottleneck_features, test_labels
        """
        # Evaluate on training set: compute cross-entropy loss and accuracy
        train_error, train_acc = self._compute_loss_and_acc(model, train_loader, device)

        # Evaluate on test set and collect activations
        test_error, test_acc, conv1_acts, conv2_acts, bottleneck_feats, test_labels = (
            self._compute_loss_and_acc(
                model, test_loader, device, collect_activations=True
            )
        )

        return {
            "train_error": train_error,
            "test_error": test_error,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "conv1_activations": conv1_acts,
            "conv2_activations": conv2_acts,
            "bottleneck_features": bottleneck_feats,
            "test_labels": test_labels,
        }

    def _compute_loss_and_acc(
        self, model, data_loader, device, collect_activations=False
    ):
        """
        Run model over a DataLoader and compute cross-entropy loss and accuracy.

        Returns:
            If collect_activations=False: (loss, accuracy)
            If collect_activations=True: (loss, accuracy, conv1_acts, conv2_acts, bottleneck_feats, labels)
        """
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        num_batches = 0

        all_conv1 = []
        all_conv2 = []
        all_bottleneck = []
        all_labels = []

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass (this populates model's activation buffers)
                logits = model(images)

                # Compute cross-entropy loss using one-hot encoded labels (experiment requirement)
                num_classes = logits.size(1)
                one_hot_labels = F.one_hot(labels, num_classes=num_classes).float()
                log_probs = F.log_softmax(logits, dim=1)
                loss = -torch.sum(one_hot_labels * log_probs) / logits.size(0)
                total_loss += loss.item()
                num_batches += 1

                # Compute accuracy using argmax of logits
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Collect activations if requested
                if collect_activations:
                    conv1_act = model.get_conv1_activations()
                    conv2_act = model.get_conv2_activations()
                    bottleneck = model.get_bottleneck_features()

                    all_conv1.append(conv1_act.cpu())
                    all_conv2.append(conv2_act.cpu())
                    all_bottleneck.append(bottleneck.cpu())
                    all_labels.append(labels.cpu())

        avg_loss = total_loss / num_batches
        accuracy = correct / total

        if not collect_activations:
            return avg_loss, accuracy

        # Concatenate all batched activations into single tensors
        conv1_acts = torch.cat(all_conv1, dim=0)  # (N, C1, H1, W1)
        conv2_acts = torch.cat(all_conv2, dim=0)  # (N, C2, H2, W2)
        bottleneck_feats = torch.cat(all_bottleneck, dim=0)  # (N, 256)
        labels = torch.cat(all_labels, dim=0).numpy()  # (N,)

        return avg_loss, accuracy, conv1_acts, conv2_acts, bottleneck_feats, labels

    def get_predictions(self, model, data_loader, device):
        """
        Get model predictions (softmax probabilities) and integer labels.

        Returns:
            all_probs: (N, 10) numpy array of softmax probabilities
            all_labels: (N,) numpy array of integer labels
        """
        model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                probs = torch.softmax(logits, dim=1)

                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        return all_probs, all_labels

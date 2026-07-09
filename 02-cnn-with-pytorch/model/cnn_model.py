import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    """
    PyTorch implementation of the CNN architecture from the NumPy study.

    Architecture:
        Input: 32x32x3
        Conv1: 4 kernels, 5x5, stride=1, pad=2, ReLU  -> 32x32x4
        MaxPool1: 2x2, stride=2                        -> 16x16x4
        Conv2: 4 kernels, 5x5, stride=1, pad=2, ReLU  -> 16x16x4
        MaxPool2: 2x2, stride=2                        -> 8x8x4
        Flatten                                        -> 256
        MLP: 256 -> 49 (ReLU) -> 10                    -> 10
        Softmax (applied separately during inference)

    Supports optional dropout at the MLP hidden layer and
    batch normalization before the MLP hidden layer ReLU.
    """

    def __init__(self, dropout_p=0.0, use_batchnorm=False):
        super(CNNModel, self).__init__()

        self.use_batchnorm = use_batchnorm

        # Conv1: 3 input channels, 4 output channels, 5x5 kernel, pad=2 for same output
        self.conv1 = nn.Conv2d(3, 4, kernel_size=5, stride=1, padding=2)

        # Conv2: 4 input channels, 4 output channels, 5x5 kernel, pad=2 for same output
        self.conv2 = nn.Conv2d(4, 4, kernel_size=5, stride=1, padding=2)

        # Max pooling: 2x2 window, stride=2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # MLP hidden layer: 8*8*4 = 256 -> 49
        self.fc1 = nn.Linear(8 * 8 * 4, 49)

        # Optional batch norm at the output of fc1 (before ReLU)
        if self.use_batchnorm:
            self.bn1 = nn.BatchNorm1d(49)
        else:
            self.bn1 = None

        # Optional dropout at the MLP hidden layer
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else None

        # MLP output layer: 49 -> 10
        self.fc2 = nn.Linear(49, 10)

        # Buffers for intermediate activations used by visualization utilities.
        self.conv1_activation = None
        self.conv2_activation = None
        self.bottleneck_features = None

    def forward(self, x):
        # Conv1 -> ReLU
        x = self.conv1(x)
        x = F.relu(x)
        self.conv1_activation = x.clone()  # Store for visualization

        # MaxPool1
        x = self.pool(x)

        # Conv2 -> ReLU
        x = self.conv2(x)
        x = F.relu(x)
        self.conv2_activation = x.clone()  # Store for visualization

        # MaxPool2
        x = self.pool(x)

        # Flatten: (N, 4, 8, 8) -> (N, 256)
        x = x.view(x.size(0), -1)

        # Store bottleneck features (before MLP) for t-SNE
        self.bottleneck_features = x.clone()

        # MLP hidden layer
        x = self.fc1(x)

        # Optional batch normalization (before ReLU)
        if self.bn1 is not None:
            x = self.bn1(x)

        # ReLU activation
        x = F.relu(x)

        # Optional dropout
        if self.dropout is not None:
            x = self.dropout(x)

        # MLP output layer (logits, no softmax)
        x = self.fc2(x)

        return x

    def get_conv1_activations(self):
        """Return stored Conv1 activation maps."""
        return self.conv1_activation

    def get_conv2_activations(self):
        """Return stored Conv2 activation maps."""
        return self.conv2_activation

    def get_bottleneck_features(self):
        """Return stored bottleneck features (flattened output before MLP)."""
        return self.bottleneck_features

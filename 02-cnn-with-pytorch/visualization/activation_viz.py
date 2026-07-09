import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import save_plot


def visualize_activation_maps(
    activation_volume,
    layer_name,
    sample_indices=None,
    subfolder="part_c",
    max_channels=4,
):
    """
    Visualize representative slices from an activation volume.
    Displays each channel as a heatmap with a colorbar.
    """
    # Default to first 4 samples if none specified
    if sample_indices is None:
        sample_indices = list(range(min(4, activation_volume.shape[0])))

    num_samples = len(sample_indices)
    num_channels = min(activation_volume.shape[1], max_channels)

    # Create a grid: rows = samples, columns = channels
    fig, axes = plt.subplots(
        num_samples, num_channels, figsize=(num_channels * 3, num_samples * 3)
    )

    # Handle edge cases where grid is 1x1, 1xN, or Nx1
    if num_samples == 1 and num_channels == 1:
        axes = np.array([[axes]])
    elif num_samples == 1:
        axes = axes.reshape(1, -1)
    elif num_channels == 1:
        axes = axes.reshape(-1, 1)

    for i, sample_idx in enumerate(sample_indices):
        for c in range(num_channels):
            # Detach from computation graph, move to CPU, convert to numpy
            act_map = activation_volume[sample_idx, c].detach().cpu().numpy()

            im = axes[i, c].imshow(act_map, cmap="viridis")
            axes[i, c].set_title(f"Sample {sample_idx}, Ch {c}", fontsize=9)
            axes[i, c].axis("off")
            fig.colorbar(im, ax=axes[i, c], fraction=0.046, pad=0.04)

        # Hide any extra columns beyond the actual number of channels
        for c in range(num_channels, axes.shape[1]):
            axes[i, c].axis("off")

    fig.suptitle(f"Activation Maps: {layer_name}", fontsize=14, y=1.01)
    plt.tight_layout()

    filename = f"activation_maps_{layer_name.lower()}.png"
    save_plot(fig, filename, subfolder)


def visualize_all_conv_activations(
    conv1_output, conv2_output, sample_indices=None, subfolder="part_c"
):
    """
    Convenience wrapper to visualize activation maps from both conv layers.
    Calls visualize_activation_maps separately for Conv1 and Conv2.
    """
    visualize_activation_maps(
        conv1_output, "Conv1", sample_indices=sample_indices, subfolder=subfolder
    )
    visualize_activation_maps(
        conv2_output, "Conv2", sample_indices=sample_indices, subfolder=subfolder
    )

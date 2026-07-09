import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class CIFAR10Dataset:
    """
    Loads CIFAR-10 images from a folder-based structure.

    Expected structure:
        root/
            train/
                airplane/0000.jpg ...
                automobile/0000.jpg ...
                ...
            test/
                airplane/0000.jpg ...
                ...
    """

    def __init__(self, root_dir, split="train", transform=None):
        # Store configuration
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        # Lists to hold image paths, integer labels, and class names
        self.image_paths = []
        self.labels = []
        self.label_names = []

        # Scan the split directory (train/ or test/) for class subfolders
        split_dir = os.path.join(self.root_dir, self.split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # Iterate over each CIFAR-10 class folder and collect image paths
        for class_idx, class_name in enumerate(CIFAR10_CLASSES):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"[Dataset] Warning: class folder not found: {class_dir}")
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(os.path.join(class_dir, fname))
                    self.labels.append(class_idx)
                    self.label_names.append(class_name)

        print(
            f"[Dataset] Loaded {len(self.image_paths)} images from split='{self.split}'"
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and label at the given index
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Open image, convert to RGB, normalize to [0, 1]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image, dtype=np.float32) / 255.0

        # Apply optional transform (e.g., ToTensor)
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def get_subset(self, samples_per_class):
        """
        Returns a new CIFAR10Dataset with exactly `samples_per_class` images
        per class, selected from the beginning of each class's sorted list.
        Used to create the 100-img train subset and 10-img test subset.
        """
        subset_paths = []
        subset_labels = []
        subset_names = []

        # Pick the first `samples_per_class` images from each class
        for class_idx, class_name in enumerate(CIFAR10_CLASSES):
            class_indices = [i for i, lbl in enumerate(self.labels) if lbl == class_idx]
            selected = class_indices[:samples_per_class]
            for i in selected:
                subset_paths.append(self.image_paths[i])
                subset_labels.append(self.labels[i])
                subset_names.append(self.label_names[i])

        # Create a lightweight shallow copy of the dataset with only selected images
        new_dataset = CIFAR10Dataset.__new__(CIFAR10Dataset)
        new_dataset.root_dir = self.root_dir
        new_dataset.split = self.split
        new_dataset.transform = self.transform
        new_dataset.image_paths = subset_paths
        new_dataset.labels = subset_labels
        new_dataset.label_names = subset_names

        print(
            f"[Dataset] Subset: {len(subset_paths)} images "
            f"({samples_per_class} per class) from split='{self.split}'"
        )
        return new_dataset

    def get_class_indices(self, class_idx):
        """Return list of dataset indices belonging to a specific class."""
        return [i for i, lbl in enumerate(self.labels) if lbl == class_idx]

    def get_one_hot_label(self, idx, num_classes=10):
        """Return one-hot encoded label for a given dataset index."""
        label = self.labels[idx]
        one_hot = np.zeros(num_classes, dtype=np.float32)
        one_hot[label] = 1.0
        return one_hot

    def get_all_one_hot_labels(self, indices=None, num_classes=10):
        """Return one-hot encoded labels for all or specified indices."""
        if indices is None:
            indices = range(len(self.labels))
        labels = []
        for idx in indices:
            one_hot = np.zeros(num_classes, dtype=np.float32)
            one_hot[self.labels[idx]] = 1.0
            labels.append(one_hot)
        return np.array(labels, dtype=np.float32)

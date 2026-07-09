# Deep Learning Architectures From Scratch

A multi-part deep-learning implementation project covering CNN primitives, PyTorch image classification, RNN/LSTM learners, transformer encoders, and long-context modeling.

## Overview

Modern deep learning stacks can hide the details of tensor transformations, optimization, and representation learning. This project keeps those details visible by implementing major building blocks in stages.
The repository moves from NumPy-level image operations to PyTorch training pipelines, then into sequence models and attention-based architectures.

## Project Design

- Implement CNN operations from scratch before using framework-level training tools.
- Use a PyTorch CIFAR-10 pipeline to compare optimizers, batch sizes, dropout, and batch normalization.
- Build recurrent sequence learners to examine gradient flow and learned embeddings.
- Implement transformer encoder components and compare long-context modeling approaches.

## Key Capabilities

- NumPy convolution, pooling, flattening, activations, MLP head, and full CNN forward pass.
- PyTorch CNN trainer with evaluation utilities, activation maps, and t-SNE visualizations.
- RNN and LSTM notebooks with tokenization, vocabulary construction, embeddings, and BPTT.
- Transformer encoder notebook with scaled dot-product attention, multi-head attention, positional encoding, and classifier head.
- Long-context modeling experiments comparing recurrent, attention-based, and state-space-inspired learners.

## Repository Structure

```text
01-cnn-from-scratch/       NumPy CNN operations and output visualizations.
02-cnn-with-pytorch/       CIFAR-10 CNN model, trainer, evaluator, and experiments.
03-optimization-study/     Optimization notes.
04-sequence-learners/      RNN/LSTM notebook and training curves.
05-transformer-encoder/    Transformer encoder implementation notebook.
06-long-context-modeling/  Long-context modeling notebook.
docs/                      Technical briefs and analysis notes.
```

## Running Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python 01-cnn-from-scratch/main.py
python 02-cnn-with-pytorch/main.py
jupyter notebook 04-sequence-learners/sequence_learners.ipynb
```

## Results and Artifacts

- `01-cnn-from-scratch/outputs/` shows intermediate CNN feature maps and tensor operations.
- `02-cnn-with-pytorch/outputs/` includes error curves, activation maps, t-SNE plots, dropout studies, and batch-normalization comparisons.
- `04-sequence-learners/part2_rnn_training_curves.png` captures recurrent training behavior.
- Transformer and long-context results are preserved in notebooks and markdown notes.

## Reproducibility Notes

- Run notebooks in sequence when comparing perceptron, MLP, CNN, and backpropagation experiments.
- Keep random seeds, learning rates, and epoch counts fixed when comparing training curves.
- Regenerate saved plots after changing architecture or optimizer settings.
- Use the lightweight examples for implementation review before scaling the experiments to larger datasets.

## Detailed Workflow

- Run the NumPy CNN component first to inspect convolution, pooling, flattening, and forward-pass tensor shapes.
- Run the PyTorch CNN component next to train a complete image-classification pipeline with saved plots.
- Open the sequence learner notebook to study recurrence, gradient flow, tokenization, and embeddings.
- Open the transformer encoder notebook to inspect attention, positional information, and encoder-block design.
- Use the long-context notebook to compare how architecture choices behave when the input sequence length increases.

## Implementation Details

- The NumPy CNN component keeps all core operations visible and avoids hiding tensor logic behind framework layers.
- The PyTorch component uses modular data, model, training, evaluation, experiment, and visualization folders.
- The CNN pipeline stores outputs for batch-size, optimizer, dropout, batch-normalization, activation-map, and t-SNE analysis.
- The sequence notebooks keep model construction and experiment logic together so the modeling flow is easy to follow.
- The docs folder preserves written explanations for optimization, sequence learning, transformer encoding, and long-context behavior.

## Data and Inputs

- The NumPy CNN demo uses a local CIFAR-style image to verify the forward pass.
- The PyTorch CNN pipeline expects CIFAR-10 formatted data under its local data directory.
- The sequence and long-context notebooks load or prepare text-style data inside the notebook workflow.
- Generated outputs are kept in the component folders so each subproject can be reviewed independently.

## How To Interpret the Results

- Use NumPy outputs to confirm the expected image-to-feature transformation pipeline.
- Use PyTorch error curves to compare optimizer and batch-size behavior.
- Use activation maps to inspect what convolutional filters respond to.
- Use t-SNE plots to compare representation geometry across training.
- Use recurrent and transformer notebooks to compare memory, attention, and long-range dependency handling.

## Limitations and Assumptions

- The repository contains multiple independent experiments, so not every component must be run together.
- The PyTorch pipeline may require a properly placed CIFAR-10 dataset before execution.
- Long-context experiments can be slower depending on hardware and dataset size.
- The project prioritizes transparent architecture study over production training infrastructure.


## Review Guide

Use this checklist when reviewing the repository on GitHub:

- Start with the overview to understand the project objective before opening notebooks or source files.
- Read the repository structure section to identify the main entry point.
- Inspect generated outputs before rerunning experiments; they provide a quick view of the current project state.
- Review scripts and notebooks together when both are present, since notebooks often explain decisions while scripts preserve reusable execution paths.
- Check the limitations section before interpreting results too strongly.
- Use the possible extensions section to understand how the project can grow without changing its core purpose.

## Troubleshooting

Common issues and fixes:

- If imports fail, confirm that the virtual environment is activated and dependencies were installed from `requirements.txt`.
- If a notebook cell fails because a variable is missing, restart the kernel and run the notebook from the first cell.
- If a script cannot find a data file, run it from the repository root unless the README gives a different working directory.
- If plots are not regenerated, check that the output folder exists and that the script has write permission.
- If a package version causes errors, create a fresh environment and reinstall dependencies before debugging the source code.
- If GPU-related packages are unavailable, run CPU-compatible sections first and treat heavier deep-learning experiments separately.

## Development Notes

The repository is organized to be understandable before it is optimized. Most code favors clear structure, visible intermediate steps, and saved artifacts over highly compressed implementations. When extending the project, prefer small changes that keep the original experiment reproducible.

Recommended development practices:

- Keep new data files under `data/`, `datasets/`, or the existing local data folder for that project.
- Keep generated figures under `figures/`, `plots/`, `outputs/`, or `results/` depending on the existing convention.
- Add new experiments as separate scripts or clearly labeled notebook sections.
- Avoid overwriting existing outputs unless the experiment is intentionally being refreshed.
- Document any new dependency in `requirements.txt`.
- Prefer relative paths so the repository works after cloning on another machine.

## Project Status

This repository is prepared as a standalone GitHub project. The current version includes the core implementation, documentation, and available artifacts needed to understand the work. Future improvements can focus on stronger automation, more tests, richer visual summaries, and cleaner packaging where appropriate.

## Possible Extensions

- Add unit tests for convolution and pooling output shapes.
- Add a single experiment runner for all PyTorch image experiments.
- Compare additional transformer variants and attention-head configurations.
- Add model checkpointing and metric logging for long-context experiments.

## Tech Stack

Python, NumPy, PyTorch, Matplotlib, scikit-learn, Hugging Face datasets, Jupyter Notebook.

## Author

Prahladha, IIT Hyderabad

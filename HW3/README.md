# Homework 3 - CS 8395: Special Topics in Deep Learning

This assignment focuses on deep learning architectures for non-Euclidean domains and sequential data, specifically set-structured inputs and vision transformers.

---

## Q1: Deep Learning on Sets (`DL_HW3_Q1.ipynb`)

Trains a neural network that processes sets of integers and predicts the **maximum element** from the set. This tests the model’s ability to handle permutation-invariant inputs.

- **Q1.1**: Implements a random data generator:
  - Samples set size `Nₖ ∈ [10, 100]`
  - Samples `Nₖ` unique integers from `[0, 999]`

- **Q1.2**: Defines a Set Neural Network:
  - Architecture uses permutation-invariant aggregation (e.g., max or mean pooling)
  - Outputs a scalar representing the set’s maximum value

- **Q1.3**: Justifies the design using geometric deep learning principles:
  - Emphasizes permutation invariance and/or equivariance

- **Q1.4**: Trains the model on generated data:
  - Reports MSE loss over iterations
  - Demonstrates convergence behavior

---

## Q2: Transformer for Vision (`DL_HW3_Q2.ipynb`)

Compares the performance of a CNN and a **custom Vision Transformer (ViT)** on MNIST digit classification.

- **Q2.1**: Theoretical explanation:
  - Discusses whether self-attention is permutation-invariant or permutation-equivariant

- **Q2.2**: Trains a CNN on MNIST:
  - Uses standard convolutional layers
  - Reports test accuracy

- **Q2.3**: Implements a Vision Transformer from scratch:
  - 6 transformer blocks with 8 attention heads
  - Patch size: 7×7
  - Custom implementation of self-attention (no PyTorch built-ins)
  - Reports test accuracy

- **Q2.4**: Compares CNN vs ViT:
  - Evaluates accuracy, model capacity, and generalization trade-offs

---

> All models are implemented in PyTorch. The transformer architecture is fully custom-built.

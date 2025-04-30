# Homework 3 - CS 8395: Special Topics in Deep Learning

This assignment focuses on deep learning architectures for non-Euclidean domains and sequential data, specifically set-structured inputs and vision transformers.

---

## Q1: Deep Learning on Sets (`DL_HW3_Q1.ipynb`)

Trains a neural network that processes sets of integers and predicts the **maximum element** from the set. This tests the model’s ability to handle permutation-invariant inputs.

- **Q1.1**: Implements a random data generator:
  - Samples set size \( N_k \in [10, 100] \)
  - Samples \( N_k \) unique integers from \( [0, 999] \)

- **Q1.2**: Defines a **Set Neural Network**:
  - Architecture uses permutation **invariant/aggregation** operations (e.g., max or mean pooling after equivariant transformations).
  - Outputs a scalar representing the set’s maximum.

- **Q1.3**: Justifies design with geometric deep learning principles:
  - Emphasizes permutation **invariance/equivariance** in architecture.

- **Q1.4**: Trains on generated data.
  - Reports MSE loss over iterations.
  - Demonstrates convergence of the model.

---

## Q2: Transformer for Vision (`DL_HW3_Q2.ipynb`)

Compares performance of CNN and a **custom-built Vision Transformer (ViT)** on MNIST digit classification.

- **Q2.1**: Theoretical Q&A:
  - Discusses whether self-attention is permutation-invariant or equivariant.

- **Q2.2**: Trains a **CNN** classifier on MNIST.
  - Reports test accuracy.
  - Uses standard convolutional layers.

- **Q2.3**: Implements a **Vision Transformer** from scratch:
  - Uses 6 transformer blocks with 8 attention heads.
  - Patch size of 7×7.
  - Writes **custom self-attention** block (not PyTorch’s built-in module).
  - Reports test accuracy.

- **Q2.4**: Compares CNN and ViT:
  - Analyzes accuracy, model capacity, and performance trade-offs.

---

> All code is implemented in PyTorch. The transformer block is written manually as required.

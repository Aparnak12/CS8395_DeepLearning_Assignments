# Homework 4 - CS 8395: Special Topics in Deep Learning

This assignment explores various aspects of generative modeling and adversarial robustness in deep learning, including GANs, diffusion models, evasion attacks, and adversarial training.

---

## Q1: Generative Adversarial Networks (GANs) (`DL_HW4_Q1.ipynb`)

Implements a 2D GAN to map samples from a standard Gaussian distribution to a target distribution.

- **Generator**: MLP with architecture [2 → 100 → 100 → 2]
- **Discriminator**: MLP with architecture [2 → 100 → 100 → 1]
- **Training Loop**:
  - Discriminator is updated for 10 steps per epoch.
  - Generator is updated for 1–5 steps (user-defined).
  - Uses Adam optimizer with `lr=1e-4`.

### Reported Outputs:
- Discriminator output over a 2D grid.
- Scatter plots of generator input and output on 2,000 samples.

**Data**: `hw4_p1.pkl`

---

## Q2: Diffusion Models (`DL_HW4_Q2.ipynb`)

Implements a 2D diffusion model with sinusoidal time embeddings to reconstruct data from noise.

- **Time Embedding**: 128D sinusoidal embedding module.
- **Noise Predictor** \( \epsilon_\theta(x_t, t) \): MLP with architecture [130 → 512 → 512 → 2]
- **Training Loop**:
  - Samples time `t ∈ {1, ..., T}` with `T = 500`
  - Uses forward diffusion equation to compute `x_t`
  - Trains on minimizing MSE between predicted and true noise

### Inference:
- Implements reverse process with denoising loop from `x_T → x_0`.
- Provides visualizations of forward and reverse diffusion trajectories.

---

## Q3: Evasion Attacks on Neural Networks (`DL_HW4_Q3.ipynb`)

Tests four adversarial attack strategies on a pre-trained ResNet18 using an input image.

- **Q3.1**: Formulates optimization objectives for:
  - Untargeted attack: maximize loss
  - Targeted attack: minimize loss for specific class
- **Q3.2**: Implements FGSM (Fast Gradient Sign Method)
- **Q3.3**: Implements Least Likely FGSM
- **Q3.4**: Implements Projected Gradient Descent (PGD)
- **Q3.5**: Implements Carlini-Wagner (CW) Attack

### Visualization:
- Plots original vs attacked image
- Shows predicted labels before and after attack

---

## Q4: Adversarial Training (`DL_HW4_Q4.ipynb`)

Trains MLP models on MNIST to evaluate robustness against adversarial perturbations.

- **Q4.1**: Trains standard MLP
  - Reports accuracy on clean and FGSM-attacked test sets
- **Q4.2**: Trains adversarially robust MLP using FGSM during training
  - Reports accuracy on clean and attacked test sets

**Perturbation Constraint**: \( ||\delta||_\infty ≤ 0.1 \)

---

> All models are implemented in PyTorch. Visualizations and training logs are included in the respective notebooks.

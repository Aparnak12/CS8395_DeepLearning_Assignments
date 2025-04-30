# Homework 4 - CS 8395: Special Topics in Deep Learning

This assignment explores various aspects of generative modeling and adversarial robustness in deep learning, including GANs, diffusion models, evasion attacks, and adversarial training.

---

## Q1: Generative Adversarial Networks (GANs) (`DL_HW4_Q1.ipynb`)

Implements a 2D GAN to map samples from a standard Gaussian distribution to a target distribution.

- **Generator**: MLP with architecture `[2 → 100 → 100 → 2]`
- **Discriminator**: MLP with architecture `[2 → 100 → 100 → 1]`
- **Training Loop**:
  - Discriminator is updated for 10 steps per epoch
  - Generator is updated for 1–5 steps (configurable)
  - Uses Adam optimizer with `lr = 1e-4`

### Reported Outputs:
- Discriminator output over a 2D grid
- Scatter plots of generator input vs output on 2,000 samples

**Data**: `hw4_p1.pkl`

---

## Q2: Diffusion Models (`DL_HW4_Q2.ipynb`)

Implements a 2D diffusion model using sinusoidal time embeddings and a noise predictor to reconstruct data from noise.

- **Time Embedding**: 128-dimensional sinusoidal embedding
- **Noise Predictor (epsilon_theta(x_t, t))**: MLP with architecture `[130 → 512 → 512 → 2]`
- **Training Loop**:
  - Samples time `t ∈ {1, ..., T}` with `T = 500`
  - Computes noisy version of input using the forward diffusion equation
  - Trains by minimizing MSE between predicted and true noise

### Inference:
- Starts from noise `x_T` and iteratively denoises back to `x_0`
- Includes visualizations of both the forward and reverse diffusion processes

---

## Q3: Evasion Attacks on Neural Networks (`DL_HW4_Q3.ipynb`)

Evaluates the robustness of a pre-trained ResNet18 on adversarial inputs using four attack methods.

- **Q3.1**: Formulates the optimization objectives for:
  - **Untargeted attack**: maximize classification loss
  - **Targeted attack**: minimize loss for a chosen target class
- **Q3.2**: Implements **FGSM (Fast Gradient Sign Method)**
- **Q3.3**: Implements **Least Likely FGSM**
- **Q3.4**: Implements **Projected Gradient Descent (PGD)**
- **Q3.5**: Implements **Carlini-Wagner (CW) Attack**

### Visualization:
- Compares original vs. attacked image
- Displays predicted labels before and after the attack

---

## Q4: Adversarial Training (`DL_HW4_Q4.ipynb`)

Trains MLP classifiers on MNIST to assess and improve robustness against adversarial perturbations.

- **Q4.1**: Trains a standard MLP on MNIST
  - Reports accuracy on both clean and FGSM-attacked test sets
- **Q4.2**: Trains an adversarially robust MLP using FGSM during training
  - Reports test performance on both clean and adversarial examples

**Perturbation Constraint**: ‖δ‖∞ ≤ 0.1

---

> All models are implemented in PyTorch. Visualizations and logs are included in each notebook.

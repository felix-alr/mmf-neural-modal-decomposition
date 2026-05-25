# MMF Neural Modal Decomposition
## Overview
![Python](https://img.shields.io/badge/Python-3.12.3-yellow?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-orange?style=flat-square)
![Python](https://img.shields.io/badge/MATLAB-R2024a-blue?style=flat-square)
![TU Dresden](https://img.shields.io/badge/TU_Dresden-Neural_Networks_for_Image_Processing-green?style=flat-square)

Multimode fibers scramble light into complex speckle patterns at their output, making it difficult to recover the underlying modal composition. This project applies deep learning to decompose speckle intensity images into their constituent fiber modes &mdash; predicting both amplitude and phase coefficients for up to 5 modes.

Two network architectures were compared:

1. MLP
2. VGG

## Results

| Model          | Modes | Γ̄     | σ      | Δρ   | Δφ    |
|----------------|-------|--------|--------|------|-------|
| MLP            | 3     | 96.7%  | 3.24%  | 9.1% | 24.1% |
| MLP            | 5     | 96.6%  | 3.30%  | 9.9% | 24.7% |
| VGG            | 3     | 99.6%  | 1.21%  | 2.2% | 6.5%  |
| VGG            | 5     | 96.4%  | 3.05%  | 8.4% | 23.5% |
| VGG + TL       | 5     | 97.5%  | 2.22%  | 7.7% | 16.4% |

## Key findings

- The VGG architecture outperformed the MLP significantly for 3 modes (99.6% vs 96.7% correlation), attributed to its ability to extract spatial features from mode profile intensity distributions.
- Phase reconstruction (Δφ) was consistently harder than amplitude (Δρ) across all experiments, consistent with the physical challenge of recovering phase from intensity-only images.
- Transfer learning from the 3-mode VGG enabled efficient adaptation to 5 modes, with accuracy scaling strongly with training data size (96.5% at 10K vs 97.5% at 50K datapoints).

## Protocol

> [Full protocol (PDF)](./protocol-mmf-modal-decomposition.pdf)
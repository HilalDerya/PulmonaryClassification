# Lung Disease Classification with CNN

A deep learning project that classifies lung diseases from audio recordings using Convolutional Neural Networks (CNN) trained on Mel-spectrograms.

## Overview

This project leverages audio signal processing and deep learning to detect and classify lung diseases from respiratory sound recordings. By converting raw audio waveforms into Mel-spectrograms, the model learns to identify patterns associated with different lung conditions — enabling automated, non-invasive disease detection.

## Dataset

- **Format:** `.wav` audio files of lung sounds
- **Labels:** Extracted directly from filenames (e.g., `patient_COPD + Pneumonia, ...`)
- **Task:** Multi-label classification — a single recording can be associated with multiple conditions

**Label extraction logic:**
Labels are parsed from the filename between the first underscore and the first comma, supporting compound labels separated by `+`.

## Technical Approach

### Audio Processing
Raw `.wav` files are transformed into **Mel-spectrograms** using the following configuration:
- Sample rate: `16,000 Hz`
- Number of Mel bands: `128`
- Hop length: `512`

Mel-spectrograms convert audio into 2D time-frequency representations that capture the characteristics of lung sounds in a format suitable for CNN processing.

### Model Architecture

A custom CNN (`CNNModel`) with the following structure:

| Layer | Details |
|---|---|
| Conv2D (1→32) | Kernel 3×3, stride 1, padding 1 + ReLU |
| MaxPool2D | Kernel 2×2, stride 2 |
| Conv2D (32→64) | Kernel 3×3, stride 1, padding 1 + ReLU |
| MaxPool2D | Kernel 2×2, stride 2 |
| Fully Connected (64×32×10→128) | ReLU + Dropout (0.5) |
| Fully Connected (128→num_classes) | Output layer |

### Training Configuration
- **Loss function:** Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`) — supports multi-label classification
- **Optimizer:** Adam (learning rate: `0.001`)
- **Epochs:** 10
- **Batch size:** 4
- **Train/val split:** 80% / 20%
- **Label encoding:** `MultiLabelBinarizer` for multi-label support

---

## Results

Training and validation metrics are tracked per epoch:

| Metric | Description |
|---|---|
| Training Loss | BCE loss on training set |
| Validation Loss | BCE loss on validation set |
| Accuracy | Exact match accuracy |
| Precision | Micro-averaged precision |
| Recall | Micro-averaged recall |
| F1 Score | Micro-averaged F1 |

Visualizations generated:
- **ROC curves** — micro-average and per-class AUC scores
- **Training/Validation loss curve** — tracks convergence across epochs

<img width="578" height="455" alt="indir (1)" src="https://github.com/user-attachments/assets/af9c12f3-bedb-4cf5-8807-792687bba4df" />

<img width="576" height="455" alt="indir (2)" src="https://github.com/user-attachments/assets/f6373440-7ac0-4e35-a0ba-5f23dad5037f" />


## Tech Stack

- **Language:** Python 3
- **Platform:** Google Colab
- **Libraries:** `PyTorch`, `torchaudio`, `scikit-learn`, `matplotlib`, `numpy`, `datasets`, `accelerate`


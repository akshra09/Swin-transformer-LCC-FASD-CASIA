# 🛡️ Leveraging Swin Transformer for Robust Deepfake Detection
### LCC-FASD + CASIA Combined Dataset · PyTorch · Transfer Learning

[![IEEE Paper](https://img.shields.io/badge/IEEE-Published%20Paper-blue?logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/document/10932042)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📖 Overview

This project implements a **Face Anti-Spoofing (FAS)** system using the **Swin Transformer** architecture — a hierarchical Vision Transformer that uses shifted window attention for efficient and powerful feature extraction. The model is fine-tuned on a combined dataset of **LCC-FASD** and **CASIA**, two widely used benchmarks in face presentation attack detection.

Face anti-spoofing is a critical defense mechanism in facial recognition systems, protecting against **presentation attacks** such as:
- 🖨️ Printed photo attacks
- 📱 Replay / screen attacks
- 🎭 3D mask attacks

By leveraging the global attention mechanism of the Swin Transformer over traditional CNNs, this model learns rich spatial and contextual facial features that are more discriminative for detecting spoofed faces.

---

## 🏗️ Architecture

```
Input Image (224×224)
        │
        ▼
┌──────────────────────────────┐
│   Swin-Tiny Transformer      │  ← Pretrained on ImageNet (frozen)
│   patch4 · window7 · 224     │
│   Hierarchical Shifted       │
│   Window Self-Attention      │
└──────────────────────────────┘
        │
        ▼  (in_features)
┌──────────────────────────────┐
│   Custom Classification Head │  ← Fine-tuned
│                              │
│   Linear(in_features → 512)  │
│   ReLU()                     │
│   Dropout(0.3)               │
│   Linear(512 → num_classes)  │
└──────────────────────────────┘
        │
        ▼
  [Real | Spoof]
```

The backbone is **frozen** during training — only the classification head is updated. This is an efficient **transfer learning** strategy that leverages the powerful representations learned on ImageNet while adapting to the anti-spoofing domain.

---

## 📂 Dataset

This project uses a **combined dataset** of LCC-FASD and CASIA, available on Kaggle:

| Dataset | Description |
|---------|-------------|
| **LCC-FASD** | Large-scale face anti-spoofing dataset with real and spoof (print, replay) facial images |
| **CASIA** | Chinese Academy of Sciences face anti-spoofing benchmark dataset |

The dataset is structured into three official splits:

```
lcc-fasd-casia/
└── LCC_FASD/
    ├── LCC_FASD_training/     # Training split
    │   ├── real/
    │   └── spoof/
    ├── LCC_FASD_development/  # Validation split
    │   ├── real/
    │   └── spoof/
    └── LCC_FASD_evaluation/   # Test split
        ├── real/
        └── spoof/
```

### Data Preprocessing & Augmentation

| Split | Transforms Applied |
|-------|--------------------|
| **Train** | `RandomResizedCrop(224)` → `RandomHorizontalFlip` → `ToTensor` |
| **Val / Test** | `Resize(224)` → `CenterCrop(224)` → `ToTensor` |

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| **Model** | `swin_tiny_patch4_window7_224` |
| **Pretrained** | ✅ ImageNet |
| **Backbone** | Frozen |
| **Batch Size** | 32 |
| **Epochs** | 200 |
| **Optimizer** | AdamW |
| **Learning Rate** | 0.005 |
| **LR Scheduler** | StepLR (step=3, γ=0.97) |
| **Loss Function** | LabelSmoothingCrossEntropy (timm) |
| **Dropout** | 0.3 |
| **Device** | CUDA (GPU) / CPU fallback |

**Why LabelSmoothingCrossEntropy?**  
Standard cross-entropy can cause a model to become overconfident on training labels. Label smoothing distributes a small probability mass to non-target classes, improving generalization and calibration — especially important in binary classification tasks like liveness detection.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision timm tqdm matplotlib
```

### Clone the Repository

```bash
git clone https://github.com/akshra09/Swin-transformer-LCC-FASD-CASIA.git
cd Swin-transformer-LCC-FASD-CASIA
```

### Dataset Setup (Kaggle)

1. Download the combined dataset from Kaggle:  
   🔗 `lcc-fasd-casia-combined`

2. Update the `data_dir` path in the notebook:
   ```python
   data_dir = '/path/to/lcc-fasd-casia/LCC_FASD'
   ```

### Run on Kaggle

The notebook is designed to run natively on **Kaggle** with GPU acceleration:
1. Upload the notebook to Kaggle
2. Attach the `lcc-fasd-casia-combined` dataset
3. Enable GPU accelerator
4. Run all cells

---

## 🔬 Code Walkthrough

### 1. Data Loading

```python
def get_swin_transformer_data_loaders(data_dir, batch_size, train=True, image_size=(224, 224)):
    if train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        dataset = datasets.ImageFolder(os.path.join(data_dir, "LCC_FASD_training/"), transform=transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        return data_loader, len(dataset)
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
        val_dataset = datasets.ImageFolder(os.path.join(data_dir, "LCC_FASD_development/"), transform=transform)
        test_dataset = datasets.ImageFolder(os.path.join(data_dir, "LCC_FASD_evaluation/"), transform=transform)
        ...
```

### 2. Model Setup (Transfer Learning)

```python
# Load pretrained Swin Transformer from torch hub
HUB_URL = "SharanSMenon/swin-transformer-hub:main"
MODEL_NAME = "swin_tiny_patch4_window7_224"
model = torch.hub.load(HUB_URL, MODEL_NAME, pretrained=True)

# Freeze all backbone parameters
for param in model.parameters():
    param.requires_grad = False

# Replace head with custom classifier
n_inputs = model.head.in_features
model.head = nn.Sequential(
    nn.Linear(n_inputs, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, len(classes))   # Binary: real vs spoof
)
```

### 3. Loss, Optimizer & Scheduler

```python
from timm.loss import LabelSmoothingCrossEntropy

criterion = LabelSmoothingCrossEntropy().to(device)
optimizer = optim.AdamW(model.head.parameters(), lr=0.005)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.97)
```

### 4. Training Loop

The `train_model` function:
- Alternates between `train` and `val` phases each epoch
- Tracks best validation accuracy and saves the best model weights
- Records losses and accuracies per epoch for visualization
- Applies the LR scheduler at the end of each training epoch

### 5. Evaluation

Per-class accuracy is computed on the held-out **evaluation split**:

```python
for i in range(len(classes)):
    print("Test Accuracy of %5s: %2d%% (%2d/%2d)" % (
        classes[i], 100 * class_correct[i] / class_total[i], ...
    ))
```

### 6. Visualization

Training and validation **loss** and **accuracy** curves are plotted side-by-side for training diagnostics:

```python
plt.subplot(1, 2, 1)  # Loss curves
plt.subplot(1, 2, 2)  # Accuracy curves
```

---

## 📊 Results

The model was evaluated on the LCC-FASD evaluation split after 200 epochs of fine-tuning the classification head.

| Metric | Score |
|--------|-------|
| **Best Validation Accuracy** | Tracked across all epochs |
| **Test Accuracy (Real)** | Per-class reported |
| **Test Accuracy (Spoof)** | Per-class reported |
| **Test Loss** | Reported at inference |

> ℹ️ Full numerical results and comparison with baselines are reported in the **IEEE paper** linked below.

---

## 🧠 Why Swin Transformer for Face Anti-Spoofing?

Traditional CNN-based FAS methods have limitations:
- **Limited receptive field** — local convolutions miss long-range spatial cues
- **Scale insensitivity** — fixed receptive fields struggle with multi-scale attacks

The **Swin Transformer** addresses these with:

| Feature | Benefit for FAS |
|---------|-----------------|
| **Shifted Window Attention** | Captures both local texture and global structure of the face |
| **Hierarchical Representation** | Multi-scale feature maps for detecting diverse spoofing artifacts |
| **Patch-based Processing** | Preserves fine-grained texture cues (printing artifacts, moiré patterns) |
| **ImageNet Pretraining** | Strong generalizable visual features, reducing need for large FAS datasets |

---

## 📁 Repository Structure

```
Swin-transformer-LCC-FASD-CASIA/
│
├── swin-transformer-lcc-fasd-casia.ipynb   # Main training notebook
└── README.md                               # This file
```

---

## 🔧 Dependencies

```
torch
torchvision
timm
numpy
pandas
matplotlib
tqdm
```

Install all at once:
```bash
pip install torch torchvision timm numpy pandas matplotlib tqdm
```

---

## 📄 Citation

If you use this work, please cite the associated IEEE paper:

```bibtex
@inproceedings{akshra2024swinFAS,
  title     = {Face Anti-Spoofing Using Swin Transformer on LCC-FASD and CASIA Datasets},
  author    = {Akshra},
  booktitle = {IEEE},
  year      = {2024},
  url       = {https://ieeexplore.ieee.org/document/10932042}
}
```

---

## 📰 Published Paper

This work has been published in **IEEE**. Read the full paper here:

> **🔗 [Face Anti-Spoofing with Swin Transformer — IEEE Xplore](https://ieeexplore.ieee.org/document/10932042)**

---

## 🤝 Acknowledgements

- [Swin Transformer](https://github.com/microsoft/Swin-Transformer) — Microsoft Research
- [timm](https://github.com/huggingface/pytorch-image-models) — Ross Wightman / HuggingFace
- [LCC-FASD Dataset](https://github.com/aniketpande/LCC-FASD)
- [CASIA Face Anti-Spoofing Dataset](http://www.cbsr.ia.ac.cn/english/FaceAntiSpoofDatabases.asp)
- Kaggle for compute and dataset hosting

---

## 📬 Contact

For questions or collaborations, open an issue or reach out via GitHub.

---

<p align="center">
  Made with ❤️ | Published in IEEE · <a href="https://ieeexplore.ieee.org/document/10932042">Read the Paper</a>
</p>

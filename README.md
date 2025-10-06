# 🧠 PyTorch Seminar – Fine-Tuning Pretrained CNNs

## 📘 Overview

This notebook demonstrates **fine-tuning convolutional neural networks (CNNs)** using **PyTorch** and **TorchVision**.  
It focuses on leveraging **pre-trained models** (such as ResNet or similar architectures) trained on the **ImageNet dataset** to perform transfer learning for new image classification tasks.

---

## 📚 Topics Covered

### 1. **Introduction to TorchVision**
- Overview of the `torchvision` library and its utilities:
  - Access to popular vision datasets.
  - Preprocessing and transformation tools.
  - Pre-trained CNN models ready for inference or fine-tuning.

### 2. **Dataset: ImageNet**
- Used pre-trained weights from ImageNet.
- ImageNet-1K dataset includes:
  - 1.28 million training images
  - 50,000 validation images
  - 100,000 test images
- Full version (ImageNet-21K) has over 14 million images and 21,000+ classes.

### 3. **Environment Setup**
- Libraries used:
  - `torch`, `torchvision`, `numpy`, `pandas`
  - `matplotlib` for visualization
  - `PIL` for image manipulation
  - `tqdm` for progress tracking
- Device check to ensure GPU (CUDA) is available for faster training:
  ```python
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

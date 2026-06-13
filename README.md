# Fault Robustness of Custom Floating-Point and Integer Formats
### Datatype Selection as a Reliability-Aware Compression Decision

**R S Haripriya, Jaynarayan T Tudu**
*Indian Institute of Technology Tirupati, India*

> ICML 2026 AdaptFM Workshop 

---

## Overview

This repository contains the complete code for our unified fault-robustness study across **14 low-precision datatypes** (custom Float16, Float8, and integer formats) under random multi-bit DRAM disturbance faults.

**Key finding:** E4M11 (1 sign + 4 exponent + 11 mantissa bits) is the Pareto-optimal 16-bit format — it matches FP16/BF16 clean accuracy while reducing worst-case exponent-fault accuracy loss by **5.7×** over FP16 and **5.9×** over BF16, with no retraining or error-correction hardware required.

---

## Repository Structure

```
├── README.md
│
├── fault_injection/                   # Main experiment scripts
│   ├── resnet18_cifar10.py
│   ├── resnet18_cifar100.py
│   ├── resnet18_tinyimagenet.py
│   ├── efficientnet_cifar10.py
│   ├── efficientnet_cifar100.py
│   ├── efficientnet_tinyimagenet.py
│   ├── mobilenetv2_cifar10.py
│   ├── mobilenetv2_cifar100.py
│   └── mobilenetv2_tinyimagenet.py
│
└── requirements.txt
```

---

## Evaluated Formats

| Family | Formats |
|--------|---------|
| Custom Float-16 | E2M13, E3M12, **E4M11**, E5M10 (FP16), E6M9, E7M8, E8M7 (BF16) |
| Custom Float-8 | F8E3M4, F8E4M3, F8E5M2 |
| Integer | INT16, INT8, INT4, INT2 |

---

## Requirements

```bash
pip install torch torchvision numpy matplotlib cupy-cudaXX
```

Replace `cupy-cudaXX` with your CUDA version, e.g.:
- CUDA 11.x → `cupy-cuda11x`
- CUDA 12.x → `cupy-cuda12x`

CuPy is optional but strongly recommended. Without it, the scripts fall back to a NumPy CPU implementation (significantly slower).

Full dependencies:
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.23.0
matplotlib>=3.6.0
cupy-cuda12x          # optional, recommended
```

---

## Dataset Setup

**CIFAR-10 and CIFAR-100** are downloaded automatically by torchvision on first run.

**Tiny ImageNet** requires a one-time setup:

```bash
# 1. Download
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip

# 2. Reorganise the validation split (run once)
python fault_injection/resnet18_tinyimagenet.py --reorganise-val
```

---

## Pre-trained Weights

Place your `.pth` weight files in the same directory as the scripts, or update the `PTH_PATH` constant at the top of each script.

Expected filenames:

| Script | Weight file |
|--------|------------|
| resnet18_cifar10.py | `resnet18_cifar10.pth` |
| resnet18_cifar100.py | `resnet18_cifar100.pth` |
| resnet18_tinyimagenet.py | `resnet18_tiny_imagenet.pth` |
| efficientnet_cifar10.py | `efficientnet_cifar10.pth` |
| efficientnet_cifar100.py | `efficientnet_cifar100.pth` |
| efficientnet_tinyimagenet.py | `efficientnet_tiny_imagenet.pth` |
| mobilenetv2_cifar10.py | `mobilenetv2_cifar10.pth` |
| mobilenetv2_cifar100.py | `mobilenetv2_cifar100.pth` |
| mobilenetv2_tinyimagenet.py | `mobilenetv2_tiny_imagenet.pth` |

**Model architectures used (modified for low-resolution inputs):**

- **ResNet-18:** 3×3 conv1 stride-1, Identity maxpool, task-specific fc
- **EfficientNet-B0:** stride-1 stem conv, task-specific classifier head
- **MobileNetV2:** stride-1 stem conv, task-specific classifier head

---

## Running Experiments

Each script is self-contained. Run any combination of model and dataset:

```bash
# ResNet-18 on CIFAR-10
python fault_injection/resnet18_cifar10.py

# EfficientNet-B0 on CIFAR-100
python fault_injection/efficientnet_cifar100.py

# MobileNetV2 on Tiny ImageNet (after --reorganise-val)
python fault_injection/mobilenetv2_tinyimagenet.py
```

Each run produces:
- A multi-panel figure (PDF + PNG) with clean accuracy bar chart and fault-drop curves per bit group
- An accuracy-drop heatmap at 8 bit-flips (PDF + PNG)
- A JSON file with all numerical results

---

## Fault Injection Protocol

| Parameter | Value |
|-----------|-------|
| Fault model | Random bit flips (DRAM disturbance / RowHammer-style) |
| Bit-flip counts | 1 – 8 simultaneous flips |
| Trials per config | 100 independent trials |
| Evaluation set | Full 10,000-sample test/val set |
| Injection target | Quantized weight buffer (before dequantization) |
| Skip threshold | Formats with PTQ accuracy drop > 30% below FP32 are excluded from fault injection |

**Fault groups evaluated per format:**

| Group | Bits targeted |
|-------|--------------|
| Group 1 | Exponent bits |
| Group 2 | Sign bit |
| Group 3 | Mantissa / Value MSB (upper 50%) |
| Group 4 | Mantissa / Value LSB (lower 50%) |

---

## E4M11 Format Specification

E4M11 is not natively available in PyTorch or NumPy. All encoding and decoding is implemented from scratch in each script.

**Bit layout:** `[S | EEEE | MMMMMMMMMMM]`  (1 + 4 + 11 bits)

| Field | Value |
|-------|-------|
| Exponent bits (E) | 4 |
| Mantissa bits (M) | 11 |
| Bias | 7 |
| Max finite value | ≈ 255.94 |
| Worst-case exponent-MSB perturbation factor ρ | 255 |

For comparison, FP16 has ρ = 65,535 and BF16 has ρ ≈ 3.4 × 10³⁸.

---

## Key Results Summary

| Format | Clean Acc. Drop vs FP32 | Exp-Fault Drop @ 8 flips (avg) | SQNR |
|--------|------------------------|-------------------------------|------|
| **E4M11** | **0.23%** | **6.1%** | **~79.8 dB** |
| FP16 (E5M10) | 0.22% | 34.6% | ~73.7 dB |
| BF16 (E8M7) | 0.25% | 36.2% | ~49.0 dB |
| INT16 | 0.0% | 0% (immune) | ~69.1 dB |
| INT8 | ~0.3% | 0% (immune) | ~21.0 dB |

Integer formats are structurally immune to exponent faults but have lower SQNR. E4M11 uniquely combines near-FP32 accuracy, highest SQNR, and strong fault tolerance.

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{haripriya2026fault,
  title     = {Fault Robustness of Custom Floating-Point and Integer Formats:
               Datatype Selection as a Reliability-Aware Compression Decision},
  author    = {Haripriya R S and Jaynarayan T Tudu},
  booktitle = {AdaptFM Workshop @ ICML 2026},
  year      = {2026},
  url       = {https://openreview.net/forum?id=t2jfdgUnQZ}
}
```



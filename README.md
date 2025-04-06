# AMF-MedIT

This repository provides a **PyTorch implementation** of **AMF-MedIT**, an efficient framework designed for multimodal learning on **medical imaging and tabular data**. 

> **AMF-MedIT: An Efficient Align-Modulation-Fusion Framework for Medical Imaging-Tabular Data**  

## Introduction

This project is built upon [MMCL-Tabular-Imaging](https://github.com/paulhager/MMCL-Tabular-Imaging).

Key implementations:
   - The proposed core module, **AMF module**, is implemented in [`./model/FusionModule.py`](./model/FusionModule.py)
   - Another contribution of our framework, the **FT-Mamba** encoder for tabular data, is implemented in [`./model/FT-Mamba.py`](./model/FT-Mamba.py)
   - The complete **AMF-MedIT model** structure can be found in [`./model/FusionMultiModalModel.py`](./model/FusionMultiModalModel.py)

## Requirements

This project was developed and tested under the following environment:

- **Operating System**: Linux (Ubuntu 22.04)
- **Kernel Version**: 5.19.0-41-generic
- **CUDA Version**: 12.0
- **Python Version**: 3.9.19
- **PyTorch Version**: 2.1.1
- **Pytorch-lightning**: 1.9.5

To install the dependencies:

```bash
conda env create --file environment.yaml
```

## Data
### 1. OL3I Dataset
- **Download from**: [Stanford AIMI Shared Datasets](https://aimi.stanford.edu/research/public-datasets)
- **Description**: A medical multimodal dataset including imaging and structured clinical/tabular data.
- **Preprocessing**:
  - Preprocessing notebook: [`./data/create_OL3I_dataset.ipynb`](./data/create_OL3I_dataset.ipynb)

### 2. DVM Dataset
- **Download from**: [DVM Car Dataset: A Large-Scale Dataset for Automotive Applications](https://deepvisualmarketing.github.io/)
- **Description**: A multimodal dataset combining images and marketing-related tabular features for automotive analysis.
- **Preprocessing**:
  - Preprocessing notebook: [`./data/create_DVM_dataset.ipynb`](./data/create_DVM_dataset.ipynb)

## Training

### Multimodal Pretraining

### MUltimodal Fine-tuning

### Supervised training for unimodal models

## Acknowledgments

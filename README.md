# AMF-MedIT

This repository provides a **PyTorch implementation** of **AMF-MedIT**, an efficient framework designed for multimodal learning on **medical imaging and tabular data**. 

> **AMF-MedIT: An Efficient Align-Modulation-Fusion Framework for Medical Imaging-Tabular Data**  

## Introduction

This project is built upon [MMCL](https://github.com/paulhager/MMCL-Tabular-Imaging).

Key implementations:
   - The proposed core module, **AMF module**, is implemented in [`./models/FusionModule.py`](./model/FusionModule.py)
   - Another contribution of our framework, the **FT-Mamba** encoder for tabular data, is implemented in [`./models/FT-Mamba.py`](./model/FT-Mamba.py)
   - The complete **AMF-MedIT model** structure can be found in [`./models/FusionMultiModalModel.py`](./model/FusionMultiModalModel.py)

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
- **Download from**: [Stanford AIMI Shared Datasets](https://stanfordaimi.azurewebsites.net/datasets/3263e34a-252e-460f-8f63-d585a9bfecfc)
- **Description**: A medical multimodal dataset including imaging and structured clinical/tabular data.
- **Preprocessing**:
  - Preprocessing notebook: [`./data/create_OL3I_dataset.ipynb`](./data/create_OL3I_dataset.ipynb)

### 2. DVM Dataset
- **Download from**: [DVM Car Dataset: A Large-Scale Dataset for Automotive Applications](https://deepvisualmarketing.github.io/)
- **Description**: A multimodal dataset combining images and marketing-related tabular features for automotive analysis.
- **Preprocessing**:
  - Preprocessing notebook: [`./data/create_DVM_dataset.ipynb`](./data/create_dvm_dataset.ipynb)

## Training
To start training, run:

```bash
python run.py
```
All training parameters and experiment configurations are defined using `.yaml` files located in the `./configs/` directory.
- You can switch between different training stages or strategies by editing the config files.
- The `run.py` script will automatically load the specified `.yaml` config and initialize the model, dataset, and training process accordingly.
### Multimodal Pretraining
We provide the config files for contrastive pretraining and pretrained checkpoints on both datasets:
- OL3I: [`config_pretrain_OL3I.yaml`](./configs/config_pretrain_OL3I.yaml), [`checkpoint_OL3I`](https://huggingface.co/Jasmine-ycj/AMF-MedIT/resolve/main/checkpoint_last_epoch_499_OL3I.ckpt)
- DVM: [`config_pretrain_DVM.yaml`](./configs/config_pretrain_DVM.yaml), [`checkpoint_DVM`](https://huggingface.co/Jasmine-ycj/AMF-MedIT/resolve/main/checkpoint_last_epoch_499_DVM.ckpt)

### Multimodal Fine-tuning
After pretraining, you can fine-tune the model for downstream classification:
- OL3I: [`config_fine_tuning_OL3I.yaml`](./configs/config_fine_tuning_OL3I.yaml)
- DVM: [`config_fine_tuning_DVM.yaml`](./configs/config_fine_tuning_DVM.yaml)

We experimented with two fine-tuning strategies:
- `finetune_strategy: frozen` — freeze the backbone and only train the fusion module and classification head
- `finetune_strategy: trainable` — fine-tune the entire model end-to-end

For each experiment, the best learning rate (`lr`) was selected from `{1e-2, 1e-3, 1e-4}` under seed `2022`.
We used five random seeds for evaluation: `2022, 2023, 2024, 2025, 2026`.

### Supervised training for unimodal models

You can also run unimodal (image-only or tabular-only) supervised experiments by modifying the config file:
- Set `eval_datatype` to either `imaging` or `tabular`
- Set `finetune_strategy` to `trainable`

## Acknowledgments

We would like to thank the following repositories for their great works:
- [MMCL](https://github.com/paulhager/MMCL-Tabular-Imaging)
- [RTDL](https://github.com/yandex-research/rtdl)
- [Mambular](https://github.com/basf/mamba-tabular)

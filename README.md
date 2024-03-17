# FSCIT

Official implementation of our paper:

> [**Few-shot Class-incremental Tuning**](https://arxiv.org/abs/) <br>
> Shuvendu Roy, Ali Etemad <br> > _Under-reivew_

## Overview

Abstract here

- main figure here
- other important figure

- Preliminiary table

## Getting started

Install the dependencies

```bash
pip install -r requirements.txt
```

## Prepare Data

CIFAR100, and all Hugging face datasets will be downloaded automatically.
For CUB-200, and miniImageNet please refer to [CEC](https://github.com/icoz69/CEC-CVPR2021) and downlaod it from [here](https://drive.google.com/drive/folders/11LxZCQj2FRCs0JTsf_dafvTHqFn2yGSN).

Extract the datasets to `./data` folder.

## Download pre-trained models

- SAM: https://storage.googleapis.com/vit_models/sam/ViT-B_16.npz
- DeiT: https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth
- CLIP: vit_base_patch16_clip_224.laion2b_ft_in12k_in1k
- MAE: https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth
- https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
- ./checkpoint/moco_v3.pth
- ./checkpoint/ibot_student.pth
- ./checkpoint/ibot_1k.pth
- ./checkpoint/clip_vit_b16.pth # clip
- https://storage.googleapis.com/vit_models/sam/ViT-B_16.npz
- https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth
- https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth

## Experiments

### Important Config Notes

- Default config for CIFAR100: 60 base classes, 5-way 5-shot incremental
- To use no-base 10-way 10-shot, used `--shot 10 --way 10 --base_class 10`

## Experiments: FSCIT

### BASELINE: No training (Prototype-based FSCIT)

```bash
# Supervised ViT-B16
python train.py \
  --update_base_classifier_with_prototypes True \
  --epochs_base 0 \
  --num_seeds 3 \
  --shot 10 --way 10 --base_class 10 \
  --pre_trained_url https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz

# Expected output
base: [94.1, 91.2, 85.4, 83.3, 81.4, 80.1, 79.6, 78.4, 77.4, 77.0]
incremental: [nan, 82.9, 79.25, 78.23, 75.28, 73.2, 72.1, 70.2, 70.29, 68.86]
all: [94.1, 87.05, 81.3, 79.5, 76.5, 74.35, 73.17, 71.22, 71.08, 69.67]
```

```bash
# DeiT
... --pre_trained_url https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth

base: [90.7, 81.5, 72.2, 68.7, 67.3, 61.6, 60.4, 59.8, 59.4]
incremental: [nan, 73.7, 72.85, 69.2, 66.05, 60.7, 58.78, 56.39, 56.55]
all: [90.7, 77.6, 72.63, 69.07, 66.3, 60.85, 59.01, 56.81, 56.87]
```

```bash
# SSL MoCo v3
...  --pre_trained_url ./checkpoint/moco_v3.pth

# Expected output
base: [51.1 50.5 37.3 36.7 33.7 33.7 32.2 32.2 31.5]
incremental: [  nan 23.1  28.55 27.63 22.43 20.82 20.47 20.2  19.65]
all: [51.1  36.8  31.47 29.9  24.68 22.97 22.14 21.7  20.97]
```

```bash
# iBOT 21K
...  --pre_trained_url ./checkpoint/ibot_student.pth

# Expected output
Base acc:  [87.3, 80.7, 72.6, 66.4, 60.9, 58.7, 57.2, 54.0, 53.7]
Inc. acc:  [nan, 69.4, 66.5, 61.3, 56.53, 54.52, 52.5, 50.93, 50.08]
Overall :  [87.3, 75.05, 68.53, 62.57, 57.4, 55.22, 53.17, 51.31, 50.48]
```

```bash
# iBOT 1K
...  --pre_trained_url ./checkpoint/ibot_1k.pth

# Expected output
base: [64.8 60.  45.8 39.6 38.4 33.1 29.6 26.8 26.3]
incremental: [  nan 45.6  33.3  33.1  31.5  28.26 25.7  23.97 22.88]
all: [64.8  52.8  37.47 34.72 32.88 29.07 26.26 24.32 23.26]
```

```bash
# DINO
... --pre_trained_url https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth

# Expected output
base: [38.  20.1 14.1  7.3  7.3  7.3  7.1  6.5  6.5]
incremental: [  nan 22.6  16.65 11.07 10.25  9.08  9.08  8.79  8.  ]
all: [38.   21.35 15.8  10.12  9.66  8.78  8.8   8.5   7.83]
```

```bash
# MAE
...  --pre_trained_url https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth

# Expected output
base: [22.2 12.4 11.3 10.7 10.4  3.   2.8  2.5  2.5]
incremental: [  nan 11.4   8.45  5.2   4.    5.08  4.3   3.93  3.45]
all: [22.2  11.9   9.4   6.57  5.28  4.73  4.09  3.75  3.34]
```

```bash
# HF google/vit-base-patch16-224-in21k
python train.py \
  --update_base_classifier_with_prototypes True \
  --epochs_base 0 \
  --num_seeds 1 \
  --shot 10 --way 10 --base_class 10 \
  --result_key _baseline \
  --hf_model_checkpoint "google/vit-base-patch16-224"

base: [92.5, 90.0, 83.4, 81.2, 78.8, 78.0, 77.4, 76.1, 75.8, 75.0]
incremental: [nan, 76.5, 73.35, 71.8, 69.0, 65.72, 64.52, 62.4, 62.59, 61.56]
all: [92.5, 83.25, 76.7, 74.15, 70.96, 67.77, 66.36, 64.11, 64.06, 62.9]
```

```bash
# google/vit-large-patch16-224-in21k
base: [95.2, 92.4, 88.5, 87.5, 86.8, 86.2, 85.7, 83.8, 83.0, 82.5]
incremental: [nan, 84.0, 80.75, 79.63, 77.23, 76.08, 75.37, 74.47, 74.66, 73.89]
all: [95.2, 88.2, 83.33, 81.6, 79.14, 77.77, 76.84, 75.64, 75.59, 74.75]
```

```bash
# HF google/vit-base-patch16-224-in21k
python train.py \
  --update_base_classifier_with_prototypes True \
  --epochs_base 0 \
  --num_seeds 1 \
  --shot 10 --way 10 --base_class 10 \
  --result_key _baseline \
  --hf_model_checkpoint "google/vit-base-patch16-224-in21k"

Base acc:  [92.9, 90.1, 83.9, 82.6, 79.3, 78.6, 77.9, 76.7, 75.8, 75.0]
Inc. acc:  [nan, 79.4, 74.3, 73.1, 70.95, 69.1, 68.37, 67.03, 66.86, 65.73]
Overall :  [92.9, 84.75, 77.5, 75.47, 72.62, 70.68, 69.73, 68.24, 67.86, 66.66]
```

### BASELINE: Start with prototype then finetune linear only

```bash
python train.py \
  --start_training_with_prototypes True \
  --epochs_base 10 \
  --num_seeds 3 \
  --shot 10 --way 10 --base_class 10 \
  --result_key _baseline_linear \
  --hf_model_checkpoint "google/vit-base-patch16-224-in21k"\
  --encoder_ft_start_layer 12 --lr_base 0.001


base: [92.8, 90.4, 83.83, 81.47, 79.67, 79.1, 78.77, 77.67, 77.17, 76.4]
incremental: [nan, 77.07, 72.72, 71.2, 68.66, 65.91, 64.67, 62.42, 62.47, 61.27]
all: [92.8, 83.73, 76.42, 73.76, 70.86, 68.11, 66.69, 64.33, 64.1, 62.78]
```

### BASELINE: Start with prototype then finetune linear and LoRA

```bash
python train.py \
  --start_training_with_prototypes True \
  --epochs_base 10 \
  --num_seeds 3 \
  --shot 10 --way 10 --base_class 10 \
  --result_key _baseline_lora \
  --hf_model_checkpoint "google/vit-base-patch16-224-in21k"\
  --encoder_ft_start_layer 12 \
  --pet_cls LoRA --adapt_blocks 3  --lr_base 0.1 # random param requires higher lr

base: [94.03, 91.4, 85.73, 84.27, 81.2, 80.33, 79.83, 78.1, 77.23, 75.53]
incremental: [nan, 81.67, 75.63, 74.08, 71.91, 69.89, 69.06, 67.71, 67.52, 66.34]
all: [94.03, 86.53, 79.0, 76.62, 73.77, 71.63, 70.6, 69.01, 68.6, 67.26]
```

### BASELINE: Tune on base session > Incremental frozen continual session

```bash
python train.py \
  --update_base_classifier_with_prototypes False  \
  --epochs_base 10 \
  --lr_base 0.1 \
  --encoder_ft_start_layer 12 \
  --num_seeds 3 \
  --pre_trained_url https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz \
  --shot 10 --way 10 --base_class 10  \
  --encoder_ft_start_epoch 0

# Expected output
base: [92.17, 46.37, 30.77, 25.27, 17.13, 15.13, 14.63, 14.23, 12.63, 10.9]
incremental: [nan, 91.73, 84.6, 81.6, 77.95, 75.44, 74.22, 72.04, 72.0, 70.33]
all: [92.17, 69.05, 66.65, 67.51, 65.79, 65.39, 65.7, 64.81, 65.4, 64.39]
```

```bash
# Tuning layer 11 and onward
python train.py \
  --update_base_classifier_with_prototypes False  \
  --epochs_base 10 \
  --lr_base 0.1 \
  --encoder_ft_start_layer 11 \
  --num_seeds 3 \
  --pre_trained_url https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz \
  --shot 10 --way 10 --base_class 10  \
  --encoder_ft_start_epoch 0

base: [91.77, 0.07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
incremental: [nan, 68.6, 58.33, 54.42, 53.18, 53.21, 51.66, 49.92, 50.23]
all: [91.77, 34.33, 38.89, 40.81, 42.54, 44.34, 44.28, 43.68, 44.64]
```

### BASELINE: Add adapter (Base session training) > Incremental Frozen

```bash
python train.py \
  --update_base_classifier_with_prototypes False  \
  --epochs_base 10 \
  --lr_base 0.1 \
  --encoder_ft_start_layer 12 \
  --num_seeds 3 \
  --pre_trained_url https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz \
  --shot 10 --way 10 --base_class 10  \
  --encoder_ft_start_epoch 0 \
  --pet_cls LoRA --adapt_blocks 3

base: [92.87, 48.67, 32.43, 25.77, 16.93, 14.67, 14.3, 13.93, 12.17, 10.37]
incremental: [nan, 91.67, 84.6, 81.6, 77.95, 75.44, 74.22, 72.03, 72.0, 70.33]
all: [92.87, 70.17, 67.21, 67.64, 65.75, 65.31, 65.66, 64.77, 65.35, 64.34]

# with --pet_cls Adapter
base: [93.27, 48.8, 33.07, 26.87, 18.2, 15.37, 15.0, 14.53, 13.03]
incremental: [nan, 91.77, 84.68, 81.7, 78.24, 75.77, 74.47, 72.32, 72.28]
all: [93.27, 70.28, 67.48, 67.99, 66.23, 65.71, 65.97, 65.1, 65.7]

# with --pet_cls Prefix
base: [93.27, 51.9, 35.8, 29.4, 20.17, 17.17, 16.83, 16.4, 14.47]
incremental: [nan, 91.93, 84.7, 81.81, 78.68, 76.35, 75.24, 73.08, 72.95]
all: [93.27, 71.92, 68.4, 68.71, 66.97, 66.49, 66.9, 65.99, 66.45]
```

## BASELINE: FSCIL

### No training (Prototyp-based FSCIT)

```bash
# Supervised ViT-B16
python train.py \
  --update_base_classifier_with_prototypes True \
  --epochs_base 0 \
  --num_seeds 3 \
  --pre_trained_url https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz

# Expected output
base: [78.92 77.92 77.23 75.85 75.43 74.05 73.32 72.87 72.  ]
incremental: [  nan 68.2  72.6  70.2  71.45 72.92 73.9  71.6  70.58]
all: [78.92 77.17 76.57 74.72 74.44 73.72 73.51 72.4  71.43]
```

```bash
# SSL MoCo v3
python train.py \
  --update_base_classifier_with_prototypes True \
  --epochs_base 0 \
  --num_seeds 3 \
  --pre_trained_url ./checkpoint/moco_v3.pth

# Expected output
base: [25.6  19.25 19.23 18.23 18.23 18.1  17.98 17.05 12.62]
incremental: [  nan 39.8  25.3  19.07 15.15 12.48 14.37 15.03 17.6 ]
all: [25.6  20.83 20.1  18.4  17.46 16.45 16.78 16.31 14.61]
```

```bash
# iBOT 21K
python train.py \
  --update_base_classifier_with_prototypes True \
  --epochs_base 0 \
  --num_seeds 3 \
  --pre_trained_url ./checkpoint/ibot_student.pth

# Expected output
base: [61.9  57.68 55.5  52.53 50.57 49.62 49.37 48.58 45.67]
incremental: [  nan 77.   62.9  54.6  54.9  54.68 52.13 51.46 51.75]
all: [61.9  59.17 56.56 52.95 51.65 51.11 50.29 49.64 48.1 ]
```

```bash
# iBOT 1K
python train.py \
  --update_base_classifier_with_prototypes True \
  --epochs_base 0 \
  --num_seeds 3 \
  --pre_trained_url ./checkpoint/ibot_1k.pth

# Expected output
base: [33.1  25.03 23.03 21.57 21.1  20.88 20.08 18.1  15.57]
incremental: [  nan 57.4  42.8  33.87 31.95 28.96 26.2  24.43 24.18]
all: [33.1  27.52 25.86 24.03 23.81 23.26 22.12 20.43 19.01]
```

```bash
# DINO
python train.py \
  --update_base_classifier_with_prototypes True \
  --epochs_base 0 \
  --num_seeds 3 \
  --pre_trained_url https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth

# Expected output
base: [8.62 8.3  8.3  7.53 7.52 7.52 7.38 7.37 4.38]
incremental: [  nan 32.6  16.5  15.07 13.55 10.88 10.17 10.94 12.18]
all: [ 8.62 10.17  9.47  9.04  9.02  8.51  8.31  8.68  7.5 ]
```

### LoRA + Asyn. MoCo

````bash
### Full fine-tune + incremental frozen

```bash
python train.py \
  --update_base_classifier_with_prototypes True \
  --start_training_with_prototypes True \
  --epochs_base 10 \
  --moco_loss_factor 1.0 \
  --lr_base 0.001 \
  --encoder_ft_start_layer -1 \
  --encoder_ft_start_epoch 0 \
  --encoder_lr_factor 2.0 \
  --num_seeds 3 \
  --pre_trained_url https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz  \
  --shot 10 --way 10 --base_class 10 \
  --pet_cls LoRA --adapt_blocks 1 \
````

## Acknowledgements # TODO

## Citing FSCIT

If you find our work useful to your research, please cite our paper:

```bash
@article{FSCIT,
  title={Few-shot Class-incremental Tuning},
  author={Roy, Shuvendu and Etemad, Ali},
  journal={arXiv preprint arXiv:2306.01229},
  year={2024}
}
```

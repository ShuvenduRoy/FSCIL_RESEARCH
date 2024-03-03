# FSCIT

Official implementation of our paper:

> [**Few-shot Class-incremental Tuning**](https://arxiv.org/abs/)
> Shuvendu Roy, Ali Etemad
> _Under-reivew_

## Getting started

Install the dependencies

```bash
pip install -r requirements.txt
```

### Download Data

- CIFAR100 - Will be downloaded automatically

### Download pre-trained models

- SAM: https://storage.googleapis.com/vit_models/sam/ViT-B_16.npz
- https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
- ./checkpoint/moco_v3.pth
- ./checkpoint/ibot_student.pth
- ./checkpoint/ibot_1k.pth
- https://storage.googleapis.com/vit_models/sam/ViT-B_16.npz
- https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth
- https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth

## Runing the Experiments

### Important Config Notes

- Default config for CIFAR100: 60 base classes, 5-way 5-shot incremental
- To use no-base 10-way 10-shot, used `--shot 10 --way 10 --base_class 10`

### Baseline: No training (Prototyp-based FSCIT)

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

### Baseline: No training (Prototype-based FSCIT) for 10-way 10-shot

```bash
# Supervised ViT-B16
python train.py \
  --update_base_classifier_with_prototypes True \
  --epochs_base 0 \
  --num_seeds 3 \
  --pre_trained_url https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz \
  --shot 10 --way 10 --base_class 10

# Expected output
Base acc:  [94.1, 91.2, 85.4, 83.3, 81.4, 80.1, 79.6, 78.4, 77.4]
Inc. acc:  [nan, 82.9, 79.25, 78.23, 75.28, 73.2, 72.1, 70.2, 70.29]
Overall :  [94.1, 87.05, 81.3, 79.5, 76.5, 74.35, 73.17, 71.22, 71.08]
```

```bash
# SSL MoCo v3
python train.py \
  --update_base_classifier_with_prototypes True \
  --epochs_base 0 \
  --num_seeds 3 \
  --pre_trained_url ./checkpoint/moco_v3.pth \
  --shot 10 --way 10 --base_class 10

# Expected output
base: [51.1 50.5 37.3 36.7 33.7 33.7 32.2 32.2 31.5]
incremental: [  nan 23.1  28.55 27.63 22.43 20.82 20.47 20.2  19.65]
all: [51.1  36.8  31.47 29.9  24.68 22.97 22.14 21.7  20.97]
```

```bash
# iBOT 21K
python train.py \
  --update_base_classifier_with_prototypes True \
  --epochs_base 0 \
  --num_seeds 3 \
  --pre_trained_url ./checkpoint/ibot_student.pth \
  --shot 10 --way 10 --base_class 10

# Expected output
Base acc:  [87.3, 80.7, 72.6, 66.4, 60.9, 58.7, 57.2, 54.0, 53.7]
Inc. acc:  [nan, 69.4, 66.5, 61.3, 56.53, 54.52, 52.5, 50.93, 50.08]
Overall :  [87.3, 75.05, 68.53, 62.57, 57.4, 55.22, 53.17, 51.31, 50.48]
```

```bash
# iBOT 1K
python train.py \
  --update_base_classifier_with_prototypes True \
  --epochs_base 0 \
  --num_seeds 3 \
  --pre_trained_url ./checkpoint/ibot_1k.pth \
  --shot 10 --way 10 --base_class 10

# Expected output
base: [64.8 60.  45.8 39.6 38.4 33.1 29.6 26.8 26.3]
incremental: [  nan 45.6  33.3  33.1  31.5  28.26 25.7  23.97 22.88]
all: [64.8  52.8  37.47 34.72 32.88 29.07 26.26 24.32 23.26]
```

```bash
# DINO
python train.py \
  --update_base_classifier_with_prototypes True \
  --epochs_base 0 \
  --num_seeds 3 \
  --pre_trained_url https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth \
  --shot 10 --way 10 --base_class 10

# Expected output
base: [38.  20.1 14.1  7.3  7.3  7.3  7.1  6.5  6.5]
incremental: [  nan 22.6  16.65 11.07 10.25  9.08  9.08  8.79  8.  ]
all: [38.   21.35 15.8  10.12  9.66  8.78  8.8   8.5   7.83]
```

```bash
# MAE
python train.py \
  --update_base_classifier_with_prototypes True \
  --epochs_base 0 \
  --num_seeds 3 \
  --pre_trained_url https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth \
  --shot 10 --way 10 --base_class 10

# Expected output
base: [22.2 12.4 11.3 10.7 10.4  3.   2.8  2.5  2.5]
incremental: [  nan 11.4   8.45  5.2   4.    5.08  4.3   3.93  3.45]
all: [22.2  11.9   9.4   6.57  5.28  4.73  4.09  3.75  3.34]
```

### Full fine-tune + incremental frozen

```
python train.py \
  --update_base_classifier_with_prototypes False  \
  --start_training_with_prototypes False \
  --epochs_base 10 \
  --lr_base 0.1 \
  --encoder_ft_start_layer 12 \
  --num_seeds 3 \
  --pre_trained_url https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz \
  --shot 10 --way 10 --base_class 10  \
  --encoder_ft_start_epoch 0
```

### LoRA + Asyn. MoCo

````bash
### Full fine-tune + incremental frozen

```bash
python train.py \
  --update_base_classifier_with_prototypes True \
  --start_training_with_prototypes True \
  --epochs_base 10 \
  --lr_base 0.001 \
  --encoder_ft_start_layer 15 \
  --num_seeds 3 \
  --pre_trained_url ./checkpoint/moco_v3.pth \
  --shot 10 --way 10 --base_class 10 \
  --pet_cls LoRA --adapt_blocks 3 \
````

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

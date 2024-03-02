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

- TODO: Add links

## Runing the Experiments

### Important Config Notes

- Default config for CIFAR100: 60 base classes, 5-way 5-shot incremental
- To use no-base 10-way 10-shot, used `--shot 10 --way 10 --base_class 10`

### Baseline

- No training (Prototyp-based FSCIT)

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

base: [8.62 8.3  8.3  7.53 7.52 7.52 7.38 7.37 4.38]
incremental: [  nan 32.6  16.5  15.07 13.55 10.88 10.17 10.94 12.18]
all: [ 8.62 10.17  9.47  9.04  9.02  8.51  8.31  8.68  7.5 ]
```

- Full fine-tune + incremental frozen

```bash
python train.py --..
```

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

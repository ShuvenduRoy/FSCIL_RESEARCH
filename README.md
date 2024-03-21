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

## Pre-trained models

Pre-trained models will be downloaded from hugging face. Here is the list of models that we tried.

- google/vit-base-patch16-224

## Experiments

### Important Config Notes

- Default config for CIFAR100: 60 base classes, 5-way 5-shot incremental
- To use no-base 10-way 10-shot, used `--shot 10 --way 10 --base_class 10`

## Experiments: FSCIT

### BASELINE: No training (Prototype-based FSCIT)

```bash
for dataset_name in sun397 dtd voc2007 stanford_cars resisc45 oxford_pets oxford_flowers gtsrb fgvc_aircraft eurosat country211 caltech101 cifar100 cub200 food101 mini_imagenet; do #

CUDA_VISIBLE_DEVICES=2 python train.py \
  --update_base_classifier_with_prototypes True \
  --epochs_base 0 \
  --num_seeds 3 \
  --shot 10 \
  --result_key baseline \
  --hf_model_checkpoint "google/vit-base-patch16-224-in21k" \
  --dataset "${dataset_name}"
done

```

### BASELINE: Start with prototype then finetune linear only

```bash
python train.py \
  --start_training_with_prototypes True \
  --epochs_base 10 \
  --num_seeds 3 \
  --shot 10 \
  --result_key baseline_linear \
  --hf_model_checkpoint "google/vit-base-patch16-224-in21k"\
  --encoder_ft_start_layer 12 --lr_base 0.001
```

### BASELINE: Start with prototype then finetune linear and LoRA

```bash
python train.py \
  --start_training_with_prototypes True \
  --epochs_base 10 \
  --num_seeds 3 \
  --shot 10 \
  --result_key baseline_lora \
  --hf_model_checkpoint "google/vit-base-patch16-224-in21k"\
  --pet_cls LoRA --adapt_blocks 3  --lr_base 0.1 # random param requires higher lr
```

# PROPOSED: Adding Contrastive Ascyncronous Prompt-tuning (CAP)

```bash
python train.py \
  --start_training_with_prototypes True \
  --epochs_base 10 \
  --num_seeds 3 \
  --shot 10 \
  --result_key PET_lora \
  --hf_model_checkpoint "google/vit-base-patch16-224-in21k"\
  --pet_cls LoRA --adapt_blocks 3  --lr_base 0.1 \
  --moco_loss_factor 1.0 \
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
```

## BASELINE: FSCIL

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

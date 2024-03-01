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
- To use no-base 10-way 10-shot, used `--shot 10 --way 10`

### Baseline

- No training (Prototyp-based FSCIT)

```bash
# Supervised ViT-16/B
python train.py \
  --update_base_classifier_with_prototypes True \
  --epochs_base 0 \
  --num_seeds 3

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

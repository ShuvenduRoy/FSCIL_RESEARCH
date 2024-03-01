"""Vision Transformer (ViT) in PyTorch.

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

The official jax code is released and available at https://github.com/google-research/vision_transformer

DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's
https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020, Ross Wightman

Credit: https://raw.githubusercontent.com/rwightman/pytorch-image-models/v0.5.4/timm/models/vision_transformer.py
"""

from typing import Any

import torch
from timm.data import (
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from timm.models.helpers import build_model_with_cfg
from timm.models.vision_transformer import VisionTransformer, checkpoint_filter_fn


def _cfg(url: str = "", **kwargs: dict) -> dict:
    """Construct a ViT config dict from a given url and kwargs.

    Parameters
    ----------
    url : str
        The url to the model weights.
    kwargs : dict
        The keyword arguments to the model.

    Returns
    -------
    dict
        The configuration dictionary.
    """
    return {
        "url": url,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "fixed_input_size": True,
        "mean": IMAGENET_INCEPTION_MEAN,
        "std": IMAGENET_INCEPTION_STD,
        "first_conv": "patch_embed.proj",
        **kwargs,
    }


def _create_vision_transformer(
    variant: Any,
    pretrained: bool = True,
    **kwargs: Any,
) -> Any:
    """Create a Vision Transformer model."""
    default_cfg = _cfg(url=kwargs.pop("pre_trained_url"))

    model = build_model_with_cfg(
        VisionTransformer,
        variant,
        pretrained=(
            False
            if default_cfg["url"] is None or "checkpoint/" in default_cfg["url"]
            else pretrained
        ),
        default_cfg=default_cfg,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load=False,
        **kwargs,
    )
    if (
        default_cfg["url"] is not None and "checkpoint/" in default_cfg["url"]
    ):  # load pretrained weights from local file
        print(
            "Loading pretrained weights from local file: {}".format(
                default_cfg["url"],
            ),
        )
        state_dict = torch.load(default_cfg["url"], map_location="cpu")
        message = model.load_state_dict(state_dict, strict=False)
        assert (
            message.missing_keys == []
        ), "Missing keys when loading pretrained weights: {}".format(
            message.missing_keys,
        )

    return model


def vit_b16(pretrained: bool = True, **kwargs: Any) -> Any:
    """ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).

    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        **kwargs,
    )
    return _create_vision_transformer(
        "vit_b16_in21k",
        pretrained=pretrained,
        **model_kwargs,
    )


def vit_h14(pretrained: bool = True, **kwargs: Any) -> Any:
    """ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929)."""
    model_kwargs = dict(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        **kwargs,
    )
    return _create_vision_transformer(
        "vit_huge_patch14_224",
        pretrained=pretrained,
        **model_kwargs,
    )

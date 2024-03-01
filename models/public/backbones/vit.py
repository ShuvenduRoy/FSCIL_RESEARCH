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

from functools import partial
from typing import Any

import torch
from timm.data import (
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from timm.models.helpers import build_model_with_cfg, named_apply
from timm.models.layers import PatchEmbed, trunc_normal_
from timm.models.vision_transformer import (
    Block,
    _init_vit_weights,
    _load_weights,
    checkpoint_filter_fn,
)
from torch import nn


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


class VisionTransformer(nn.Module):
    """Vision Transformer.

    A PyTorch impl of : `An Image is Worth 16x16 Words:
        Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for
        `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        distilled: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        embed_layer: Any = PatchEmbed,
        norm_layer: Any = None,
        act_layer: Any = None,
        weight_init: Any = "",
    ) -> None:
        """Vision Transformer forward.

        Parameters
        ----------
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            distilled (bool): model includes a distillation token
                and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme.
        """
        super().__init__()

        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dim),
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ],
        )
        self.norm = norm_layer(embed_dim)

        self.init_weights(weight_init)

    def init_weights(self, mode: str = "") -> None:
        """Initialize weights."""
        assert mode in ("jax", "jax_nlhb", "nlhb", "")
        trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=0.02)
        if mode.startswith("jax"):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=0.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m: Any) -> None:
        """Initialize weights."""
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path: str, prefix: str = "") -> None:
        """Load pretrained weights."""
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self) -> Any:
        """No weight decay."""
        return {"pos_embed", "cls_token", "dist_token"}

    def forward_features(self, x: Any) -> Any:
        """Forward features."""
        x = self.patch_embed(x)
        # stole cls_tokens impl from Phil Wang, thanks
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            dist_token = self.dist_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, dist_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        x = self.blocks(x)
        return self.norm(x)

    def forward(self, x: Any) -> Any:
        """Forward method."""
        return self.forward_features(x)  # shape: B N C


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

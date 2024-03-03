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

import numpy as np
import torch
from timm.data import (
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from timm.models.helpers import adapt_input_conv, build_model_with_cfg, named_apply
from timm.models.layers import (
    DropPath,
    PatchEmbed,
    trunc_normal_,
)
from timm.models.layers.helpers import to_2tuple
from timm.models.vision_transformer import (
    _init_vit_weights,
    resize_pos_embed,
)
from torch import nn

from models.pet_mixin import AdapterMixin, PrefixMixin, PromptMixin


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


def checkpoint_filter_fn(state_dict: Any, model: Any) -> dict:
    """Convert patch embedding weight from manual patchify + linear proj to conv."""
    out_dict = {}
    if "model" in state_dict:
        # For deit models
        state_dict = state_dict["model"]
    for k, v in state_dict.items():
        # ignore head and pre_logits
        if k.startswith("head.") or k.startswith("pre_logits"):
            continue
        if "patch_embed.proj.weight" in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape  # noqa: N806, E741
            v = v.reshape(O, -1, H, W)  # noqa: PLW2901
        elif k == "pos_embed" and v.shape != model.pos_embed.shape:
            # To resize pos emb when using model at different size of pretrained weight
            v = resize_pos_embed(  # noqa: PLW2901
                v,
                model.pos_embed,
                getattr(model, "num_tokens", 1),
                model.patch_embed.grid_size,
            )
        out_dict[k] = v
    return out_dict


@torch.no_grad()
def _load_weights(  # noqa: PLR0915
    model: Any,
    checkpoint_path: str,
    prefix: str = "",
) -> None:
    """Load weights from .npz checkpoints for Google Brain Flax implementation."""

    def _n2p(w: Any, t: bool = True) -> torch.Tensor:
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and "opt/target/embedding/kernel" in w:
        prefix = "opt/target/"

    if hasattr(model.patch_embed, "backbone"):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, "stem")
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(
            adapt_input_conv(
                stem.conv.weight.shape[1],
                _n2p(w[f"{prefix}conv_root/kernel"]),
            ),
        )
        stem.norm.weight.copy_(_n2p(w[f"{prefix}gn_root/scale"]))
        stem.norm.bias.copy_(_n2p(w[f"{prefix}gn_root/bias"]))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f"{prefix}block{i + 1}/unit{j + 1}/"
                    for r in range(3):
                        getattr(block, f"conv{r + 1}").weight.copy_(
                            _n2p(w[f"{bp}conv{r + 1}/kernel"]),
                        )
                        getattr(block, f"norm{r + 1}").weight.copy_(
                            _n2p(w[f"{bp}gn{r + 1}/scale"]),
                        )
                        getattr(block, f"norm{r + 1}").bias.copy_(
                            _n2p(w[f"{bp}gn{r + 1}/bias"]),
                        )
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(
                            _n2p(w[f"{bp}conv_proj/kernel"]),
                        )
                        block.downsample.norm.weight.copy_(
                            _n2p(w[f"{bp}gn_proj/scale"]),
                        )
                        block.downsample.norm.bias.copy_(
                            _n2p(w[f"{bp}gn_proj/bias"]),
                        )
        embed_conv_w = _n2p(w[f"{prefix}embedding/kernel"])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1],
            _n2p(w[f"{prefix}embedding/kernel"]),
        )
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f"{prefix}embedding/bias"]))
    model.cls_token.copy_(_n2p(w[f"{prefix}cls"], t=False))
    pos_embed_w = _n2p(
        w[f"{prefix}Transformer/posembed_input/pos_embedding"],
        t=False,
    )
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w,
            model.pos_embed,
            getattr(model, "num_tokens", 1),
            model.patch_embed.grid_size,
        )
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f"{prefix}Transformer/encoder_norm/scale"]))
    model.norm.bias.copy_(_n2p(w[f"{prefix}Transformer/encoder_norm/bias"]))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f"{prefix}Transformer/encoderblock_{i}/"
        mha_prefix = block_prefix + "MultiHeadDotProductAttention_1/"
        block.norm1.weight.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/scale"]))
        block.norm1.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_0/bias"]))
        block.attn.qkv.weight.copy_(
            torch.cat(
                [
                    _n2p(w[f"{mha_prefix}{n}/kernel"], t=False).flatten(1).T
                    for n in ("query", "key", "value")
                ],
            ),
        )
        block.attn.qkv.bias.copy_(
            torch.cat(
                [
                    _n2p(w[f"{mha_prefix}{n}/bias"], t=False).reshape(-1)
                    for n in ("query", "key", "value")
                ],
            ),
        )
        block.attn.proj.weight.copy_(
            _n2p(w[f"{mha_prefix}out/kernel"]).flatten(1),
        )
        block.attn.proj.bias.copy_(_n2p(w[f"{mha_prefix}out/bias"]))
        for r in range(2):
            getattr(block.mlp, f"fc{r + 1}").weight.copy_(
                _n2p(w[f"{block_prefix}MlpBlock_3/Dense_{r}/kernel"]),
            )
            getattr(block.mlp, f"fc{r + 1}").bias.copy_(
                _n2p(w[f"{block_prefix}MlpBlock_3/Dense_{r}/bias"]),
            )
        block.norm2.weight.copy_(_n2p(w[f"{block_prefix}LayerNorm_2/scale"]))
        block.norm2.bias.copy_(_n2p(w[f"{block_prefix}LayerNorm_2/bias"]))


class Mlp(nn.Module, AdapterMixin):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Any = None,
        out_features: Any = None,
        act_layer: Any = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        x = self.adapt_module("fc1", x)  # x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.adapt_module("fc2", x)  # x = self.fc2(x)
        return self.drop2(x)


class Attention(nn.Module, PromptMixin, PrefixMixin, AdapterMixin):
    """Attention layer of Transformer."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Any) -> torch.Tensor:
        """Forward function."""
        x = self.add_prompt(x)

        B, N, C = x.shape  # noqa: N806
        qkv = self.adapt_module("qkv", x)

        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv.chunk(3, dim=-1)
        k, v = self.add_prefix(k, v)

        q = q.reshape(B, N, self.num_heads, C // self.num_heads)
        q = q.permute(0, 2, 1, 3)

        k = k.reshape(B, -1, self.num_heads, C // self.num_heads)
        k = k.permute(0, 2, 1, 3)

        v = v.reshape(B, -1, self.num_heads, C // self.num_heads)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = self.compensate_prefix(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.adapt_module("proj", x)  # x = self.proj(x)
        x = self.proj_drop(x)

        return self.reduce_prompt(x)


class Block(nn.Module, AdapterMixin):
    """Transformer block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Any = nn.GELU,
        norm_layer: Any = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # drop path for stoch. depth, we shall see if this is better than dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        x = x + self.drop_path(
            self.adapt_module("attn", self.norm1(x)),  # self.attn(self.norm1(x))
        )
        return x + self.drop_path(
            self.adapt_module("mlp", self.norm2(x)),  # self.mlp(self.norm2(x))
        )


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
        pretrained_custom_load=default_cfg["url"] and "npz" in default_cfg["url"],
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

"""Encoder model for FSCIL."""

import argparse
from typing import Any, Optional, Tuple

import torch
from torch import nn

from models.backbones.vit import vit_b16
from models.pet.adapter import Adapter
from models.pet.lora import KVLoRA
from models.pet.prefix import Prefix


class EncoderWrapper(nn.Module):
    """Encoder Wrapper encoders."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize the EncoderWrapper.

        Parameters
        ----------
        num_classes : int
            Number of classes.
        args : argparse.ArgumentParser
            Arguments passed to the model.

        Returns
        -------
        None
        """
        super(EncoderWrapper, self).__init__()
        self.args = args
        if self.args.encoder == "vit-b16":
            print("Encoder: ViT-B16")
            self.model = vit_b16(
                True,
                pre_trained_url=args.pre_trained_url,
            )
            self.num_features = 768
        if args.num_mlp == 2:
            self.fc = nn.Sequential(
                nn.Linear(self.num_features, self.num_features),
                nn.ReLU(),
                nn.Linear(self.num_features, self.args.moco_dim),
            )

        elif args.num_mlp == 3:
            self.fc = nn.Sequential(
                nn.Linear(self.num_features, self.num_features),
                nn.ReLU(),
                nn.Linear(self.num_features, self.num_features),
                nn.ReLU(),
                nn.Linear(self.num_features, self.args.moco_dim),
            )
        else:
            self.fc = nn.Sequential(nn.Linear(self.num_features, self.args.moco_dim))

        self.classifier = nn.Linear(
            self.num_features,
            self.args.num_classes,
            bias=self.args.add_bias_in_classifier,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            patch embeddings, [b, embed_dim]
            projecting embedding, [b, moco_dim]
            output logits, [b, n_classes]
        """
        x = self.model(x).mean(1)  # [b, n_tokens=197, embed_dim=768] -> [b, embed_dim]
        return (
            x,  # [b, embed_dim=768]
            self.fc(x),  # [b, moco_dim=128]
            self.classifier(x),  # [b, n_classes]
        )


class FSCILencoder(nn.Module):
    """FSCIL Model class."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize FSCIL Model.

        Parameters
        ----------
        args : argparse.ArgumentParser
            Arguments passed to the model.

        Returns
        -------
        None
        """
        super().__init__()

        self.args = args
        self.encoder_q = EncoderWrapper(args)
        self.num_features = 768

        # By default all pre-trained parameters are frozen
        # except the MLP and the classifier layers
        for param in self.encoder_q.parameters():
            param.requires_grad = False
        for param in self.encoder_q.fc.parameters():
            param.requires_grad = True
        for param in self.encoder_q.classifier.parameters():
            param.requires_grad = True

        self.encoder_k = EncoderWrapper(args)

        # Add PET modules. Doing after initializing the both encoders
        if self.args.pet_cls is not None:
            self.args.pet_kwargs = {}

            self.pets = self.create_pets()
            self.attach_pets(self.pets, self.encoder_q)

            if self.args.pet_on_teacher:
                self.args.pet_kwargs = {}

                self.pets = self.create_pets()
                self.attach_pets(self.pets, self.encoder_k)

        # Handle the case that EMA can be different in terms of parameters
        encoder_q_params = self.encoder_q.state_dict()
        for name, param in self.encoder_k.named_parameters():
            param.requires_grad = False
            if name in encoder_q_params:
                param.data.copy_(encoder_q_params[name])

        # print param name and status
        print("\nencoder_q parameters:")
        for name, param in self.encoder_q.named_parameters():
            print("model.encoder_q", name, param.requires_grad)

        print("\nencoder_q parameters:")
        for name, param in self.encoder_k.named_parameters():
            print("model.encoder_k", name, param.requires_grad)

        pet_name = "none" if self.args.pet_cls is None else self.args.pet_cls.lower()
        self.params_with_lr = [
            # Higher LR for newly initalized parameters
            {
                "params": [
                    p
                    for n, p in self.encoder_q.named_parameters()
                    if pet_name in n or n.startswith("fc")
                ],
                "lr": args.lr_base,
            },
            # (Optional) reduced LR for pre-trained model parameters
            {
                "params": [
                    p
                    for n, p in self.encoder_q.named_parameters()
                    if pet_name not in n and not n.startswith("fc")
                ],
                "lr": args.lr_base * args.encoder_lr_factor,
            },
        ]

        # create the queue
        self.register_buffer(
            "queue",
            torch.randn(self.args.moco_dim, self.args.moco_k, dtype=torch.float32),
        )
        self.queue: torch.Tensor = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("label_queue", torch.zeros(int(args.moco_k)).long() - 1)

    def create_pets(self) -> nn.ModuleList:
        """Create PETs."""
        return self.create_pets_vit()

    def attach_pets(self, pets: nn.ModuleList, encoder: nn.Module) -> None:
        """Attach PETs."""
        return self.attach_pets_vit(pets, encoder)

    def create_pets_vit(self) -> nn.ModuleList:
        """Create PETs for ViT."""
        assert self.args.pet_cls in ["Adapter", "LoRA", "Prefix"]

        n = len(self.args.adapt_blocks)
        embed_dim = self.num_features

        kwargs = dict(**self.args.pet_kwargs)
        if self.args.pet_cls == "Adapter":
            kwargs["embed_dim"] = embed_dim
            return nn.ModuleList([Adapter(**kwargs) for _ in range(n)])

        if self.args.pet_cls == "LoRA":
            kwargs["in_features"] = embed_dim
            kwargs["out_features"] = embed_dim
            kwargs["rank"] = self.args.rank
            return nn.ModuleList([KVLoRA(**kwargs) for _ in range(n)])

        kwargs["dim"] = embed_dim
        return nn.ModuleList([Prefix(**kwargs) for i in range(n)])

    def attach_pets_vit(self, pets: nn.ModuleList, encoder: Any) -> None:
        """Attach PETs for ViT.

        Parameters
        ----------
        pets : nn.ModuleList
            PETs.
        encoder : Any
            Encoder to which PETs are attached.

        Returns
        -------
        None
        """
        assert self.args.pet_cls in ["Adapter", "LoRA", "Prefix"]

        if self.args.pet_cls == "Adapter":
            for i, b in enumerate(self.args.adapt_blocks):
                encoder.model.blocks[b].attach_adapter(attn=pets[i])
            return

        if self.args.pet_cls == "LoRA":
            for i, b in enumerate(self.args.adapt_blocks):
                encoder.model.blocks[b].attn.attach_adapter(qkv=pets[i])
            return

        for i, b in enumerate(self.args.adapt_blocks):
            encoder.model.blocks[b].attn.attach_prefix(pets[i])

    @torch.no_grad()
    def _momentum_update_key_encoder(self, base_sess: bool) -> None:
        """Momentum update of the key encoder.

        Parameters
        ----------
        base_sess : bool
            Whether the current session is a base session.

        Returns
        -------
        None
        """
        encoder_q_params = self.encoder_q.state_dict()
        for name, param in self.encoder_k.named_parameters():
            param.data = param.data * self.args.moco_m + encoder_q_params[name] * (
                1.0 - self.args.moco_m
            )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor, labels: torch.Tensor) -> None:
        """Update the queue.

        Parameters
        ----------
        keys : torch.Tensor
            The keys.
        labels : torch.Tensor
            The labels.

        Returns
        -------
        None
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr.item())  # type: ignore

        # replace the keys and labels at ptr (dequeue and enqueue)
        if ptr + batch_size > self.args.moco_k:
            remains = ptr + batch_size - self.args.moco_k
            self.queue[:, ptr:] = keys.T[:, : batch_size - remains]
            self.queue[:, :remains] = keys.T[:, batch_size - remains :]
            self.label_queue[ptr:] = labels[: batch_size - remains]  # type: ignore
            self.label_queue[:remains] = labels[batch_size - remains :]  # type: ignore
        else:
            self.queue[:, ptr : ptr + batch_size] = (
                keys.T
            )  # this queue is feature queue
            self.label_queue[ptr : ptr + batch_size] = labels  # type: ignore
        ptr = (ptr + batch_size) % self.args.moco_k  # move pointer
        self.queue_ptr[0] = ptr  # type: ignore

    def forward(
        self,
        im_q: torch.Tensor,
        im_k: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        base_sess: bool = True,
    ) -> Any:
        """Forward pass of the model.

        Parameters
        ----------
        im_q : torch.Tensor, optional
            The query image, or Input image at test time
        im_k : torch.Tensor, optional
            The key image, by default None.
        labels : torch.Tensor, optional
            The labels, by default None.
        base_sess : bool, optional
            Whether the current session is a base session, by default True.

        Returns
        -------
        Any
            The output tensor.
        """
        if isinstance(im_q, list):
            im_q = torch.cat(im_q, dim=0)
        token_embeding, embedding_q, logits = self.encoder_q(
            im_q,
        )  # [b, embed_dim=768], [b, moco_dim=128], [b, n_classes]
        assert len(logits.shape) == 2
        assert logits.shape[1] == self.args.num_classes
        assert embedding_q.shape[1] == self.args.moco_dim

        if labels is None:  # during evaluation, im_q should be a single image
            return (token_embeding, logits)
        embedding_q = nn.functional.normalize(embedding_q, dim=1)

        # foward key
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder(base_sess)  # update the key encoder
            _, embedding_k, _ = self.encoder_k(im_k)  # keys: bs x dim
            embedding_k = nn.functional.normalize(embedding_k, dim=1)

        if embedding_q.shape[0] != embedding_k.shape[0]:  # multiple views
            embedding_q = embedding_q.view(
                embedding_k.shape[0],
                -1,
                embedding_k.shape[1],
            )
        else:
            embedding_q = embedding_q.unsqueeze(1)
        embedding_k = embedding_k.unsqueeze(1)

        return logits[: embedding_q.shape[0]], embedding_q, embedding_k

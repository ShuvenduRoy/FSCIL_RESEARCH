"""Encoder model for FSCIL."""

import argparse
import importlib
import math
from typing import Any, List, Optional, Tuple

import torch
from torch import nn


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
        if self.args.encoder == "vit-16":
            print("Loading ViT-16")
            net_module = importlib.import_module("models.public.backbones.vit")
            self.model = net_module.vit_b16_in21k(
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Output tensor.
        """
        x = self.model(x).mean(1)
        return self.fc(x), self.classifier(x)


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
        for param in self.encoder_q.parameters():
            param.requires_grad = False
        for param in self.encoder_q.fc.parameters():
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
        for name, param in self.encoder_q.named_parameters():
            print(name, param.requires_grad)
        for name, param in self.encoder_k.named_parameters():
            print(name, param.requires_grad)

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
        pet_module = importlib.import_module("models.public.pet")
        assert self.args.pet_cls in ["Adapter", "LoRA", "Prefix"]

        n = len(self.args.adapt_blocks)
        embed_dim = self.num_features

        kwargs = dict(**self.args.pet_kwargs)
        if self.args.pet_cls == "Adapter":
            kwargs["embed_dim"] = embed_dim
            return nn.ModuleList([pet_module.Adapter(**kwargs) for _ in range(n)])

        if self.args.pet_cls == "LoRA":
            kwargs["in_features"] = embed_dim
            kwargs["out_features"] = embed_dim
            kwargs["rank"] = self.args.rank
            return nn.ModuleList([pet_module.KVLoRA(**kwargs) for _ in range(n)])

        kwargs["dim"] = embed_dim
        return nn.ModuleList([pet_module.Prefix(**kwargs) for i in range(n)])

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
        if (
            base_sess
        ):  # TODO this is horrible hard coded logic. also need to handle EMA with diff params
            for param_q, param_k in zip(
                self.encoder_q.parameters(),
                self.encoder_k.parameters(),
            ):
                param_k.data = param_k.data * self.args.moco_m + param_q.data * (
                    1.0 - self.args.moco_m
                )
        else:
            for k, v in self.encoder_q.named_parameters():
                if (
                    k.startswith("fc")
                    or k.startswith("layer4")
                    or k.startswith("layer3")
                ):
                    self.encoder_k.state_dict()[k].data = self.encoder_k.state_dict()[
                        k
                    ].data * self.args.moco_m + v.data * (1.0 - self.args.moco_m)

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
        ptr = int(self.queue_ptr.item())

        # replace the keys and labels at ptr (dequeue and enqueue)
        if ptr + batch_size > self.K:
            remains = ptr + batch_size - self.K
            self.queue[:, ptr:] = keys.T[:, : batch_size - remains]
            self.queue[:, :remains] = keys.T[:, batch_size - remains :]
            self.label_queue[ptr:] = labels[: batch_size - remains]
            self.label_queue[:remains] = labels[batch_size - remains :]
        else:
            self.queue[:, ptr : ptr + batch_size] = (
                keys.T
            )  # this queue is feature queue
            self.label_queue[ptr : ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    def forward(
        self,
        im_q: torch.Tensor,
        im_k: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        base_sess: bool = True,
        last_epochs_new: bool = False,
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
        last_epochs_new : bool, optional
            Whether the last epoch an incremental session, by default False.

        Returns
        -------
        Any
            The output tensor.
        """
        embedding_q, logits = self.encoder_q(im_q)  # [b, embed_dim] [b, n_classes]
        assert len(logits.shape) == 2
        assert logits.shape[1] == self.args.num_classes
        assert embedding_q.shape[1] == self.args.moco_dim

        if labels is None:  # during evaluation
            return logits  # [b, n_classes]
        embedding_q = nn.functional.normalize(embedding_q, dim=1)
        embedding_q = embedding_q.unsqueeze(1)  # [b, 1, embed_dim]

        # foward key
        b = im_q.shape[0]
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder(base_sess)  # update the key encoder
            embedding_k, _ = self.encoder_k(im_k)  # keys: bs x dim
            embedding_k = nn.functional.normalize(embedding_k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = (embedding_q * embedding_k.unsqueeze(1)).sum(2).view(-1, 1)

        # negative logits: NxK
        l_neg = torch.einsum(
            "nc,ck->nk",
            [embedding_q.view(-1, self.args.moco_dim), self.queue.clone().detach()],
        )

        # logits with shape Nx(1+K)``
        logits_global = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits_global /= self.args.moco_t

        # one-hot target from augmented image
        positive_target = torch.ones((b, 1)).to(logits_global.device)

        # find same label images from label queue
        # for the query with -1, all
        targets = (
            ((labels[:, None] == self.label_queue[None, :]) & (labels[:, None] != -1))
            .float()
            .to(logits_global.device)
        )

        targets_global = torch.cat([positive_target, targets], dim=1)

        # dequeue and enqueue
        if base_sess or (not base_sess and last_epochs_new):
            self._dequeue_and_enqueue(embedding_k, labels)

        return logits, embedding_q, logits_global, targets_global

    def update_fc(self, dataloader: Any, class_list: List, transform: Any) -> None:
        """Update the fully connected layer.

        Parameters
        ----------
        dataloader : Any
            The dataloader.
        class_list : list
            The class list.
        transform : Any
            The transformations

        Returns
        -------
        None
        """
        data_all = []
        all_labels = []

        for batch in dataloader:
            if torch.cuda.is_available():
                data, label = [_.cuda() for _ in batch]
            else:
                data, label = batch
            data = transform(data)
            labels = torch.stack([label], 1).view(-1)
            data, _ = self.encode_q(data)
            data.detach()

            data_all.append(data)
            all_labels.append(labels)

        labels = torch.cat(all_labels, 0)
        data = torch.cat(data_all, 0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device=device),
                requires_grad=True,
            )
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, labels, class_list)

    def update_fc_avg(self, data: Any, labels: Any, class_list: list) -> torch.Tensor:
        """Update the fully connected layer.

        Parameters
        ----------
        data : torch.Tensor
            The data.
        labels : torch.Tensor
            The labels.
        class_list : list
            The class list.

        Returns
        -------
        torch.Tensor
            The new fully connected layer weight.
        """
        new_fc = []  # TODO need to fix this. This should be the classifier weight
        for index in class_list:
            data_index = (labels == index).nonzero().squeeze(-1)
            embedding = data[data_index]
            proto = embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[index] = proto
        return torch.stack(new_fc, dim=0)

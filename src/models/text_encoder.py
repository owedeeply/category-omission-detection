from __future__ import annotations
from contextlib import nullcontext
from typing import Any
import torch
from torch import nn


class CXRTextEncoder(nn.Module):
    """CXR-BERT text encoder using last_hidden_state and attention_mask."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install transformers to use CXRTextEncoder.") from exc
        self.cfg = cfg
        self.name = cfg["name"]
        self.max_length = int(cfg.get("max_length", 256))
        self.freeze_backbone = bool(cfg.get("freeze", True))
        trust_remote_code = bool(cfg.get("trust_remote_code", False))
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.name, trust_remote_code=trust_remote_code
        )
        self.model = AutoModel.from_pretrained(
            self.name, trust_remote_code=trust_remote_code
        )
        self.output_dim = int(self.model.config.hidden_size)
        if self.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def train(self, mode: bool = True) -> "CXRProjectedTextEncoder":
        super().train(mode)
        if self.freeze_backbone:
            self.model.eval()
        return self

    def unfreeze_last_layers(self, n_layers: int) -> None:
        if n_layers <= 0:
            return
        self.freeze_backbone = False
        encoder_layers = getattr(getattr(self.model, "encoder", None), "layer", None)
        if encoder_layers is None:
            return
        for layer in list(encoder_layers)[-n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def forward(
        self,
        texts: list[str] | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.model.parameters()).device
        if texts is not None:
            encoded = self.tokenize(texts)
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
        if input_ids is None or attention_mask is None:
            raise ValueError("Provide either texts or input_ids with attention_mask.")
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        context = torch.no_grad() if self.freeze_backbone else nullcontext()
        with context:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            tokens = outputs.last_hidden_state
        return (tokens, attention_mask)

    @torch.no_grad()
    def encode_prompts(self, prompts: list[str]) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        tokens, _ = self.forward(texts=prompts)
        prompt_embeddings = tokens[:, 0, :].detach().cpu()
        if was_training:
            self.model.train()
        return prompt_embeddings


class SyntheticTextEncoder(nn.Module):

    def __init__(
        self, vocab_size: int = 4096, output_dim: int = 768, max_length: int = 256
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, output_dim)

    def forward(
        self,
        texts: list[str] | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids is None:
            batch_size = len(texts or [""])
            input_ids = torch.zeros(
                batch_size,
                self.max_length,
                dtype=torch.long,
                device=self.embedding.weight.device,
            )
            attention_mask = torch.ones_like(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return (
            self.embedding(input_ids.to(self.embedding.weight.device)),
            attention_mask.to(self.embedding.weight.device),
        )

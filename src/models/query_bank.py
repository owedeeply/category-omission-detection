from __future__ import annotations
from pathlib import Path
from typing import Mapping, Sequence
import torch
from torch import nn
from src.data.parquet_dataset import CLASSES, CLASS_SLUGS
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)
CANONICAL_PROMPTS = {
    "Atelectasis": "There is atelectasis.",
    "Cardiomegaly": "There is cardiomegaly.",
    "Consolidation": "There is consolidation.",
    "Fracture": "There is a fracture.",
    "Pulmonary Nodule/Mass": "There is a pulmonary nodule or mass.",
    "Pleural Effusion": "There is pleural effusion.",
    "Pneumothorax": "There is pneumothorax.",
    "Emphysema": "There is emphysema.",
    "Edema": "There is pulmonary edema.",
}


def build_query_count_vector(
    class_names: Sequence[str],
    queries_per_class: int,
    queries_per_class_by_class: Mapping[str, int] | None = None,
) -> list[int]:
    default_count = int(queries_per_class)
    if default_count <= 0:
        raise ValueError("queries_per_class must be positive.")
    overrides = dict(queries_per_class_by_class or {})
    unknown = sorted(set(overrides) - set(class_names))
    if unknown:
        raise ValueError(f"Unknown query-count override classes: {unknown}")
    counts = [
        int(overrides.get(class_name, default_count)) for class_name in class_names
    ]
    bad = {
        class_name: count
        for class_name, count in zip(class_names, counts)
        if count <= 0
    }
    if bad:
        raise ValueError(f"Per-class query counts must be positive: {bad}")
    return counts


class CategoryQueryBank(nn.Module):

    def __init__(
        self,
        prompt_embeddings: torch.Tensor,
        hidden_dim: int = 768,
        queries_per_class: int = 3,
        queries_per_class_by_class: Mapping[str, int] | None = None,
        class_names: Sequence[str] | None = None,
        query_init_path: str | Path | None = None,
        query_init_trainable: bool = True,
        query_init_required: bool = False,
        query_init_random_seed: int = 42,
        query_init_random_reference_max_queries: int | None = None,
        image_residual_enable: bool = False,
        image_residual_init_std: float = 0.02,
        image_residual_scale: float = 1.0,
        image_residual_trainable: bool = True,
        use_projection: bool = False,
        use_modality_adaptive: bool = False,
        dropout: float = 0.1,
        query_bank_type: str = "class_queries",
    ) -> None:
        super().__init__()
        if prompt_embeddings.ndim != 2:
            raise ValueError(
                "prompt_embeddings must have shape [num_classes, text_dim]."
            )
        self.num_classes, text_dim = prompt_embeddings.shape
        self.hidden_dim = hidden_dim
        self.queries_per_class = queries_per_class
        self.use_projection = bool(use_projection)
        self.use_modality_adaptive = bool(use_modality_adaptive)
        self.query_bank_type = str(query_bank_type or "class_queries").lower()
        self.query_init_random_seed = int(query_init_random_seed)
        self.query_init_random_reference_max_queries = (
            None
            if query_init_random_reference_max_queries is None
            else int(query_init_random_reference_max_queries)
        )
        self.image_residual_enable = bool(image_residual_enable)
        self.image_residual_init_std = float(image_residual_init_std)
        self.image_residual_scale = float(image_residual_scale)
        self.image_residual_trainable = bool(image_residual_trainable)
        if self.image_residual_init_std < 0.0:
            raise ValueError("image_residual_init_std must be non-negative.")
        if self.image_residual_scale < 0.0:
            raise ValueError("image_residual_scale must be non-negative.")
        if self.query_bank_type in {"legacy", "prompt_offset"}:
            self.query_bank_type = "legacy_prompt_offset"
        if self.query_bank_type not in {"class_queries", "legacy_prompt_offset"}:
            raise ValueError(f"Unsupported query_bank_type={query_bank_type!r}.")
        if self.image_residual_enable and self.query_bank_type != "class_queries":
            raise ValueError(
                "image query residual is supported only for query_bank_type='class_queries'."
            )
        self.class_names = list(class_names or CLASS_SLUGS)
        if len(self.class_names) != self.num_classes:
            raise ValueError(
                f"class_names length ({len(self.class_names)}) must match num_classes ({self.num_classes})."
            )
        self.query_counts = build_query_count_vector(
            self.class_names,
            queries_per_class,
            queries_per_class_by_class=queries_per_class_by_class,
        )
        self.max_queries_per_class = max(self.query_counts)
        if (
            self.query_init_random_reference_max_queries is not None
            and self.query_init_random_reference_max_queries
            < self.max_queries_per_class
        ):
            raise ValueError(
                f"query_init_random_reference_max_queries must be at least the configured maximum query count ({self.max_queries_per_class}), got {self.query_init_random_reference_max_queries}."
            )
        query_mask = torch.zeros(
            self.num_classes, self.max_queries_per_class, dtype=torch.bool
        )
        for class_idx, count in enumerate(self.query_counts):
            query_mask[class_idx, :count] = True
        self.register_buffer("query_mask", query_mask, persistent=True)
        self.register_buffer(
            "prompt_embeddings", prompt_embeddings.float(), persistent=True
        )
        initial_queries = None
        loaded_evidence_anchor = initial_queries is not None
        if self.image_residual_enable and (not loaded_evidence_anchor):
            raise ValueError(
                "model.query.image_residual.enable=true requires a successfully loaded Evidence-KMeans query_init anchor."
            )
        if initial_queries is None:
            reference_max_queries = (
                self.query_init_random_reference_max_queries
                or self.max_queries_per_class
            )
            reference_shape = (self.num_classes, reference_max_queries, hidden_dim)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.query_init_random_seed)
            reference_queries = (
                torch.randn(reference_shape, generator=generator, dtype=torch.float32)
                * 0.02
            )
            initial_queries = reference_queries[
                :, : self.max_queries_per_class, :
            ].contiguous()
            initial_queries = initial_queries.masked_fill(
                ~self.query_mask.unsqueeze(-1), 0.0
            )
            LOGGER.info(
                "Initialized CGQ-Net category queries | reference_shape=%s registered_shape=%s query_counts=%s trainable=%s seed=%d std=0.02 independent_generator=true",
                tuple(reference_shape),
                tuple(initial_queries.shape),
                self.query_counts,
                bool(query_init_trainable),
                self.query_init_random_seed,
            )
        self.class_queries = nn.Parameter(
            initial_queries.float(), requires_grad=bool(query_init_trainable)
        )
        LOGGER.info(
            "Final per-class Query counts | %s | total_valid_queries=%d | registered_shape=%s",
            dict(zip(self.class_names, self.query_counts)),
            int(self.query_mask.sum().item()),
            tuple(self.class_queries.shape),
        )
        self.register_parameter("image_query_residual", None)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.query_proj_norm = nn.LayerNorm(hidden_dim)
        self.img_query_adapter = self._build_query_adapter(hidden_dim, dropout)
        self.txt_query_adapter = self._build_query_adapter(hidden_dim, dropout)
        self.img_query_norm = nn.LayerNorm(hidden_dim)
        self.txt_query_norm = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _build_query_adapter(hidden_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self, batch_size: int | None = None, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_img, q_txt = self.modality_queries(device=device)
        query_mask = self.query_mask
        if device is not None:
            query_mask = query_mask.to(device)
        if batch_size is None:
            return (q_img, q_txt, query_mask)
        return (
            q_img.unsqueeze(0).expand(batch_size, -1, -1, -1),
            q_txt.unsqueeze(0).expand(batch_size, -1, -1, -1),
            query_mask.unsqueeze(0).expand(batch_size, -1, -1),
        )

    def modality_queries(
        self, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_q = self.class_queries
        if device is not None:
            base_q = base_q.to(device)
        if self.use_projection:
            q0 = self.query_proj_norm(self.query_proj(base_q))
        else:
            q0 = base_q
        q_img_seed = q0
        if self.use_modality_adaptive:
            q_img = self.img_query_norm(q_img_seed + self.img_query_adapter(q_img_seed))
            q_txt = self.txt_query_norm(q0 + self.txt_query_adapter(q0))
        else:
            q_img = q_img_seed
            q_txt = q0
        return (q_img, q_txt)


def build_query_bank_from_text_encoder(
    text_encoder: nn.Module,
    hidden_dim: int,
    queries_per_class: int,
    queries_per_class_by_class: Mapping[str, int] | None = None,
    query_init_path: str | Path | None = None,
    query_init_trainable: bool = True,
    query_init_required: bool = False,
    query_init_random_seed: int = 42,
    query_init_random_reference_max_queries: int | None = None,
    image_residual_enable: bool = False,
    image_residual_init_std: float = 0.02,
    image_residual_scale: float = 1.0,
    image_residual_trainable: bool = True,
    use_projection: bool = False,
    use_modality_adaptive: bool = False,
    dropout: float = 0.1,
    query_bank_type: str = "class_queries",
) -> CategoryQueryBank:
    prompts = [CANONICAL_PROMPTS[name] for name in CLASSES]
    prompt_embeddings = text_encoder.encode_prompts(prompts)
    return CategoryQueryBank(
        prompt_embeddings=prompt_embeddings,
        hidden_dim=hidden_dim,
        queries_per_class=queries_per_class,
        queries_per_class_by_class=queries_per_class_by_class,
        class_names=CLASS_SLUGS,
        query_init_path=query_init_path,
        query_init_trainable=query_init_trainable,
        query_init_required=query_init_required,
        query_init_random_seed=query_init_random_seed,
        query_init_random_reference_max_queries=query_init_random_reference_max_queries,
        image_residual_enable=image_residual_enable,
        image_residual_init_std=image_residual_init_std,
        image_residual_scale=image_residual_scale,
        image_residual_trainable=image_residual_trainable,
        use_projection=use_projection,
        use_modality_adaptive=use_modality_adaptive,
        dropout=dropout,
        query_bank_type=query_bank_type,
    )


def build_synthetic_query_bank(
    text_dim: int,
    hidden_dim: int,
    queries_per_class: int,
    queries_per_class_by_class: Mapping[str, int] | None = None,
    query_init_path: str | Path | None = None,
    query_init_trainable: bool = True,
    query_init_required: bool = False,
    query_init_random_seed: int = 42,
    query_init_random_reference_max_queries: int | None = None,
    image_residual_enable: bool = False,
    image_residual_init_std: float = 0.02,
    image_residual_scale: float = 1.0,
    image_residual_trainable: bool = True,
    use_projection: bool = False,
    use_modality_adaptive: bool = False,
    dropout: float = 0.1,
    query_bank_type: str = "class_queries",
) -> CategoryQueryBank:
    generator = torch.Generator().manual_seed(17)
    prompt_embeddings = torch.randn(len(CLASSES), text_dim, generator=generator) * 0.02
    return CategoryQueryBank(
        prompt_embeddings,
        hidden_dim,
        queries_per_class,
        queries_per_class_by_class=queries_per_class_by_class,
        class_names=CLASS_SLUGS,
        query_init_path=query_init_path,
        query_init_trainable=query_init_trainable,
        query_init_required=query_init_required,
        query_init_random_seed=query_init_random_seed,
        query_init_random_reference_max_queries=query_init_random_reference_max_queries,
        image_residual_enable=image_residual_enable,
        image_residual_init_std=image_residual_init_std,
        image_residual_scale=image_residual_scale,
        image_residual_trainable=image_residual_trainable,
        use_projection=use_projection,
        use_modality_adaptive=use_modality_adaptive,
        dropout=dropout,
        query_bank_type=query_bank_type,
    )

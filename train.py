from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import torch

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
from src.lightning.lit_omission import (
    CxrOmissionDataModule,
    LitOmissionModule,
    save_latest_thresholds,
)
from src.losses.class_weights import compute_omit_pos_weight, compute_weak_pos_weight
from src.models.omission_model import build_model_from_config
from src.utils.diagnostics import (
    log_pair_weight_summary,
    log_parameter_summary,
    log_pos_weights,
    log_training_label_statistics,
)
from src.utils.io import (
    assert_training_paths_filled,
    load_yaml,
    resolve_output_dir,
    validate_config_structure,
)
from src.utils.logging import configure_logging, get_logger
from src.utils.seed import seed_everything

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a Lightning checkpoint for resume training.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path for a plain-text training log. Defaults to outputs/logs/train_<timestamp>.log.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    errors = validate_config_structure(cfg)
    if errors:
        raise ValueError("\n".join(errors))
    assert_training_paths_filled(cfg)
    seed_everything(int(cfg["train"].get("seed", 42)))
    torch.set_float32_matmul_precision("high")
    output_dir = resolve_output_dir(cfg)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_file = (
        Path(args.log_file)
        if args.log_file
        else output_dir / "logs" / f"train_{datetime.now():%Y%m%d_%H%M%S}.log"
    )
    configure_logging(log_file)
    LOGGER.info("Writing plain-text training log to %s", log_file)
    if args.resume:
        LOGGER.info("Resuming training from checkpoint: %s", args.resume)
    data_module = CxrOmissionDataModule(cfg)
    data_module.setup("fit")
    if data_module.train_dataset is None:
        raise RuntimeError("Training dataset was not initialized.")
    omit_clip = tuple(
        (float(v) for v in cfg["loss"].get("omit_pos_weight_clip", [1.0, 5.0]))
    )
    weak_clip = tuple(
        (float(v) for v in cfg["loss"].get("weak_pos_weight_clip", [1.0, 3.0]))
    )
    log_training_label_statistics(data_module.train_dataset.df)
    log_pair_weight_summary(data_module.train_dataset.df, cfg)
    omit_pos_weight = compute_omit_pos_weight(
        data_module.train_dataset.df, clip=omit_clip
    )
    weak_mask_source = str(
        cfg["loss"].get(
            "weak_mask_source",
            "omit" if bool(cfg["loss"].get("weak_use_omit_mask", False)) else "all",
        )
    ).lower()
    if weak_mask_source not in {"all", "omit", "valid"}:
        raise ValueError("loss.weak_mask_source must be one of: all, omit, valid.")
    weak_pos_weight = compute_weak_pos_weight(
        data_module.train_dataset.df,
        clip=weak_clip,
        mask_with_omit_mask=weak_mask_source == "omit",
        mask_column_prefix="weak_valid_mask" if weak_mask_source == "valid" else None,
    )
    log_pos_weights(omit_pos_weight, weak_pos_weight)
    model = build_model_from_config(cfg, require_checkpoint=True)
    query_bank = getattr(model, "query_bank", None)
    log_parameter_summary(model, cfg)
    lit_module = LitOmissionModule(
        model=model,
        cfg=cfg,
        omit_pos_weight=omit_pos_weight,
        weak_pos_weight=weak_pos_weight,
    )
    early_cfg = cfg["train"]["early_stopping"]
    callbacks = [
        EarlyStopping(
            monitor=early_cfg.get("monitor", "val/omit_macro_auprc"),
            mode=early_cfg.get("mode", "max"),
            patience=int(early_cfg.get("patience", 12)),
            min_delta=float(early_cfg.get("min_delta", 0.0005)),
        ),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="epoch{epoch:03d}-val_omit_macro_auprc{val/omit_macro_auprc:.4f}",
            monitor=early_cfg.get("monitor", "val/omit_macro_auprc"),
            mode=early_cfg.get("mode", "max"),
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False,
        ),
    ]
    logger = TensorBoardLogger(save_dir=str(output_dir), name="tensorboard")
    trainer = pl.Trainer(
        max_epochs=int(cfg["train"].get("max_epochs", 100)),
        accelerator="auto",
        devices="auto",
        precision=cfg["train"].get("precision", "bf16-mixed"),
        accumulate_grad_batches=int(cfg["train"].get("accumulate_grad_batches", 2)),
        gradient_clip_val=float(cfg["train"].get("gradient_clip_val", 1.0)),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=20,
    )
    trainer.fit(lit_module, datamodule=data_module, ckpt_path=args.resume)
    save_latest_thresholds(lit_module, output_dir)


if __name__ == "__main__":
    main()

"""
lineage_receipt.py — Run Lineage Receipt
=========================================

Every checkpoint carries a cryptographic lineage header so runs can be
audited, reproduced, and traced across resumptions.

Fields:
    positional_encoding_mode  : RoPE / learned / none
    architecture_hash         : SHA-256 of model hyperparameters (truncated)
    training_loop_hash        : SHA-256 of training configuration (truncated)
    configuration_hash        : SHA-256 of governor settings (truncated)
    checkpoint_hash           : SHA-256 of model state dict structure (truncated)
    step_range                : "start-end" steps covered by this checkpoint
    epoch                     : Training epoch
    creation_date             : ISO-8601 timestamp
    author                    : "Mycelia-v4.2-Auto" or human name
    version                   : "lineage-v1"

Usage:
    from lineage_receipt import compute_lineage_receipt
    receipt = compute_lineage_receipt(model, step=1000, epoch=0, cfg=cfg)
    checkpoint_dict['lineage_receipt'] = receipt
"""

import hashlib
from datetime import datetime
from typing import Dict, Any


def compute_lineage_receipt(
    model,
    step: int,
    epoch: int,
    cfg: Any,
    batch_size: int = 2,
    accum_steps: int = 16,
    max_seq_len: int = 512,
    peak_lr: float = 3e-4,
    min_lr: float = 3e-5,
    warmup_steps: int = 500,
    grad_clip: float = 2.0,
    weight_decay: float = 0.01,
    ffn_target_start: float = 50.0,
    ffn_target_end: float = 150.0,
    alpha_target_start: float = 100.0,
    alpha_target_end: float = 150.0,
    transition_duration: int = 10000,
    version: str = "lineage-v1",
    author: str = "Mycelia-v4.2-Auto",
) -> Dict[str, Any]:
    """Compute a lineage receipt dict with hashes of the run's identity.

    All hyperparameters are passed explicitly to avoid implicit dependencies
    on module-level globals. This makes the function pure and testable.
    """
    # Positional encoding mode
    pe_mode = "RoPE"
    if hasattr(cfg, 'positional_encoding_mode'):
        pe_mode = cfg.positional_encoding_mode
    elif hasattr(model, 'pos_encoding_mode'):
        pe_mode = model.pos_encoding_mode

    # Architecture hash: model class + key hyperparameters
    arch_components = [
        type(model).__name__,
        str(getattr(cfg, 'num_layers', 12)),
        str(getattr(cfg, 'num_heads', 12)),
        str(getattr(cfg, 'hidden_size', 768)),
        str(getattr(cfg, 'intermediate_size', 3072)),
        str(getattr(cfg, 'vocab_size', 151643)),
        str(getattr(cfg, 'max_seq_len', 512)),
        pe_mode,
        str(getattr(cfg, 'use_compression', False)),
        str(getattr(cfg, 'consensus_rounds', 2)),
    ]
    arch_hash = hashlib.sha256("|".join(arch_components).encode()).hexdigest()[:16]

    # Training loop hash: version + key loop parameters
    loop_components = [
        "v9.1",
        str(batch_size),
        str(accum_steps),
        str(max_seq_len),
        str(peak_lr),
        str(min_lr),
        str(warmup_steps),
        str(grad_clip),
        str(weight_decay),
        str(ffn_target_start),
        str(ffn_target_end),
        str(alpha_target_start),
        str(alpha_target_end),
        str(transition_duration),
    ]
    loop_hash = hashlib.sha256("|".join(loop_components).encode()).hexdigest()[:16]

    # Configuration hash: all governor + transition settings
    config_components = [
        str(getattr(cfg, 'ffn_norm_target', 150.0)),
        str(getattr(cfg, 'alpha_norm_target', 150.0)),
        str(getattr(cfg, 'soft_cap', 400.0)),
        str(getattr(cfg, 'instability_target', 0.45)),
        str(getattr(cfg, 'control_gain', 1.0)),
        str(getattr(cfg, 'control_factor_floor', 0.7)),
        str(getattr(cfg, 'predictive_scale', True)),
        str(getattr(cfg, 'use_rate_governor', False)),
        str(getattr(cfg, 'ffn_growth_ratio_max', 2.0)),
        str(getattr(cfg, 'residual_growth_ratio_max', 1.5)),
        str(getattr(cfg, 'max_simultaneous_governors', 2)),
    ]
    config_hash = hashlib.sha256("|".join(config_components).encode()).hexdigest()[:16]

    # Checkpoint hash: deterministic from model state dict keys + shapes
    state_dict = model.state_dict()
    ckpt_components = []
    for k in sorted(state_dict.keys()):
        shape_str = "x".join(str(d) for d in state_dict[k].shape)
        ckpt_components.append(f"{k}:{shape_str}")
    checkpoint_hash = hashlib.sha256("|".join(ckpt_components).encode()).hexdigest()[:16]

    return {
        "positional_encoding_mode": pe_mode,
        "architecture_hash": arch_hash,
        "training_loop_hash": loop_hash,
        "configuration_hash": config_hash,
        "checkpoint_hash": checkpoint_hash,
        "step_range": f"0-{step}" if step > 0 else "0-0",
        "epoch": epoch,
        "creation_date": datetime.now().isoformat(),
        "author": author,
        "version": version,
    }

from dataclasses import dataclass


@dataclass
class CUREConfig:
    feature_dim: int = 768
    embedding_dim: int = 64
    feature_size: int = 68
    dropout_rate: float = 0.5
    warmup_epochs: int = 1
    lambda_fp: float = 0.5
    lambda_ent: float = 0.05
    ema_weight: float = 0.99
    roi_threshold: float = 0.2
    scale_power: float = 2.0
    clip_min: float = 0.1
    clip_max: float = 2.0
    eps: float = 1e-5

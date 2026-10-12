from dataclasses import dataclass
import torch

@dataclass
class DatasetConfig:
    num_nodes: int = 50
    motif_ratio: float = 0.3
    noise_multiplier: int = 4
    time_window_delta: float = 5.0
    num_distractors: int = 3
    num_partial_motifs: int = 2
    seed: int = 42

@dataclass
class ModelConfig:
    memory_dim: int = 128
    time_dim: int = 16
    embedding_dim: int = 64
    num_heads: int = 2
    num_layers: int = 1
    dropout: float = 0.108
    message_dim: int = 32

@dataclass
class TrainingConfig:
    epochs: int = 200
    lr: float = 0.002
    batch_size: int = 32
    checkpoint_interval: int = 5
    patience: int = 20
    num_seeds: int = 5
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    completion_loss_weight: float = 0.5

@dataclass
class ExperimentConfig:
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    experiment_name: str
    output_dir: str

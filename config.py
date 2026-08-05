from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    atom_dim: int = 11
    bond_dim: int = 6
    protein_vocab_size: int = 26
    hidden_dim: int = 256
    gine_layers: int = 4
    mamba_layers: int = 2
    mamba_state_dim: int = 16
    token_count: int = 5
    token_temperature: float = 0.2
    prior_dim: int = 64
    attention_dim: int = 128
    prior_bias: float = 0.5
    csd_weight: float = 0.1
    dropout: float = 0.1

    def as_dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    batch_size: int = 16
    epochs: int = 120
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    patience: int = 20
    seed: int = 2026
    threshold: float = 0.5

    def as_dict(self):
        return asdict(self)

from dataclasses import dataclass


@dataclass(frozen=True)
class HyperParameters:
    learning_rate: float = 0.0001
    gamma: float = 0.99
    entropy_beta: float = 1e-4
    reward_steps: int = 4


@dataclass(frozen=True)
class Config:
    environment: str = 'CarRacing-v0'
    num_episodes: int = 1000
    batch_size: int = 1024


hyper_parameters = HyperParameters()
config = Config()

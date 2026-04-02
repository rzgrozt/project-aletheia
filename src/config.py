"""Configuration dataclasses for the Fast Weight Programmer network.

These configurations parameterize the network architecture and training loop.
Many parameters have direct biological analogues in synaptic plasticity and
neural circuit dynamics, noted in the field docstrings below.

References:
    Ba, J., Hinton, G., Mnih, V., Leibo, J. Z., & Ionescu, C. (2016).
        Using Fast Weights to Attend to the Recent Past.
    Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020).
        Transformers are RNNs: Fast Autoregressive Transformers with
        Linear Attention.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Architecture configuration for the Fast Weight Programmer.

    Attributes:
        d_model: Dimensionality of token embeddings and hidden states.
            Analogous to the width of a cortical column — higher values
            allow richer distributed representations.
        n_head: Number of parallel attention heads. Each head maintains
            its own fast-weight memory matrix, analogous to independent
            synaptic sub-populations encoding different relational patterns.
        n_layers: Number of stacked transformer blocks. Corresponds to
            the depth of hierarchical processing stages in the cortical
            hierarchy (V1 -> V2 -> V4 -> IT in visual cortex).
        d_ff: Inner dimensionality of the position-wise feed-forward
            network. Acts as an expansion factor for nonlinear feature
            mixing between attention stages.
        vocab_size: Size of the token vocabulary (number of discrete
            input symbols). No default — must be set per dataset.
        decay_rate: Lambda for exponential Hebbian decay (0 < lambda <= 1).
            Models the biological phenomenon of synaptic fading: without
            reinforcement, synaptic traces exponentially decay toward zero.
            A value of 0.95 means each memory trace retains 95% of its
            strength per timestep, providing a ~20-step effective horizon.
        epsilon: Small constant for numerical stability in the normalizer
            denominator, preventing division by zero when the accumulated
            key mass is negligible — analogous to a baseline tonic firing
            rate ensuring neurons remain responsive.
        dropout: Dropout probability applied to attention weights and
            feed-forward activations. Models stochastic synaptic failure,
            which in biological systems promotes robust, redundant coding.
        max_len: Maximum sequence length for learnable positional embeddings.
            Defines the upper bound of the temporal receptive field — analogous
            to the maximum span of working memory in biological systems.
    """

    d_model: int = 128
    n_head: int = 4
    n_layers: int = 4
    d_ff: int = 512
    vocab_size: int = 0  # Must be set before use
    max_len: int = 512
    decay_rate: float = 0.95
    epsilon: float = 1e-6
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer")
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_head ({self.n_head})"
            )
        if not (0.0 < self.decay_rate <= 1.0):
            raise ValueError(
                f"decay_rate must be in (0, 1], got {self.decay_rate}"
            )


@dataclass
class TrainConfig:
    """Training loop configuration.

    Attributes:
        batch_size: Number of sequences processed in parallel per
            gradient step.
        block_size: Maximum sequence length (context window). Determines
            how far back the fast-weight memory can attend — analogous
            to the temporal horizon of working memory.
        learning_rate: Step size for the optimizer. Controls the rate of
            slow-weight (long-term) parameter updates, in contrast to
            the fast-weight (short-term) updates within forward passes.
        epochs: Number of full passes over the training dataset.
        grad_clip: Maximum gradient norm for clipping. Prevents
            catastrophic weight updates analogous to excitotoxicity
            in biological systems.
        device: Compute device string (e.g. 'cuda', 'cpu', 'mps').
    """

    batch_size: int = 32
    block_size: int = 256
    learning_rate: float = 3e-4
    epochs: int = 10
    grad_clip: float = 1.0
    device: str = "cuda"

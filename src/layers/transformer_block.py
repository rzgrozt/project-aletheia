"""Transformer Block — assembling the dual-pathway processing unit.

This module defines the building blocks that compose the full Fast Weight
Programmer stack: a feed-forward network (static pathway) and a complete
transformer block combining fast-weight attention with feed-forward processing.

Biological Analogy — Dual Pathway Architecture:

    The brain processes information through complementary pathways operating
    on different timescales and with different computational roles:

    1. **Fast Weight Attention** (Dynamic / Episodic pathway):
       Analogous to the hippocampal system and prefrontal working memory —
       rapidly binds novel associations within a single episode via Hebbian
       plasticity.  This is the "what just happened" pathway.

    2. **Feed-Forward Network** (Static / Reflexive pathway):
       Analogous to the neocortical feedforward hierarchy — applies stable,
       learned nonlinear transformations that extract features and patterns
       consolidated from long-term experience.  This is the "what does this
       mean" pathway.

    The Pre-LayerNorm residual design mirrors homeostatic regulation in
    biological circuits: normalization before each processing stage keeps
    activations within a functional range, while residual connections allow
    raw signals to bypass processing stages (analogous to thalamic relay
    nuclei providing skip connections across cortical layers).

References:
    Ba et al. (2016) - Using Fast Weights to Attend to the Recent Past
    Katharopoulos et al. (2020) - Transformers are RNNs
    Xiong et al. (2020) - On Layer Normalization in the Transformer Architecture
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch.nn as nn
from torch import Tensor

from src.config import ModelConfig
from src.layers.fast_weight import FastWeightAttention
from src.types import FastWeightState


class FeedForward(nn.Module):
    """Position-wise feed-forward network — the Static/Reflexive pathway.

    This is the "long-term memory processing" component of each transformer
    block.  While the Fast Weight Attention pathway captures transient,
    episode-specific associations (short-term potentiation), the feed-forward
    network applies stable nonlinear transformations learned over many
    training episodes (long-term potentiation).

    Biological Analogy:
        The expansion-compression architecture (d_model -> d_ff -> d_model)
        mirrors the divergent-convergent connectivity observed in cortical
        columns: layer 4 receives compressed input, projects broadly to
        layers 2/3 (expansion), which then converge back to layer 5/6
        (compression) for output.  The GELU activation approximates the
        smooth, probabilistic firing curves of biological neurons, unlike
        the hard thresholding of ReLU.

    Architecture:
        Linear(d_model -> d_ff) -> GELU -> Dropout -> Linear(d_ff -> d_model)

    Attributes:
        net: The sequential feed-forward pipeline.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the static feed-forward transformation.

        Args:
            x: Input tensor of shape (B, L, d_model).

        Returns:
            Tensor of shape (B, L, d_model).
        """
        return self.net(x)


class FastWeightBlock(nn.Module):
    """A single transformer block with Pre-LayerNorm and fast-weight attention.

    Combines the dynamic (fast-weight attention) and static (feed-forward)
    pathways into a complete processing unit with residual connections and
    pre-normalization.

    Biological Analogy:
        Each block represents one level of the cortical hierarchy (e.g.,
        V1 -> V2 -> V4 -> IT in the ventral visual stream).  Information
        flows upward through the hierarchy, with each level:

        1. Normalizing its input (homeostatic regulation),
        2. Forming transient associations via fast weights (working memory),
        3. Passing the result through stable feature detectors (long-term memory),
        4. Adding the raw input back via residual connections (thalamic relay).

    Architecture (Pre-LayerNorm):
        x_norm = LayerNorm1(x)
        attn_out, new_state = FastWeightAttention(x_norm, state)
        x = x + Dropout(attn_out)          # Residual 1
        x_norm = LayerNorm2(x)
        ffn_out = FeedForward(x_norm)
        x = x + Dropout(ffn_out)           # Residual 2

    Attributes:
        ln1: First layer normalization (before attention).
        ln2: Second layer normalization (before feed-forward).
        attn: The fast-weight attention mechanism.
        ffn: The static feed-forward network.
        drop: Dropout applied to sub-layer outputs.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.ln1: nn.LayerNorm = nn.LayerNorm(config.d_model)
        self.ln2: nn.LayerNorm = nn.LayerNorm(config.d_model)
        self.attn: FastWeightAttention = FastWeightAttention(config)
        self.ffn: FeedForward = FeedForward(config)
        self.drop: nn.Dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        state: Optional[FastWeightState] = None,
    ) -> Tuple[Tensor, FastWeightState]:
        """Forward pass through one transformer block.

        Args:
            x: Input tensor of shape (B, L, d_model).
            state: Optional recurrent fast-weight state from previous call.
                If ``None``, a fresh zero state is initialized internally.

        Returns:
            A tuple ``(output, new_state)`` where:
                - ``output`` has shape (B, L, d_model).
                - ``new_state`` is the updated :class:`FastWeightState`.
        """
        # --- Dynamic pathway: Fast Weight Attention ---
        x_norm: Tensor = self.ln1(x)
        attn_out: Tensor
        new_state: FastWeightState
        attn_out, new_state = self.attn(x_norm, state)
        x = x + self.drop(attn_out)

        # --- Static pathway: Feed-Forward Network ---
        x_norm = self.ln2(x)
        ffn_out: Tensor = self.ffn(x_norm)
        x = x + self.drop(ffn_out)

        return x, new_state

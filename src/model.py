"""Fast Weight Language Model — the complete Project Aletheia architecture.

This module assembles the full language model by stacking Fast Weight
Transformer blocks with token/positional embeddings and a tied output head.

Biological Analogy — The Complete Neural Circuit:

    The assembled model mirrors a hierarchical neural processing pipeline:

    1. **Token Embedding** (Sensory Encoding):
       Discrete symbols are mapped to dense distributed representations,
       analogous to how retinal ganglion cells encode visual stimuli into
       population codes transmitted to higher cortical areas.

    2. **Positional Embedding** (Temporal Context):
       Learnable position signals provide a sense of "when" in the sequence,
       analogous to hippocampal time cells that encode the temporal position
       of events within an episode.

    3. **Stacked Fast Weight Blocks** (Cortical Hierarchy):
       Each block adds a level of abstraction — from surface-level lexical
       patterns (early layers) to abstract semantic relationships (deep
       layers), mirroring the ventral stream hierarchy (V1 → V2 → V4 → IT).

    4. **Weight-Tied Output Head** (Motor Output / Decision):
       The output projection shares weights with the input embedding,
       reflecting the biological principle that recognition and generation
       share neural substrates (mirror neuron system).  This also acts as
       a powerful regularizer for small-data regimes.

References:
    Ba et al. (2016) - Using Fast Weights to Attend to the Recent Past
    Katharopoulos et al. (2020) - Transformers are RNNs
    Press & Wolf (2017) - Using the Output Embedding to Improve Language Models
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config import ModelConfig
from src.layers.transformer_block import FastWeightBlock
from src.types import FastWeightState


class FastWeightLM(nn.Module):
    """Language model built on stacked Fast Weight Transformer blocks.

    Combines token embeddings, learnable positional embeddings, a stack of
    Pre-LayerNorm Fast Weight Blocks, and a weight-tied output head to form
    a complete autoregressive language model.

    Attributes:
        config: The model configuration dataclass.
        token_embedding: Maps token indices to dense vectors.
        position_embedding: Learnable positional encoding parameter.
        layers: Stack of :class:`FastWeightBlock` modules.
        ln_final: Final layer normalization before the output head.
        head: Linear projection to vocabulary logits (weight-tied).
        drop: Dropout applied after embedding sum.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config: ModelConfig = config

        # --- Sensory Encoding: Token + Position Embeddings ---
        self.token_embedding: nn.Embedding = nn.Embedding(
            config.vocab_size, config.d_model
        )
        self.position_embedding: nn.Parameter = nn.Parameter(
            torch.randn(1, config.max_len, config.d_model) * 0.02
        )
        self.drop: nn.Dropout = nn.Dropout(config.dropout)

        # --- Cortical Hierarchy: Stacked Fast Weight Blocks ---
        self.layers: nn.ModuleList = nn.ModuleList(
            [FastWeightBlock(config) for _ in range(config.n_layers)]
        )

        # --- Final Normalization ---
        self.ln_final: nn.LayerNorm = nn.LayerNorm(config.d_model)

        # --- Output Head (weight-tied with token embedding) ---
        self.head: nn.Linear = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

    def forward(
        self,
        idx: Tensor,
        targets: Optional[Tensor] = None,
        states: Optional[List[Optional[FastWeightState]]] = None,
    ) -> Tuple[Tensor, Optional[Tensor], List[FastWeightState]]:
        """Forward pass: embed, process through hierarchy, predict.

        Args:
            idx: Input token indices of shape (B, L), dtype ``torch.long``.
            targets: Optional target token indices of shape (B, L) for
                computing cross-entropy loss.  If ``None``, loss is not
                computed.
            states: Optional list of per-layer :class:`FastWeightState`.
                Length must equal ``n_layers``.  If ``None``, fresh states
                are initialized for all layers.

        Returns:
            A tuple ``(logits, loss, new_states)`` where:
                - ``logits`` has shape (B, L, vocab_size).
                - ``loss`` is a scalar tensor if ``targets`` is provided,
                  otherwise ``None``.
                - ``new_states`` is a list of :class:`FastWeightState`,
                  one per layer, for continued generation.
        """
        B, L = idx.shape

        # --- Step 1: Embed tokens + add positional encoding ---
        tok_emb: Tensor = self.token_embedding(idx)  # (B, L, d_model)
        pos_emb: Tensor = self.position_embedding[:, :L, :]  # (1, L, d_model)
        x: Tensor = self.drop(tok_emb + pos_emb)

        # --- Step 2: Initialize per-layer states if needed ---
        if states is None:
            states = [None] * self.config.n_layers

        # --- Step 3: Process through cortical hierarchy ---
        new_states: List[FastWeightState] = []
        for i, layer in enumerate(self.layers):
            x, layer_state = layer(x, states[i])
            new_states.append(layer_state)

        # --- Step 4: Final normalization ---
        x = self.ln_final(x)

        # --- Step 5: Project to vocabulary logits ---
        logits: Tensor = self.head(x)  # (B, L, vocab_size)

        # --- Step 6: Compute loss if targets provided ---
        loss: Optional[Tensor] = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss, new_states


if __name__ == "__main__":
    # --- Full Model Assembly Verification ---
    print("=" * 60)
    print("Project Aletheia — Full Model Verification")
    print("=" * 60)

    # Create configuration
    config: ModelConfig = ModelConfig(
        vocab_size=256,
        d_model=128,
        n_head=4,
        n_layers=4,
        d_ff=512,
        max_len=512,
        dropout=0.1,
    )
    print(f"\nConfig: {config}")

    # Instantiate model
    model: FastWeightLM = FastWeightLM(config)
    n_params: int = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    # Create dummy inputs
    B: int = 2
    L: int = 50
    idx: Tensor = torch.randint(0, config.vocab_size, (B, L))
    targets: Tensor = torch.randint(0, config.vocab_size, (B, L))

    # Forward pass with loss
    logits: Tensor
    loss: Optional[Tensor]
    new_states: List[FastWeightState]
    logits, loss, new_states = model(idx, targets=targets)

    # --- Assertions ---
    assert logits.shape == (B, L, config.vocab_size), (
        f"Logits shape mismatch: expected {(B, L, config.vocab_size)}, "
        f"got {logits.shape}"
    )
    assert loss is not None, "Loss should be computed when targets provided"
    assert loss.dim() == 0, f"Loss should be scalar, got dim={loss.dim()}"
    assert len(new_states) == config.n_layers, (
        f"States list length mismatch: expected {config.n_layers}, "
        f"got {len(new_states)}"
    )

    # Verify weight tying
    assert model.head.weight is model.token_embedding.weight, (
        "Weight tying failed: head and embedding weights are not shared"
    )

    # Verify gradient flow
    loss.backward()
    grad_ok: bool = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
        if p.requires_grad
    )
    assert grad_ok, "Gradients did not flow through all parameters"

    # Forward pass without targets (inference mode)
    logits_inf: Tensor
    loss_inf: Optional[Tensor]
    logits_inf, loss_inf, _ = model(idx)
    assert loss_inf is None, "Loss should be None when no targets provided"

    # Forward pass with state continuation
    logits2: Tensor
    logits2, _, new_states2 = model(idx, states=new_states)
    assert logits2.shape == (B, L, config.vocab_size), "Stateful forward failed"

    # --- Report ---
    print(f"\nOutput Logits Shape : {logits.shape}")
    print(f"Loss Value          : {loss.item():.4f}")
    print(f"States List Length  : {len(new_states)}")
    print(f"Weight Tying        : OK")
    print(f"Gradient Flow       : OK")
    print(f"Stateful Forward    : OK")
    print(f"Inference (no loss) : OK")
    print(f"\n{'=' * 60}")
    print("Full Model Verification Passed")
    print(f"{'=' * 60}")

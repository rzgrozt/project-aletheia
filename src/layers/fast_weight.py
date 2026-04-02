"""Fast Weight Attention — the core engine of Project Aletheia.

This module implements the linearized attention mechanism with recurrent
fast-weight updates, following Ba et al. (2016) and Katharopoulos et al. (2020).

Biological Analogy — Slow Weights vs. Fast Weights:

    In biological neural circuits, learning operates on two timescales:

    1. **Slow Weights** (``nn.Linear`` projections W_Q, W_K, W_V, W_O):
       Analogous to long-term synaptic strength shaped by gradient descent
       over many episodes — like long-term potentiation (LTP) that encodes
       stable knowledge in the connectome over hours to years.

    2. **Fast Weights** (the recurrent ``memory_matrix`` S_t):
       Analogous to short-term potentiation (STP) and activity-dependent
       synaptic facilitation that forms *within a single episode*.  These
       transient traces allow the network to rapidly bind novel associations
       (e.g., "the key is under the mat") without modifying the slow weights.

    The forward pass thus mirrors a biological circuit where stable synaptic
    pathways (slow weights) project inputs into a representational space,
    while a rapidly-updating Hebbian memory (fast weights) captures the
    transient relational structure of the current context.

The Neuromodulated Hebbian Update Rule:
    w_t = σ(W_w · x_t)                 (dopaminergic write gate)
    e_t = σ(W_e · x_t)                 (active erase gate)
    S_t = (1 - e_t) ⊙ (λ · S_{t-1}) + w_t ⊙ (v_t ⊗ k_t)
    z_t = (1 - e_t) ⊙ (λ · z_{t-1}) + w_t ⊙ k_t
    y_t = S_t · q_t / (z_t^T · q_t + ε) (memory retrieval)

    where λ is the exponential decay rate, w_t is the write gate (salience),
    and e_t is the erase gate (active clearance).

References:
    Ba, J., Hinton, G., Mnih, V., Leibo, J. Z., & Ionescu, C. (2016).
        Using Fast Weights to Attend to the Recent Past. NeurIPS.
    Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020).
        Transformers are RNNs: Fast Autoregressive Transformers with
        Linear Attention. ICML.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config import ModelConfig
from src.types import FastWeightState


class FeatureMap(nn.Module):
    """Non-negative feature map: φ(x) = elu(x) + 1.

    Standard softmax attention computes:
        Attention(Q, K, V) = softmax(Q K^T / √d) V

    The kernel trick (Katharopoulos et al., 2020) replaces this with:
        Attention(Q, K, V) ≈ φ(Q) (φ(K)^T V) / (φ(Q) φ(K)^T 1)

    The feature map φ must be **non-negative** so that the resulting
    kernel k(q, k) = φ(q)^T φ(k) ≥ 0, which:

    1. Guarantees a valid (positive semi-definite) kernel, allowing us
       to interpret attention weights as a probability-like distribution
       over keys — analogous to how biological synaptic weights are
       non-negative conductances.

    2. Ensures numerical stability in the recurrent accumulation: since
       all contributions to the memory matrix S_t are non-negative, the
       normalizer z_t^T q_t remains positive, preventing sign flips and
       catastrophic cancellation during long sequences.

    The ELU+1 choice specifically maps the negative domain smoothly to
    (0, 1) and the positive domain to (1, ∞), preserving gradient flow
    while maintaining strict positivity.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Apply the non-negative feature map.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor of same shape with values in (0, ∞).
        """
        return F.elu(x) + 1.0


class FastWeightAttention(nn.Module):
    """Multi-head linearized attention with recurrent fast-weight memory.

    This module replaces the standard O(L²) softmax attention with an
    O(L·d²) recurrent formulation.  At each timestep t, the network:

    1. Projects the input through slow-weight matrices to obtain Q, K, V.
    2. Applies a non-negative feature map φ to Q and K.
    3. Updates the fast-weight memory matrix via the Hebbian rule:
           S_t = λ · S_{t-1} + v_t ⊗ k_t
       This is the "cells that fire together, wire together" principle:
       the value v_t (post-synaptic activity) is associated with the
       key k_t (pre-synaptic activity) through their outer product.
    4. Retrieves a value by querying the memory:
           y_t = S_t · q_t / (z_t^T · q_t + ε)

    The explicit sequential loop over timesteps makes the "Fast Weight
    Programmer" interpretation transparent: the network is literally
    *programming* its own weight matrix S_t at each step.

    Neuromodulatory Gating (Phase 1 — Biological Realism):

        The static decay λ is replaced by learned, input-dependent gates
        that modulate the write and erase dynamics of working memory:

        - **Write gate** (``w_write``): Acts as a **dopaminergic salience
          signal**.  In biological circuits, dopamine release in the
          prefrontal cortex gates whether a new stimulus is "important
          enough" to be stored in working memory.  The write gate σ(W_w · x_t)
          scales the Hebbian outer-product association, allowing the
          network to selectively strengthen storage for salient inputs.

        - **Erase gate** (``w_erase``): Models **active clearance of
          working memory**.  Biological prefrontal circuits can actively
          suppress or flush outdated information (e.g., when task context
          switches).  The erase gate σ(W_e · x_t) attenuates the existing
          memory before the new association is written, enabling the
          network to "forget on demand" rather than relying solely on
          passive exponential decay.

        The gated Hebbian update becomes:
            S_t = (1 - e_t) ⊙ (λ · S_{t-1}) + w_t ⊙ (v_t ⊗ k_t)
            z_t = (1 - e_t) ⊙ (λ · z_{t-1}) + w_t ⊙ k_t

        where w_t = σ(W_w · x_t) and e_t = σ(W_e · x_t) are per-head
        scalar gates derived from the raw input before Q/K/V projection.

    Attributes:
        d_head: Dimensionality per attention head.
        n_head: Number of parallel attention heads.
        decay_rate: Exponential decay λ for Hebbian memory fading.
        epsilon: Numerical stability constant for normalization.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.d_model: int = config.d_model
        self.n_head: int = config.n_head
        self.d_head: int = config.d_model // config.n_head
        self.decay_rate: float = config.decay_rate
        self.epsilon: float = config.epsilon

        # --- Slow Weights (learned via gradient descent / LTP) ---
        # These linear projections are the stable, long-term synaptic
        # connections that transform raw inputs into the query / key / value
        # representational space.
        self.w_query: nn.Linear = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_key: nn.Linear = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_value: nn.Linear = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_out: nn.Linear = nn.Linear(config.d_model, config.d_model, bias=False)

        # --- Neuromodulatory Gates (learned via gradient descent) ---
        # These gates modulate the write/erase dynamics of the fast-weight
        # memory, replacing the static decay with input-dependent gating.
        #
        # w_write: Dopaminergic salience signal — determines whether the
        #   current input should be stored in working memory.  In biological
        #   prefrontal cortex, dopamine D1 receptor activation gates the
        #   maintenance of new representations in persistent activity.
        #   Output shape: (B, H) — one scalar gate per attention head.
        #
        # w_erase: Active memory clearance — determines whether existing
        #   working memory contents should be flushed.  Models the
        #   biological mechanism by which prefrontal interneuron activity
        #   can reset sustained firing patterns when task context changes.
        #   Output shape: (B, H) — one scalar gate per attention head.
        self.w_write: nn.Linear = nn.Linear(config.d_model, config.n_head)
        self.w_erase: nn.Linear = nn.Linear(config.d_model, config.n_head)

        # Non-negative feature map for kernel linearization
        self.feature_map: FeatureMap = FeatureMap()

    def _init_state(self, batch_size: int, device: torch.device) -> FastWeightState:
        """Create a zero-initialized fast-weight state.

        Analogous to a "blank slate" synaptic matrix at the start of a
        new episode — no prior associations have been formed.

        Args:
            batch_size: Number of sequences in the batch.
            device: Torch device for tensor allocation.

        Returns:
            A :class:`FastWeightState` with zeroed memory and normalizer.
        """
        memory_matrix: Tensor = torch.zeros(
            batch_size, self.n_head, self.d_head, self.d_head,
            device=device,
        )
        normalizer: Tensor = torch.zeros(
            batch_size, self.n_head, self.d_head,
            device=device,
        )
        return FastWeightState(memory_matrix=memory_matrix, normalizer=normalizer)

    def _step_fast_weights(
        self,
        k_t: Tensor,
        v_t: Tensor,
        state: FastWeightState,
        write_gate: Tensor,
        erase_gate: Tensor,
    ) -> FastWeightState:
        """Perform one neuromodulated Hebbian update step on the fast-weight memory.

        Implements the gated Hebbian rule with dopaminergic write control
        and active memory clearance:

            S_t = (1 - e_t) ⊙ (λ · S_{t-1}) + w_t ⊙ (v_t ⊗ k_t)
            z_t = (1 - e_t) ⊙ (λ · z_{t-1}) + w_t ⊙ k_t

        The write gate w_t models dopaminergic salience — only associations
        deemed "important" by the learned gating network are stored.  The
        erase gate e_t models active clearance — the network can flush
        stale memories when context shifts, rather than waiting for passive
        exponential decay.

        Args:
            k_t: Key vector at timestep t, shape (B, H, d_head).
            v_t: Value vector at timestep t, shape (B, H, d_head).
            state: Previous fast-weight state.
            write_gate: Dopaminergic write signal, shape (B, H).
                Values in [0, 1] from sigmoid.  Reshaped internally to
                (B, H, 1, 1) for memory matrix and (B, H, 1) for normalizer
                to broadcast across d_head dimensions.
            erase_gate: Active clearance signal, shape (B, H).
                Values in [0, 1] from sigmoid.  Same reshaping as write_gate.

        Returns:
            Updated :class:`FastWeightState`.
        """
        # Hebbian outer product: v_t ⊗ k_t  →  (B, H, d_head, d_head)
        # v_t: (B, H, d_head) → (B, H, d_head, 1)
        # k_t: (B, H, d_head) → (B, H, 1, d_head)
        association: Tensor = v_t.unsqueeze(-1) * k_t.unsqueeze(-2)

        # Reshape gates for broadcasting:
        #   memory_matrix is (B, H, d_head, d_head) → gates need (B, H, 1, 1)
        #   normalizer is    (B, H, d_head)          → gates need (B, H, 1)
        write_gate_mem: Tensor = write_gate.unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
        erase_gate_mem: Tensor = erase_gate.unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
        write_gate_norm: Tensor = write_gate.unsqueeze(-1)               # (B, H, 1)
        erase_gate_norm: Tensor = erase_gate.unsqueeze(-1)               # (B, H, 1)

        # Gated Hebbian update:
        #   Erase: attenuate existing memory (active clearance)
        #   Write: selectively store new association (dopaminergic gating)
        new_memory: Tensor = (
            (1.0 - erase_gate_mem) * (self.decay_rate * state.memory_matrix)
            + write_gate_mem * association
        )
        new_normalizer: Tensor = (
            (1.0 - erase_gate_norm) * (self.decay_rate * state.normalizer)
            + write_gate_norm * k_t
        )

        return FastWeightState(memory_matrix=new_memory, normalizer=new_normalizer)

    def _forward_parallel(
        self,
        x: Tensor,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
    ) -> Tuple[Tensor, FastWeightState]:
        """Parallel forward pass for training (no sequential loop).

        Replaces the O(L) sequential recurrence with O(L²) parallel matrix
        operations that fully saturate the GPU.  The gated decay recurrence
        is unrolled into a causal decay matrix computed via cumulative sums
        in log-space, allowing the entire sequence to be processed in one
        batched matmul.

        This is mathematically equivalent to the sequential loop when no
        prior state is carried over (fresh episode).

        Args:
            x: Raw input, shape (B, L, D).
            Q: Feature-mapped queries,  shape (B, H, L, d_head).
            K: Feature-mapped keys,     shape (B, H, L, d_head).
            V: Values (unmapped),       shape (B, H, L, d_head).

        Returns:
            ``(output, dummy_state)`` — output is (B, L, D); state is
            zeroed (not meaningful during training).
        """
        B, L, D = x.shape

        # --- Neuromodulatory gates for every position (parallel) ---
        # x: (B, L, D) → gates: (B, L, H) → transpose → (B, H, L)
        write_gate: Tensor = torch.sigmoid(self.w_write(x)).transpose(1, 2)  # (B, H, L)
        erase_gate: Tensor = torch.sigmoid(self.w_erase(x)).transpose(1, 2)  # (B, H, L)

        # --- Parallel Gated Decay Matrix ---
        # The sequential recurrence  S_t = (1-e_t)·λ·S_{t-1} + w_t·(v_t⊗k_t)
        # has effective per-step decay  g_t = (1-e_t)·λ.
        # The cumulative decay from step j to step i (i >= j) is
        #   Π_{s=j+1}^{i} g_s  =  exp( Σ_{s=j+1}^{i} log g_s )
        # which we compute as  exp( C_i - C_j )  via a prefix sum in
        # log-space.
        g: Tensor = (1.0 - erase_gate) * self.decay_rate          # (B, H, L)
        log_g: Tensor = torch.log(g + 1e-8)
        C: Tensor = torch.cumsum(log_g, dim=-1)                   # (B, H, L)

        # decay_matrix[i, j] = exp(C_i - C_j)  for i >= j, else 0
        C_i: Tensor = C.unsqueeze(3)                               # (B, H, L, 1)
        C_j: Tensor = C.unsqueeze(2)                               # (B, H, 1, L)
        decay_logits: Tensor = C_i - C_j                          # (B, H, L, L)

        # Mask future positions with -inf BEFORE exp() so they evaluate
        # to exactly 0.  The old approach (exp first, then multiply by 0)
        # produced inf * 0 = NaN when C_i - C_j was large and positive.
        causal_mask: Tensor = torch.tril(
            torch.ones(L, L, device=x.device, dtype=x.dtype),
        ).unsqueeze(0).unsqueeze(0)                                # (1, 1, L, L)
        decay_logits = decay_logits.masked_fill(causal_mask == 0, float('-inf'))
        decay_matrix: Tensor = torch.exp(decay_logits)             # (B, H, L, L)

        # --- Parallel attention scores ---
        # scores[i, j] = φ(q_i)^T φ(k_j)
        scores: Tensor = torch.matmul(Q, K.transpose(-1, -2))     # (B, H, L, L)

        # Apply write gate (scales the *key* side: association strength)
        # write_gate: (B, H, L) → (B, H, 1, L)  broadcasts over query dim
        effective_scores: Tensor = scores * write_gate.unsqueeze(2) * decay_matrix

        # --- Retrieve and normalize ---
        retrieved: Tensor = torch.matmul(effective_scores, V)      # (B, H, L, d_head)
        denominator: Tensor = (
            effective_scores.sum(dim=-1, keepdim=True) + self.epsilon
        )                                                          # (B, H, L, 1)
        output: Tensor = retrieved / denominator                   # (B, H, L, d_head)

        # --- Reshape back to (B, L, D) ---
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.w_out(output)

        # Dummy state — not used during training, satisfies API contract
        dummy_state: FastWeightState = self._init_state(B, x.device)
        return output, dummy_state

    def _forward_sequential(
        self,
        x: Tensor,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        state: FastWeightState,
    ) -> Tuple[Tensor, FastWeightState]:
        """Sequential forward pass for autoregressive inference.

        Preserves the original recurrent loop that updates the fast-weight
        memory matrix step-by-step.  Used when a prior ``state`` is carried
        forward or when ``L == 1`` (single-token generation).

        Args:
            x: Raw input, shape (B, L, D).
            Q: Feature-mapped queries,  shape (B, H, L, d_head).
            K: Feature-mapped keys,     shape (B, H, L, d_head).
            V: Values (unmapped),       shape (B, H, L, d_head).
            state: Recurrent fast-weight state from the previous call.

        Returns:
            ``(output, final_state)`` with shapes (B, L, D) and
            :class:`FastWeightState`.
        """
        B, L, D = x.shape
        outputs: List[Tensor] = []

        for t in range(L):
            q_t: Tensor = Q[:, :, t, :]  # (B, H, d_head)
            k_t: Tensor = K[:, :, t, :]
            v_t: Tensor = V[:, :, t, :]

            x_t: Tensor = x[:, t, :]  # (B, D)
            write_gate: Tensor = torch.sigmoid(self.w_write(x_t))  # (B, H)
            erase_gate: Tensor = torch.sigmoid(self.w_erase(x_t))  # (B, H)

            state = self._step_fast_weights(k_t, v_t, state, write_gate, erase_gate)

            retrieved: Tensor = torch.matmul(
                state.memory_matrix, q_t.unsqueeze(-1)
            ).squeeze(-1)  # (B, H, d_head)

            denominator: Tensor = (
                torch.sum(state.normalizer * q_t, dim=-1, keepdim=True)
                + self.epsilon
            )  # (B, H, 1)

            y_t: Tensor = retrieved / denominator  # (B, H, d_head)
            outputs.append(y_t)

        output: Tensor = torch.stack(outputs, dim=2)  # (B, H, L, d_head)
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.w_out(output)

        return output, state

    def forward(
        self,
        x: Tensor,
        state: Optional[FastWeightState] = None,
    ) -> Tuple[Tensor, FastWeightState]:
        """Forward pass: project, update fast weights, retrieve.

        Uses a **parallel** path during training (no prior state, L > 1)
        to fully saturate the GPU via batched matmuls, and falls back to
        the **sequential** recurrent loop during autoregressive inference
        (prior state or L == 1).

        Args:
            x: Input tensor of shape (B, L, d_model).
            state: Optional recurrent state from a previous call.
                If ``None``, a fresh zero state is created (new episode).

        Returns:
            A tuple ``(output, final_state)`` where:
                - ``output`` has shape (B, L, d_model).
                - ``final_state`` is the :class:`FastWeightState` after
                  processing the full sequence, ready to be fed into
                  the next call for continued generation.
        """
        B, L, D = x.shape

        # --- Slow-weight projections ---
        Q: Tensor = self.w_query(x)  # (B, L, D)
        K: Tensor = self.w_key(x)
        V: Tensor = self.w_value(x)

        # Reshape to multi-head: (B, L, D) → (B, L, H, d_head) → (B, H, L, d_head)
        Q = Q.view(B, L, self.n_head, self.d_head).transpose(1, 2)
        K = K.view(B, L, self.n_head, self.d_head).transpose(1, 2)
        V = V.view(B, L, self.n_head, self.d_head).transpose(1, 2)

        # Apply non-negative feature map
        Q = self.feature_map(Q)  # (B, H, L, d_head)
        K = self.feature_map(K)

        # --- Branching: Parallel Training vs. Sequential Inference ---
        is_training: bool = (state is None and L > 1)

        if is_training:
            return self._forward_parallel(x, Q, K, V)
        else:
            if state is None:
                state = self._init_state(B, x.device)
            return self._forward_sequential(x, Q, K, V, state)


if __name__ == "__main__":
    from src.config import ModelConfig

    # --- Verification ---
    cfg = ModelConfig(vocab_size=256, d_model=128, n_head=4)
    layer = FastWeightAttention(cfg)

    B, L, D = 2, 10, cfg.d_model
    x = torch.randn(B, L, D)

    # === 1. Parallel path (training): no state, L > 1 ===
    output, final_state = layer(x)

    assert output.shape == (B, L, D), (
        f"[parallel] Output shape mismatch: expected {(B, L, D)}, got {output.shape}"
    )
    assert final_state.memory_matrix.shape == (B, cfg.n_head, cfg.d_model // cfg.n_head, cfg.d_model // cfg.n_head), (
        f"[parallel] Memory matrix shape mismatch: {final_state.memory_matrix.shape}"
    )
    assert final_state.normalizer.shape == (B, cfg.n_head, cfg.d_model // cfg.n_head), (
        f"[parallel] Normalizer shape mismatch: {final_state.normalizer.shape}"
    )

    # Verify gradients flow through the parallel path
    loss = output.sum()
    loss.backward()
    grad_ok = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in layer.parameters()
    )
    assert grad_ok, "[parallel] Gradients did not flow through all parameters"
    layer.zero_grad()

    # === 2. Sequential path (inference): with state ===
    output2, final_state2 = layer(x, state=final_state)
    assert output2.shape == (B, L, D), "[sequential] Stateful forward shape mismatch"

    # Verify state is non-trivially updated
    assert final_state2.memory_matrix.abs().sum() > 0, "[sequential] Memory matrix is all zeros"
    assert final_state2.normalizer.abs().sum() > 0, "[sequential] Normalizer is all zeros"

    # Verify gradients flow through the sequential path
    loss2 = output2.sum()
    loss2.backward()
    grad_ok2 = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in layer.parameters()
    )
    assert grad_ok2, "[sequential] Gradients did not flow through all parameters"
    layer.zero_grad()

    # === 3. Sequential path: single token (L=1, no state) ===
    x_single: Tensor = torch.randn(B, 1, D)
    output3, state3 = layer(x_single)
    assert output3.shape == (B, 1, D), "[single-token] Output shape mismatch"
    assert state3.memory_matrix.abs().sum() > 0, "[single-token] Memory matrix is all zeros"

    print(f"Parallel  output shape : {output.shape}")
    print(f"Sequential output shape: {output2.shape}")
    print(f"Single-token shape     : {output3.shape}")
    print(f"Gradient flow (parallel)  : OK")
    print(f"Gradient flow (sequential): OK")
    print(f"\nFast Weight Layer Verification Passed (parallel + sequential)")

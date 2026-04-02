"""Core type definitions for the Fast Weight Programmer.

This module defines the recurrent state types that flow through the
linearized attention mechanism. The design mirrors biological fast-weight
synaptic plasticity: at each timestep, the network updates a Hebbian
memory matrix (outer-product association) and a normalizer (accumulated
key mass), which together implement a content-addressable associative
memory.

The state is structured as a NamedTuple for immutability and clarity —
each forward step produces a *new* state rather than mutating in place,
following the functional style recommended for recurrent computations
in PyTorch.

References:
    Ba et al. (2016) - Fast Weights to Attend to the Recent Past
    Katharopoulos et al. (2020) - Transformers are RNNs
"""

from typing import NamedTuple

from torch import Tensor


class FastWeightState(NamedTuple):
    """Recurrent state for a single linearized attention head.

    This state is passed from one timestep to the next, accumulating
    key-value associations in a manner analogous to Hebbian synaptic
    modification in biological neural circuits.

    Attributes:
        memory_matrix: The Hebbian synaptic matrix of shape
            (batch, d_head, d_head). Stores the accumulated outer
            products of (transformed) keys and values:
                W_t = lambda * W_{t-1} + v_t @ k_t^T
            This is the "fast weight" matrix — a rapidly-updated
            associative memory analogous to short-term potentiation
            (STP) at biological synapses. Querying with a key vector
            retrieves the associated value via matrix-vector product.

        normalizer: The normalization denominator of shape
            (batch, d_head). Accumulates the transformed key vectors:
                z_t = lambda * z_{t-1} + k_t
            Used to normalize retrieved values, ensuring that the
            output magnitude remains stable regardless of sequence
            length — analogous to synaptic scaling / homeostatic
            plasticity that maintains neural firing rates within a
            functional range.
    """

    memory_matrix: Tensor
    normalizer: Tensor

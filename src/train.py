"""Training loop and inference for Project Aletheia — the Fast Weight Programmer.

This module implements the complete training pipeline and stateful text
generation, demonstrating the key advantage of the linearized attention
architecture: O(1) per-step inference via recurrent fast-weight state.

Biological Analogy — Learning and Recall:

    Training (Slow Weight Update):
        Each gradient step adjusts the slow-weight parameters (W_Q, W_K, W_V,
        W_O, FFN weights) via backpropagation — analogous to long-term
        potentiation (LTP) that consolidates knowledge over many exposures.
        The optimizer (AdamW) acts as a homeostatic mechanism, maintaining
        stable learning rates per-parameter (adaptive moment estimation)
        while weight decay prevents runaway synaptic strengthening.

    Generation (Fast Weight Recall):
        At inference time, the model processes a prompt to "prime" its
        fast-weight memory matrices (context priming), then generates
        token-by-token by feeding only the latest token and the accumulated
        state.  This mirrors how biological working memory maintains a
        compressed representation of recent context without re-processing
        the entire episode — the synaptic traces *are* the context.

    The O(1) per-step generation cost (vs. O(L) for standard Transformers)
    directly follows from the recurrent formulation: the fast-weight matrix
    S_t already encodes all prior key-value associations, so querying it
    with a new key requires only a matrix-vector product, not a scan over
    all previous keys.

References:
    Ba et al. (2016) - Using Fast Weights to Attend to the Recent Past
    Katharopoulos et al. (2020) - Transformers are RNNs
    Loshchilov & Hutter (2019) - Decoupled Weight Decay Regularization (AdamW)
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.config import ModelConfig, TrainConfig
from src.data.loader import Tokenizer, create_dataloaders
from src.model import FastWeightLM
from src.types import FastWeightState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
_BANNER: str = r"""
================================================================
    ___    __     __  __           _
   /   |  / /__  / /_/ /_  ___   (_)___ _
  / /| | / / _ \/ __/ __ \/ _ \ / / __ `/
 / ___ |/ /  __/ /_/ / / /  __// / /_/ /
/_/  |_/_/\___/\__/_/ /_/\___//_/\__,_/

    Fast Weight Programmer — Training Protocol
================================================================
"""


# ---------------------------------------------------------------------------
# Generation — O(1) per-step recurrent inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model: FastWeightLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    device: torch.device = torch.device("cpu"),
) -> str:
    """Generate text using stateful fast-weight inference.

    Demonstrates the key advantage of the linearized attention architecture:
    after an initial O(L) context-priming pass, each subsequent token is
    generated in O(1) time by feeding only the last token and the accumulated
    fast-weight state.

    Biological Analogy:
        Context priming loads the "episode" into working memory (fast weights).
        Generation then proceeds as a stream-of-consciousness recall — each
        new output token modifies the synaptic landscape, influencing the
        next retrieval, without re-scanning the entire episode.

    Args:
        model: Trained :class:`FastWeightLM` in eval mode.
        tokenizer: The :class:`Tokenizer` used during training.
        prompt: Seed text to prime the fast-weight memory.
        max_new_tokens: Number of new tokens to generate.
        temperature: Sampling temperature.  Lower values produce more
            deterministic output; higher values increase diversity.
        device: Torch device for computation.

    Returns:
        The generated text (prompt + continuation) as a string.
    """
    model.eval()

    # --- Tokenize prompt ---
    prompt_ids: List[int] = tokenizer.encode(prompt)
    if len(prompt_ids) == 0:
        prompt_ids = [0]  # fallback to <unk> if prompt is empty

    input_ids: Tensor = torch.tensor(
        [prompt_ids], dtype=torch.long, device=device
    )  # (1, L_prompt)

    # --- Context Priming: build fast-weight state from full prompt ---
    # This is the O(L_prompt) pass — processes the entire prompt to
    # accumulate Hebbian associations in the fast-weight matrices.
    logits: Tensor
    states: List[FastWeightState]
    logits, _, states = model(input_ids)

    # Take logits for the last prompt token to predict the first new token
    next_logits: Tensor = logits[:, -1, :]  # (1, vocab_size)

    generated_ids: List[int] = list(prompt_ids)

    # --- Generation Loop: O(1) per step ---
    for _ in range(max_new_tokens):
        # Apply temperature scaling and sample
        scaled_logits: Tensor = next_logits / max(temperature, 1e-8)
        probs: Tensor = torch.softmax(scaled_logits, dim=-1)
        next_token: Tensor = torch.multinomial(probs, num_samples=1)  # (1, 1)

        token_id: int = next_token.item()  # type: ignore[assignment]
        generated_ids.append(token_id)

        # Feed ONLY the new token + previous state → O(1) per step
        # The fast-weight matrices already encode all prior context.
        next_logits_tuple: tuple = model(next_token, states=states)
        next_logits = next_logits_tuple[0][:, -1, :]  # (1, vocab_size)
        states = next_logits_tuple[2]

    return tokenizer.decode(generated_ids)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(
    train_config: TrainConfig,
    model_config: ModelConfig,
    data_dir: str = "data",
    checkpoint_dir: str = "checkpoints",
) -> FastWeightLM:
    """Train the Fast Weight Language Model on WikiText-2.

    Implements a standard training loop with:
    - AdamW optimizer with cosine learning rate schedule
    - Gradient clipping for RNN stability
    - Validation with perplexity tracking
    - Best-model checkpointing
    - Per-epoch text generation sanity checks

    Biological Analogy:
        Each epoch is an "exposure" to the full training corpus.  The slow
        weights gradually consolidate statistical patterns (LTP), while
        gradient clipping prevents excitotoxic weight explosions.  The
        cosine schedule models a natural decay of learning plasticity —
        large updates early (critical period), fine-tuning later (mature).

    Args:
        train_config: Training hyperparameters.
        model_config: Model architecture hyperparameters.  ``vocab_size``
            will be overwritten by the dataset's actual vocabulary size.
        data_dir: Directory for dataset storage/download.
        checkpoint_dir: Directory for saving model checkpoints.

    Returns:
        The trained :class:`FastWeightLM` model.
    """
    print(_BANNER)
    print("Initializing Project Aletheia Training Protocol...\n")

    # --- Device setup ---
    device: torch.device = torch.device(
        train_config.device
        if torch.cuda.is_available() or train_config.device == "cpu"
        else "cpu"
    )
    print(f"Device           : {device}")

    # --- Data pipeline ---
    print("Loading data...")
    train_loader: DataLoader
    val_loader: DataLoader
    vocab_size: int
    train_loader, val_loader, vocab_size = create_dataloaders(
        train_config, data_dir=data_dir
    )
    print(f"Vocab size       : {vocab_size:,}")
    print(f"Train batches    : {len(train_loader):,}")
    print(f"Val batches      : {len(val_loader):,}")

    # Retrieve tokenizer for generation sanity checks
    tokenizer: Tokenizer = train_loader.dataset.tokenizer  # type: ignore[union-attr]

    # --- Model ---
    model_config.vocab_size = model_config.__class__(
        **{**model_config.__dict__, "vocab_size": vocab_size}
    ).vocab_size  # re-validate via __post_init__

    # Re-instantiate to apply validations properly
    model_config = ModelConfig(
        d_model=model_config.d_model,
        n_head=model_config.n_head,
        n_layers=model_config.n_layers,
        d_ff=model_config.d_ff,
        vocab_size=vocab_size,
        max_len=model_config.max_len,
        decay_rate=model_config.decay_rate,
        epsilon=model_config.epsilon,
        dropout=model_config.dropout,
    )
    model: FastWeightLM = FastWeightLM(model_config).to(device)

    n_params: int = sum(p.numel() for p in model.parameters())
    print(f"Model parameters : {n_params:,}")
    print(f"Config           : {model_config}\n")

    # --- Optimizer & Scheduler ---
    optimizer: AdamW = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=1e-2,
    )
    total_steps: int = len(train_loader) * train_config.epochs
    scheduler: CosineAnnealingLR = CosineAnnealingLR(
        optimizer, T_max=total_steps
    )

    # --- Checkpoint directory ---
    ckpt_path: Path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    best_val_loss: float = float("inf")

    # --- Training loop ---
    print("=" * 60)
    print("Beginning Training")
    print("=" * 60)

    for epoch in range(1, train_config.epochs + 1):
        epoch_start: float = time.time()

        # ---- Train ----
        model.train()
        train_loss_sum: float = 0.0
        train_batches: int = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits: Tensor
            loss: Optional[Tensor]
            logits, loss, _ = model(x, targets=y)
            assert loss is not None

            loss.backward()

            # Gradient clipping — vital for RNN / recurrent stability
            nn.utils.clip_grad_norm_(
                model.parameters(), train_config.grad_clip
            )

            optimizer.step()
            scheduler.step()

            train_loss_sum += loss.item()
            train_batches += 1

            if (batch_idx + 1) % 50 == 0:
                avg_loss: float = train_loss_sum / train_batches
                lr_current: float = scheduler.get_last_lr()[0]
                print(
                    f"  Epoch {epoch}/{train_config.epochs} | "
                    f"Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {lr_current:.2e}"
                )

        avg_train_loss: float = train_loss_sum / max(train_batches, 1)

        # ---- Validation ----
        model.eval()
        val_loss_sum: float = 0.0
        val_batches: int = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                _, loss, _ = model(x, targets=y)
                assert loss is not None
                val_loss_sum += loss.item()
                val_batches += 1

        avg_val_loss: float = val_loss_sum / max(val_batches, 1)
        val_perplexity: float = math.exp(min(avg_val_loss, 20.0))  # cap to avoid overflow

        epoch_time: float = time.time() - epoch_start

        print(
            f"\n  Epoch {epoch}/{train_config.epochs} Summary:\n"
            f"    Train Loss   : {avg_train_loss:.4f}\n"
            f"    Val Loss     : {avg_val_loss:.4f}\n"
            f"    Val Perplexity: {val_perplexity:.2f}\n"
            f"    Time         : {epoch_time:.1f}s"
        )

        # ---- Checkpoint ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path: Path = ckpt_path / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "model_config": model_config.__dict__,
                    "train_config": train_config.__dict__,
                },
                save_path,
            )
            print(f"    Checkpoint saved (val_loss={avg_val_loss:.4f})")

        # ---- Sanity Check: Generate sample text ----
        sample_prompt: str = "The meaning of life is"
        sample_text: str = generate(
            model,
            tokenizer,
            prompt=sample_prompt,
            max_new_tokens=30,
            temperature=0.8,
            device=device,
        )
        print(f"\n    Generation sample:\n    > {sample_text}\n")
        print("-" * 60)

    # --- Final report ---
    print("\n" + "=" * 60)
    print("Training Complete")
    print(f"  Best Val Loss  : {best_val_loss:.4f}")
    print(f"  Best Perplexity: {math.exp(min(best_val_loss, 20.0)):.2f}")
    print("=" * 60)

    return model


# ---------------------------------------------------------------------------
# Main Execution (CLI Interface)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Project Aletheia: Fast Weight Programmer")

    # Directory arguments
    parser.add_argument("--data_dir", type=str, default="data", help="Directory for dataset storage")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory for saving models")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per step")
    parser.add_argument("--block_size", type=int, default=128, help="Context window length (sequence length)")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Peak learning rate for AdamW")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device (cuda/cpu/mps)")

    # Model architecture arguments
    parser.add_argument("--d_model", type=int, default=128, help="Dimensionality of token embeddings")
    parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of transformer blocks")
    parser.add_argument("--d_ff", type=int, default=512, help="Feed-forward network inner dimension")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum positional encoding length")
    parser.add_argument("--decay_rate", type=float, default=0.95, help="Hebbian memory exponential decay rate (lambda)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # --- Configuration ---
    train_cfg: TrainConfig = TrainConfig(
        batch_size=args.batch_size,
        block_size=args.block_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        grad_clip=args.grad_clip,
        device=args.device,
    )

    model_cfg: ModelConfig = ModelConfig(
        d_model=args.d_model,
        n_head=args.n_head,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        vocab_size=1,  # placeholder — overwritten by dataset vocab
        max_len=args.max_len,
        decay_rate=args.decay_rate,
        dropout=args.dropout,
    )

    trained_model: FastWeightLM = train(
        train_config=train_cfg,
        model_config=model_cfg,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
    )

<div align="center">

```
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║     █████╗ ██╗     ███████╗████████╗██╗  ██╗███████╗██╗ █████╗    ║
║    ██╔══██╗██║     ██╔════╝╚══██╔══╝██║  ██║██╔════╝██║██╔══██╗   ║
║    ███████║██║     █████╗     ██║   ███████║█████╗  ██║███████║   ║
║    ██╔══██║██║     ██╔══╝     ██║   ██╔══██║██╔══╝  ██║██╔══██║   ║
║    ██║  ██║███████╗███████╗   ██║   ██║  ██║███████╗██║██║  ██║   ║
║    ╚═╝  ╚═╝╚══════╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝╚═╝  ╚═╝   ║
║                                                                   ║
║          F A S T   W E I G H T   P R O G R A M M E R              ║
║                  — Project Aletheia —                             ║
╚═══════════════════════════════════════════════════════════════════╝
```

*ἀλήθεια — the uncovering of what is hidden; the emergence of truth.*

---

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Built with Passion](https://img.shields.io/badge/Built_with-Passion-ff6b6b?style=for-the-badge&logo=heart&logoColor=white)]()
[![AI-Challenged](https://img.shields.io/badge/Challenged-Current_AI_Limits-blueviolet?style=for-the-badge&logo=openai&logoColor=white)]()

</div>

---

## ✦ The Story Behind This Project

> *"The code did not come from my hands — it came from my mind."*
> — Ruzgar Ozturk

This project was born from a specific kind of ambition: to build something at the genuine intersection of **neuroscience**, **machine learning theory**, and **engineering** — and to use the challenge as a deliberate test of what current AI can do when pushed into non-trivial mathematical and architectural territory.

**Project Aletheia is a proof of concept and a personal statement.** The architecture, the biological analogies, the mathematical formulations, the research choices — all of it originated as a mental model. The translation into code was a collaborative experiment with AI, a challenge issued to test whether modern language models could faithfully implement a real cognitive-science-grounded architecture without collapsing into generic solutions.

They could. Barely. It required the kind of precise, uncompromising guidance that only comes from someone who actually understands the domain.

**The code was not written by human hands. But the vision, the architecture, and the intellectual authorship are entirely human — entirely mine.**

---

## ✦ What Is Project Aletheia?

Project Aletheia implements a **Fast Weight Programmer** — a language model architecture where memory operates on **two distinct timescales**, mirroring biological neural circuits:

| Timescale | Biological Analogue | Implementation |
|-----------|---------------------|----------------|
| **Slow Weights** | Long-term potentiation (LTP), stable synaptic strength shaped over episodes | Standard `nn.Linear` projections W_Q, W_K, W_V — updated via gradient descent |
| **Fast Weights** | Short-term potentiation (STP), episode-specific Hebbian traces | Recurrent `memory_matrix` S_t — updated within a single forward pass |

The key insight: a standard Transformer discards its context once it ends. This model **stores it** — not by re-reading tokens, but by maintaining a compressed associative memory that evolves with every token it processes.

This yields **O(1) per-step inference** versus the O(L) cost of standard attention — a computational and conceptual advantage that mirrors how biological working memory actually functions.

---

## ✦ Architecture

```
INPUT TOKENS
     │
     ▼
┌─────────────────────────────────────┐
│       Token Embedding               │  ← Sensory encoding
│   + Learnable Positional Encoding   │  ← Hippocampal time cells
└────────────────┬────────────────────┘
                 │
     ┌───────────▼────────────┐
     │   Fast Weight Block ×N │  ← Cortical hierarchy (V1→IT)
     │ ┌─────────────────────┐│
     │ │  Pre-LayerNorm      ││  ← Homeostatic regulation
     │ │  FastWeightAttention││  ← Hippocampal/episodic pathway
     │ │   ┌──────────────┐  ││
     │ │   │ FeatureMap φ │  ││  ← Non-negative kernel: ELU + 1
     │ │   │ W_t update   │  ││  ← Hebbian write gate (dopamine)
     │ │   │ e_t erase    │  ││  ← Active forgetting
     │ │   │ S_t memory   │  ││  ← Fast weight matrix
     │ │   └──────────────┘  ││
     │ │  + Residual         ││
     │ │  Pre-LayerNorm      ││
     │ │  FeedForward (FFN)  ││  ← Neocortical/reflexive pathway
     │ │  + Residual         ││
     │ └─────────────────────┘│
     └───────────┬────────────┘
                 │
     ┌───────────▼────────────┐
     │   Final LayerNorm      │
     │   Output Head (tied)   │  ← Weight-tied with embedding
     └───────────┬────────────┘
                 │
                 ▼
           LOGITS / LOSS
```

### The Neuromodulated Hebbian Update Rule

At every timestep *t*, the fast-weight memory updates as:

```
φ(·)  =  ELU(·) + 1                         (non-negative feature map)

w_t   =  σ(W_w · x_t)                       (write gate — salience signal)
e_t   =  σ(W_e · x_t)                       (erase gate — active forgetting)

S_t   =  (1 − e_t) ⊙ (λ · S_{t−1})  +  w_t ⊙ (v_t ⊗ k_t)
z_t   =  (1 − e_t) ⊙ (λ · z_{t−1})  +  w_t ⊙ k_t

y_t   =  S_t · q_t  /  (z_t^T · q_t  +  ε)  (normalized memory retrieval)
```

Where λ is the **exponential decay rate** — modelling synaptic fading. A value of 0.95 gives approximately a 20-step effective memory horizon. The erase gate provides active clearance; the write gate selects what is worth remembering.

---

## ✦ Project Structure

```
project-aletheia/
│
├── src/
│   ├── __init__.py
│   ├── config.py              # ModelConfig & TrainConfig dataclasses
│   ├── model.py               # FastWeightLM — full assembled model
│   ├── train.py               # Training loop + stateful text generation
│   ├── types.py               # FastWeightState (memory_matrix, normalizer)
│   │
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── fast_weight.py     # FastWeightAttention + FeatureMap — the core
│   │   └── transformer_block.py  # FeedForward + FastWeightBlock
│   │
│   └── data/
│       ├── __init__.py
│       └── loader.py          # WikiText-2 downloader, tokenizer, DataLoader
│
├── checkpoints/               # Saved model states
├── data/                      # Dataset cache (auto-downloaded)
├── generate.py                # Standalone text generation script
├── requirements.txt
├── CLAUDE.md
└── README.md
```

---

## ✦ Getting Started

### Requirements

```bash
pip install -r requirements.txt
```

**Core dependencies:**

| Package | Purpose |
|---------|---------|
| `torch >= 2.0` | Model, autograd, DataLoader |
| `numpy` | Numerical utilities |

No HuggingFace, no heavy frameworks. The data pipeline downloads WikiText-2 directly and tokenizes it in pure Python.

### Training

```bash
python -m src.train \
  --epochs 10 \
  --batch_size 32 \
  --block_size 256 \
  --lr 3e-4 \
  --device cuda
```

The training loop uses **AdamW** with **Cosine Annealing LR** and gradient clipping (norm = 1.0), with checkpoint saving at configurable intervals.

### Text Generation

```bash
python generate.py \
  --checkpoint checkpoints/model_epoch10.pt \
  --prompt "The theory of" \
  --max_tokens 200 \
  --temperature 0.8
```

Generation is **stateful** — the model primes its fast-weight matrices from the prompt, then generates token-by-token using only the accumulated state. No re-reading of context. Pure recurrent inference.

---

## ✦ Key Design Decisions

### Why Weight Tying?
The output head shares weights with the token embedding matrix (`head.weight is token_embedding.weight`). This reflects the biological principle that recognition and generation share neural substrates — the mirror neuron system — and acts as a powerful regularizer for data-limited regimes.

### Why ELU + 1 as the Feature Map?
The kernel trick replacing softmax requires a **non-negative** feature map to guarantee:
1. A valid positive semi-definite kernel (probability-like attention weights)
2. Numerical stability in recurrent accumulation (no sign flips)

`φ(x) = ELU(x) + 1` maps the negative domain smoothly to `(0, 1)` and positive domain to `(1, ∞)`, preserving gradient flow while maintaining strict positivity.

### Why Pre-LayerNorm?
Pre-LayerNorm (normalization *before* each sub-layer) provides more stable training dynamics than the original Post-LayerNorm formulation — mirroring homeostatic regulation in biological circuits that keeps activations within functional range.

---

## ✦ References

> Ba, J., Hinton, G., Mnih, V., Leibo, J. Z., & Ionescu, C. (2016).
> **Using Fast Weights to Attend to the Recent Past.** *NeurIPS.*

> Katharopoulos, A., Vyas, A., Pappas, N., & Fleuret, F. (2020).
> **Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.** *ICML.*

> Press, O., & Wolf, L. (2017).
> **Using the Output Embedding to Improve Language Models.** *EACL.*

> Loshchilov, I., & Hutter, F. (2019).
> **Decoupled Weight Decay Regularization (AdamW).** *ICLR.*

> Merity, S., Xiong, C., Bradbury, J., & Socher, R. (2016).
> **Pointer Sentinel Mixture Models.** *(WikiText-2 dataset)*

> Xiong, R., Yang, Y., He, D., et al. (2020).
> **On Layer Normalization in the Transformer Architecture.** *ICML.*

---

## ✦ Attribution & License

**Intellectual Author:** Ruzgar Ozturk
*(Psychology & Cognitive Science, AI & Data Analysis)*

This project was conceived, designed, and directed by Ruzgar Ozturk.
The architecture, biological framing, research grounding, and all conceptual decisions originate from the author.
AI tools were used strictly as implementation assistants under the author's direction.

> *The code came from AI. The mind behind it is human.*

```
MIT License

Copyright (c) 2025 Ruzgar Ozturk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

Attribution Requirement: Any public use, derivative work, or academic
reference must credit the original author: Ruzgar Ozturk — Project Aletheia.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---

<div align="center">

*"What the slow weights know, the fast weights remember."*

**Project Aletheia** — built with passion, grounded in science.

</div>

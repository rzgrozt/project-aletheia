"""
Project Aletheia: Neural Inference Interface
--------------------------------------------
This script initializes a trained FastWeightLM and provides a terminal-based
chat interface to interact with the model's working memory.

Biological Analogy:
    The "Prompt" primes the Prefrontal Cortex (working memory).
    The "Generation" is a stream-of-consciousness recall where the synaptic
    landscape (Fast Weights) shifts with every word spoken.
"""

import os
import torch
import logging
from typing import List, Optional
from src.model import FastWeightLM
from src.config import ModelConfig
from src.data.loader import Tokenizer, create_dataloaders
from src.config import TrainConfig
from src.types import FastWeightState

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_aletheia_model(checkpoint_path: str, device: torch.device):
    """Loads the model and configuration from a saved checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    logger.info(f"Loading brain states from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct configs
    model_cfg_dict = checkpoint['model_config']
    model_cfg = ModelConfig(**model_cfg_dict)

    # Initialize architecture
    model = FastWeightLM(model_cfg).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info("Aletheia is awake and optimized for O(1) inference.")
    return model, checkpoint['train_config']

@torch.no_grad()
def interactive_chat():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "checkpoints/best_model.pth"

    try:
        model, train_cfg_dict = load_aletheia_model(checkpoint_path, device)
        train_cfg = TrainConfig(**train_cfg_dict)

        # We need the tokenizer. In a real-world scenario, you'd save the vocab
        # as a separate JSON, but here we can quickly rebuild it from the loaders.
        logger.info("Reconstructing vocabulary from data directory...")
        _, _, vocab_size = create_dataloaders(train_cfg, data_dir="data")

        # Assuming the tokenizer is part of the dataset object as built in train.py
        from src.data.loader import WikiTextDataset
        tokenizer = Tokenizer()
        # Note: In a production repo, you should save/load tokenizer.json
        # Here we re-read the training file to ensure vocab mapping is identical.
        with open("data/wikitext-2-raw/wiki.train.raw", 'r', encoding='utf-8') as f:
            tokenizer.build(f.read())

    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return

    print("\n" + "="*60)
    print(" PROJECT ALETHEIA: INTERACTIVE INFERENCE ".center(60, "="))
    print("="*60)
    print("Type your prompt and press Enter. Type 'exit' to quit.")
    print("The model uses a gated Hebbian memory for context retention.")
    print("-"*60)

    while True:
        prompt = input("\n[USER]> ")
        if prompt.lower() in ['exit', 'quit']:
            break

        print("[ALETHEIA]> ", end="", flush=True)

        # 1. Tokenize prompt
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        # 2. Context Priming (O(L) pass)
        # This pass sets the initial fast-weight matrix state based on the prompt.
        logits, _, states = model(input_ids)
        next_token_logits = logits[:, -1, :]

        # 3. Stream Generation (O(1) recurrent steps)
        # We only feed the LAST generated token + the CARRY-OVER state.
        for _ in range(50): # Generate up to 50 tokens
            probs = torch.softmax(next_token_logits / 0.8, dim=-1) # Temp = 0.8
            next_token = torch.multinomial(probs, num_samples=1)

            word = tokenizer.decode([next_token.item()])
            print(word + " ", end="", flush=True)

            if word == ".": # Stop at sentence end for brevity
                break

            # Recurrent update: Only feed (B=1, T=1) + current synaptic state
            logits, _, states = model(next_token, states=states)
            next_token_logits = logits[:, -1, :]

        print("\n" + "-"*30)

if __name__ == "__main__":
    interactive_chat()

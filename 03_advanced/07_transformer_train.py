"""
19. Transformer Training Pipeline in PyTorch

This script demonstrates how to configure and train a complete sequence-to-sequence 
Transformer model. It implements causal self-attention masks (look-ahead masks) 
to prevent looking at future tokens during decoding, generates synthetic translation 
sentence batches, and verifies gradient updates.

Learning Objectives:
1. Construct causal look-ahead and source padding masks.
2. Build training pipelines for Sequence-to-Sequence (Seq2Seq) tasks.
3. Validate tensor shapes across Transformer encoders and decoders.
"""

# We import the core torch library.
import torch

# nn contains loss metrics (CrossEntropyLoss).
import torch.nn as nn

# optim contains optimization modules (Adam).
import torch.optim as optim

# We import importlib to dynamically load our Transformer module.
import importlib
model_module = importlib.import_module("03_advanced.06_transformer_model")
Transformer = model_module.Transformer

# ==========================================
# 1. CAUSAL MASK GENERATION UTILITIES
# ==========================================
def make_src_mask(src, src_pad_idx=0):
    """Generates a source mask that ignores padding indices."""
    # src shape: [Batch_Size, Src_Seq_Len]
    # Output shape: [Batch_Size, 1, 1, Src_Seq_Len]
    return (src != src_pad_idx).unsqueeze(1).unsqueeze(2)

def make_trg_mask(trg, trg_pad_idx=0):
    """Generates a target mask combining padding masking and causal look-ahead masking."""
    # trg shape: [Batch_Size, Trg_Seq_Len]
    batch_size, trg_len = trg.size()
    
    # 1.1 Padding mask: [Batch_Size, 1, 1, Trg_Seq_Len]
    trg_pad_mask = (trg != trg_pad_idx).unsqueeze(1).unsqueeze(2)
    
    # 1.2 Look-ahead causal mask: a lower-triangular matrix of ones.
    # Prevents step 't' from attending to future steps > 't' during autoregressive decoding.
    # Documentation: https://pytorch.org/docs/stable/generated/torch.tril.html
    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
        batch_size, 1, trg_len, trg_len
    ).to(trg.device)
    
    # Combined mask requires BOTH conditions to be True (non-padded and not future)
    return trg_pad_mask & (trg_sub_mask == 1)

def main():
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Running Transformer Seq2Seq Training on: {device} ---")

    # ==========================================
    # 2. INITIALIZE TOY DATA AND HYPERPARAMETERS
    # ==========================================
    # Vocabulary parameters
    src_vocab_size = 10  # Toy French vocab size
    trg_vocab_size = 12  # Toy English vocab size
    
    # Instantiate custom Transformer
    model = Transformer(
        src_vocab_size=src_vocab_size, 
        trg_vocab_size=trg_vocab_size, 
        d_model=64, 
        num_heads=4, 
        num_layers=1
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding in loss
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 2.1 Setup 2 synthetic sentence pairs (e.g., French -> English translation)
    # Batch size = 2, Max Seq length = 5
    src_sentences = torch.tensor([
        [1, 3, 4, 5, 2],  # French sentence 1
        [1, 3, 6, 2, 0]   # French sentence 2 (padded with 0 at the end)
    ], dtype=torch.long).to(device)
    
    trg_sentences = torch.tensor([
        [1, 7, 8, 9, 2],  # English translation 1
        [1, 7, 10, 2, 0]  # English translation 2 (padded with 0)
    ], dtype=torch.long).to(device)

    # ==========================================
    # 3. TRANSFOMER TRAINING ROUTINE
    # ==========================================
    print("--- Simulating Seq2Seq Optimization Epochs ---")
    model.train()
    
    epochs = 20
    for epoch in range(epochs):
        # Generate masking criteria
        src_mask = make_src_mask(src_sentences)
        trg_mask = make_trg_mask(trg_sentences)
        
        # Forward pass: targets shifted by 1 token for training predictions
        outputs = model(src_sentences, trg_sentences, src_mask, trg_mask)
        
        # Reshape for cross entropy mapping
        loss = criterion(outputs.view(-1, trg_vocab_size), trg_sentences.view(-1))
        
        # Optimize parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Translation Loss: {loss.item():.4f}")

    # ==========================================
    # 4. TRANSLATION VALIDATION CHECK
    # ==========================================
    model.eval()
    with torch.no_grad():
        src_mask = make_src_mask(src_sentences)
        trg_mask = make_trg_mask(trg_sentences)
        outputs = model(src_sentences, trg_sentences, src_mask, trg_mask)
        # Select the most probable English token indexes
        predictions = torch.argmax(outputs, dim=-1)
        
        print("\nTranslation Verification Results:")
        print(f"  Predicted Translation Batch 1: {predictions[0].tolist()}")
        print(f"  Expected Translation Batch 1: {trg_sentences[0].tolist()}")

if __name__ == "__main__":
    main()

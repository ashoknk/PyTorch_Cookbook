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
import sys
import importlib
sys.path.append(".")  # Adds the current working directory to Python's module search path

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


def decode_tokens(token_ids, id_to_token):
    """Convert token IDs to readable tokens, omitting padding and boundary markers."""
    special_tokens = {"<pad>", "<sos>", "<eos>"}
    return " ".join(
        id_to_token[token_id]
        for token_id in token_ids
        if id_to_token[token_id] not in special_tokens
    )


def greedy_decode(model, src, src_mask, start_idx, end_idx, max_new_tokens):
    """Generate target tokens one at a time, feeding each prediction back in."""
    generated = torch.full(
        (src.size(0), 1), start_idx, dtype=torch.long, device=src.device
    )
    finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            trg_mask = make_trg_mask(generated)
            logits = model(src, generated, src_mask, trg_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            next_token = next_token.masked_fill(finished.unsqueeze(1), 0)
            generated = torch.cat((generated, next_token), dim=1)
            finished |= next_token.squeeze(1) == end_idx
            if torch.all(finished):
                break

    return generated


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Running Transformer Seq2Seq Training on: {device} ---")

    # ==========================================
    # 2. INITIALIZE TOY DATA AND HYPERPARAMETERS
    # ==========================================
    src_id_to_token = {
        0: "<pad>",
        1: "<sos>",
        2: "<eos>",
        3: "j'aime",
        4: "les",
        5: "chats",
        6: "chiens",
    }
    trg_id_to_token = {
        0: "<pad>",
        1: "<sos>",
        2: "<eos>",
        3: "I",
        4: "like",
        5: "cats",
        6: "dogs",
    }
    src_vocab_size = len(src_id_to_token)
    trg_vocab_size = len(trg_id_to_token)
    
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

    # 2.1 Two hand-written toy pairs, not a real translation dataset.
    # Batch size = 2, Max sequence length = 6.
    src_sentences = torch.tensor([
        [1, 3, 4, 5, 2, 0],  # "j'aime les chats", padded
        [1, 3, 4, 6, 2, 0]   # "j'aime les chiens", padded
    ], dtype=torch.long).to(device)
    
    trg_sentences = torch.tensor([
        [1, 3, 4, 5, 2, 0],  # "I like cats", padded
        [1, 3, 4, 6, 2, 0]   # "I like dogs", padded
    ], dtype=torch.long).to(device)

    # ==========================================
    # 3. TRANSFOMER TRAINING ROUTINE
    # ==========================================
    print("--- Simulating Seq2Seq Optimization Epochs ---")
    model.train()
    
    epochs = 200
    for epoch in range(epochs):
        # Generate masking criteria
        src_mask = make_src_mask(src_sentences)
        
        # Feed the target prefix and predict the next token at every position.
        trg_input = trg_sentences[:, :-1]
        trg_expected = trg_sentences[:, 1:]
        trg_mask = make_trg_mask(trg_input)
        outputs = model(src_sentences, trg_input, src_mask, trg_mask)
        
        # Reshape for cross entropy mapping
        loss = criterion(
            outputs.reshape(-1, trg_vocab_size), trg_expected.reshape(-1)
        )
        
        # Optimize parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Translation Loss: {loss.item():.4f}")

    # ==========================================
    # 4. TRANSLATION VALIDATION CHECK
    # ==========================================
    src_mask = make_src_mask(src_sentences)
    predictions = greedy_decode(
        model,
        src_sentences,
        src_mask,
        start_idx=1,
        end_idx=2,
        max_new_tokens=trg_sentences.size(1) - 1,
    )

    print("\nSource vocabulary (class ID -> token):")
    for token_id, token in src_id_to_token.items():
        print(f"  {token_id}: {token}")

    print("\nTarget vocabulary (class ID -> token):")
    for token_id, token in trg_id_to_token.items():
        print(f"  {token_id}: {token}")

    print("\nGreedy translation results (generated one token at a time):")
    for batch_index, predicted_ids in enumerate(predictions.tolist()):
        source_ids = src_sentences[batch_index].tolist()
        expected_ids = trg_sentences[batch_index].tolist()
        print(f"  Batch {batch_index + 1}:")
        print(f"    Source:       {decode_tokens(source_ids, src_id_to_token)}")
        print(f"    Predicted IDs: {predicted_ids}")
        print(f"    Predicted:    {decode_tokens(predicted_ids, trg_id_to_token)}")
        print(f"    Expected IDs: {expected_ids}")
        print(f"    Expected:     {decode_tokens(expected_ids, trg_id_to_token)}")

if __name__ == "__main__":
    main()

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
"""
Generates a source mask that ignores padding indices.
   Creates a binary mask that flags blank padding tokens in the French source sentence.
   This prevents the Encoder from wasting computational effort paying attention to empty space."""
def make_src_mask(src, src_pad_idx=0):
    # src shape: [Batch_Size, Src_Seq_Len]
    # Output shape: [Batch_Size, 1, 1, Src_Seq_Len]
    return (src != src_pad_idx).unsqueeze(1).unsqueeze(2)

"""
Generates a target mask combining padding masking and causal look-ahead masking.
   Creates a combined mask for the target English sentence that hides both padding tokens 
   and future unseen words (causal look-ahead mask). This prevents the Decoder from 
   "cheating" by looking ahead at upcoming words during step-by-step generation.
 
1. What torch.tril does:
   Generates a lower-triangular matrix of 1s (e.g., for trg_len=3):
   [ [1, 0, 0],
     [1, 1, 0],
     [1, 1, 1] ]
   - Row 1 (Word 1): Can only attend to position 1.
   - Row 2 (Word 2): Can attend to positions 1 and 2.
   - Row 3 (Word 3): Can attend to positions 1, 2, and 3.

2. What .expand(batch_size, 1, trg_len, trg_len) does:
   Broadcasts the 2D triangular mask into a 4D tensor matching the shape 
   expected by Multi-Head Attention: [Batch, 1, Trg_Len, Trg_Len].

3. Final Combination (trg_pad_mask & (trg_sub_mask == 1)):
   Ensures attention is allowed ONLY if a token is both non-padded AND not in the future.

   """
def make_trg_mask(trg, trg_pad_idx=0):
    
    # trg shape: [Batch_Size, Trg_Seq_Len]
    batch_size, trg_len = trg.size()
    
    # 1.1 Padding mask: [Batch_Size, 1, 1, Trg_Seq_Len]
    trg_pad_mask = (trg != trg_pad_idx).unsqueeze(1).unsqueeze(2)
    
    # 1.2 Look-ahead causal mask: a lower-triangular matrix of ones.
    # Prevents step 't' from attending to future steps > 't' during autoregressive decoding.
    # Documentation: https://pytorch.org/docs/stable/generated/torch.tril.html
    #torch.tril: Returns the lower triangular part of the matrix (2-D tensor) , the other elements of the result tensor out are set to 0.
    #   This prevents the model from looking ahead at future words during training.
    # expand(batch_size, 1, trg_len, trg_len): matrix across the entire batch (and head dimensions) without duplicating memory.
    #   If batch_size = 2, the shape becomes [2, 1, 3, 3], repeating that same lower-triangular mask for every sentence in the batch.
    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
        batch_size, 1, trg_len, trg_len
    ).to(trg.device)
    
    # Combined mask requires BOTH conditions to be True (non-padded and not future)
    return trg_pad_mask & (trg_sub_mask == 1)

""" 
Convert token IDs to readable tokens, omitting padding and boundary markers.
   Takes an array of predicted integer token IDs, filters out non-word structural 
   markers (<pad>, <sos>, <eos>), and converts the remaining numbers into a clean, 
   human-readable English sentence string."""
def decode_tokens(token_ids, id_to_token):
    special_tokens = {"<pad>", "<sos>", "<eos>"}
    return " ".join(
        id_to_token[token_id]
        for token_id in token_ids
        if id_to_token[token_id] not in special_tokens
    )

"""
Simulates real-world translation (Inference) where true answer sentences do not exist.

Instead of getting the answer beforehand, the function generates translations word-by-word:
    1. Starts with an initial <sos> (start of sentence) token for every item in the batch.
    2. Applies make_trg_mask to hide future tokens.
    3. Asks the model to predict the next single highest-probability word token.
    4. Appends that newly predicted word back into the input sequence and feeds it right 
    back into the model to predict the next word.
    5. Repeats this loop step-by-step until every sentence generates an <eos> (end of sentence) token.

1. torch.full((src.size(0), 1), start_idx): Initializes the output sequence container with the <sos> token index for 
     every item in the batch.
   - What `generated` contains: A matrix holding all generated token IDs so far. At step 0, 
     it is just a column vector of start tokens: [[start_idx], [start_idx]].

2. finished = torch.zeros(..., dtype=torch.bool): Tracks which sentences in the batch have completed generation.
   - Why zeros? Boolean 0 represents False, meaning no sentences have finished yet  because we are just starting!

3. logits[:, -1, :].argmax(dim=-1, keepdim=True): Extracts the predicted scores for the VERY LAST predicted word step (index -1) 
     and picks the token ID with the highest score (greedy choice).

4. next_token.masked_fill(finished.unsqueeze(1), 0): Handles completed sentences cleanly. If a sentence in the batch already 
     hit the <eos> (end-of-sequence) marker, its subsequent predicted tokens are 
     overwritten with 0 (the <pad> token index) so it stops generating new words.

5. torch.cat((generated, next_token), dim=1): Appends the newly predicted token to the right side of the existing sequence.
   - Input for the next iteration: The newly expanded `generated` tensor (containing the 
     original start token plus all predicted words so far) is fed directly back into 
     `model(src, generated, ...)` in the next loop cycle so the model can predict word step t+1!

6. finished |= next_token.squeeze(1) == end_idx:
   - Purpose: Checks if the newly predicted token is the <eos> token. If yes, it flips 
     that sentence's `finished` status to True. Once `torch.all(finished)` is True for all 
     sentences in the batch, the loop exits early.
"""
def greedy_decode(model, src, src_mask, start_idx, end_idx, max_new_tokens):
    
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

"""
    == How They Connect to main() and Each Other:== 
    During training in main(), make_src_mask and make_trg_mask ensure the Transformer 
    only learns valid word relationships. After training completes, main() passes the 
    trained model into greedy_decode (which uses make_trg_mask internally) to perform 
    step-by-step translation, and then passes those predictions into decode_tokens 
    to print the final readable English text.""" 
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
        # During training, we use Teacher Forcing. We feed the entire target sentence prefix (trg_input) to the decoder all at once, 
        # masked so it can't see future tokens.
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
        #The 0 at the end of expected IDs is padding, not part of the actual translation. 

if __name__ == "__main__":
    main()

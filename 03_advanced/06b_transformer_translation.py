"""
18 - 6b. Complete Transformer Training Using PyTorch's Built-in nn.Transformer


This script demonstrates how to use PyTorch's built-in nn.Transformer directly for a complete English-French translation task.

1. Utility functions for embeddings, positional encoding, and masking
2. Direct transformer trainer class using torch.nn.Transformer
3. Complete training pipeline with dataset creation
4. Inference pipeline with greedy decoding
5. Full English-French translation example

Learning Objectives:
1. Use torch.nn.Transformer directly without wrappers
2. Understand proper tensor formats and masking
3. Build complete seq2seq translation pipeline
4. Implement teacher forcing and greedy decoding
"""

import os
import warnings

# Set MPS fallback BEFORE importing torch to avoid nested tensor issues
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Optional: Suppress known non-critical warnings
warnings.filterwarnings("ignore", message="Support for mismatched key_padding_mask")
warnings.filterwarnings("ignore", message=".*not currently supported on the MPS backend.*")

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import math

# ==========================================
# 1. POSITIONAL ENCODING
# ==========================================

class PositionalEncoding(nn.Module):
    """
    Adds positional information to every word slot using sinusoidal patterns.
    A Transformer processes all words in a sentence simultaneously in parallel. 
    Because of this, the Transformer has no inherent sense of word order. 
    
    forward(x) : It takes the raw word embeddings x (shape: [Batch, Seq_Len, d_model]) 
        and adds the positional wave patterns directly to them.
    Output : Vectors that contain both what the word means and where the word sits in the sentence.

    Used in Transformer right after converting French and English word IDs into embedding vectors.
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        # torch.arange  returns a 1-D tensor containing evenly spaced values within a specified [start, end]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # torch.exp returns a new tensor with the exponential of the elements of the input tensor.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        # Apply sine to even indices, cosine to odd indices
        # Using a combination of sin and cos gives every single position in the sentence a unique, 
        # fixed signature while making it easy for the model to calculate relative distances between words
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: [max_len, 1, d_model]

        # Register as buffer (saved with model but not updated during training). Saves memory/computation
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
            x: Embeddings tensor [seq_len, batch_size, d_model]
            x.size(0) gets the actual length of your current sentence (e.g., 10 words)
        Returns: Tensor with positional encoding added (both what the word means and where the word is located in the sentence).
        """
        return x + self.pe[:x.size(0), :]

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================

def generate_square_subsequent_mask(sz):
    """
    Generate causal (look-ahead) mask for decoder.
    It forces the decoder to calculate attention using only past and current tokens (< t), 
    while completely blocking access to future tokens (> t).   
    torch.triu (Triangle Upper): Keeps elements on and above the diagonal; sets elements below it to 0
    sz: Sequence length
    Returns:Upper triangular matrix with -inf above diagonal [sz, sz]
    """
    mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return mask

def create_padding_mask(seq, pad_idx):
    """
    Create padding mask for sequences. Do element-wise comparison
        seq: Input sequence [batch_size, seq_len]
        pad_idx: Padding token index
    Returns:Padding mask [batch_size, seq_len] where True indicates padding
    """
    return seq == pad_idx

def create_embeddings(vocab_size, d_model, pad_idx=0):
    """
    Create embedding layer with proper initialization.
        vocab_size: Size of vocabulary
        d_model: Embedding dimension
        pad_idx: Padding token index (will be zero-initialized)
    Returns:nn.Embedding layer with Xavier initialization
    """
    embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
    # Initialize with Xavier uniform (excluding padding). Prevents Vanishing and Exploding Gradients:
    # replaces default random numbers with smartly scaled initial random numbers.
    nn.init.xavier_uniform_(embedding.weight[1:])  # Skip padding index 0
    return embedding

def scale_embeddings(embeddings, d_model):
    """
    Scale embeddings by sqrt(d_model) as per transformer paper.
        embeddings: Embedding tensor [batch_size, seq_len, d_model] or [seq_len, batch_size, d_model]
        d_model: Model dimension

    Multiplying embeddings by sqrt(d_model) scales up the word vector magnitudes, 
    ensuring the model focuses primarily on WHAT the word is, while using 
    position values as a minor hint for WHERE the word sits        

    Returns:Scaled embeddings
    """
    return embeddings * math.sqrt(d_model)

# ==========================================
# 3. DIRECT TRANSFORMER TRAINER CLASS
# ==========================================

class DirectTransformerTrainer(nn.Module):
    """
    Direct usage of torch.nn.Transformer without wrapper classes.

    This class demonstrates how to use PyTorch's transformer directly by managing
    all components (embeddings, positional encoding, transformer, output) separately.
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048,
                 dropout=0.1, pad_idx=0):
        super(DirectTransformerTrainer, self).__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx

        # Create separate components (no wrapper)
        self.src_embedding = create_embeddings(src_vocab_size, d_model, pad_idx)
        self.tgt_embedding = create_embeddings(tgt_vocab_size, d_model, pad_idx)
        self.pos_encoding = PositionalEncoding(d_model)

        # Direct PyTorch Transformer - this is the core component
        # Disable nested tensor optimization to avoid MPS issues
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Use batch_first for better performance
        )

        # Disable nested tensor optimization for MPS compatibility
        # Nested Tensor acceleration is fully supported on modern NVIDIA GPUs (cuda) and x86 CPUs.
        if hasattr(self.transformer, 'enable_nested_tensor'):
            self.transformer.enable_nested_tensor = False
        if hasattr(self.transformer.encoder, 'enable_nested_tensor'):
            self.transformer.encoder.enable_nested_tensor = False
        if hasattr(self.transformer.decoder, 'enable_nested_tensor'):
            self.transformer.decoder.enable_nested_tensor = False

        # Output projection
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None,
                tgt_mask=None):
        """
        Forward pass using torch.nn.Transformer directly.

        Args:
            src: [batch_size, src_len] - source sequences (batch_first=True)
            tgt: [batch_size, tgt_len] - target sequences (batch_first=True)
            src_key_padding_mask: [batch_size, src_len] - source padding mask
            tgt_key_padding_mask: [batch_size, tgt_len] - target padding mask
            tgt_mask: [tgt_len, tgt_len] - causal mask for target
        """
        # Step 1: Apply embeddings and scale
        src_emb = scale_embeddings(self.src_embedding(src), self.d_model)
        tgt_emb = scale_embeddings(self.tgt_embedding(tgt), self.d_model)

        # Step 2: Add positional encoding (need to transpose for pos_encoding)
        src_emb = self.pos_encoding(src_emb.transpose(0, 1)).transpose(0, 1)
        tgt_emb = self.pos_encoding(tgt_emb.transpose(0, 1)).transpose(0, 1)

        # Step 3: Apply PyTorch transformer directly. It performs contextual translation processing
        # It mixes the context of all source words (French) with all generated target words (English).
        transformer_output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_key_padding_mask  # encoder output mask
        )

        # Step 4: Project to vocabulary scores (logits)
        return self.output_proj(transformer_output)

# ==========================================
# 4. DATASET CREATION
# ==========================================

def create_translation_dataset():
    """
    Create a comprehensive English-French translation dataset.
    Returns vocabularies and training data.
    """
    # English-French translation pairs
    translation_pairs = [
        ("I love cats", "J'aime les chats"),
        ("I love dogs", "J'aime les chiens"),
        ("The cat is black", "Le chat est noir"),
        ("The dog is white", "Le chien est blanc"),
        ("She likes music", "Elle aime la musique"),
        ("He reads books", "Il lit des livres"),
        ("We eat apples", "Nous mangeons des pommes"),
        ("They speak French", "Ils parlent français"),
        ("The house is big", "La maison est grande"),
        ("The car is red", "La voiture est rouge"),
        ("I study English", "J'étudie l'anglais"),
        ("You are nice", "Tu es gentil"),
        ("The weather is good", "Le temps est bon"),
        ("I drink water", "Je bois de l'eau"),
        ("She sings well", "Elle chante bien"),
    ]

    # Build vocabularies
    src_vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
    tgt_vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2}

    # Extract unique words
    for eng, fr in translation_pairs:
        for word in eng.lower().split():
            if word not in src_vocab:
                src_vocab[word] = len(src_vocab)
        for word in fr.lower().split():
            if word not in tgt_vocab:
                tgt_vocab[word] = len(tgt_vocab)

    # Create reverse vocabularies
    src_id_to_token = {v: k for k, v in src_vocab.items()}
    tgt_id_to_token = {v: k for k, v in tgt_vocab.items()}

    return translation_pairs, src_vocab, tgt_vocab, src_id_to_token, tgt_id_to_token

def encode_sentence(sentence, vocab, max_len=20):
    """Convert sentence to token IDs with padding."""
    tokens = sentence.lower().split()
    ids = [vocab["<sos>"]] + [vocab[word] for word in tokens] + [vocab["<eos>"]]

    # Pad or truncate to max_len
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len-1] + [vocab["<eos>"]]

    return ids


def prepare_batch(translation_pairs, src_vocab, tgt_vocab, max_len=20):
    """Convert translation pairs to padded tensor batches. Helper function used in main()"""
    src_batch = []
    tgt_batch = []

    for eng, fr in translation_pairs:
        src_ids = encode_sentence(eng, src_vocab, max_len)
        tgt_ids = encode_sentence(fr, tgt_vocab, max_len)
        src_batch.append(src_ids)
        tgt_batch.append(tgt_ids)

    return torch.tensor(src_batch), torch.tensor(tgt_batch)

# ==========================================
# 5. TRAINING FUNCTIONS
# ==========================================

def train_epoch(model, data_loader, optimizer, criterion, device, pad_idx):
    """Train model for one epoch using direct transformer approach."""
    model.train()
    total_loss = 0
    num_batches = 0

    for src_batch, tgt_batch in data_loader:
        src_batch = src_batch.to(device)  # [batch_size, seq_len] - batch_first
        tgt_batch = tgt_batch.to(device)  # [batch_size, seq_len] - batch_first

        # Prepare decoder input (exclude last token) and target (exclude first token)
        tgt_input = tgt_batch[:, :-1]  # [batch_size, seq_len-1]
        tgt_output = tgt_batch[:, 1:]  # [batch_size, seq_len-1]

        # Create masks using utility functions
        src_padding_mask = create_padding_mask(src_batch, pad_idx)
        tgt_padding_mask = create_padding_mask(tgt_input, pad_idx)
        tgt_causal_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)

        # Forward pass with direct transformer
        optimizer.zero_grad()
        output = model(
            src=src_batch,
            tgt=tgt_input,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            tgt_mask=tgt_causal_mask
        )

        # Calculate loss - use reshape instead of view for non-contiguous tensors
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches

"""
  Handles real-world sequence generation (Inference/Evaluation).
  In real life, we do not have the answer beforehand. This function generates 
  a translated sequence word-by-word, taking its own past predictions and feeding 
  them back into the model to predict the next word.

model: evaluates the input sentence and calculate probability scores (logits) for 
   the next candidate word at every step.

create_padding_mask(ys, model.pad_idx) [Padding Mask]:
   Tells the model which slots in 'ys' are actual generated words vs. padding tokens. 
   
STEP-BY-STEP FLOW:
1. Start sequence 'ys' with just the <sos> token: [[start_token]].
2. Loop up to max_len - 1 times:
   a. Create the causal look-ahead mask for 'ys'.
   b. Pass 'src' and 'ys' through the model.
   c. Pick the highest-probability token ID from the last time step (greedy choice).
   d. Append (torch.cat) the predicted token to 'ys'.
   e. Exit early if all sentences hit the <eos> token.

Return the fully generated token sequence 'ys'.
"""
def greedy_decode_direct(model, src, src_mask, max_len, start_token, end_token, device):
    """Greedy decoding using direct transformer approach."""
    model.eval()

    src = src.to(device)
    batch_size = src.size(0)  # batch_first format

    # Start with start token [batch_size, 1]. Create a tensor of a specified shape filled entirely 
    # with `start_token` value
    ys = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)

    with torch.no_grad():
        for i in range(max_len - 1):
            # Create causal mask for current sequence
            tgt_mask = generate_square_subsequent_mask(ys.size(1)).to(device)

            # Create target padding mask (for generated sequence, no padding yet)
            tgt_padding_mask = create_padding_mask(ys, model.pad_idx)

            # Forward pass using direct transformer with proper masks
            out = model(
                src=src,
                tgt=ys,
                src_key_padding_mask=src_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                tgt_mask=tgt_mask
            )

            # Get next token probabilities and select most likely
            prob = out[:, -1, :]  # Last time step [batch_size, vocab_size]
            next_word = prob.argmax(dim=1, keepdim=True)  # [batch_size, 1]

            # Append to sequence
            ys = torch.cat([ys, next_word], dim=1)

            # Check if all sequences have ended
            if (next_word.squeeze() == end_token).all():
                break

    return ys  # [batch_size, seq_len]

def decode_tokens(token_ids, id_to_token):
    """Convert token IDs back to readable text."""
    special_tokens = {"<pad>", "<sos>", "<eos>"}
    return " ".join(
        id_to_token[token_id]
        for token_id in token_ids
        if token_id in id_to_token and id_to_token[token_id] not in special_tokens
    )

# ==========================================
# 6. MAIN TRAINING PIPELINE
# ==========================================

def main():
    """Complete training pipeline using direct PyTorch transformer."""
    # Set seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # For MPS compatibility, fallback is set at import time
    device = torch.device("cuda" if torch.cuda.is_available() else
                         ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"🚀 Training Transformer on device: {device}")

    if device.type == "mps":
        print("💡 MPS fallback enabled for unsupported operations")

    # Create dataset
    translation_pairs, src_vocab, tgt_vocab, src_id_to_token, tgt_id_to_token = create_translation_dataset()

    print(f"📊 Dataset: {len(translation_pairs)} translation pairs")
    print(f"📝 Source vocabulary size: {len(src_vocab)}")
    print(f"📝 Target vocabulary size: {len(tgt_vocab)}")

    # Prepare data
    src_data, tgt_data = prepare_batch(translation_pairs, src_vocab, tgt_vocab)

    # Create simple data loader (using all data as one batch for this example)
    dataset = [(src_data, tgt_data)]

    # Model parameters
    d_model = 256
    nhead = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    dim_feedforward = 512

    # Initialize direct transformer model (NO WRAPPER CLASS)
    model = DirectTransformerTrainer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=0.1,
        pad_idx=src_vocab["<pad>"]
    ).to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

    # Training loop
    print("\n🎯 Starting training...")
    num_epochs = 100

    for epoch in range(num_epochs):
        avg_loss = train_epoch(model, dataset, optimizer, criterion, device, src_vocab["<pad>"])

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
    
    

    # ==========================================
    # 7. EVALUATION AND TRANSLATION EXAMPLES
    # ==========================================

    print("\n🔍 Testing translations...")

    # Test on a few examples
    test_sentences = [
        "I love cats",
        "The car is red",
        "She sings well"
    ]

    model.eval()

    for test_sentence in test_sentences:
        # Encode test sentence
        test_ids = encode_sentence(test_sentence, src_vocab)
        test_tensor = torch.tensor([test_ids]).to(device)  # [1, seq_len] - batch_first

        # Create source mask
        src_mask = create_padding_mask(test_tensor, src_vocab["<pad>"])

        # Generate translation using direct transformer
        translated = greedy_decode_direct(
            model, test_tensor, src_mask,
            max_len=20,
            start_token=tgt_vocab["<sos>"],
            end_token=tgt_vocab["<eos>"],
            device=device
        )

        # Decode result
        translated_text = decode_tokens(translated[0].tolist(), tgt_id_to_token)

        print(f"English: {test_sentence}")
        print(f"French:  {translated_text}")
        print("-" * 40)

    print("✅ Training completed!")

# ==========================================
# 8. RUN TRAINING
# ==========================================

if __name__ == "__main__":
    main()

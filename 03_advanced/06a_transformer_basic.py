"""
18 - 6a. Transformer Basic - THE ABSOLUTE BEGINNER'S STEPPING STONE

`nn.Transformer` is PyTorch’s built-in block that mixes information between words 
in a sentence using attention mechanisms. 
It takes raw sequences and turns them into contextualized feature vectors so 
the model can learn relationships like word order, grammar, and translation.

"What tensor shapes go into a Transformer, and what tensor shapes come out?"

1. Initialize Model:
   We instantiate PyTorch's native `nn.Transformer` layer. 
   Using `batch_first=True` configures the model to accept 3D data in the standard format.

2. Create Synthetic Inputs:
   We generate random 3D numbers representing: `src`: Source sequences and `tgt`: Target sequences 

3. Perform Forward Pass & Print Shapes:
   We pass `src` and `tgt` into the Transformer. The output matches the batch size 
   and target sequence length, projecting features for every single word step.
"""
import torch
import torch.nn as nn


"""
# 1. Initialize the Transformer model

- d_model=512: 
  The "depth" or size of the vector representation for every word token. 
  A larger number allows the model to store richer details about each word, but requires more memory.

- nhead=8: 
  The number of parallel "attention heads" looking at the sequence at the same time. 
  Dividing d_model into 8 heads allows the model to simultaneously focus on 8 different relationships 
  (e.g., subject-verb vs. adjective-noun).

- num_encoder_layers=6: 
  The number of stacked encoder blocks used to process and understand the source sequence. 
  More layers allow the model to learn deeper, more complex relationships in the input text.

- num_decoder_layers=6: 
  The number of stacked decoder blocks used to generate the target output sequence. 
  Multiple layers refine the generated text step-by-step before producing the final word scores.

- batch_first=True: 
  Tells PyTorch to format tensors as [Batch Size, Sequence Length, Feature Size]. 
  Setting this to True makes tensor shapes far more intuitive for human readability and standard data loaders.
"""
transformer_model = nn.Transformer(
    d_model=512, 
    nhead=8, 
    num_encoder_layers=6, 
    num_decoder_layers=6,
    batch_first=True
)

"""
2. Create dummy continuous input data
Shapes: (batch_size, sequence_length, embedding_dimension)
   - `src`: Source sequences (e.g., 32 sentences, 10 words each, 512 features per word).
   - `tgt`: Target sequences (e.g., 32 sentences, 20 words each, 512 features per word).
"""
batch_size = 32
sequence_length = 10
embedding_dim = 512
src = torch.rand(batch_size, sequence_length, embedding_dim)  # Source sequence (Encoder input)
tgt = torch.rand(batch_size, sequence_length * 2, embedding_dim)  # Target sequence (Decoder input)

# 3. Forward pass
out = transformer_model(src, tgt)

# print("Output shape:", out.shape)
# Expected output: torch.Size([32, 20, 512])
print("--- TENSOR SHAPE BREAKDOWN ---")
print(f"Source Input Shape (src):      {src.shape}")
print(f"Target Input Shape (tgt):      {tgt.shape}")
print(f"Transformer Output Shape:      {out.shape}\n")

print("--- WHAT THESE DIMENSIONS MEAN ---")
print(f"Batch Size (Sentences per batch): {out.shape[0]}")
print(f"Target Sequence Length (Words):    {out.shape[1]}")
print(f"Embedding Dimension (Features):    {out.shape[2]}")


# =========torch.triu==========
sz = 4
# 1. BEFORE: Creating a square grid of 1s
ones_matrix = torch.ones(sz, sz)

# 2. INTERMEDIATE: Multiplying by -inf
inf_matrix = ones_matrix * float('-inf')

# 3. AFTER: Applying torch.triu with diagonal=1
# diagonal=1 keeps values ABOVE the main diagonal and turns everything on/below it to 0.0
causal_mask = torch.triu(inf_matrix, diagonal=1)

# === DISPLAY RESULTS ===

print("\n--- BEFORE (torch.ones) ---")
print(ones_matrix)

print("--- AFTER (torch.triu with diagonal=1) ---")
print(causal_mask)
"""
Output shape: [4, 4]
[[ 0., -inf, -inf, -inf],   # Step 0: Can see position 0, future steps 1,2,3 masked out
 [ 0.,   0., -inf, -inf],   # Step 1: Can see positions 0,1, future steps 2,3 masked out
 [ 0.,   0.,   0., -inf],   # Step 2: Can see positions 0,1,2, future step 3 masked out
 [ 0.,   0.,   0.,   0.]]   # Step 3: Can see all past and current positions
"""

# =========padding mask==========
# 1. Define the padding token index (e.g., 0)
pad_idx = 0

# 2. Sample input sequence tensor: 2 sentences, 4 tokens each
# Batch size = 2, Sequence length = 4
seq = torch.tensor([
    [1, 3, 4, 0],  # Sentence 1: <sos>, "I", "cats", <pad>
    [1, 5, 0, 0]   # Sentence 2: <sos>, "dog", <pad>, <pad>
])

# 3. Create the padding mask using element-wise comparison (==)
mask = (seq == pad_idx)

print(f"\n--- PADDING MASK (seq == {pad_idx}) ---")
print(mask)
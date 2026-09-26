"""
18. Transformer Architecture (From Scratch) in PyTorch

This script implements the complete, original Encoder-Decoder Transformer model 
from scratch. It structures all core mathematical modules, including 
Scaled Dot-Product Attention, Multi-Head Attention blocks, Sinusoidal Positional 
Encoding layers, and fully stacked Encoder/Decoder layers with residual LayerNorm pathways.

Learning Objectives:
1. Implement the mathematical formulation of Scaled Dot-Product Attention.
2. Structure Multi-Head Attention projecting query, key, and value vectors.
3. Compute and inject frequency-based Sinusoidal Positional Encodings.
4. Assemble a complete autoregressive Encoder-Decoder Transformer network.
"""

import math

# We import the core torch library.
import torch

# nn contains the container, normalization, and linear mapping classes.
# Documentation: https://pytorch.org/docs/stable/nn.html
import torch.nn as nn

# ==========================================
# 1. SINUSOIDAL POSITIONAL ENCODING
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        # We compute sinusoidal frequency waves to inject sequence order awareness.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine waves to even indexes and cosine waves to odd indexes
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # We save this as a non-trainable register so it is automatically saved with our weights
        # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [Batch, Seq_Len, d_model]
        return x + self.pe[:, :x.size(1)]

# ==========================================
# 2. MULTI-HEAD ATTENTION MODULE
# ==========================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Define projection matrices for Queries, Keys, and Values
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 2.1 Project inputs and split into heads: [Batch, Num_Heads, Seq_Len, d_k]
        Q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2.2 Scaled Dot-Product Attention: Softmax( (Q * K^T) / sqrt(d_k) ) * V
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask to zero out unwanted connections (padding or future tokens)
        if mask is not None:
            # We fill masked indexes with a large negative value so they map to 0 in Softmax
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V) # Shape: [Batch, Num_Heads, Seq_Len, d_k]
        
        # 2.3 Concatenate heads back and project: [Batch, Seq_Len, d_model]
        concat = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(concat)

# ==========================================
# 3. ENCODER AND DECODER BLOCKS
# ==========================================
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Self-attention with residual connection + LayerNorm
        attn_out = self.mha(x, x, x, mask)
        x = self.layernorm1(x + attn_out)
        # Position-wise feed-forward with residual connection + LayerNorm
        ffn_out = self.ffn(x)
        x = self.layernorm2(x + ffn_out)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048):
        super(DecoderLayer, self).__init__()
        self.mha_self = MultiHeadAttention(d_model, num_heads)
        self.mha_cross = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask=None, trg_mask=None):
        # Masked self-attention (look-ahead mask prevents attending to future steps)
        self_attn = self.mha_self(x, x, x, trg_mask)
        x = self.layernorm1(x + self_attn)
        # Cross-attention over encoder output
        cross_attn = self.mha_cross(x, enc_output, enc_output, src_mask)
        x = self.layernorm2(x + cross_attn)
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.layernorm3(x + ffn_out)
        return x

# ==========================================
# 4. FULL ENCODER-DECODER TRANSCRIPT ASSEMBLY
# ==========================================
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model=256, num_heads=8, num_layers=2):
        super(Transformer, self).__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.pe = PositionalEncoding(d_model)
        
        # Build encoder and decoder stacks
        self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, num_heads) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, trg_vocab_size)

    def forward(self, src, trg, src_mask=None, trg_mask=None):
        # 4.1 Process Source: Embed + PE -> Encoder stack
        enc_out = self.pe(self.src_embedding(src))
        for layer in self.encoder:
            enc_out = layer(enc_out, src_mask)
            
        # 4.2 Process Target: Embed + PE -> Decoder stack
        dec_out = self.pe(self.trg_embedding(trg))
        for layer in self.decoder:
            dec_out = layer(dec_out, enc_out, src_mask, trg_mask)
            
        # 4.3 Project to target vocabulary probability scores
        return self.fc_out(dec_out)

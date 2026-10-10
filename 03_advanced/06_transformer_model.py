"""
18. Transformer Architecture (From Scratch) in PyTorch

1. What Are We Trying to Achieve Overall?
   We are building an original, complete Transformer model (Encoder-Decoder) 
   from scratch to perform Sequence-to-Sequence (Seq2Seq) tasks. 
   Specifically, these scripts demonstrate translating text from a source language 
   into a target language.

2. A Simple Real-World Example: French to English Translation
   - Input (Source Language - French): "j'aime les chats"
   - The Encoder (06_transformer_model.py): Reads the entire French sentence, 
     understands the context using Multi-Head Attention, and converts it into a 
     meaningful mathematical representation.
   - The Decoder (06_transformer_model.py): Takes that mathematical representation 
     and generates the English translation word-by-word.
   - Output (Target Language - English): "I like cats"

3. Role of Each File:
   - 06_transformer_model.py: Defines the core architecture components from scratch, 
     including Positional Encoding, Multi-Head Attention, Encoder Layers, 
     Decoder Layers, and the full Transformer class.
   - 07_transformer_train.py: Imports the Transformer model, builds the causal 
     look-ahead masks, sets up a toy French-to-English dataset, trains the model 
     using CrossEntropyLoss, and executes greedy decoding to generate the final translated text.


    1. PositionalEncoding:Injects sine/cosine wave position markers into embeddings so the 
        model knows word order in parallel processing 

    2. MultiHeadAttention: Computes Q, K, V attention matrices to dynamically highlight 
        relationships between words across multiple representation heads 

    3. EncoderLayer: Uses Self-Attention to encode the source sentence (French) into 
        a deep contextual feature matrix (enc_out) 

    4. DecoderLayer: Uses Masked Self-Attention on target tokens (English) and 
        Cross-Attention on enc_out to align English target words with French source words 

    5. Full Assembly Chain:
        French Text -> PositionalEncoding -> EncoderLayers -> enc_out
        English Text -> PositionalEncoding -> DecoderLayers (Cross-Attending to enc_out) -> FC -> Output Logits 
        

    [French Word Tokens] 
        │
        ▼
    1.  src_embedding + PositionalEncoding 
        │ (Adds word order awareness to French tokens)
        ▼
    2.  EncoderLayer(s) ──> MultiHeadAttention (Self-Attention)
        │ (Every French word analyzes all other French words)
        ▼
    enc_out (Complete French Context Matrix) ─────────────────────┐
                                                                  │
    [English Word Tokens so far]                                  │
        │                                                         │
        ▼                                                         │
    3.  trg_embedding + PositionalEncoding                        │
        │ (Adds word order awareness to English tokens)           │
        ▼                                                         │
    4.  DecoderLayer(s)                                           │
        │                                                         │
        ├───> mha_self (Masked Self-Attention)                    │
        │     (English tokens analyze past English tokens)        │
        │                                                         │
        └───> mha_cross (Cross-Attention) <───────────────────────┘
                (Query = English, Key/Value = French enc_out)
                ("What French words translate to the next English word?")
        │
        ▼
    5.  self.fc_out (Linear Projection) ──> Predicts Next English Word Token Index

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

"""
PositionalEncoding
- What the Class Does:Unlike LSTMs or RNNs that process words step-by-step, a Transformer processes all words in a sentence simultaneously in parallel. Because of this, the Transformer has no inherent sense of word order. This class creates a static "map" of sine and cosine math waves that assign a unique position signature to every word slot.
- What forward(x) Does: It takes the raw word embeddings x (shape: [Batch, Seq_Len, d_model]) and adds the positional wave patterns directly to them.
- What Output It Gives: Vectors that contain both what the word means and where the word sits in the sentence.
- How It Connects: Used in Transformer right after converting French and English word IDs into embedding vectors.
"""
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
"""
- What the Class Does:
  Acts as the main "thinking engine" of the Transformer. It splits word representations 
  into three roles: Queries (what am I looking for?), Keys (what content is available?), 
  and Values (what information do I pass along?). It divides these across multiple "heads" 
  so the model can focus on different word connections simultaneously—for example, 
  one head can connect the verb "runs" to its subject "dog", while another connects 
  the adjective "brown" to the noun "dog".

- What forward(q, k, v, mask) Does:
  Compares every word in Query (Q) against every word in Key (K) to figure out which 
  words are most relevant to each other. It turns those scores into percentages and 
  uses them to blend the information in Value (V). If a mask is provided, it hides 
  unwanted locations, such as blank padding tokens or future words in the sentence.

- What Output It Gives:Returns an updated tensor of shape [Batch, Seq_Len, d_model] where every word vector 
  now contains rich context from all the related words around it.

- How It Connects: Serves as the foundational building block inside both the EncoderLayer (for understanding 
  source text) and the DecoderLayer (for generating target text).
"""
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

"""
EncoderLayer
- What the Class Does:Acts as a single processing layer inside the French-reading Encoder stack. 
    It combines MultiHeadAttention with a position-wise Feed-Forward Network (ffn) and Layer Normalization (layernorm).
- What forward(x, mask) Does:
- Runs x through MultiHeadAttention where Q, K, V are all x (Self-Attention). Every French word looks at every other French word to understand full sentence context.
- Adds residual connections (x + attn_out) and applies Layer Normalization.
- Passes through the Feed-Forward Network to refine features, followed by another Layer Normalization.
- What Output It Gives: An updated, context-rich feature representation of the input French sentence.
- How It Connects:Stacking multiple EncoderLayer blocks inside Transformer.encoder produces the final enc_out memory matrix that represents the complete meaning of the French text.
"""
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

"""
DecoderLayer
- What the Class Does:
- Acts as a single processing layer inside the English-generating Decoder stack. It contains two MultiHeadAttention blocks: Masked Self-Attention (focuses on English words generated so far without looking ahead at future target words) and Cross-Attention (connects the generated English words to the French encoder features).
- What forward(x, enc_output, src_mask, trg_mask) Does:
- Self-Attention on Target: Runs x (English text) through mha_self with a causal trg_mask so current step cannot see future steps.
- Cross-Attention: Runs mha_cross where Query comes from the English decoder state (x), but Key and Value come from the French encoder output (enc_output). This asks: "Based on the English words I've written so far, which French words should I focus on next?"
- Passes through the Feed-Forward Network and Layer Normalization pathways.
- What Output It Gives:
- Refined decoder vectors that blend the English sequence history with focused French context.
- How It Connects:
- Stacking multiple DecoderLayer blocks inside Transformer.decoder outputs dec_out, which is projected by self.fc_out into vocabulary logits to predict the next English word.
"""
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
# 4. FULL ENCODER-DECODER TRANSFORMER ASSEMBLY
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
        """Return target-vocabulary logits with shape [batch, target_length, trg_vocab_size].

        The final dimension contains one score per target token ID. Convert the
        highest-scoring ID to a word with the vocabulary used by the caller.
        """
        # 4.1 Process Source: Embed + PE -> Encoder stack
        enc_out = self.pe(self.src_embedding(src))
        for layer in self.encoder:
            enc_out = layer(enc_out, src_mask)
            
        # 4.2 Process Target: Embed + PE -> Decoder stack
        dec_out = self.pe(self.trg_embedding(trg))
        for layer in self.decoder:
            dec_out = layer(dec_out, enc_out, src_mask, trg_mask)
            
        # 4.3 Project to one logit per target-vocabulary token ID
        return self.fc_out(dec_out)

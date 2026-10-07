"""
09 - 3b : Comparing Vanilla RNN, GRU, and LSTM (RECURRENT LAYER ENGINES)

1. Standard Vanilla RNN (nn.RNN)
--------------------------------
- How it works:
  Processes data step-by-step from left to right. At each step, it updates a single 
  hidden state (memory) using the current word and the memory from the previous step.
- Pros & Cons:
  Very simple and fast, but suffers from the "vanishing gradient" problem—it quickly 
  forgets information from earlier words in long sequences.
- Best used for:
  Simple, short sequence tasks or basic educational models.

2. Gated Recurrent Unit (nn.GRU)
--------------------------------
- How it works:
  An improved version of the Vanilla RNN that uses internal "gates" (Update and 
  Reset gates) to control how much past memory to keep or discard at each step.
- Pros & Cons:
  Maintains longer context than a Vanilla RNN while keeping a single hidden state 
  tensor, making it computationally fast and memory-efficient.
- Best used for:
  Medium-to-long text sequences when you need a balance of strong performance 
  and fast training speeds.

3. Long Short-Term Memory (nn.LSTM)
-----------------------------------
- How it works:
  Uses a dual-memory system managed by three gates (Input, Forget, and Output gates):
    a) Hidden State (hn): Short-term working memory used for immediate predictions.
    b) Cell State (cn): Long-term highway memory that lets information flow across 
       long sequences without fading.
- Pros & Cons:
  The gold standard for handling long dependencies, though it requires more memory 
  and returns two memory tensors `(hn, cn)` instead of one.
- Best used for:
  Complex NLP tasks, long documents, or datasets where early word context heavily 
  influences final predictions.
===============================================================================

Purpose:
    This script compares the three major recurrent layer engines in PyTorch side by side:
    1. Standard Vanilla RNN (nn.RNN) (Recurrent Neural Network)
    2. Gated Recurrent Unit (nn.GRU)
    3. Long Short-Term Memory (nn.LSTM)

    The goal is to visibly demonstrate why LSTM returns TWO memory tensors (Hidden State + Cell State)
    while RNN and GRU return only ONE memory tensor (Hidden State).

Data Processing Flow:
    1. Create identical synthetic token sequences and embed them.
    2. Pass the exact same embedded tensor through nn.RNN, nn.GRU, and nn.LSTM.
    3. Print and compare return tuples, output shapes, and internal memory mechanisms.
"""

import torch
import torch.nn as nn

def main():
    # Setup consistent dimensions across all three models for fair comparison
    vocab_size = 50        # Small dictionary of 50 possible words
    embedding_dim = 8      # Vector size per word
    hidden_size = 16       # Memory size of each recurrent layer
    
    # Create identical test input: 3 sentences, each with 5 tokens
    inputs = torch.randint(low=0, high=vocab_size, size=(3, 5))
    
    # Prepare shared embedding lookup table
    embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
    embedded_inputs = embedding(inputs)
    
    print("=" * 60)
    print(f"Input Shape: {inputs.shape} [Batch, Seq_Len]")
    print(f"Embedded Input Shape: {embedded_inputs.shape} [Batch, Seq_Len, Embedding_Dim]")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. VANILLA RNN (nn.RNN)
    # ------------------------------------------------------------------
    print("\n[1] Testing Vanilla RNN (nn.RNN)")
    rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
    rnn_out, rnn_hn = rnn(embedded_inputs)
    
    print(f"  Returns 2 values: (outputs, final_hidden_state)")
    print(f"  Outputs Shape:      {rnn_out.shape} [Batch, Seq_Len, Hidden_Size]")
    print(f"  Hidden State Shape: {rnn_hn.shape}  [Num_Layers, Batch, Hidden_Size]")
    # Practical Note: Simple, but forgets early information on longer sequences (vanishing gradient).
    
    # ------------------------------------------------------------------
    # 2. GATED RECURRENT UNIT (nn.GRU)
    # ------------------------------------------------------------------
    print("\n[2] Testing Gated Recurrent Unit (nn.GRU)")
    gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
    gru_out, gru_hn = gru(embedded_inputs)
    
    print(f"  Returns 2 values: (outputs, final_hidden_state)")
    print(f"  Outputs Shape:      {gru_out.shape} [Batch, Seq_Len, Hidden_Size]")
    print(f"  Hidden State Shape: {gru_hn.shape}  [Num_Layers, Batch, Hidden_Size]")
    # Practical Note: Uses internal update and reset gates to remember key words efficiently with fast speed.

    # ------------------------------------------------------------------
    # 3. LONG SHORT-TERM MEMORY (nn.LSTM)
    # ------------------------------------------------------------------
    print("\n[3] Testing Long Short-Term Memory (nn.LSTM)")
    lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
    # Notice the tuple unpacking for LSTM: (hn, cn)
    lstm_out, (lstm_hn, lstm_cn) = lstm(embedded_inputs)
    
    print(f"  Returns a tuple of outputs and TWO memory states: (outputs, (hn, cn))")
    print(f"  Outputs Shape:      {lstm_out.shape} [Batch, Seq_Len, Hidden_Size]")
    print(f"  Hidden State (hn):  {lstm_hn.shape}  [Num_Layers, Batch, Hidden_Size]")
    print(f"  Cell State (cn):    {lstm_cn.shape}  [Num_Layers, Batch, Hidden_Size]")
    # Practical Note: 'hn' handles short-term working memory; 'cn' acts as a long-term highway memory.
    # Extract the final layer using lstm_hn[-1] to pass into your linear classifier. (just like you would with rnn_hn[-1] or gru_hn[-1]).
    # The cell state acts like an internal conveyor belt that helps carry long-term information through the sequence without losing it (using gates).

    print("\n" + "=" * 60)
    print("SUMMARY FOR BEGINNERS:")
    print("- Use RNN:  For basic learning / academic toy examples.")
    print("- Use GRU:  When training speed and parameter efficiency are key.")
    print("- Use LSTM: When handling complex text with long dependencies.")
    print("=" * 60)

if __name__ == "__main__":
    main()
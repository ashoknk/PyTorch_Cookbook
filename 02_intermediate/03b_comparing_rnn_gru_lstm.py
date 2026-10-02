"""
Purpose:
    This script compares the three major recurrent layer engines in PyTorch side by side:
    1. Standard Vanilla RNN (nn.RNN)
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

    print("\n" + "=" * 60)
    print("SUMMARY FOR BEGINNERS:")
    print("- Use RNN:  For basic learning / academic toy examples.")
    print("- Use GRU:  When training speed and parameter efficiency are key.")
    print("- Use LSTM: When handling complex text with long dependencies.")
    print("=" * 60)

if __name__ == "__main__":
    main()
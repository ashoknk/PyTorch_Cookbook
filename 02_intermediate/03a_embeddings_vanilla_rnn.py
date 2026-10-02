"""
Purpose:
    This script introduces the fundamental building blocks of sequence processing:
    1. Mapping word numbers (tokens) to dense continuous numbers using an Embedding layer.
    2. Passing sequential data through a basic Vanilla RNN (nn.RNN).
    3. Understanding how the RNN output and hidden state can produce a final classification score.

Data Processing Flow:
    1. Tokens: Words are represented as integer IDs (e.g., [1, 2, 3]).
    2. Embedding: Converts integer IDs into a matrix of float vectors (e.g., [Batch, Seq_Len, Embedding_Dim]).
    3. Recurrent Processing: Passes vectors through nn.RNN to extract sequential memory across time steps.
    4. Prediction: Uses the last hidden state to generate class prediction scores via nn.Linear.
"""

import torch
import torch.nn as nn

def main():
    print("--- Step 1: Text Tokens & Padding ---")
    # Define vocabulary size (total unique words in our dictionary) and batch dimensions
    vocab_size = 100       # Total unique words recognized by our dictionary (IDs 0 to 99)
    embedding_dim = 16     # Each word ID will be mapped to a feature vector of length 16
    hidden_size = 32       # The internal memory state size of the RNN
    num_classes = 2        # Output classes (e.g., 0 = Negative, 1 = Positive)

    # Simulate a mini-batch of 2 sentences, each padded to a sequence length of 4 words
    # Sentence 1: Words [1, 2, 3, 0] (0 represents padding for shorter text)
    # Sentence 2: Words [4, 5, 0, 0]
    raw_tokens = torch.tensor([
        [1, 2, 3, 0],
        [4, 5, 0, 0]
    ])
    print(f"Raw Input Token Tensor Shape: {raw_tokens.shape} (Format: [Batch Size, Sequence Length])")

    print("\n--- Step 2: The Role of nn.Embedding ---")
    # nn.Embedding transforms discrete integer IDs into continuous floating-point vectors.
    # It acts as a lookup table where row index = word ID.
    embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
    
    embedded_x = embedding_layer(raw_tokens)
    print(f"Embedded Tensor Shape: {embedded_x.shape} (Format: [Batch Size, Sequence Length, Embedding Dim])")
    print("Notice how each word integer is now expanded into 16 floating-point values!")

    print("\n--- Step 3: Processing Sequences with nn.RNN ---")
    # batch_first=True ensures inputs/outputs match [Batch, Seq_Len, Feature_Dim] order
    rnn_layer = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
    
    # nn.RNN returns two values:
    # 1. outputs: The hidden state vectors for EVERY step in the sequence
    # 2. hn: The FINAL hidden state vector after reading the last word
    outputs, hn = rnn_layer(embedded_x)
    
    print(f"RNN Full Step Outputs Shape: {outputs.shape} (Format: [Batch, Seq_Len, Hidden_Size])")
    print(f"RNN Final Hidden State Shape: {hn.shape} (Format: [Num_Layers, Batch, Hidden_Size])")

    print("\n--- Step 4: Classification Prediction ---")
    # Extract the last layer's final hidden state for each sample in the batch
    last_hidden = hn[-1]  # Shape: [Batch Size, Hidden_Size] (2, 32)
    
    # Linear classification head maps hidden memory representation down to class output scores
    classifier = nn.Linear(in_features=hidden_size, out_features=num_classes)
    logits = classifier(last_hidden)
    
    print(f"Final Prediction Logits Shape: {logits.shape} (Format: [Batch Size, Num_Classes])")
    print(f"Sample Output Predictions:\n{logits}")

if __name__ == "__main__":
    main()
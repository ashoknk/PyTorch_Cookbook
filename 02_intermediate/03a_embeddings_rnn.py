"""
09 - 3a : Embeddings + Vanilla RNN for Sequence Classification

What is an RNN (Recurrent Neural Network)?
    - Standard neural networks look at inputs all at once and have no memory of the past.
    - An RNN processes data step-by-step in order (like reading a sentence word by word).
    - At each step, it takes the current input PLUS its "memory" from previous steps
      (called the hidden state) to update its understanding.

Why use an RNN?
    - Order and context matter: In language, "not bad" means the opposite of "bad, not good at all".
    - Variable length inputs: Sentences and time series can be short or long; RNNs handle
      arbitrary sequence lengths gracefully.

Where to use an RNN?
    - Natural Language Processing (NLP): Sentiment analysis, text classification, machine translation.
    - Time-Series & Forecasting: Stock prices, weather prediction, sensor readings.
    - Audio & Speech: Speech recognition, music generation.

Purpose:
    This script introduces the fundamental building blocks of sequence processing:
    1. Mapping word numbers (tokens) to dense continuous numbers using an Embedding layer.
    2. Passing sequential data through a basic RNN (nn.RNN).
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
    # Map each word to an integer ID; nn.Embedding uses these IDs as row indices.
    vocab = {"<PAD>": 0, "i": 1, "love": 2, "this": 3, "movie": 4, "hate": 5, "it": 6}
    sentence_1 = ["i", "love", "this", "movie"]  # Positive sentence
    sentence_2 = ["i", "hate", "it"]             # Negative sentence

    tokens_1 = [vocab[word] for word in sentence_1] #[1, 2, 3, 4]
    tokens_2 = [vocab[word] for word in sentence_2] #[1, 5, 6]
    tokens_2 += [vocab["<PAD>"]] * (4 - len(tokens_2)) #[1, 5, 6, 0]
    raw_tokens = torch.tensor([tokens_1, tokens_2]) 

    # Define vocabulary size (total unique words in our dictionary) and batch dimensions
    vocab_size = len(vocab)
    embedding_dim = 16     # Each word ID will be mapped to a feature vector of length 16
    hidden_size = 32       # The internal memory state size of the RNN
    num_classes = 2        # Output classes (e.g., 0 = Negative, 1 = Positive)

    print(f"Raw Input Token Tensor Shape: {raw_tokens.shape} (Format: [Batch Size, Sequence Length])")

    print("\n--- Step 2: The Role of nn.Embedding ---")
    # nn.Embedding transforms discrete integer IDs into continuous floating-point vectors.
    # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
    # It acts as a lookup table where row index = word ID.
    embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
    
    embedded_x = embedding_layer(raw_tokens)
    print(f"Embedded Tensor Shape: {embedded_x.shape} (Format: [Batch Size, Sequence Length, Embedding Dim])")
    print(f"Embedded Tensor: {embedded_x}")

    print("\n--- Step 3: Processing Sequences with nn.RNN ---")
    # batch_first=True ensures inputs/outputs match [Batch, Seq_Len, Feature_Dim] order
    rnn_layer = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
    
    # nn.RNN returns two values:
    # 1. outputs: The hidden state vectors for EVERY step in the sequence
    # 2. hn: The FINAL hidden state vector after reading the last word
    # Because this script is performing Sentence Classification (predicting a single sentiment label for the whole sentence), you only need a single vector that summarizes the entire text.
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
    #             Class 0, Class 1 
    #             Negative   Positive
    #   tensor([[ -0.7358,  0.0517], ===> Sentence 1 : ["i", "love", "this", "movie"]. 0.0517 > -0.7358.
    #           [ -0.4474, -0.3162]] ===> Sentence 2 : ["i", "hate", "it"]. -0.3162 > -0.4474
    # These raw numbers are called logits (unnormalized class scores). 
    # The model has not been trained on any actual data or loss function yet, so its weights are completely random.   

if __name__ == "__main__":
    main()

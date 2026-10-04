"""
09 - 3c : Recurrent Neural Network (RNN) with PyTorch

This script demonstrates how to construct, train, and validate recurrent sequence 
classifiers. We build a customizable network containing an Embedding layer, 
a recurrent engine (supporting RNN, LSTM, or GRU), and a final linear prediction 
head. We simulate a binary sentiment classification task with padded token sequences.

The vocabulary maps words to integer IDs (for example, "very" -> 1 and
"good" -> 2). nn.Embedding uses each ID to look up a trainable dense vector
(e.g., 16 numbers per word). The vectors are illustrative, not values from
this run. Training adjusts them to help the model classify sentiment; it
does not guarantee that words with similar meanings will be close together.

Learning Objectives:
1. Build sequence classifiers using PyTorch's recurrent modules (nn.LSTM, nn.GRU).
2. Map token indices to dense continuous vector spaces using nn.Embedding.
3. Understand how recurrent layers return complete outputs and final step hidden states.
4. Process dynamic input dimensions and batch sequence shapes.
"""

# We import the core torch library.
import torch

# nn houses sequential modules: Embedding, LSTM, GRU, Linear.
# Documentation: https://pytorch.org/docs/stable/nn.html
import torch.nn as nn

# optim contains optimization algorithms.
import torch.optim as optim

# ==========================================
# 1. DEFINE SEQUENCE CLASSIFIER MODULE
# ==========================================
class SequenceClassifier(nn.Module):
    def __init__(self, vocab_size=1000, embedding_dim=64, hidden_size=128, num_classes=2, rnn_type="LSTM"):
        """
        Args:
            vocab_size (int): Size of vocabulary.
            embedding_dim (int): Vector space dimensions.
            hidden_size (int): Hidden state dimensions.
            num_classes (int): Category output counts.
            rnn_type (str): Type of recurrent layer (RNN, LSTM, or GRU).
        """
        super(SequenceClassifier, self).__init__()
        # Embedding layer converts discrete token indices to dense continuous vectors.
        # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        
        # Select recurrent engine
        self.rnn_type = rnn_type.upper()
        if self.rnn_type == "LSTM":
            # nn.LSTM handles vanishing gradients via gated cell states.
            # batch_first=True specifies input shapes as [Batch, Sequence_Length, Embedding_Dim]
            # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
            self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        elif self.rnn_type == "GRU":
            # nn.GRU is a simplified variant of LSTM combining cell and hidden states.
            # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
            self.rnn = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        else:
            # Standard vanilla RNN
            # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
            self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
            
        # Linear layer mapping the final step hidden representation to category outputs
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 1.1 Map inputs of shape [Batch, Seq_Len] to [Batch, Seq_Len, Embedding_Dim]
        embedded = self.embedding(x)
        
        # 1.2 Feed to recurrent layer
        # LSTM maintains two states, while RNN and GRU maintain one.vanilla RNN or GRU, it returns (out, hn). They don’t have a separate cell state.
        # out: hidden output at every sequence step.
        # hn: final hidden state.
        # cn: final cell state, the LSTM’s additional memory state.

        if self.rnn_type == "LSTM":
            # LSTM returns: (outputs, (final_hidden_state, final_cell_state))
            out, (hn, cn) = self.rnn(embedded)
        else:
            # GRU and RNN return: (outputs, final_hidden_state)
            out, hn = self.rnn(embedded)
            
        # hn shape: [1, Batch, Hidden_Size] (for single-layer RNNs)
        # We extract the last layer's hidden state and reshape to [Batch, Hidden_Size]
        last_hidden = hn[-1]
        
        # 1.3 Project to classes
        logits = self.fc(last_hidden)
        return logits

def main():
    print("--- 1. Initializing LSTM Sequence Classifier ---")
    vocab = {
        "<PAD>": 0,
        "very": 1,
        "good": 2,
        "awesome": 3,
        "extremely": 4,
        "bad": 5,
        "review": 6,
        "indeed": 7,
        "terrible": 8,
        "awful": 9,
        "movie": 10,
    }
    vocab_size = len(vocab)
    model = SequenceClassifier(vocab_size=vocab_size, embedding_dim=16, hidden_size=32, rnn_type="LSTM")
    print(model)

    # ==========================================
    # 2. RUN SIMULATED SEQUENCE BATCH CHECK
    # ==========================================
    print("\n--- Running Dimensional Check on Sequences ---")
    # Simulate 4 sequences of length 10 with token IDs from 0 to vocab_size - 1.
    synthetic_sequences = torch.randint(0, vocab_size, (4, 10))
    print(f"Batch sequence input shape: {synthetic_sequences.shape} (Format: [Batch, Seq_Len])")
    
    # Forward pass
    logits = model(synthetic_sequences)
    print(f"Output predictions shape: {logits.shape} (Format: [Batch, Num_Classes])")
    assert logits.shape == (4, 2), "Sequence classifier forward pass dimensional mismatch."

    # ==========================================
    # 3. TRAINING LOOP ON TOY SENTIMENT DATA
    # ==========================================
    print("\n--- Training on Toy Sentiments (0=Negative, 1=Positive) ---")
    # Each sentence is converted to token IDs and padded to the same sequence length.
    sentences = [
        ["very", "good"],                  # Positive
        ["extremely", "bad", "review"],    # Negative
        ["very", "awesome", "indeed"],     # Positive
        ["terrible"],                      # Negative
        ["good", "good", "awesome"],       # Positive
        ["awful", "bad", "movie"],         # Negative
    ]
    toy_y = torch.tensor([1, 0, 1, 0, 1, 0])  # Binary labels
    max_seq_len = 5
    encoded_sentences = [
        [vocab[word] for word in sentence][:max_seq_len]
        for sentence in sentences
    ]
    toy_X = torch.tensor([
        sentence + [vocab["<PAD>"]] * (max_seq_len - len(sentence))
        for sentence in encoded_sentences
    ])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 40
    for epoch in range(epochs):
        # Forward pass
        outputs = model(toy_X)
        loss = criterion(outputs, toy_y)
        
        # Backward optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Evaluate predictions
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(toy_X), dim=-1)
        print(f"Predictions: {preds.tolist()}")
        print(f"Ground Truth: {toy_y.tolist()}")

if __name__ == "__main__":
    main()

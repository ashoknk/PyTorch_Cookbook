"""
10. Bidirectional Recurrent Neural Network (BiLSTM) with PyTorch

This script demonstrates how to construct and train a Bidirectional Recurrent 
Neural Network (BiLSTM). In sequence classification or sequence tagging, reading 
the input from both left-to-right (forward) and right-to-left (backward) 
simultaneously allows the network to incorporate full temporal context. We 
illustrate the concatentation mechanics and evaluate a BiLSTM model on a synthetic classification task.

Learning Objectives:
1. Initialize a bidirectional LSTM module using nn.LSTM(..., bidirectional=True).
2. Retrieve and concatenate forward and backward hidden state vectors.
3. Classify sequence representations of shape [Batch, 2 * Hidden_Size].
4. Verify gradient flows and computational layers.
"""

# We import the core torch library.
import torch

# nn houses architectural layers: Embedding, LSTM, Linear.
# Documentation: https://pytorch.org/docs/stable/nn.html
import torch.nn as nn

# optim contains optimizer engines.
import torch.optim as optim

# ==========================================
# 1. DEFINE BILSTM CLASSIFIER MODULE
# ==========================================
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size=1000, embedding_dim=64, hidden_size=128, num_classes=2):
        """
        Args:
            vocab_size (int): Size of vocabulary.
            embedding_dim (int): Dense vector space dimensions.
            hidden_size (int): Size of hidden representations for each direction.
            num_classes (int): Category output counts.
        """
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # We specify bidirectional=True. The LSTM will internally maintain 
        # separate forward and backward weight matrices.
        # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_size, 
            batch_first=True, 
            bidirectional=True
        )
        
        # The output layer must receive the combined context of BOTH directions.
        # Since we concatenate forward and backward representations, the input dimension is 2 * hidden_size.
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # 1.1 Map token indices to vectors. Shape: [Batch, Seq_Len, Embedding_Dim]
        embedded = self.embedding(x)
        
        # 1.2 Feed to bidirectional LSTM.
        # out: [Batch, Seq_Len, 2 * Hidden_Size] (contains combined forward/backward outputs at each timestep)
        # hn: [num_layers * 2, Batch, Hidden_Size] (contains last step hidden states)
        out, (hn, cn) = self.lstm(embedded)
        
        # We extract the last layer's forward hidden state (index -2) and backward hidden state (index -1)
        # and concatenate them horizontally along the feature dimension.
        # hn[-2] shape: [Batch, Hidden_Size] (forward final state)
        # hn[-1] shape: [Batch, Hidden_Size] (backward final state)
        # Documentation: https://pytorch.org/docs/stable/generated/torch.cat.html
        forward_final = hn[-2]
        backward_final = hn[-1]
        
        combined_hidden = torch.cat((forward_final, backward_final), dim=-1) # Shape: [Batch, 2 * Hidden_Size]
        
        # 1.3 Project concatenated state to category logits
        logits = self.fc(combined_hidden)
        return logits

def main():
    print("--- 1. Initializing Bidirectional LSTM Classifier ---")
    vocab_size = 150
    model = BiLSTMClassifier(vocab_size=vocab_size, embedding_dim=16, hidden_size=32, num_classes=3)
    print(model)

    # ==========================================
    # 2. RUN SIMULATED DIMENSIONAL CHECK
    # ==========================================
    print("\n--- Running Dimensional Check with Synthetic Sequences ---")
    # Simulate a batch of 4 sequences, each of length 8
    synthetic_batch = torch.randint(0, vocab_size, (4, 8))
    print(f"Batch sequence input shape: {synthetic_batch.shape}")
    
    # Forward pass
    logits = model(synthetic_batch)
    print(f"Output predictions shape: {logits.shape} (Expected: [4, 3])")
    assert logits.shape == (4, 3), "Bidirectional classifier forward pass dimensional mismatch."

    # ==========================================
    # 3. TRAINING LOOP RUN ON TOY SENTENCE DATA
    # ==========================================
    print("\n--- Training Model on 3-Class Sentiment Prompts ---")
    # Toy dataset: 6 sentences, tokenized and padded to sequence length 6
    # Classes: 0 = Negative, 1 = Neutral, 2 = Positive
    toy_X = torch.tensor([
        [10, 11, 12, 0, 0, 0],  # "Very positive day"
        [20, 21, 22, 0, 0, 0],  # "Extremely boring post"
        [30, 31, 32, 0, 0, 0],  # "Worst film ever"
        [10, 12, 13, 0, 0, 0],  # "Positive beautiful weather"
        [20, 23, 0, 0, 0, 0],  # "Boring book"
        [30, 34, 35, 0, 0, 0]   # "Awful bad terrible"
    ])
    toy_y = torch.tensor([2, 1, 0, 2, 1, 0])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 30
    for epoch in range(epochs):
        outputs = model(toy_X)
        loss = criterion(outputs, toy_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Print final verified class predictions
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(toy_X), dim=-1)
        print(f"Predictions: {preds.tolist()}")
        print(f"Ground Truth: {toy_y.tolist()}")

if __name__ == "__main__":
    main()

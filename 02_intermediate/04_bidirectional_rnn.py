"""
10 - 4 : Bidirectional Recurrent Neural Network (BiLSTM) with PyTorch

This script demonstrates how to construct and train a Bidirectional Recurrent 
Neural Network (BiLSTM). In sequence classification or sequence tagging, reading 
the input from both left-to-right (forward) and right-to-left (backward) 
simultaneously allows the network to incorporate full temporal context. We 
illustrate the concatenation mechanics and evaluate a BiLSTM model on a small,
hand-labeled English sentiment dataset.

Real-World Applications - To classify sequences by reading text in both directions so the network understands 
        every word using context from both what came before it and what comes after it.
    Text / NLP: Sentences, articles, or audio transcriptions (Most common).   
    DNA / Protein Sequences: Biological sequences where information flows both ways along a strand.
    Time-Series / Sensors: Recorded sensor logs where entire sequences are analyzed retroactively.

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
    def __init__(self, vocab_size=1000, embedding_dim=16, hidden_size=32, num_classes=3):
        """
        Args:
            vocab_size (int): Size of vocabulary.
            embedding_dim (int): Dense vector space dimensions.
            hidden_size (int): Size of hidden representations for each direction.
            num_classes (int): Category output counts.
        """
        super(BiLSTMClassifier, self).__init__()
        # Maps word integer IDs (e.g., 10) to continuous float vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Reads the sequence and creates context representations
        # We specify bidirectional=True. The LSTM will internally maintain 
        # separate forward and backward weight matrices.
        # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_size, 
            batch_first=True, 
            bidirectional=True # <--- THIS IS THE SWITCH
        )
        
        # The output layer must receive the combined context of BOTH directions.
        # Since we concatenate forward and backward representations, the input dimension is 2 * hidden_size.(NOT just hidden_size)
        # Takes the combined memory vector and projects it into final class scores (logits)
        # Receives 2x Features. Because forward and backward outputs are concatenated, this layer receives an input size of hidden_size * 2
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
        # hn[-2] shape: [Batch, Hidden_Size] (forward final state). Final state of the left-to-right pass
        # hn[-1] shape: [Batch, Hidden_Size] (backward final state). Final state of the right-to-left pass
        
        forward_final = hn[-2] # Shape: [4, 32]
        backward_final = hn[-1] # Shape: [4, 32]
        
        # concatenate a sequence of tensors along an existing dimension
        # Documentation: https://pytorch.org/docs/stable/generated/torch.cat.html
        combined_hidden = torch.cat((forward_final, backward_final), dim=-1) # Shape: [Batch, 2 * Hidden_Size] -> [4, 32 + 32] = [4, 64]   <-- HERE IS YOUR DOUBLED DIMENSION!
        
        # 1.3 Project concatenated state to category logits
        logits = self.fc(combined_hidden)
        return logits

def main():
    print("--- 1. Initializing Bidirectional LSTM Classifier ---")

    # Each English word is mapped to the integer ID expected by nn.Embedding.
    vocab = {
        "<PAD>": 0,
        "very": 1, "positive": 2, "day": 3,
        "extremely": 4, "boring": 5, "post": 6,
        "worst": 7, "film": 8, "ever": 9,
        "beautiful": 10, "weather": 11,
        "book": 12, "awful": 13, "bad": 14, "terrible": 15
    }
    vocab_size = len(vocab)
    model = BiLSTMClassifier(vocab_size=vocab_size, embedding_dim=16, hidden_size=32, num_classes=3)
    print(model)

    # ==========================================
    # 2. RUN SIMULATED DIMENSIONAL CHECK
    # ==========================================
    print("\n--- Running Dimensional Check with Synthetic Sequences ---")
    
    # Shape-check with random valid token IDs; these do not represent sentences.
    synthetic_batch = torch.randint(0, vocab_size, (4, 8))
    print(f"Batch sequence input shape: {synthetic_batch.shape}")
    
    # Forward pass
    logits = model(synthetic_batch)
    print(f"Output predictions shape: {logits.shape} (Expected: [4, 3])")
    assert logits.shape == (4, 3), "Bidirectional classifier forward pass dimensional mismatch."

    # ==========================================
    # 3. TRAINING LOOP ON ENGLISH SENTENCES
    # ==========================================
    print("\n--- Training Model on 3-Class Sentiment Prompts ---")
    text_sentences = [
        ["very", "positive", "day"],
        ["extremely", "boring", "post"],
        ["worst", "film", "ever"],
        ["positive", "beautiful", "weather"],
        ["boring", "book"],
        ["awful", "bad", "terrible"]
    ]

    # Labels are class IDs: 0 = Negative, 1 = Neutral, 2 = Positive.
    # 1. Define human-readable label dictionary. num_classes=3
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    toy_y = torch.tensor([2, 1, 0, 2, 1, 0])

    # Convert each word to its vocabulary ID and pad sentences to length 6.
    max_seq_len = 6
    tokenized_batch = []
    for sentence in text_sentences:
        tokens = [vocab[word] for word in sentence]
        padded_tokens = tokens + [vocab["<PAD>"]] * (max_seq_len - len(tokens))
        tokenized_batch.append(padded_tokens)
    # Convert to PyTorch Tensor
    toy_X = torch.tensor(tokenized_batch)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 30
    # model -> criterion (loss calculation) ->  optimizer step
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
        preds = torch.argmax(model(toy_X), dim=-1) #returns the indices of the maximum value of all elements in the tensor
        # print(f"Predictions: {preds.tolist()}")
        # print(f"Ground Truth: {toy_y.tolist()}")
        # Convert prediction indices [2, 1, 0, 2, 1, 0] to English words
        pred_words = [label_map[p.item()] for p in preds]
        truth_words = [label_map[t.item()] for t in toy_y]
        
        print(f"Predicted Labels: {pred_words}")
        print(f"Ground Truth:     {truth_words}")

if __name__ == "__main__":
    main()

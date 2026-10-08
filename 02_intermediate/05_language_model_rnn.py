"""
11 - 5 : RNN Language Model (Character-Level) with PyTorch

This script demonstrates how to train an autoregressive, character-level 
Language Model using an LSTM. Given historical character contexts, the model 
learns to predict the next character. It showcases tokenization, sequence shifting, 
truncated Backpropagation Through Time (BPTT), and generation with Softmax temperature scaling.

Learning Objectives:
1. Process raw text to build a character-to-index mapping (vocabulary).
2. Format sequence inputs and target indices for next-token prediction tasks.
3. Manage and detach recurrent hidden state tensors between epochs.
4. Generate new text dynamically starting from seed prompts using temperature scaling.
"""

import numpy as np

# We import the core torch library.
import torch

# nn houses recurrent layers, embeddings, and loss criteria.
import torch.nn as nn

# optim contains parameter optimization engines.
import torch.optim as optim

# ==========================================
# 1. DEFINE RNN LANGUAGE MODEL MODULE
# ==========================================
class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_size=64):
        super(RNNLanguageModel, self).__init__()
        # Embedding layer projects character indices to dense spaces
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM processes sequence vectors. batch_first=True specifies shape as [Batch, Seq_Len, Features]
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        # Final projection mapping outputs back to vocabulary scores
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        # x shape: [Batch, Seq_Len]
        embedded = self.embedding(x)
        # lstm returns outputs and a tuple containing final hidden and cell states
        # hidden: Holds the LSTM’s memory vectors (h_t and c_t) containing context from all previously processed characters.
        # The Purpose: During generation, characters are predicted one by one. Returning hidden allows you to pass the model's 
        # running memory vector directly into the next step. Without passing hidden, the model would reset its memory every 
        # time you feed it a single character and forget what it previously generated!

        out, hidden = self.lstm(embedded, hidden)
        # Map sequential outputs across all steps to vocabulary logits. Shape: [Batch, Seq_Len, Vocab_Size]
        logits = self.fc(out)
        return logits, hidden

def sample_next_char(logits, temperature=1.0):
    """Applies Softmax temperature scaling and samples an index from the resulting probability distribution."""
    # Scale logits by temperature. Higher temperature increases randomness; lower increases confidence.
    logits = logits / max(temperature, 1e-5) #1e-5 is scientific notation for 0.00001
    #softmax will raw numbers and convert them to probabilities that sum to 1.0
    probs = torch.softmax(logits, dim=-1).numpy()
    # Draw index randomly according to computed probabilities (avoid boring and rigid argmax selection)
    return np.random.choice(len(probs), p=probs)

def main():
    print("--- 1. Processing Text and Building Vocabulary ---")
    # Toy training text corpus
    text = "hello world! welcome to pytorch character language models."
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    # Generate bidirectional mappings: character <-> integer index
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for idx, char in enumerate(chars)}
    print("chars :", list(chars[:10]))
    print("char_to_idx sample:", dict(list(char_to_idx.items())[:10]))
    print("idx_to_char sample:", dict(list(idx_to_char.items())[:10]))
    print(f"Text length: {len(text)}, Unique characters: {vocab_size}")
    
    # ==========================================
    # 2. SEQUENCE PREPARATION FOR AUTOREGRESION
    # ==========================================
    # Translate entire text string to indices
    encoded_text = [char_to_idx[c] for c in text]
    print("encoded_text sample:", dict(list(enumerate(encoded_text))[:10])) #maps to "hello worl"
    print("encoded_text 1: :", dict(list(enumerate(encoded_text))[1:])) #maps to "ello worl..."
    print("encoded_text :-1:", dict(list(enumerate(encoded_text))[:-1])) #maps to "hello worl..."
    
    # For autoregressive learning, target outputs are shifted right by 1 index:
    # Input sequence:  "h" "e" "l" "l" "o"
    # Target sequence: "e" "l" "l" "o" " "
    # At every step index t, the model tries to predict the target index t+1:
    # Given "h", predict "e".
    # Given "h" "e", predict "l".
    # Given "h" "e" "l", predict "l".
    inputs = torch.tensor(encoded_text[:-1], dtype=torch.long).unsqueeze(0)  # Shape: [1, Seq_Len-1]
    targets = torch.tensor(encoded_text[1:], dtype=torch.long).unsqueeze(0)  # Shape: [1, Seq_Len-1]
    print(f"inputs shape: {inputs.shape}, targets shape: {targets.shape}")
    
    # Initialize model, optimizer, and standard CrossEntropyLoss.
    # CrossEntropyLoss expects target shapes matching flattened sequence classifications.
    model = RNNLanguageModel(vocab_size=vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # ==========================================
    # 3. TRAINING LOOP WITH BPTT STATE MANAGEMENT
    # ==========================================
    epochs = 150
    print(f"--- Training Autoregressive LM for {epochs} epochs ---")
    
    for epoch in range(epochs):
        # In complete text training, we often preserve the recurrent state across sequence windows.
        # But we must detach these states from their historical autograd nodes to prevent 
        # backpropagating infinitely through time, which would exhaust memory (BPTT Backpropagation Through Time algorithm).
        # With .detach(): You cut the flowchart links to the past. The model keeps its memories, but forgets the mathematical baggage. GPU memory usage stays low and constant.
        outputs, hidden = model(inputs)
        hidden = (hidden[0].detach(), hidden[1].detach())
        
        # Flatten outputs and targets for CrossEntropyLoss computation
        # outputs shape: [Batch * Seq_Len, Vocab_Size], targets shape: [Batch * Seq_Len]
        #tensor.view(-1, size) will reshape the tensor into a 2D grid (a matrix) where the second dimension is exactly size
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 30 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # ==========================================
    # 4. TESTING AND TEXT GENERATION (SAMPLING)
    # ==========================================
    print("\n--- Generating Text Starting with Seed Prompt 'hello' ---")
    model.eval()
    seed = "hello"
    generated_text = seed
    
    # Encode and feed seed string to establish the recurrent history (hidden state)
    current_input = torch.tensor([[char_to_idx[c] for c in seed]], dtype=torch.long)
    
    with torch.no_grad():
        logits, hidden = model(current_input)
        
        # Generate subsequent characters one-by-one
        number_of_chars_to_generate = 31
        for _ in range(number_of_chars_to_generate):
            # Pick the final step logit corresponding to our last generated character
            # During character-by-character text generation:
            # You feed the current input into the model. The output tensor logits has shape [batch_size, sequence_length, vocab_size].
            # Index 0 selects the first (and only) sequence batch.
            # Index -1 selects the very last time step in that sequence.

            last_logit = logits[0, -1]
            next_idx = sample_next_char(last_logit, temperature=0.8) # 0.8 is "sweet spot" low to moderate
            
            # Map index back to character and append
            char = idx_to_char[next_idx]
            generated_text += char
            
            # Prepare single-character input tensor for the next step
            current_input = torch.tensor([[next_idx]], dtype=torch.long)
            # Pass the running hidden state forward to retain text history context
            logits, hidden = model(current_input, hidden)

    print(f"Generated Text:\n\"{generated_text}\"")

if __name__ == "__main__":
    main()

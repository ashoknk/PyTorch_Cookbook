"""
Morse Code to Text Translation using PyTorch Transformer

PURPOSE:
This script implements a complete sequence-to-sequence translation system that converts
Morse code patterns to readable text using a Transformer neural network architecture.

FLOW:
1. DATA PREPARATION:
   - Define Morse code dictionary mapping letters to binary patterns (0=dot, 1=dash)
   - Create synthetic dataset of random Morse sequences with corresponding text
   - Build vocabularies for both input (Morse) and output (text) tokens
   - Convert sequences to padded tensors with special tokens (SOS, EOS, PAD)

2. MODEL ARCHITECTURE:
   - PositionalEncoding: Adds position information to embeddings
   - TransformerModel: Encoder-decoder architecture with attention mechanisms
   - Input embedding for Morse code tokens
   - Target embedding for text characters
   - Multi-head attention and feed-forward layers

3. TRAINING PROCESS:
   - Split data into train/validation sets
   - Train model using teacher forcing (providing correct previous tokens)
   - Use cross-entropy loss with padding token ignored
   - Track training/validation loss and accuracy

4. INFERENCE:
   - Translate new Morse sequences to text using greedy decoding
   - Generate tokens one by one until EOS token is produced

APPLICATIONS:
- Educational tool for learning Morse code
- Historical communication decoding
- Emergency communication systems
- Pattern recognition and sequence-to-sequence learning demonstration
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def seed_everything(seed=512):
    """Set random seeds for reproducible results across all libraries."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()

"""We use a dictionary to map letters to Morse Code. This acts as the foundational mapping for our translation task."""

MORSE_CODE_DICT = {
    'A': '01', 'B': '1000', 'C': '1010', 'D': '100', 'E': '0',
    'F': '0010', 'G': '110', 'H': '0000', 'I': '00', 'J': '0111',
    'K': '101', 'L': '0100', 'M': '11', 'N': '10', 'O': '111',
    'P': '0110', 'Q': '1101', 'R': '010', 'S': '000', 'T': '1',
    'U': '001', 'V': '0001', 'W': '011', 'X': '1001', 'Y': '1011',
    'Z': '1100',
}

"""We also define special tokens to handle sequence boundaries and padding:

<SOS>: Start of Sequence.
<EOS>: End of Sequence.
<PAD>: For padding sequences to uniform length.
"""
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
PADDING_TOKEN = '<PAD>'
SPACE_CHAR = ' '


"""We create a synthetic dataset by randomly selecting characters, converting them to Morse Code, and adding space characters for clarity.

"""
def generate_dataset(num_samples=1000, max_len=10):
    """Generates random Morse code sequences with text translations."""
    dataset = []
    for _ in range(num_samples):
        seq_length = np.random.randint(1, max_len)
        text_seq = [np.random.choice(list(MORSE_CODE_DICT.keys())) for _ in range(seq_length)]
        morse_seq = [MORSE_CODE_DICT[char] for char in text_seq]
        dataset.append((SPACE_CHAR.join(morse_seq), text_seq))
    return dataset

"""The raw sequences are converted into tensors for model training. Here’s the key process:

Convert Morse Code and text sequences to vocabulary indices.
Add special tokens like <SOS> and <EOS>.
Pad the sequences to a uniform length."""

def prepare_data(dataset, morse_vocab, text_vocab, max_morse_len, max_text_len):
    """Converts sequences into padded tensors with indexes."""
    morse_sequences, text_sequences = [], []
    for morse_seq, text_seq in dataset:
        morse_tensor = [morse_vocab[SOS_TOKEN]] + \
                       [morse_vocab[ch] for ch in morse_seq if ch in morse_vocab] + \
                       [morse_vocab[EOS_TOKEN]]
        text_tensor = [text_vocab[SOS_TOKEN]] + \
                      [text_vocab[ch] for ch in text_seq] + \
                      [text_vocab[EOS_TOKEN]]
        morse_tensor += [morse_vocab[PADDING_TOKEN]] * (max_morse_len - len(morse_tensor))
        text_tensor += [text_vocab[PADDING_TOKEN]] * (max_text_len - len(text_tensor))
        morse_sequences.append(morse_tensor)
        text_sequences.append(text_tensor)
    return torch.tensor(morse_sequences), torch.tensor(text_sequences)

"""Vocabulary Creation
We define vocabularies for both Morse Code and text tokens."""

morse_vocab = {ch: idx for idx, ch in enumerate([PADDING_TOKEN, SOS_TOKEN, EOS_TOKEN, SPACE_CHAR, '0', '1'])}
text_vocab = {ch: idx for idx, ch in enumerate([PADDING_TOKEN, SOS_TOKEN, EOS_TOKEN] + list(MORSE_CODE_DICT.keys()))}
reverse_text_vocab = {idx: ch for ch, idx in text_vocab.items()}        


"""To ensure the model generalizes well, we split the dataset into training and validation sets:"""

NUM_SAMPLES = 10000
TRAIN_RATIO = 0.8

# Generate dataset
full_dataset = generate_dataset(NUM_SAMPLES)
# Split into train and validation sets
train_data = full_dataset[:int(NUM_SAMPLES * TRAIN_RATIO)]
val_data = full_dataset[int(NUM_SAMPLES * TRAIN_RATIO):]

# Determine maximum sequence lengths for padding
max_morse_len = max(len(morse_seq) for morse_seq, _ in full_dataset) + 2  # +2 for SOS/EOS
max_text_len = max(len(text_seq) for _, text_seq in full_dataset) + 2    # +2 for SOS/EOS

# Prepare training and validation data
train_morse, train_text = prepare_data(train_data, morse_vocab, text_vocab, max_morse_len, max_text_len)
val_morse, val_text = prepare_data(val_data, morse_vocab, text_vocab, max_morse_len, max_text_len)

print(f"Training data shape: Morse {train_morse.shape}, Text {train_text.shape}")
print(f"Validation data shape: Morse {val_morse.shape}, Text {val_text.shape}")

"""Now that we have prepared the dataset, let’s build the Transformer model. The architecture includes the following components:

Input Embedding: Converts Morse code tokens to dense vector representations.
Target Embedding: Converts text tokens to dense vector representations.
Positional Encoding: Adds sequence position information to embeddings.
Transformer Module: Implements sequence-to-sequence translation using attention mechanisms.
Output Layer: Maps transformer outputs to the target vocabulary.
"""

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)
    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)


"""The Transformer model integrates embeddings, positional encoding, and the core Transformer module."""

class TransformerModel(nn.Module):
    def __init__(self, input_vocab_size, target_vocab_size, d_model=512, nhead=8, num_layers=3, dim_feedforward=2048):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        # Embedding layers
        self.input_embedding = nn.Embedding(input_vocab_size, d_model)
        self.target_embedding = nn.Embedding(target_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
            num_decoder_layers=num_layers, dim_feedforward=dim_feedforward,
            batch_first=True
        )
        # Output layer
        self.fc_out = nn.Linear(d_model, target_vocab_size)
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
    def forward(self, src, tgt):
        src_mask = None
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
        src_emb = self.positional_encoding(self.input_embedding(src) * np.sqrt(self.d_model))
        tgt_emb = self.positional_encoding(self.target_embedding(tgt) * np.sqrt(self.d_model))
        # With batch_first=True, no need to transpose
        output = self.transformer(src_emb, tgt_emb, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.fc_out(output)            

"""Let’s instantiate the Transformer model and define the optimizer and loss function."""

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Instantiate the model
model = TransformerModel(
    input_vocab_size=len(morse_vocab),
    target_vocab_size=len(text_vocab),
    d_model=256,  # Reduced for faster training
    nhead=8,
    num_layers=3,
    dim_feedforward=512  # Reduced for simplicity
).to(device)
# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(ignore_index=text_vocab[PADDING_TOKEN])

"""The training loop involves:

Forward propagation to compute predictions.
Calculating loss using cross-entropy.
Backpropagation to compute gradients.
Optimizer step to update weights."""

def train(model, train_morse, train_text, optimizer, criterion, num_epochs=10, batch_size=32):
    """Train the transformer model on Morse code to text translation."""
    # Create DataLoader for training data
    train_dataset = torch.utils.data.TensorDataset(train_morse, train_text)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]  # Decoder input
            tgt_output = tgt[:, 1:]  # Ground truth
            optimizer.zero_grad()
            output = model(src, tgt_input)
            # Compute loss
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

"""We evaluate the model on the validation set by calculating loss and accuracy without updating model weights."""
def evaluate(model, val_morse, val_text, criterion, batch_size=32):
    """Evaluate the model on validation data."""
    val_dataset = torch.utils.data.TensorDataset(val_morse, val_text)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            output = model(src, tgt_input)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            total_loss += loss.item()
            pred = output.argmax(2)
            correct += (pred == tgt_output).sum().item()
            total += tgt_output.numel()
    val_loss = total_loss / len(val_loader)
    val_acc = correct / total
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    return val_loss, val_acc

"""After training, visualize loss and accuracy trends using Matplotlib."""
import matplotlib.pyplot as plt

def plot_metrics(history):
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.title("Loss Trends")
    plt.show()
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.legend()
    plt.title("Accuracy Trends")
    plt.show()    

"""Finally, we use the trained model for inference."""

def translate_sequence(sequence, model, morse_vocab, text_vocab, reverse_text_vocab):
    """Translate a Morse code sequence to text using the trained model."""
    model.eval()
    morse_tensor = [morse_vocab[ch] for ch in sequence if ch in morse_vocab]
    morse_tensor = torch.tensor(morse_tensor, dtype=torch.long).unsqueeze(0).to(device)
    tgt_input = torch.tensor([text_vocab[SOS_TOKEN]], dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(20):  # Maximum sequence length
            output = model(morse_tensor, tgt_input)
            next_token = output.argmax(2)[:, -1:]
            tgt_input = torch.cat((tgt_input, next_token), dim=1)
            if next_token.item() == text_vocab[EOS_TOKEN]:
                break

    translated_text = ''.join(reverse_text_vocab[idx.item()] for idx in tgt_input[0, 1:-1])
    return translated_text


# Example usage and training
if __name__ == "__main__":
    print("Starting Morse Code to Text Translation Training...")

    # Train the model
    print("\n=== TRAINING ===")
    train(model, train_morse, train_text, optimizer, criterion, num_epochs=50, batch_size=32)

    # Evaluate the model
    print("\n=== EVALUATION ===")
    val_loss, val_acc = evaluate(model, val_morse, val_text, criterion)

    
    print("\n=== TRANSLATION EXAMPLES ===")
    test_cases = [
        ("000 111 000", "SOS"),
        ("01 00", "AI"),
        ("0000 0 01 0100 0100 111", "HELLO"),
    ]

    for morse_seq, expected_text in test_cases:
        try:
            translated = translate_sequence(morse_seq, model, morse_vocab, text_vocab, reverse_text_vocab)
            status = "PASSED" if translated == expected_text else "FAILED"
            print(f"[{status}]")
            print(f"  Morse Input:    '{morse_seq}'")
            print(f"  Expected Text:  '{expected_text}'")
            print(f"  Predicted Text: '{translated}'\n")
        except Exception as e:
            print(f"Translation failed for '{morse_seq}': {e}")
"""
16. Image Captioning Architecture (CNN-RNN with Attention) in PyTorch

This script implements the neural network modules for an attention-based image 
captioning system. It bridges Computer Vision and Natural Language Processing by 
using a pre-trained ResNet convolutional backbone to extract spatial visual features 
(Encoder) and a Recurrent Neural Network with custom Bahdanau (additive) Attention 
to decode features into sentences (Decoder).

Learning Objectives:
1. Extract spatial visual feature grids by removing classifier heads from pre-trained CNNs.
2. Build a Bahdanau (additive) Attention module to align visual regions with sequence states.
3. Design a recurrent decoder merging text embeddings and visual context vectors.
"""

# We import the core torch library.
import torch

# nn contains the activation, convolution, recurrent, and linear layer layers.
# Documentation: https://pytorch.org/docs/stable/nn.html
import torch.nn as nn

# We import torchvision models to load our pre-trained convolutional backbone.
import torchvision.models as models

# ==========================================
# 1. ENCODER CNN: SPATIAL FEATURE EXTRACTION
# ==========================================
class EncoderCNN(nn.Module):
    def __init__(self, embed_size=256):
        super(EncoderCNN, self).__init__()
        # Load pre-trained ResNet-18 (pre-cached during transfer learning topic)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # We strip off the last average pooling and fully connected classification layers, 
        # extracting raw spatial activations.
        # This yields shape: [Batch, 512, 7, 7] for a 224x224 input image.
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        
        # We project the 512 channels down to the target embedding dimension (embed_size)
        self.projection = nn.Conv2d(in_channels=512, out_channels=embed_size, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, images):
        # 1.1 Extract visual features: [Batch, 512, 7, 7]
        features = self.resnet(images)
        # 1.2 Project channels: [Batch, embed_size, 7, 7]
        features = self.relu(self.projection(features))
        
        # 1.3 Flatten the spatial height and width into a sequential grid representation
        # Shape: [Batch, embed_size, 49]
        features = features.view(features.size(0), features.size(1), -1)
        
        # 1.4 Transpose so that spatial locations act as a sequence: [Batch, 49, embed_size]
        features = features.transpose(1, 2)
        return features

# ==========================================
# 2. ADDITIVE BAHDANAU ATTENTION MODULE
# ==========================================
class Attention(nn.Module):
    def __init__(self, encoder_dim=256, decoder_dim=256, attention_dim=256):
        super(Attention, self).__init__()
        # Linear layer projecting spatial encoder features
        self.encoder_linear = nn.Linear(encoder_dim, attention_dim)
        # Linear layer projecting decoder hidden state
        self.decoder_linear = nn.Linear(decoder_dim, attention_dim)
        # Alignment vector projecting combined features to scalar alignment scores
        self.v = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, encoder_features, decoder_hidden):
        """
        Args:
            encoder_features: spatial visual sequence, shape [Batch, 49, encoder_dim]
            decoder_hidden: current decoder state, shape [Batch, decoder_dim]
        """
        # Project encoder features: [Batch, 49, attention_dim]
        enc_proj = self.encoder_linear(encoder_features)
        
        # Project decoder state and expand along the temporal sequence dimension
        # Shape: [Batch, 1, attention_dim]
        dec_proj = self.decoder_linear(decoder_hidden).unsqueeze(1)
        
        # Sum both projected states and pass through Tanh to calculate energy scores.
        # Shape: [Batch, 49, 1]
        energy = self.v(self.tanh(enc_proj + dec_proj))
        
        # Compute softmax over sequence steps to obtain normalized attention weights
        # Shape: [Batch, 49, 1]
        attention_weights = torch.softmax(energy, dim=1)
        
        # Calculate visual context vector as a weighted sum of encoder spatial features
        # Shape: [Batch, encoder_dim]
        context_vector = (encoder_features * attention_weights).sum(dim=1)
        
        return context_vector, attention_weights

# ==========================================
# 3. ATTENTION-BASED LSTM DECODER
# ==========================================
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size=256, encoder_dim=256):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        
        # Word embedding layer mapping word tokens to vector space
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Additive attention module
        self.attention = Attention(encoder_dim=encoder_dim, decoder_dim=hidden_size)
        
        # Recurrent layer: processes combination of word embedding and visual context vector
        # Thus input dimension = embed_size + encoder_dim
        # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html
        self.lstm_cell = nn.LSTMCell(input_size=embed_size + encoder_dim, hidden_size=hidden_size)
        
        # Output prediction layer projecting hidden states to vocabulary distributions
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, encoder_features, captions):
        """
        Args:
            encoder_features: visual spatial features, shape [Batch, 49, embed_size]
            captions: ground-truth target word indices, shape [Batch, Seq_Len]
        """
        batch_size = encoder_features.size(0)
        seq_length = captions.size(1)
        
        # Embed captions: [Batch, Seq_Len, Embed_Size]
        embedded = self.embedding(captions)
        
        # Initialize decoder LSTM cell states with zeros
        h = torch.zeros(batch_size, self.lstm_cell.hidden_size).to(encoder_features.device)
        c = torch.zeros(batch_size, self.lstm_cell.hidden_size).to(encoder_features.device)
        
        # Container to store output vocabulary logits at each sequence step
        outputs = torch.zeros(batch_size, seq_length, self.vocab_size).to(encoder_features.device)
        
        # Loop through sequence timesteps (autoregressively predicting next words)
        for t in range(seq_length):
            # Compute visual attention weights and context vector using previous hidden state 'h'
            context_vector, _ = self.attention(encoder_features, h)
            
            # Combine current step word embedding with visual context vector
            lstm_input = torch.cat((embedded[:, t, :], context_vector), dim=1)
            
            # Step the LSTM cell
            h, c = self.lstm_cell(lstm_input, (h, c))
            
            # Project to vocabulary distribution and save
            outputs[:, t, :] = self.fc(h)
            
        return outputs

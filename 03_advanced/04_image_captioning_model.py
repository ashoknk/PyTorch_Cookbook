"""
16. Image Captioning Architecture (CNN-RNN with Attention) in PyTorch

This script implements the neural network modules for an attention-based image 
captioning system. An AI system that looks at a photograph and automatically describes 
what it sees in a written sentence (e.g., "A dog running through a grassy field").

It bridges Computer Vision and Natural Language Processing by 
using a pre-trained ResNet convolutional backbone to extract spatial visual features 
(Encoder) and a RNN with custom Bahdanau (additive) Attention 
to decode features into sentences (Decoder).

It bridges two major areas of AI by joining two specialized components:
1. The Encoder (CNN / ResNet):
   - Role: Acts as the "Eyes".
   - Function: Scans the image and breaks it down into a grid of 49 visual feature regions.
2. The Decoder (RNN / LSTM + Attention):
   - Role: Acts as the "Brain & Mouth".
   - Function: At each step, the Attention module acts like a spotlight looking 
     at specific grid regions while the LSTM generates the next word in the sentence.

Learning Objectives:
1. Extract spatial visual feature grids by removing classifier heads from pre-trained CNNs.
2. Build a Bahdanau (additive) Attention module to align visual regions with sequence states.
3. Design a recurrent decoder merging text embeddings and visual context vectors.
"""

import os
import ssl
# Import ssl to prevent certificate verification errors on macOS model downloads
ssl._create_default_https_context = ssl._create_unverified_context
# Point the Torch Home folder directly to your current working directory ('.')
os.environ['TORCH_HOME'] = './data/'

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
        # Load pre-trained ResNet-18 (pre-cached during transfer learning topic). 
        #Documentation: https://pytorch.org/vision/stable/models.html#torchvision.models.resnet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # We strip off the last average pooling and fully connected classification layers, extracting raw spatial activations.
        # models.resnet18().children()  model returns an iterator over its 10 immediate top-level module instances 
        # (Conv2d, BatchNorm2d, ReLU, MaxPool2d, etc.). The last 2 modules are AvgPool2d and Linear layers.
        # *list(...): Unpacks the sliced layer list
        # This yields shape: [Batch, 512, 7, 7] for a 224x224 input image.
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        
        # We project the 512 channels down to the target embedding dimension (embed_size)
        # self.projection uses a 1x1 Conv2d to reduce/project channels from 512 down to embed_size (e.g., 256),
        # yielding shape: [Batch, embed_size, 7, 7]
        self.projection = nn.Conv2d(in_channels=512, out_channels=embed_size, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, images):
        # 1.1 Extract visual features: [Batch, 512, 7, 7] (assuming 224x224 input image)
        features = self.resnet(images)
        
        # 1.2 Project 512 channels to target embedding 256 dimension: [Batch, embed_size, 7, 7]
        features = self.relu(self.projection(features))
        
        # 1.3 Flatten spatial height & width (7x7=49) into a flattened spatial grid
        # Shape: [Batch, embed_size, 49]
        features = features.view(features.size(0), features.size(1), -1)
        
        # 1.4 Transpose so spatial regions act as sequence timesteps for Attention/LSTM. transpose swaps two dimensions (dimension 1 and dimension 2)
        # Shape: [Batch, 49, embed_size]
        features = features.transpose(1, 2)
        return features

# ==========================================
# 2. ADDITIVE BAHDANAU ATTENTION MODULE
# ==========================================
#At each step, the Attention module acts like a spotlight looking 
#at specific grid regions while the LSTM generates the next word in the sentence.
class Attention(nn.Module):
    def __init__(self, encoder_dim=256, decoder_dim=256, attention_dim=256):
        super(Attention, self).__init__()
        # Linear layer projecting spatial encoder features
        self.encoder_linear = nn.Linear(encoder_dim, attention_dim)
        # Linear layer projecting decoder hidden state
        self.decoder_linear = nn.Linear(decoder_dim, attention_dim)
        # Alignment vector projecting combined features to scalar alignment scores
        self.v = nn.Linear(attention_dim, 1)
        #Output -Zero-centered between (-1, 1).
        self.tanh = nn.Tanh()

    def forward(self, encoder_features, decoder_hidden):
        """
        Args:
            encoder_features: spatial visual sequence, shape [Batch, 49, encoder_dim]
            decoder_hidden: current decoder state, shape [Batch, decoder_dim]
            The Attention class acts as a dynamic searchlight. At every word step, 
            it compares what the text decoder is currently thinking (decoder_hidden) 
            with all 49 image regions (encoder_features) to calculate where to look next
        """
        # Project encoder features: [Batch, 49, attention_dim]
        enc_proj = self.encoder_linear(encoder_features)
        
        # Project decoder state and expand along the temporal sequence dimension
        # Shape: [Batch, 1, attention_dim]
        dec_proj = self.decoder_linear(decoder_hidden).unsqueeze(1)
        
        # Sum both projected states and pass through Tanh to calculate energy scores.
        # Reduces the combined features down to a single raw score (an alignment score) for each of the 49 image locations
        # Shape: [Batch, 49, 1]
        energy = self.v(self.tanh(enc_proj + dec_proj))
        
        # Compute softmax over sequence steps to obtain normalized attention weights
        # Converts those raw scores into percentages/probabilities that sum to 1.0 (100%). 
        # These 49 numbers are the attention_weights—telling us what percentage of the network's focus 
        # is placed on each spot in the image grid ex. Region 12 might get 0.85 (85% focus), while other regions get 0.003
        # Shape: [Batch, 49, 1]
        attention_weights = torch.softmax(energy, dim=1)
        
        # Calculate visual context vector as a weighted sum of encoder spatial features
        # We multiply these percentages against the original encoder_features so that 
        # irrelevant regions get multiplied near 0.0, while the relevant visual region gets highlighted
        # Shape: [Batch, encoder_dim]
        context_vector = (encoder_features * attention_weights).sum(dim=1)
        
        # Return (context_vector, attention_weights):
        # - context_vector -> Fed into LSTM cell to predict next word.
        # - attention_weights -> Used to generate visual heatmaps for AI explainability.
        return context_vector, attention_weights

# ==========================================
# 3. ATTENTION-BASED LSTM DECODER
# ==========================================
    """The DecoderRNN acts as the Brain and Mouth of the system.   
    Its job is to take the raw visual features extracted from the image (by the CNN Encoder) and 
    generate a meaningful sentence word-by-word. It acts like a storyteller describing a photo one word at a time. 
    Acts as the language generator that converts visual features into written word sequences
    - Execution Flow per Timestep t:
        Previous State (h) + Visuals -> Attention -> Context Vector
        Context Vector + Current Word -> LSTMCell -> New State (h) -> FC -> Predicted Word Logits

    """

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, vocab_size, hidden_size=256, encoder_dim=256):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        
        # Word embedding layer mapping word tokens to vector space
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Additive attention module
        #At each step, the Attention module acts like a spotlight looking 
        #at specific grid regions while the LSTM generates the next word in the sentence.
        self.attention = Attention(encoder_dim=encoder_dim, decoder_dim=hidden_size)
        
        # Recurrent layer: processes combination of word embedding and visual context vector
        # Maintains recurrent memory (h, c) across time to track sentence grammar and context
        # Thus input dimension = embed_size + encoder_dim
        # Documentation: https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html
        self.lstm_cell = nn.LSTMCell(input_size=embed_size + encoder_dim, hidden_size=hidden_size)
        
        # Output prediction layer projecting hidden states to vocabulary distributions
        # Maps LSTM hidden states to vocabulary probability logits
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, encoder_features, captions):
        """
        Args:
            encoder_features: visual spatial features, shape [Batch, 49, embed_size]
            captions: ground-truth target word indices, shape [Batch, Seq_Len]
            When starting a brand-new caption at timestep t=0, there is no past memory yet. 
            Creating vectors filled with zeros provides a clean, neutral starting memory state for the batch
        """
        #Process groups (batches) of samples in parallel. batch_size tells how many images/captions are being processed simultaneously 
        batch_size = encoder_features.size(0) 
        # Represents the number of word timesteps in the sentence sequence (e.g., 5 words)
        seq_length = captions.size(1)
        
        # Embed captions: [Batch, Seq_Len, Embed_Size]
        embedded = self.embedding(captions)
        
        # Initialize decoder LSTM cell states with zeros
        h = torch.zeros(batch_size, self.lstm_cell.hidden_size).to(encoder_features.device)
        c = torch.zeros(batch_size, self.lstm_cell.hidden_size).to(encoder_features.device)
        
        # Container to store output vocabulary logits at each sequence step
        outputs = torch.zeros(batch_size, seq_length, self.vocab_size).to(encoder_features.device)
        
        # Loop through sequence timesteps (autoregressively predicting next words)
        # Iterations (steps) it must perform to finish generating the full caption.
        # The image input is inside encoder_features. Passing encoder_features into self.attention(encoder_features, h) gives Attention full access to all 49 image spots
        for t in range(seq_length):
            # Compute visual attention weights and context vector using previous hidden state 'h'
            context_vector, _ = self.attention(encoder_features, h)
            
            # Combine current step word embedding with visual context vector
            # Multimodal Concatenation (torch.cat):Glues current word embedding + visual context vector together so LSTM receives 
            # both text history and visual spotlight simultaneously. 
            # embedded : textual embedding of the current word at timestep t, context_vector : visual image context vector from attention
            lstm_input = torch.cat((embedded[:, t, :], context_vector), dim=1)
            
            # Step the LSTM cell
            h, c = self.lstm_cell(lstm_input, (h, c))
            
            # Project to vocabulary distribution. Contains raw numbers (logits) for each word in the vocabulary ex.[3, 4, 5] ->["dog", "runs", "outside"]
            # :: Selects ALL elements along the first two dimensions (batch and sequence) and all vocabulary logits along the last dimension
            # Shape: Batch Size, Sequence Length, Vocabulary Size. ex. 2 images, 5 words per caption, and a vocabulary of 6 words -> [2, 5, 6].
            outputs[:, t, :] = self.fc(h)
            
        return outputs
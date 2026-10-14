# PyTorch Cookbook: A Comprehensive Learning Curriculum (Vision & NLP)

Welcome to the **PyTorch Cookbook**, an exhaustive, progressive, hands-on learning curriculum designed to take you from fundamental tensor operations and deep learning building blocks to designing, training, and deploying advanced state-of-the-art Computer Vision (CV), Natural Language Processing (NLP), and Reinforcement Learning (RL) models.

---

## Curriculum Overview

This repository is structured into four sequential phases across **40 practical Python scripts**, each building upon the concepts, design patterns, and programming interfaces introduced in previous sections:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                1. BASICS                                 │
│  Tensors & Autograd ➔ Linear/Logistic Regression ➔ MLP ➔ Data Pipelines  │
└─────────────────────────────────────┬────────────────────────────────────┘
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                             2. INTERMEDIATE                              │
│   CNNs ➔ Autoencoders ➔ RGB & Augmentation ➔ Multi-Task ➔ RNNs ➔ Transfer │
└─────────────────────────────────────┬────────────────────────────────────┘
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                               3. ADVANCED                                │
│   Generative Models (GANs/VAEs) ➔ Style Transfer ➔ Captioning ➔ Transf.  │
│   ➔ Reinforcement Learning (CartPole & DQN)                              │
└─────────────────────────────────────┬────────────────────────────────────┘
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          4. UTILITIES & PROJECTS                         │
│   Monitoring (TensorBoard) ➔ Scaling (DDP, AMP, MPS) ➔ Export & Serving  │
│   ➔ Parameter-Efficient Fine-Tuning (LoRA from Scratch)                  │
└──────────────────────────────────────────────────────────────────────────┘
```

Below is the complete blueprint of the curriculum. For each topic, we provide a conceptual overview, associated file breakdown, key modules/classes, and execution flows.

---

# Table of Contents
1. [Phase 1: Basics](#phase-1-basics)
   - [1.1 PyTorch Tensors & Autograd Basics](#11-pytorch-tensors--autograd-basics)
   - [1.2 Linear Regression](#12-linear-regression)
   - [1.3 Logistic Regression](#13-logistic-regression)
   - [1.4 Feedforward Neural Network (MLP) & Vision Pipelines](#14-feedforward-neural-network-mlp--vision-pipelines)
   - [1.5 Custom Datasets & DataLoaders](#15-custom-datasets--dataloaders)
   - [1.6 Model Saving, Loading & Checkpointing](#16-model-saving-loading--checkpointing)
2. [Phase 2: Intermediate](#phase-2-intermediate)
   - [2.1 Convolutional Neural Network (CNN)](#21-convolutional-neural-network-cnn)
   - [2.2 Intermediate Vision & Representation Learning](#22-intermediate-vision--representation-learning)
   - [2.3 Recurrent Neural Networks: Foundations](#23-recurrent-neural-networks-foundations)
   - [2.4 Bidirectional Recurrent Neural Network](#24-bidirectional-recurrent-neural-network)
   - [2.5 Language Model (RNN-LM)](#25-language-model-rnn-lm)
   - [2.6 Transfer Learning & Fine-Tuning](#26-transfer-learning--fine-tuning)
3. [Phase 3: Advanced](#phase-3-advanced)
   - [3.1 Generative Adversarial Networks (DCGAN)](#31-generative-adversarial-networks-dcgan)
   - [3.2 Variational Autoencoder (VAE)](#32-variational-autoencoder-vae)
   - [3.3 Neural Style Transfer](#33-neural-style-transfer)
   - [3.4 Image Captioning (CNN-RNN with Attention)](#34-image-captioning-cnn-rnn-with-attention)
   - [3.5 Transformers: Native Architecture & Seq2Seq Translation](#35-transformers-native-architecture--seq2seq-translation)
   - [3.6 Reinforcement Learning: CartPole & Deep Q-Networks (DQN)](#36-reinforcement-learning-cartpole--deep-q-networks-dqn)
4. [Phase 4: Utilities & Projects](#phase-4-utilities--projects)
   - [4.1 TensorBoard Integration & Monitoring](#41-tensorboard-integration--monitoring)
   - [4.2 Learning Rate Schedulers & Hyperparameter Tuning](#42-learning-rate-schedulers--hyperparameter-tuning)
   - [4.3 Distributed, Mixed Precision & Hardware Acceleration](#43-distributed-mixed-precision--hardware-acceleration)
   - [4.4 Modern Model Compilation & Export (torch.export, ONNX)](#44-modern-model-compilation--export-torchexport-onnx)
   - [4.5 Web API Model Serving (FastAPI)](#45-web-api-model-serving-fastapi)
   - [4.6 Parameter-Efficient Fine-Tuning (LoRA from Scratch)](#46-parameter-efficient-fine-tuning-lora-from-scratch)
5. [Getting Started & Installation](#getting-started--installation)

---

# Phase 1: Basics

This phase introduces PyTorch's native data structures, automatic differentiation, optimization primitives, dataset abstractions, and model serialization.

---

### 1.1 PyTorch Tensors & Autograd Basics

#### Topic Overview & Description
Before building deep learning models, one must master the basic unit of PyTorch: the `Tensor`. This module introduces PyTorch tensor initialization, math operations, broadcasting, device routing (CPU, CUDA, and Apple Silicon MPS), and the central engine of PyTorch: **Autograd** (automatic differentiation). Learners will understand how PyTorch tracks operations dynamically to construct a Directed Acyclic Graph (DAG) for gradient computation.

#### Associated Python File Breakdown
*   **`01_basics/01a_pytorch_basics.py`**: A dedicated guide to initializing tensors without copying memory (`torch.from_numpy`, `torch.as_tensor`), creating constant/uninitialized tensors (`zeros`, `ones`, `full`, `empty`), generating sequences and ranges (`arange`, `linspace`, `eye`), and sampling from uniform, integer, and normal distributions.
*   **`01_basics/01b_pytorch_basics.py`**: Practical entry point covering tensor manipulations, device placement (`cuda`/`mps`), tensor math (`view`, `reshape`, `@`, `matmul`), Autograd computational graphs, `.backward()`, gradient accumulation, `.detach()`, and `torch.no_grad()`.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `numpy as np`
*   **Execution Flow**:
    1. **Memory-Shared Tensors**: Creating tensors from existing NumPy arrays without copying memory (`from_numpy`, `as_tensor`).
    2. **Device Routing**: Dynamically selecting and migrating tensors to `cuda` or `mps` if hardware accelerators are available.
    3. **Tensor Math**: Reshaping, slicing, transposing, and matrix multiplications (`matmul`, `@`).
    4. **Autograd Demonstration**: Creating a tensor with `requires_grad=True`, building forward functions, invoking `.backward()`, and examining `.grad`.
    5. **Gradient Detachment**: Optimizing inference memory using `.detach()` and `with torch.no_grad():` contexts.

---

### 1.2 Linear Regression

#### Topic Overview & Description
Linear Regression is the "Hello World" of parametric model training. Learners explore single-layer linear regression mechanics and fit a line $y = wx + b$ to synthetic noisy data. It demonstrates manual parameter inspections as well as custom optimization and loss interfaces provided by `torch.nn` and `torch.optim`.

#### Associated Python File Breakdown
*   **`01_basics/02a_linear_regression.py`**: Demonstrates the mechanics of a single `nn.Linear(in_features=1, out_features=1)` layer, inspecting weight matrices, bias vectors, and raw scalar predictions.
*   **`01_basics/02b_linear_regression.py`**: A complete training script fitting a line to noisy data using `nn.Module`, `nn.MSELoss`, and `optim.SGD`. Exports the regression plot to `01_basics/linear_regression_fit.png`.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn as nn`, `torch.optim as optim`, `matplotlib`, `matplotlib.pyplot as plt`
*   **Modules / Classes**:
    *   `class LinearRegressionModel(nn.Module)`: Inherits from `nn.Module`. Encapsulates `nn.Linear(1, 1)`.
*   **Execution Flow**:
    1. **Data Prep**: Generate synthetic data $y = 2x + 1 + \epsilon$ using `torch.randn`.
    2. **Instantiation**: Initialize model, select `nn.MSELoss()`, and configure `optim.SGD` with learning rate $0.01$.
    3. **Training Loop**: Standard forward pass, compute loss, zero gradients (`optimizer.zero_grad()`), backpropagate (`loss.backward()`), and update weights (`optimizer.step()`).
    4. **Evaluation & Plotting**: Compare true vs. learned weight and bias; plot prediction line over ground-truth points.

---

### 1.3 Logistic Regression

#### Topic Overview & Description
Logistic Regression extends linear modeling to binary classification by passing linear logits through a Sigmoid activation function, mapping predictions to probabilities between 0 and 1. This topic introduces binary classification loss functions, decision thresholds, and decision boundary visualization.

#### Associated Python File Breakdown
*   **`01_basics/03_logistic_regression.py`**: Generates synthetic 2D cluster data using scikit-learn, trains a single-layer logistic classifier, applies a 0.5 decision threshold, and visualizes the decision boundary in `01_basics/logistic_regression_boundary.png`.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn as nn`, `torch.optim as optim`, `sklearn.datasets.make_blobs`, `matplotlib.pyplot as plt`, `numpy as np`
*   **Modules / Classes**:
    *   `class LogisticRegressionModel(nn.Module)`: Combines an `nn.Linear(num_features, 1)` layer with `nn.Sigmoid()`.
*   **Execution Flow**:
    1. **Data Prep**: Create two-dimensional clustered data for binary labels ($0$ or $1$) via `make_blobs`.
    2. **Instantiation**: Instantiate `LogisticRegressionModel`, use `nn.BCELoss()`, and use `optim.Adam` / `optim.SGD`.
    3. **Training Loop**: Forward pass, loss computation, backward step, and parameter update.
    4. **Decision Boundary**: Apply a $0.5$ threshold to output probabilities to assign binary classes.
    5. **Visualization**: Plot loss curves and 2D linear decision boundaries separating the classes.

---

### 1.4 Feedforward Neural Network (MLP) & Vision Pipelines

#### Topic Overview & Description
Multi-Layer Perceptrons (MLPs) introduce non-linear activation functions (such as ReLU) and hidden layers, allowing the network to approximate complex non-linear functions. This module covers the vision preparation pipeline, dataset exploration, and multi-class classification on both MNIST and FashionMNIST.

#### Associated Python File Breakdown
*   **`01_basics/04a_feedfwd_neuralnw_normalize.py`**: Visual preprocessing and augmentation pipeline with `torchvision.transforms` (Resize, RandomCrop, RandomHorizontalFlip, ToTensor, and Channel Normalization), explaining why standardizing pixel decimals to zero-centered spreads $[-2.0, 2.0]$ stabilizes training.
*   **`01_basics/04b_feedfwd_neuralnw_MNIST.py`**: Demonstrates decoupled dataset loading and mini-batching on MNIST grayscale digits using `torch.utils.data.Dataset` and `DataLoader`.
*   **`01_basics/04c_feedfwd_neuralnw_FashionMNIST1.py`**: Parses the FashionMNIST test dataset to extract and display a unique exemplar for each of the 10 clothing categories in a 2x5 grid saved to `data/fashion_mnist_examples.png`.
*   **`01_basics/04d_feedfwd_neuralnw_FashionMNIST2.py`**: Full training and evaluation of a Multi-Layer Perceptron on FashionMNIST. Breaks down ReLU non-linearities and `nn.CrossEntropyLoss`.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn as nn`, `torch.optim as optim`, `torchvision`, `torchvision.transforms as transforms`, `torch.utils.data.DataLoader`
*   **Modules / Classes**:
    *   `class FeedforwardNet(nn.Module)`: Stacks `nn.Linear(784, hidden_size)`, `nn.ReLU()`, and `nn.Linear(hidden_size, num_classes)`.
*   **Execution Flow**:
    1. **Data Loading**: Download and normalize FashionMNIST into train and test `DataLoader` streams.
    2. **Architecture**: Flatten $28 \times 28$ image inputs, project through hidden dimensions with ReLU, and output 10-class logits.
    3. **Training & Metrics**: Train using `nn.CrossEntropyLoss()` and `optim.Adam`, logging epoch loss and training accuracy.
    4. **Evaluation**: Run inference on the test set under `torch.no_grad()` to compute classification accuracy.

---

### 1.5 Custom Datasets & DataLoaders

#### Topic Overview & Description
Real-world datasets rarely arrive pre-packaged. PyTorch provides flexible abstractions via `torch.utils.data.Dataset` and `DataLoader`. Learners implement lazy-loading data pipelines capable of handling custom file formats (e.g., CSV indexes, raw images, on-the-fly transforms).

#### Associated Python File Breakdown
*   **`01_basics/05_custom_dataset.py`**: Creates a mock disk-based dataset (CSV metadata and directory of images), implements a custom `Dataset` subclass with PIL lazy loading, chains torchvision transforms, and loads batches with shuffling.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.utils.data.Dataset`, `torch.utils.data.DataLoader`, `pandas as pd`, `PIL.Image`, `torchvision.transforms as transforms`, `os`
*   **Modules / Classes**:
    *   `class CustomDataset(Dataset)`: Implements `__init__`, `__len__`, and `__getitem__`.
*   **Execution Flow**:
    1. **Mock Data Generation**: Generates synthetic image files and a metadata CSV index file.
    2. **Dataset Subclassing**: Implement `__getitem__` to read images with PIL, read labels from pandas, and apply transforms.
    3. **Transforms Pipeline**: Chain augmentations (`transforms.Compose`) including crop, flip, and tensor normalization.
    4. **DataLoader Iteration**: Wrap in a `DataLoader` specifying `batch_size`, `shuffle=True`, and `num_workers`.

---

### 1.6 Model Saving, Loading & Checkpointing

#### Topic Overview & Description
Training robust models requires proper serialization routines. This module covers best practices for saving and loading model weights (`state_dict`), serializing complete training checkpoints (weights, optimizer state, epoch, and loss history) to resume training after interruptions, and cross-device loading across CUDA, Apple Silicon MPS, and CPU.

#### Associated Python File Breakdown
*   **`01_basics/06_model_save_load.py`**: Interactive pipeline demonstrating `state_dict` extraction, checkpoint creation (`checkpoint.tar`), resumed training loops, and cross-hardware deserialization (`map_location`).

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn as nn`, `torch.optim as optim`, `os`
*   **Modules / Classes**: `class SimpleMLP(nn.Module)`
*   **Execution Flow**:
    1. **Weight Serialization**: Extract `model.state_dict()` and serialize to disk via `torch.save()`.
    2. **Weight Restoration**: Reinstantiate architecture and load weights with `model.load_state_dict()`.
    3. **Full Checkpointing**: Save a dictionary containing `'epoch'`, `'model_state_dict'`, `'optimizer_state_dict'`, and `'loss'`.
    4. **Resumed Training**: Reload full checkpoints, map states back to model and optimizer, and resume training.
    5. **Cross-Device Deserialization**: Load models trained on CUDA/MPS onto CPU using `map_location=torch.device('cpu')`.

---

# Phase 2: Intermediate

This phase covers Convolutional Neural Networks for Vision, advanced image pipelines (denoising autoencoders, multi-channel processing, modern augmentations, multi-task learning), and Recurrent Neural Networks for sequence modeling, ending with transfer learning.

---

### 2.1 Convolutional Neural Network (CNN)

#### Topic Overview & Description
Convolutional layers use translation invariance and spatial locality to extract hierarchical feature maps from visual grids. This module explores convolution filters, batch normalization, max pooling, and dense classification heads on FashionMNIST.

#### Associated Python File Breakdown
*   **`02_intermediate/01_convolutional_neural_network.py`**: Custom deep CNN architecture using Conv2D, BatchNorm2d, MaxPool2d, and linear classification heads. Accompanied by the structural diagram `02_intermediate/01_convolutional_neural_network.png`.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn as nn`, `torch.optim as optim`, `torchvision`, `torchvision.transforms as transforms`, `torch.utils.data.DataLoader`
*   **Modules / Classes**:
    *   `class SimpleCNN(nn.Module)`: 2-stage convolutional feature extractor followed by flattened linear classifier.
*   **Execution Flow**:
    1. **Data Loading**: Load FashionMNIST with tensor conversions and normalization.
    2. **Model Definition**: $3 \times 3$ conv kernels, batch normalization to stabilize activations, max pooling downsampling, and linear classifier projection.
    3. **Training & Metrics**: Train with Adam optimizer and CrossEntropyLoss, logging batch loss and epoch statistics.
    4. **Evaluation**: Compute per-class and test set accuracy.

---

### 2.2 Intermediate Vision & Representation Learning

#### Topic Overview & Description
Real-world computer vision extends far beyond basic single-label grayscale classification. This comprehensive topic covers unsupervised representation learning, multi-channel RGB processing, modern data augmentation to combat overfitting, multi-task joint loss optimization, and visual dataset inspection.

#### Associated Python File Breakdown
*   **`02_intermediate/02a_autoencoder.py`**: Unsupervised representation learning and image denoising. Encodes FashionMNIST images into a compact bottleneck representation `(32, 7, 7)` and reconstructs clean images from noisy inputs using Transposed Convolutions (`nn.ConvTranspose2d`) and `nn.MSELoss`.
*   **`02_intermediate/02b_rgb_cnn.py`**: Multi-channel (3-channel RGB) CNN classification on CIFAR-10 ($32 \times 32 \times 3$). Implements a 3-block hierarchical feature extractor ($3 \rightarrow 32 \rightarrow 64 \rightarrow 128$) with Dropout and linear heads.
*   **`02_intermediate/02c_data_aug.py`**: Modern data augmentation with `torchvision.transforms.v2`. Combats overfitting using `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`, and `RandomErasing`. Compares non-augmented vs. augmented training pipelines.
*   **`02_intermediate/02d_multitask.py`**: Multi-Task Learning combining categorical classification and bounding box regression. Features a shared CNN backbone feeding two distinct output heads, dynamic synthetic shape generation (circles and rectangles), and a weighted multi-task loss combining `nn.CrossEntropyLoss` and `nn.SmoothL1Loss`.
*   **`02_intermediate/02e_cifar.py`**: CIFAR-10 dataset preview tool that extracts and plots a labeled 2x5 grid of sample images saved to `data/cifar10_examples.png`.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn as nn`, `torch.optim as optim`, `torchvision`, `torchvision.transforms.v2 as v2`, `torch.utils.data.Dataset`, `torch.utils.data.DataLoader`
*   **Modules / Classes**:
    *   `class DenoisingAutoencoder(nn.Module)`: Encoder with Conv2d bottleneck and Decoder with ConvTranspose2d.
    *   `class ColorCNN(nn.Module)`: 3-block RGB feature extractor with Dropout.
    *   `class RegularizedCNN(nn.Module)`: CNN model tested with modern v2 visual transformations.
    *   `class SyntheticShapeDataset(Dataset)`: Procedurally generates shapes with class labels and bounding box coordinates.
    *   `class MultiTaskShapeDetector(nn.Module)`: Shared CNN backbone branching into classification and bounding box regression heads.
*   **Execution Flow**:
    1. **Bottleneck & Reconstruction**: Compress images into latent channels, reconstruct using transposed convolutions, and compute MSE loss against clean ground truth.
    2. **Multi-Channel Hierarchies**: Process 3-channel RGB depth through progressive convolutional blocks to capture low-level edges, textures, and object components.
    3. **Augmentation Regularization**: Distort batches dynamically on-the-fly with v2 transforms, proving higher test generalization.
    4. **Multi-Task Optimization**: Forward images through shared backbones; calculate $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{class}} + \lambda \mathcal{L}_{\text{box}}$ and backpropagate jointly.

---

### 2.3 Recurrent Neural Networks: Foundations

#### Topic Overview & Description
Recurrent Neural Networks (RNNs) process sequential data by maintaining an evolving hidden state vector summarizing historical context. To counter vanishing gradients across long sequences, gated variants (LSTM and GRU) regulate information flow. This topic introduces sequence embeddings, provides an architectural comparison of recurrent engines, and trains a sentiment sequence classifier.

#### Associated Python File Breakdown
*   **`02_intermediate/03a_embeddings_rnn.py`**: Introduces sequence processing fundamentals: mapping token IDs to dense vectors via `nn.Embedding`, feeding sequences through `nn.RNN`, and extracting hidden state representations.
*   **`02_intermediate/03b_rnn_gru_lstm.py`**: Architectural comparison of Vanilla RNN (`nn.RNN`), GRU (`nn.GRU`), and LSTM (`nn.LSTM`). Analyzes gating mechanisms (Update, Reset, Input, Forget, Output gates), vanishing gradients, cell state $c_t$ mechanics, and computational efficiency. Accompanied by diagram `02_intermediate/03b_rnn_gru_lstm.png`.
*   **`02_intermediate/03c_recurrent_neuralnw.py`**: End-to-end recurrent sequence classifier with interchangeable engines (`RNN`, `LSTM`, `GRU`) trained on padded text sequences for sentiment analysis.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn as nn`, `torch.optim as optim`
*   **Modules / Classes**:
    *   `class SequenceClassifier(nn.Module)`: Houses an `nn.Embedding` layer, a customizable recurrent engine (`nn.RNN`, `nn.LSTM`, or `nn.GRU`), and a linear classification head.
*   **Execution Flow**:
    1. **Token Embeddings**: Map discrete token IDs to dense continuous representation spaces using `nn.Embedding`.
    2. **Recurrent Computation**: Pass sequences through chosen recurrent engines, returning output tensors and final hidden states $h_t$ (plus cell state $c_t$ for LSTM).
    3. **Classification**: Pass the final timestep representation to a linear layer for classification.
    4. **Batch Processing**: Handle variable-length sequences using padding and dynamic shape batching.

---

### 2.4 Bidirectional Recurrent Neural Network

#### Topic Overview & Description
Standard recurrent networks read sequences chronologically (left-to-right), leaving states at timestep $t$ unaware of upcoming context. Bidirectional RNNs process sequences forward and backward simultaneously, merging historical and future temporal dependencies to dramatically boost sequence classification and tagging performance.

#### Associated Python File Breakdown
*   **`02_intermediate/04_bidirectional_rnn.py`**: Constructs a Bidirectional LSTM (`nn.LSTM(..., bidirectional=True)`), concatenates forward and backward hidden states, and evaluates sentiment classification.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn as nn`, `torch.optim as optim`
*   **Modules / Classes**:
    *   `class BiLSTMClassifier(nn.Module)`: Implements a bidirectional LSTM sequence model.
*   **Execution Flow**:
    1. **Bidirectional Aggregation**: Unpack forward hidden state $h_f$ and backward hidden state $h_b$.
    2. **Feature Concatenation**: Concatenate both states (`torch.cat([h_f, h_b], dim=-1)`), producing representations of shape `[batch, 2 * hidden_size]`.
    3. **Classification Header**: Feed concatenated context to output classification layers.
    4. **Optimization**: Train on sequence classifications and verify bidirectional gradient flow.

---

### 2.5 Language Model (RNN-LM)

#### Topic Overview & Description
Language Modeling is an autoregressive task where a model learns to predict the next token given historical sequence context. This topic introduces character-level vocabularies, sequence target shifting, Truncated Backpropagation Through Time (BPTT), and generation with Softmax temperature scaling.

#### Associated Python File Breakdown
*   **`02_intermediate/05_language_model_rnn.py`**: Complete character-level language modeling script that parses text, constructs vocabularies, trains an LSTM with detached hidden states, and generates text from seed prompts.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn as nn`, `torch.optim as optim`, `numpy as np`
*   **Modules / Classes**:
    *   `class RNNLanguageModel(nn.Module)`: Combines `nn.Embedding`, multi-layer `nn.LSTM`, and linear projections to vocabulary size.
*   **Execution Flow**:
    1. **Vocab Construction**: Build character-to-index mapping dictionaries from input text corpora.
    2. **Autoregressive Formatting**: Shift targets right by one token ($X[t]$ predicts $y[t+1]$).
    3. **Truncated BPTT**: Feed batch sequences to the LSTM, detaching hidden states (`h.detach()`) between steps to prevent memory explosion.
    4. **Temperature Sampling**: Sample predictions using Softmax temperature scaling to control generation diversity and randomness.

---

### 2.6 Transfer Learning & Fine-Tuning

#### Topic Overview & Description
Training large deep networks from scratch is computationally expensive. Transfer learning leverages representations learned on vast datasets (e.g., ImageNet) by transferring weights to specialized downstream tasks. This topic covers feature extraction (freezing pre-trained convolutional backbones) versus full fine-tuning with differential learning rates.

#### Associated Python File Breakdown
*   **`02_intermediate/06_transfer_learning.py`**: Loads pre-trained ResNet-18 weights, freezes backbone parameters (`requires_grad = False`), swaps out the final fully connected classification head (`model.fc`), and configures optimizer parameter groups.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn as nn`, `torch.optim as optim`, `torchvision.models`, `torchvision.transforms as transforms`, `os`
*   **Execution Flow**:
    1. **Pretrained Backbone**: Load `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)`.
    2. **Freezing Weights**: Iterate through backbone parameters, setting `param.requires_grad = False`.
    3. **Head Replacement**: Replace `model.fc` with a newly initialized linear layer matching downstream class counts.
    4. **Targeted Optimization**: Provide only trainable parameters (`requires_grad=True`) to the optimizer.
    5. **Training & Verification**: Train and evaluate downstream adaptation speed.

---

# Phase 3: Advanced

This phase explores deep generative models (DCGAN, VAE), Neural Style Transfer, multi-modal image captioning with attention, native Transformer architectures, and Reinforcement Learning with Deep Q-Networks.

---

### 3.1 Generative Adversarial Networks (DCGAN)

#### Topic Overview & Description
Generative Adversarial Networks (GANs) formulate generative modeling as a minimax zero-sum game between two competing networks: a Generator ($G$) synthesizing images from random noise vectors, and a Discriminator ($D$) distinguishing real images from synthetic fabrications. Deep Convolutional GANs (DCGAN) use spatial convolutions and transposed convolutions to generate realistic 2D image grids.

#### Associated Python File Breakdown
*   **`03_advanced/01_generative_adversarial_network.py`**: DCGAN implementation trained on FashionMNIST to synthesize realistic clothing images from standard normal noise.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn as nn`, `torch.optim as optim`, `torchvision`, `torchvision.transforms as transforms`, `torchvision.utils as vutils`
*   **Modules / Classes**:
    *   `class Generator(nn.Module)`: Maps noise vectors $z$ to 2D image grids using `nn.ConvTranspose2d`, BatchNorm2d, and Tanh.
    *   `class Discriminator(nn.Module)`: Maps image grids to a real/fake probability using strided `nn.Conv2d`, LeakyReLU, and Sigmoid.
*   **Execution Flow**:
    1. **Minimax Formulation**: Use Binary Cross Entropy (`nn.BCELoss()`).
    2. **Discriminator Step**: Train $D$ on real and generated fake images; maximize $\log D(x) + \log(1 - D(G(z)))$.
    3. **Generator Step**: Train $G$ to fool the discriminator; maximize $\log D(G(z))$.
    4. **Sample Generation**: Periodically save grids of generated images to visualize image quality progression.

---

### 3.2 Variational Autoencoder (VAE)

#### Topic Overview & Description
Variational Autoencoders are generative models that map input data to a continuous, probabilistic latent space using variational inference. By enforcing a Gaussian prior on the latent distribution, VAEs allow smooth latent-space interpolations and stable synthesis of novel data points.

#### Associated Python File Breakdown
*   **`03_advanced/02_variational_autoencoder.py`**: VAE trained on MNIST digits featuring the Reparameterization Trick, combined Reconstruction and KL Divergence loss, and latent space sampling.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn as nn`, `torch.optim as optim`, `torchvision`, `torchvision.transforms as transforms`
*   **Modules / Classes**:
    *   `class VAE(nn.Module)`: Includes encoder predicting mean $\mu$ and log-variance $\log(\sigma^2)$, reparameterization logic, and decoder reconstruction.
*   **Execution Flow**:
    1. **Encoder Pass**: Map flattened images to latent parameters $\mu$ and $\log(\sigma^2)$.
    2. **Reparameterization Trick**: Sample $z = \mu + \epsilon \odot \sigma$ with standard normal noise $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ to maintain backprop differentiability.
    3. **Decoder Pass**: Reconstruct original images from latent sample $z$.
    4. **Combined Loss**: Optimize $\mathcal{L} = \text{ReconstructionLoss} + \text{KLDivergence}$.

---

### 3.3 Neural Style Transfer

#### Topic Overview & Description
Neural Style Transfer (NST) blends the semantic content of a "Content" image with the artistic texture and colors of a "Style" image. Using feature representations extracted from a pre-trained VGG-19 network, NST optimizes canvas pixel values directly using Gram matrices and the L-BFGS optimizer.

#### Associated Python File Breakdown
*   **`03_advanced/03_neural_style_transfer.py`**: Feature extraction using pre-trained VGG-19, Gram matrix computation for spatial texture correlations, and image pixel optimization using L-BFGS.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn as nn`, `torch.optim as optim`, `torchvision.models`, `torchvision.transforms as transforms`, `PIL.Image`, `copy`, `os`
*   **Modules / Classes**:
    *   `class VGGFeatureExtractor(nn.Module)`: Extracts intermediate activations across VGG-19 convolutional stages (`conv1_1` through `conv5_1`).
*   **Execution Flow**:
    1. **Initialization**: Initialize a target canvas image with `requires_grad=True`.
    2. **Loss Equations**:
        - Content Loss: Mean Squared Error between target and content feature representations.
        - Style Loss: Mean Squared Error between Gram matrices of target and style feature representations.
    3. **L-BFGS Optimization**: Execute optimization loops using `optim.LBFGS` to update pixel values directly.
    4. **Export**: Save stylistically blended output images.

---

### 3.4 Image Captioning (CNN-RNN with Attention)

#### Topic Overview & Description
Image Captioning bridges computer vision and natural language processing. The network extracts visual features using a pre-trained CNN backbone (Encoder) and decodes them into a textual caption using a recurrent neural network with Bahdanau additive attention (Decoder), dynamically focusing on relevant visual regions for each token.

#### Associated Python File Breakdown
*   **`03_advanced/04_image_captioning_model.py`**: Complete architecture implementation: spatial feature extractor (`EncoderCNN`), Bahdanau additive attention (`Attention`), and recurrent decoder (`DecoderRNN`).
*   **`03_advanced/05_image_captioning_train.py`**: Multi-module training pipeline handling tokenization, sequence masking, temporal cross-entropy loss, and end-to-end backpropagation.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn as nn`, `torch.nn.functional as F`, `torch.optim as optim`, `torchvision.models`
*   **Modules / Classes**:
    *   `class EncoderCNN(nn.Module)`: Strips classification layers from ResNet-50 to yield a $7 \times 7$ spatial grid of feature vectors.
    *   `class Attention(nn.Module)`: Computes alignment scores between visual spatial vectors and decoder hidden states.
    *   `class DecoderRNN(nn.Module)`: Merges visual context vectors and token embeddings using LSTM cells.
*   **Execution Flow**:
    1. **Visual Feature Extraction**: Extract spatial feature grids from input images.
    2. **Attention Weighting**: Compute attention coefficients across the visual grid at each sequence step.
    3. **Recurrent Decoding**: Feed combined visual context and word embeddings into the LSTM decoder.
    4. **Masked Loss**: Compute cross-entropy loss over unpadded token steps and update both models jointly.

---

### 3.5 Transformers: Native Architecture & Seq2Seq Translation

#### Topic Overview & Description
The Transformer architecture revolutionized sequence modeling by replacing recurrence with multi-head self-attention. This topic covers the native PyTorch Transformer architecture (`nn.Transformer`), tensor dimensionality conventions, positional encoding, attention masks, and full sequence-to-sequence translation.

#### Associated Python File Breakdown
*   **`03_advanced/06a_transformer_basic.py`**: Entry-level stepping stone demonstrating PyTorch native `nn.Transformer` tensor shape mechanics with `batch_first=True` and 3D input/output transformations.
*   **`03_advanced/06b_transformer_translation.py`**: Complete Seq2Seq English-to-French translation pipeline built with PyTorch's native `nn.Transformer`. Features positional encoding (`PositionalEncoding`), source padding masks, causal look-ahead target masks, teacher forcing, and greedy decoding.
*   **`03_advanced/06c_transformer_morsecode.py`**: Morse Code to Text Seq2Seq translation using a full Transformer model. Maps binary/dot-dash patterns to readable English text characters with custom vocabularies, multi-head attention, teacher forcing, and translation evaluations.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn as nn`, `torch.optim as optim`, `math`, `random`, `warnings`, `os`
*   **Modules / Classes**:
    *   `class PositionalEncoding(nn.Module)`: Injects sine/cosine position embeddings into token vectors.
    *   `class DirectTransformerTrainer`: Complete Seq2Seq training framework wrapping `nn.Transformer`.
    *   `class TransformerModel(nn.Module)`: Full encoder-decoder Seq2Seq translation architecture.
*   **Execution Flow**:
    1. **Shape Understanding**: Explore 3D tensor transitions through `nn.Transformer`.
    2. **Mask Preparation**: Generate source padding masks and causal subsequent masks (`generate_square_subsequent_mask`).
    3. **Teacher Forcing Training**: Feed shifted target sequences alongside source representations to optimize cross-entropy loss.
    4. **Autoregressive Decoding**: Implement greedy sequence inference starting from `<SOS>` until `<EOS>` tokens are predicted.

---

### 3.6 Reinforcement Learning: CartPole & Deep Q-Networks (DQN)

#### Topic Overview & Description
Reinforcement Learning (RL) trains agents to maximize cumulative rewards through environmental interactions. This topic introduces the standardized Gymnasium API, explores the classic CartPole control task, and implements a complete Deep Q-Network (DQN) agent with experience replay and target networks.

#### Associated Python File Breakdown
*   **`03_advanced/07a_cartpole.py`**: Exploration of Gymnasium's CartPole-v1 environment. Breaks down observation state spaces, discrete action spaces, episode reward structures, and environment stepping loops.
*   **`03_advanced/07b_dqn_reinforcement_learning.py`**: Full Deep Q-Network (DQN) implementation solving CartPole-v1. Features a neural Q-network, an experience replay buffer, an $\epsilon$-greedy exploration schedule, and target network synchronization.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn as nn`, `torch.optim as optim`, `gymnasium as gym`, `random`, `collections.deque`
*   **Modules / Classes**:
    *   `class QNetwork(nn.Module)`: Neural network mapping state vectors to Q-values for all discrete actions.
    *   `class ReplayBuffer`: Buffer class storing transition tuples `(state, action, reward, next_state, done)`.
*   **Execution Flow**:
    1. **Environment Stepping**: Agent selects actions using $\epsilon$-greedy policy (exploring randomly vs. exploiting Q-network).
    2. **Replay Buffer**: Store transition tuples to break temporal correlations between successive steps.
    3. **Bellman Optimization**: Sample random mini-batches and compute target $r + \gamma \max_{a'} Q_{\text{target}}(s', a')$.
    4. **Target Network Sync**: Periodically update target network weights to stabilize training.

---

# Phase 4: Utilities & Projects

This phase covers developer tooling, including experiment tracking, learning rate scheduling, distributed multi-GPU training, Apple Silicon hardware acceleration, modern model compilation (`torch.export`), ONNX export, FastAPI model serving, and parameter-efficient fine-tuning (LoRA).

---

### 4.1 TensorBoard Integration & Monitoring

#### Topic Overview & Description
Tracking deep learning experiments via console prints is difficult and error-prone. PyTorch's native TensorBoard integration enables real-time visual monitoring. This module covers logging scalar metrics (loss, accuracy), computational graphs, parameter weight/gradient histograms, and input image grids.

#### Associated Python File Breakdown
*   **`04_utilities/01_tensorboard_visualization.py`**: Training pipeline with TensorBoard callbacks streaming metrics, model graphs, weight histograms, and image grids to `./runs/fashion_mnist_experiment`.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn as nn`, `torch.utils.tensorboard.SummaryWriter`, `torchvision`, `torchvision.utils as vutils`, `shutil`
*   **Modules / Classes**: `class SimpleCNN(nn.Module)`
*   **Execution Flow**:
    1. **SummaryWriter Setup**: Point writer to run directory.
    2. **Graph Serialization**: Trace model computational graph using `writer.add_graph(model, dummy_input)`.
    3. **Image Grids**: Upload sample batch grids using `writer.add_image("input_images", img_grid)`.
    4. **Metric Logging**: Stream batch and epoch loss/accuracy via `writer.add_scalar()`.
    5. **Weight Histograms**: Monitor gradient distributions over epochs with `writer.add_histogram()`.

---

### 4.2 Learning Rate Schedulers & Hyperparameter Tuning

#### Topic Overview & Description
Static learning rates often lead to suboptimal convergence. Learning rate schedulers dynamically adapt learning rates during training. This topic covers standard PyTorch scheduler policies—including `StepLR`, `CosineAnnealingLR`, and `ReduceLROnPlateau`—and compares their convergence trajectories.

#### Associated Python File Breakdown
*   **`04_utilities/02_lr_schedulers.py`**: Demonstrates the impact of different schedulers on optimization and compares their convergence behaviors across epochs.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn as nn`, `torch.optim as optim`, `torch.optim.lr_scheduler`
*   **Execution Flow**:
    1. **Scheduler Instantiation**: Configure `StepLR` (interval decay), `CosineAnnealingLR` (cosine cycles), and `ReduceLROnPlateau` (metric-driven decay).
    2. **Epoch Stepping**: Call `scheduler.step()` per epoch or pass validation metrics for plateau detection.
    3. **Comparison Routine**: Compare learning rate trajectories and loss curves.

---

### 4.3 Distributed, Mixed Precision & Hardware Acceleration

#### Topic Overview & Description
Scaling deep learning models requires efficient hardware utilization. Automatic Mixed Precision (AMP) accelerates computation and cuts memory usage using FP16/BF16 precision. Distributed Data Parallel (DDP) distributes training across multiple GPUs with synchronized gradient updates. Furthermore, hardware acceleration on Apple Silicon (MPS) provides local GPU acceleration.

#### Associated Python File Breakdown
*   **`04_utilities/03a_distributed_amp_training.py`**: Production-grade distributed training with `DistributedDataParallel`, `DistributedSampler` data partitioning, and Automatic Mixed Precision (`torch.amp.autocast` and `GradScaler`).
*   **`04_utilities/03b_distributed_amp_training.py`**: Benchmarks matrix math acceleration on Apple Silicon GPUs (`mps`) against CPU execution using large matrix multiplications ($4096 \times 4096$).

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn as nn`, `torch.optim as optim`, `torch.distributed as dist`, `torch.nn.parallel.DistributedDataParallel`, `torch.utils.data.distributed.DistributedSampler`, `torch.amp.autocast`, `torch.amp.GradScaler`, `torch.multiprocessing as mp`, `time`, `os`
*   **Execution Flow**:
    1. **Process Group Initialization**: Initialize distributed process groups with `dist.init_process_group`.
    2. **Model & Data Wrapping**: Wrap models with `DistributedDataParallel` and partition datasets with `DistributedSampler`.
    3. **AMP Training Loop**: Execute forward passes in `autocast()`, scale loss with `GradScaler`, and execute scaled optimizer steps.
    4. **MPS Benchmarking**: Check `torch.backends.mps.is_available()`, allocate high-dimensional tensors on MPS, and compare runtime against CPU.

---

### 4.4 Modern Model Compilation & Export (torch.export, ONNX)

#### Topic Overview & Description
Deploying PyTorch models into production environments (C++ applications, mobile devices, TensorRT, or ONNX Runtime) requires language-agnostic compiled representations. This module covers the modern PyTorch 2.x `torch.export` graph capture framework (exporting `.pt2` packages), dynamic batch shapes, standard ONNX export, and runtime verification using ONNX Runtime.

#### Associated Python File Breakdown
*   **`04_utilities/04_model_export.py`**: Exports a trained image classifier using PyTorch 2.x `torch.export.export()`, saves compiled `.pt2` packages, defines dynamic shapes for variable batch sizes, exports to `.onnx`, and verifies numerical parity using `onnxruntime`.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn as nn`, `torch.export`, `onnx`, `onnxruntime as ort`, `os`
*   **Modules / Classes**: `class SimpleImageClassifier(nn.Module)`
*   **Execution Flow**:
    1. **Model Training**: Train a lightweight image classifier.
    2. **PyTorch 2.x torch.export**: Capture computational graphs using `torch.export.export()` with dynamic batch dimensions (`torch.export.Dim`).
    3. **Artifact Serialization**: Save and reload compiled `.pt2` packages via `torch.export.save()` and `load()`.
    4. **ONNX Export**: Call `torch.onnx.export()` to generate standalone `.onnx` model graphs.
    5. **ONNX Runtime Validation**: Load the ONNX model using `onnxruntime.InferenceSession` and verify outputs match PyTorch predictions.

---

### 4.5 Web API Model Serving (FastAPI)

#### Topic Overview & Description
Serving deep learning models requires low latency, asynchronous request handling, and robust input validation. This module implements a production-ready REST API using FastAPI that loads compiled models during startup and serves image classification requests.

#### Associated Python File Breakdown
*   **`04_utilities/05_model_serve_api.py`**: FastAPI application featuring an asynchronous lifespan startup handler, multipart image upload endpoint (`/predict`), real-time PIL/torchvision preprocessing, sub-millisecond inference, and JSON responses.

#### Code Architecture & Implementation Details
*   **Key Imports**: `fastapi`, `uvicorn`, `torch`, `torchvision.transforms as transforms`, `PIL.Image`, `io`, `contextlib.asynccontextmanager`
*   **Execution Flow**:
    1. **Lifespan Startup**: Load model weights into memory during application startup using `@asynccontextmanager`.
    2. **Request Routing**: Define `/health` check and `/predict` POST endpoints handling multipart form data.
    3. **Image Preprocessing**: Convert uploaded image byte buffers into PIL images and normalized tensors.
    4. **Inference & Serialization**: Execute model inference and return top predicted classes and probabilities as JSON.

---

### 4.6 Parameter-Efficient Fine-Tuning (LoRA from Scratch)

#### Topic Overview & Description
Fine-tuning modern large foundation models is computationally demanding. Parameter-Efficient Fine-Tuning (PEFT) methods freeze base weights and insert a tiny fraction of trainable parameters. Low-Rank Adaptation (LoRA) decomposes weight update matrices into two low-rank matrices ($A$ and $B$, where $r \ll d$). This module walks through building a custom LoRA layer from scratch and verifying parameter efficiency.

#### Associated Python File Breakdown
*   **`04_utilities/06_lora_peft_finetuning.py`**: Educational implementation of custom LoRA layers from scratch, parameter audits (trainable vs. frozen parameters), and fine-tuning verification.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn as nn`, `torch.optim as optim`, `math`
*   **Modules / Classes**:
    *   `class LoraLinear(nn.Module)`: Wrapper layer replacing standard linear layers:
        - Frozen base linear layer (`nn.Linear`).
        - Trainable low-rank matrix $A$ (dim $d_{\text{in}} \times r$) and matrix $B$ (dim $r \times d_{\text{out}}$, initialized to zero).
        - Scaling factor $\alpha / r$.
        - Forward pass: $h = W_{\text{base}}x + \frac{\alpha}{r}(x A) B$.
    *   `class PretrainedClassifier(nn.Module)`: Base neural network used to inject LoRA adapters.
*   **Execution Flow**:
    1. **Adapter Injection**: Replace linear layers with custom `LoraLinear` wrappers.
    2. **Freezing Base Parameters**: Set `param.requires_grad = False` across base weights, leaving only matrices $A$ and $B$ trainable.
    3. **Parameter Audit**: Verify that trainable parameters constitute $< 1\%$ of total parameters.
    4. **Fine-Tuning Loop**: Train adapters on new data, confirming that base weights remain untouched.

---

# Getting Started & Installation

To run this cookbook locally:

### 1. Clone the repository
```bash
git clone https://github.com/ashoknk/PyTorch_Cookbook.git
cd PyTorch_Cookbook
```

### 2. Set up environment

#### Using `uv` (Recommended)
This repository uses `pyproject.toml` managed with [`uv`](https://github.com/astral-sh/uv):
```bash
# Create virtual environment and install all dependencies from lockfile
uv sync

# Or create a venv manually:
uv venv
source .venv/bin/activate
uv add torch torchvision matplotlib scikit-learn pandas tensorboard gymnasium onnx onnxruntime onnxscript fastapi uvicorn python-multipart
```

#### Using standard `pip`
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision matplotlib scikit-learn pandas tensorboard gymnasium onnx onnxruntime onnxscript fastapi uvicorn python-multipart
```

---

### Dependency Overview

The project dependencies defined in `pyproject.toml` are categorized as follows:

1. **Core Deep Learning & Computer Vision**
   - `torch` ($\ge 2.14.0$): Core tensor operations, Autograd engine, and neural network primitives.
   - `torchvision` ($\ge 0.29.0$): Datasets (MNIST, FashionMNIST, CIFAR-10), pre-trained architectures (ResNet, VGG), and vision transforms (including modern `v2`).

2. **Scientific Computing & Data Processing**
   - `matplotlib` ($\ge 3.11.2$): Plotting loss trajectories, regression lines, and visual dataset grids.
   - `scikit-learn` ($\ge 1.9.1$): Generating synthetic cluster data (`make_blobs`) for classification tasks.
   - `pandas` ($\ge 3.0.6$): CSV parsing and metadata indexing for custom dataset pipelines.

3. **Model Monitoring, Reinforcement Learning & Export**
   - `tensorboard` ($\ge 2.21.0$): Streaming scalars, model graphs, images, and histograms.
   - `gymnasium` ($\ge 1.3.0$): Standardized reinforcement learning environments (CartPole-v1).
   - `onnx` ($\ge 1.23.0$) & `onnxscript` ($\ge 0.7.2$): Exporting PyTorch computational graphs to ONNX formats.
   - `onnxruntime` ($\ge 1.31.0$): Cross-platform, high-performance execution engine for validating ONNX models.

4. **Production Web Serving**
   - `fastapi` ($\ge 0.141.1$): Asynchronous REST API framework for model serving endpoints.
   - `uvicorn` ($\ge 0.53.0$): High-performance ASGI server for production deployment.
   - `python-multipart` ($\ge 0.0.32$): Streaming parser for handling multipart image file uploads.

---

### Running the Scripts

Run any script directly using your virtual environment Python interpreter:
```bash
# Phase 1: Basics
python 01_basics/01a_pytorch_basics.py
python 01_basics/04d_feedfwd_neuralnw_FashionMNIST2.py

# Phase 2: Intermediate
python 02_intermediate/02a_autoencoder.py
python 02_intermediate/02d_multitask.py
python 02_intermediate/06_transfer_learning.py

# Phase 3: Advanced
python 03_advanced/06b_transformer_translation.py
python 03_advanced/07b_dqn_reinforcement_learning.py

# Phase 4: Utilities
python 04_utilities/04_model_export.py
python 04_utilities/05_model_serve_api.py
python 04_utilities/06_lora_peft_finetuning.py
```

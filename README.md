# PyTorch Cookbook: A Comprehensive Learning Curriculum (Vision & NLP)

Welcome to the **PyTorch Cookbook**, an exhaustive, progressive, hands-on learning curriculum designed to take you from a complete beginner in tensor operations and deep learning to an expert capable of designing, training, and deploying advanced state-of-the-art Computer Vision (CV) and Natural Language Processing (NLP) models.

---

## Curriculum Overview

This repository is structured into four sequential phases, each building upon the concepts, design patterns, and programming interfaces introduced in the previous sections:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                1. BASICS                                 │
│  Tensors & Autograd ➔ Linear/Logistic Regression ➔ MLP ➔ Data pipelines   │
└─────────────────────────────────────┬────────────────────────────────────┘
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                             2. INTERMEDIATE                              │
│       CNNs ➔ Custom ResNets ➔ RNNs, LSTMs, GRUs ➔ Transfer Learning      │
└─────────────────────────────────────┬────────────────────────────────────┘
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                               3. ADVANCED                                │
│   Generative Models (GANs/VAEs) ➔ Style Transfer ➔ Transformers & DQNs   │
└─────────────────────────────────────┬────────────────────────────────────┘
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          4. UTILITIES & PROJECTS                         │
│   Monitoring (TensorBoard) ➔ Scaling (DDP, AMP) ➔ serving ➔ PEFT (LoRA)  │
└──────────────────────────────────────────────────────────────────────────┘
```

Below is the complete blueprint of our **23-topic curriculum**. For each topic, we provide a deep conceptual overview, the target Python file paths, and the exact code architectures and execution flows.

---

# Table of Contents
1. [Phase 1: Basics](#phase-1-basics)
   - [1.1 PyTorch Tensors & Autograd Basics](#11-pytorch-tensors--autograd-basics)
   - [1.2 Linear Regression](#12-linear-regression)
   - [1.3 Logistic Regression](#13-logistic-regression)
   - [1.4 Feedforward Neural Network (MLP)](#14-feedforward-neural-network-mlp)
   - [1.5 Custom Datasets & DataLoaders](#15-custom-datasets--dataloaders)
   - [1.6 Model Saving, Loading & Checkpointing](#16-model-saving-loading--checkpointing)
2. [Phase 2: Intermediate](#phase-2-intermediate)
   - [2.1 Convolutional Neural Network (CNN)](#21-convolutional-neural-network-cnn)
   - [2.2 Deep Residual Network (ResNet)](#22-deep-residual-network-resnet)
   - [2.3 Recurrent Neural Network (RNN)](#23-recurrent-neural-network-rnn)
   - [2.4 Bidirectional Recurrent Neural Network](#24-bidirectional-recurrent-neural-network)
   - [2.5 Language Model (RNN-LM)](#25-language-model-rnn-lm)
   - [2.6 Transfer Learning & Fine-Tuning](#26-transfer-learning--fine-tuning)
3. [Phase 3: Advanced](#phase-3-advanced)
   - [3.1 Generative Adversarial Networks (GANs & DCGAN)](#31-generative-adversarial-networks-gans--dcgan)
   - [3.2 Variational Autoencoder (VAE)](#32-variational-autoencoder-vae)
   - [3.3 Neural Style Transfer](#33-neural-style-transfer)
   - [3.4 Image Captioning (CNN-RNN with Attention)](#34-image-captioning-cnn-rnn-with-attention)
   - [3.5 Transformers from Scratch](#35-transformers-from-scratch)
   - [3.6 Deep Q-Networks (DQN) for Reinforcement Learning](#36-deep-q-networks-dqn-for-reinforcement-learning)
4. [Phase 4: Utilities & Projects](#phase-4-utilities--projects)
   - [4.1 TensorBoard Integration & Monitoring](#41-tensorboard-integration--monitoring)
   - [4.2 Learning Rate Schedulers & Hyperparameter Tuning](#42-learning-rate-schedulers--hyperparameter-tuning)
   - [4.3 Distributed & Mixed Precision Training (DDP & AMP)](#43-distributed--mixed-precision-training-ddp--amp)
   - [4.4 Serving PyTorch Models (TorchScript, ONNX, and FastAPI)](#44-serving-pytorch-models-torchscript-onnx-and-fastapi)
   - [4.5 Parameter-Efficient Fine-Tuning (LoRA from Scratch)](#45-parameter-efficient-fine-tuning-lora-from-scratch)

---

# Phase 1: Basics

This phase introduces PyTorch's native data structures, automatic differentiation, optimization primitives, dataset abstractions, and basic serialization.

---

### 1.1 PyTorch Tensors & Autograd Basics

#### Topic Overview & Description
Before building deep learning models, one must master the basic unit of PyTorch: the `Tensor`. This module introduces PyTorch tensor initialization, math operations, broadcasting, device management (CPU vs. GPU/CUDA vs. Apple Silicon MPS), and the central engine of PyTorch: **Autograd** (automatic differentiation). Learners will understand how PyTorch tracks operations dynamically to construct a Directed Acyclic Graph (DAG) for gradient computation.

#### Associated Python File Breakdown
*   **`01_basics/01_pytorch_basics.py`**: A clean, educational script showcasing tensor manipulations and the Autograd engine. Contains self-contained, commented verification code.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`
*   **Modules / Classes**: No complex model classes needed here. Focused on mathematical demonstrations.
*   **Execution Flow**:
    1. **Creation**: Initializing tensors from lists, numpy arrays, and probability distributions (rand, randn).
    2. **Device Routing**: Dynamically selecting and migrating tensors to `cuda` or `mps` if available.
    3. **Tensor Math**: Slicing, reshaping (`view`, `reshape`), transposing, matrix multiplication (`@`, `matmul`).
    4. **Autograd Demonstration**: Creating a tensor with `requires_grad=True`, building a forward equation (e.g., $y = 3x^2 + 2x$), invoking `.backward()`, and examining `.grad`.
    5. **Gradient Detachment**: Understanding the use of `.detach()` and `with torch.no_grad():` block contexts.

---

### 1.2 Linear Regression

#### Topic Overview & Description
Linear Regression is the "Hello World" of parametric model training. Learners will fit a line $y = wx + b$ to synthetic, noisy data. It demonstrates how parameter updates can be executed manually and introduces the custom optimization and loss interfaces provided by `torch.nn` and `torch.optim`.

#### Associated Python File Breakdown
*   **`01_basics/02_linear_regression.py`**: A script generating data, initializing parameters, running a training loop from scratch, and plotting the convergence.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn`, `torch.optim`, `matplotlib.pyplot`
*   **Modules / Classes**:
    *   `class LinearRegressionModel(nn.Module)`: Inherits from `nn.Module`. Defines a single `nn.Linear(1, 1)` layer.
*   **Execution Flow**:
    1. **Data Prep**: Generate synthetic data $y = 2x + 1 + \epsilon$ using `torch.randn`.
    2. **Instantiation**: Initialize model, select `nn.MSELoss()`, and select the `optim.SGD` optimizer with a learning rate of $0.01$.
    3. **Training Loop** (e.g., 100 epochs):
        - Forward pass: Predict $\hat{y}$.
        - Loss computation.
        - Backward pass: `optimizer.zero_grad()` to reset computed gradients, followed by `loss.backward()`.
        - Parameter update: `optimizer.step()`.
    4. **Evaluation & Plotting**: Compare true vs. learned weight and bias; plot prediction line over ground-truth points.

---

### 1.3 Logistic Regression

#### Topic Overview & Description
Logistic Regression extends linear modeling to binary classification by piping linear logits through a logistic Sigmoid function, mapping output boundaries to probabilities between 0 and 1. This topic introduces binary classification loss functions and decision threshold boundaries.

#### Associated Python File Breakdown
*   **`01_basics/03_logistic_regression.py`**: A complete classification pipeline with a training loop and boundary visualization.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn`, `torch.optim`, `sklearn.datasets` (optional, for toy classification data generation), `matplotlib.pyplot`
*   **Modules / Classes**:
    *   `class LogisticRegressionModel(nn.Module)`: Combines an `nn.Linear(num_features, 1)` layer with a final `nn.Sigmoid()`.
*   **Execution Flow**:
    1. **Data Prep**: Create two-dimensional clustered data for binary labels ($0$ or $1$).
    2. **Instantiation**: Instantiate `LogisticRegressionModel`, use `nn.BCELoss()` (Binary Cross Entropy), and use `optim.Adam` (or SGD).
    3. **Training Loop**: Execute standard forward, loss, backward, step routine.
    4. **Inference**: Apply a $0.5$ threshold to output probabilities to assign classes.
    5. **Visualization**: Plot the training loss curve and the linear decision boundary separating the classes.

---

### 1.4 Feedforward Neural Network (MLP)

#### Topic Overview & Description
An extension of single-layer linear models, Multi-Layer Perceptrons (MLPs or FNNs) introduce non-linear activation functions (ReLU, GeLU) and hidden layers, allowing the network to approximate complex non-linear functions. Learners will implement an MLP for multi-class digit classification on the classic MNIST dataset.

#### Associated Python File Breakdown
*   **`01_basics/04_feedforward_neural_network.py`**: A self-contained script that downloads MNIST, defines a multi-layer network, trains it, and computes multi-class test metrics.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn`, `torch.optim`, `torchvision.datasets`, `torchvision.transforms`
*   **Modules / Classes**:
    *   `class FeedforwardNet(nn.Module)`: Contains:
        - `nn.Linear(784, hidden_size)` (flattened $28 \times 28$ image inputs)
        - `nn.ReLU()`
        - `nn.Linear(hidden_size, num_classes)` (10 classes for digits 0-9)
*   **Execution Flow**:
    1. **Data Loading**: Retrieve MNIST via `torchvision.datasets.MNIST`. Transform inputs to tensors and normalize them. Use default PyTorch `DataLoader` for batching.
    2. **Instantiation**: Setup `FeedforwardNet`, `nn.CrossEntropyLoss()` (note: standard PyTorch cross-entropy handles log-softmax internally, so logits are directly output), and `optim.Adam`.
    3. **Training Loop**: Iterate over epochs, process mini-batches, update weights, and compute training accuracy.
    4. **Testing Loop**: Put model in `model.eval()`, disable gradients (`torch.no_grad()`), and compute final test set accuracy.

---

### 1.5 Custom Datasets & DataLoaders

#### Topic Overview & Description
For real-world projects, data does not arrive pre-packaged. PyTorch offers excellent extensibility using the `Dataset` and `DataLoader` abstractions. Learners will learn to build memory-safe, lazy-loading data pipelines for custom file formats (e.g., CSV, raw image directories, JSON text files).

#### Associated Python File Breakdown
*   **`01_basics/05_custom_dataset.py`**: A complete pipeline simulation that writes a mock disk dataset (CSV + images), custom loads them, applies augmentations, and loads batches.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.utils.data.Dataset`, `torch.utils.data.DataLoader`, `pandas`, `PIL.Image`, `torchvision.transforms`
*   **Modules / Classes**:
    *   `class CustomDataset(Dataset)`: Must implement:
        - `__init__(self, csv_file, img_dir, transform=None)`: Save metadata path references.
        - `__len__(self)`: Return size of dataset.
        - `__getitem__(self, idx)`: Load image from path, load target from CSV, apply transform, return `(tensor_image, tensor_label)`.
*   **Execution Flow**:
    1. **Synthetic Generation**: Automatically write a few dummy images and a CSV index file locally.
    2. **Transforms Pipeline**: Chain augmentations (`transforms.Compose`) like random crop, flip, and normalization.
    3. **Loader Setup**: Instantiate custom `Dataset`, pass to a `DataLoader` specifying `batch_size=4`, `shuffle=True`, and `num_workers=2`.
    4. **Iteration Validation**: Run a dummy loop validating tensor shapes yielded by the loader.

---

### 1.6 Model Saving, Loading & Checkpointing

#### Topic Overview & Description
Training robust models takes hours or days. This module covers best practices for saving and loading model weights (`state_dict`), exporting complete model architectures, and serializing full training checkpoints (weights, optimizer state, epoch index, historical loss) to resume training after interruptions.

#### Associated Python File Breakdown
*   **`01_basics/06_model_saving_loading.py`**: An interactive pipeline showcasing saving and loading processes under various hardware contexts.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn`, `torch.optim`
*   **Modules / Classes**: Uses a simple FNN for demonstration.
*   **Execution Flow**:
    1. **Saving Weights**: Extract `model.state_dict()`, serialize to disk (`torch.save(..., 'model.pth')`).
    2. **Loading Weights**: Re-instantiate model architecture, call `model.load_state_dict(torch.load('model.pth'))`.
    3. **Comprehensive Checkpointing**:
        - Store dictionary containing `'epoch'`, `'model_state_dict'`, `'optimizer_state_dict'`, and `'loss'`.
        - Save as `'checkpoint.tar'`.
    4. **Resume Flow**: Load checkpoint, map back to model and optimizer state, and continue synthetic training loop.
    5. **Cross-Device Loading**: Load models trained on CUDA to CPU using the `map_location=torch.device('cpu')` argument.

---

# Phase 2: Intermediate

This phase covers Convolutional Neural Networks for Vision and Recurrent Neural Networks for sequential/text data, culminating in advanced transfer learning.

---

### 2.1 Convolutional Neural Network (CNN)

#### Topic Overview & Description
Convolutional layers leverage inductive biases—specifically translation invariance and spatial locality—to process visual grids. This module covers convolution filters, pooling operations, batch normalization, and dense classification heads. It trains a custom classifier from scratch on CIFAR-10.

#### Associated Python File Breakdown
*   **`02_intermediate/01_convolutional_neural_network.py`**: CIFAR-10 vision classification using a custom deep CNN architecture.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn`, `torch.optim`, `torchvision.datasets`, `torchvision.transforms`
*   **Modules / Classes**:
    *   `class SimpleCNN(nn.Module)`: Implements a sequence of Conv2D layers, Batch Normalization, ReLU activations, MaxPool2D, followed by standard linear classification layers.
*   **Execution Flow**:
    1. **Data Fetching**: Load CIFAR-10 images ($32 \times 32 \times 3$). Apply normalization.
    2. **Model Definition**: Use $3 \times 3$ conv kernels to double feature maps (e.g., $3 \rightarrow 32 \rightarrow 64 \rightarrow 128$), downsample with MaxPool2D, flatten, and map to 10 linear outputs.
    3. **Training & Metrics**: Train using Adam optimizer. Print batch loss and overall training loop metrics.
    4. **Evaluation**: Compute validation accuracy per-class to identify performance discrepancies.

---

### 2.2 Deep Residual Network (ResNet)

#### Topic Overview & Description
As neural networks get deeper, they suffer from the "vanishing/exploding gradient" problem. ResNets introduce residual (shortcut/skip) connections that bypass intermediate layers, allowing gradients to flow directly back through the network. This topic covers designing a deep residual network from scratch.

#### Associated Python File Breakdown
*   **`02_intermediate/02_deep_residual_network.py`**: Implementing ResNet blocks and stacking them to build a customizable ResNet-9 or ResNet-18 variant for image classification.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn`, `torch.optim`
*   **Modules / Classes**:
    *   `class ResidualBlock(nn.Module)`: Standard block with two Conv2D layers, Batch Norm, an optional downsample pathway (1x1 conv to match channels), and an element-wise addition step.
    *   `class ResNet(nn.Module)`: Stackable architecture using multiple layers of `ResidualBlock` followed by global average pooling and a fully connected layer.
*   **Execution Flow**:
    1. **Block Assembly**: Instantiate `ResidualBlock` and demonstrate the skip connection logic.
    2. **Stacking Architecture**: Initialize a complete ResNet with custom channel depths.
    3. **Training Loop**: Execute a mini-epoch loop with synthetic image tensors to verify structural and mathematical correctness.

---

### 2.3 Recurrent Neural Network (RNN)

#### Topic Overview & Description
Recurrent Neural Networks (RNNs) process sequential data by maintaining a recurrent hidden state vector that summarizes historical step information. To resolve vanishing gradients over long sequences, LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) architectures introduce gate mechanisms. This topic introduces sequential inputs for multi-class sentiment classification.

#### Associated Python File Breakdown
*   **`02_intermediate/03_recurrent_neural_network.py`**: A clean, sequence classification script.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn`, `torch.optim`
*   **Modules / Classes**:
    *   `class SequenceClassifier(nn.Module)`: Houses an `nn.Embedding` layer, a customizable recurrent engine (`nn.RNN`, `nn.LSTM`, or `nn.GRU`), and a linear layer mapping hidden states to logit dimensions.
*   **Execution Flow**:
    1. **Sequence Modeling**: Simulate sequence processing where inputs are tokens indices representing text sequences.
    2. **Recurrent Computation**: Map word indices to dense vector spaces. Feed to LSTM, retrieve final hidden step state $h_t$ (or cell state $c_t$ if LSTM).
    3. **Classification**: Pass the final state to a dense layer for binary classification.
    4. **Validation**: Test input shapes, padding handling, and sequence mask validations.

---

### 2.4 Bidirectional Recurrent Neural Network

#### Topic Overview & Description
Standard Recurrent networks read sequences in a single direction (left-to-right), meaning states at step $t$ lack knowledge of context at step $t+1$. Bidirectional RNNs process sequences forwards and backwards concurrently. The combined hidden representation captures complete temporal dependencies, significantly improving classification tasks.

#### Associated Python File Breakdown
*   **`02_intermediate/04_bidirectional_rnn.py`**: A script evaluating bidirectional sequence models for token or sequence-level tagging.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn`, `torch.optim`
*   **Modules / Classes**:
    *   `class BiLSTMClassifier(nn.Module)`: Defines an `nn.LSTM` with `bidirectional=True`.
*   **Execution Flow**:
    1. **Bidirectional Aggregation**: Unpack forward hidden state $h_f$ and backward hidden state $h_b$.
    2. **Feature Concatenation**: Concatenate both states (`torch.cat([h_f, h_b], dim=-1)`), resulting in a layer shape of $2 \times \text{hidden\_size}$.
    3. **Classification Header**: Feed concatenated context to output classification layers.
    4. **Loop Evaluation**: Train on synthetic sequence classifications to verify gradient flow.

---

### 2.5 Language Model (RNN-LM)

#### Topic Overview & Description
Unlike sequence classification, Language Modeling is an autoregressive task where a model learns to predict the next word/token given historical sequence context. This topic introduces generative character-level text generation and Truncated Backpropagation Through Time (BPTT).

#### Associated Python File Breakdown
*   **`02_intermediate/05_language_model_rnn.py`**: A complete text training script that accepts text, builds vocab mapping, trains an LSTM language model, and dynamically generates new text.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn`, `torch.optim`
*   **Modules / Classes**:
    *   `class RNNLanguageModel(nn.Module)`: Embedding layer, multiple LSTM layers, and a dense output projection layer map to vocabulary sizes.
*   **Execution Flow**:
    1. **Vocab Building**: Extract character or token vocab from input corpora.
    2. **Autoregressive Prep**: Slice inputs so target outputs are shifted right by 1 token ($X[1:]$ matches $y[:-1]$).
    3. **BPTT Optimization**: Feed batch data to LSTM, detaching recurrent hidden state variables between backprop steps to prevent unbounded memory utilization.
    4. **Generation & Sampling**: Create a sampling routine that accepts a seed string, applies Softmax temperature scaling, draws token indices, and translates output indices back to characters.

---

### 2.6 Transfer Learning & Fine-Tuning

#### Topic Overview & Description
Training models from scratch is computationally expensive and requires vast amounts of data. Transfer learning leverages rich features learned on huge baseline datasets (e.g., ImageNet) by transferring weights to new, specialized down-stream tasks. This topic covers feature extraction (freezing backbone weights) versus full model fine-tuning.

#### Associated Python File Breakdown
*   **`02_intermediate/06_transfer_learning.py`**: Load pre-trained structures, apply differential learning rates, and target specific head adjustments.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn`, `torch.optim`, `torchvision.models`, `torchvision.transforms`
*   **Modules / Classes**: Uses models like `torchvision.models.resnet50` or `efficientnet_b0`.
*   **Execution Flow**:
    1. **Backbone Loading**: Load pre-trained model instances (`weights=ResNet50_Weights.DEFAULT`).
    2. **Backbone Freezing**: Iterate through parameters (`param.requires_grad = False`) to prevent gradient updates.
    3. **Classifier Replacement**: Swap out the final fully connected layer (`model.fc`) to match target dataset class outputs (e.g., 2 classes).
    4. **Differential Optimizers**: Setup optimizer parameters assigning a larger learning rate to the classifier head and a zero/smaller learning rate to the core feature layers.
    5. **Evaluation**: Compare baseline target adaptation rates.

---

# Phase 3: Advanced

This phase explores deep generative models, cross-domain visual-text alignment, sequence-to-sequence Transformers from scratch, and Reinforcement Learning agents.

---

### 3.1 Generative Adversarial Networks (GANs & DCGAN)

#### Topic Overview & Description
Generative Adversarial Networks (GANs) represent generative modeling as a zero-sum game between two competing networks: a Generator ($G$), which synthesizes realistic images from random noise vectors, and a Discriminator ($D$), which evaluates images to distinguish real inputs from synthesized fabrications.

#### Associated Python File Breakdown
*   **`03_advanced/01_generative_adversarial_network.py`**: A script training a Deep Convolutional GAN (DCGAN) on MNIST to generate synthetic handwritten digits.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn`, `torch.optim`, `torchvision.utils`
*   **Modules / Classes**:
    *   `class Generator(nn.Module)`: Maps standard normal distribution noise vectors $z$ to visual pixel outputs using a series of transposed convolution layers (`nn.ConvTranspose2d`), Batch Norm, and ReLU activations, ending in a Tanh activation.
    *   `class Discriminator(nn.Module)`: Standard convolutional classifier mapping visual grids to a single binary confidence output (real vs. fake) using Strided Convolution layers, LeakyReLU activations, and a Sigmoid output.
*   **Execution Flow**:
    1. **Minimax Optimization Setup**: Use standard Binary Cross Entropy Loss (`nn.BCELoss()`).
    2. **Discriminator Step**: Train $D$ on both real and fake data; maximize objective function $\log D(x) + \log(1 - D(G(z)))$.
    3. **Generator Step**: Train $G$ to fool the discriminator; maximize objective $\log D(G(z))$.
    4. **Artifact Export**: Periodically compile grid images of synthetic outputs to track generation quality.

---

### 3.2 Variational Autoencoder (VAE)

#### Topic Overview & Description
Variational Autoencoders are generative models that map input data to a continuous, probabilistic latent space using variational inference. Unlike standard autoencoders, VAEs ensure the latent space is smooth and regularized by enforcing a Gaussian prior. This allows for stable latent-space interpolation and image synthesis.

#### Associated Python File Breakdown
*   **`03_advanced/02_variational_autoencoder.py`**: A VAE script that trains on MNIST, generates novel handwritten digits, and plots latent-space interpolations.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn`, `torch.optim`
*   **Modules / Classes**:
    *   `class VAE(nn.Module)`: Includes:
        - Encoder: Maps inputs to latent space parameters $\mu$ (mean) and $\log(\sigma^2)$ (log variance).
        - Reparameterization: Samples $z = \mu + \epsilon \odot \sigma$ using standard normal noise $\epsilon$.
        - Decoder: Maps $z$ back to reconstructed inputs.
*   **Execution Flow**:
    1. **Reparameterization Trick**: Crucial step allowing backpropagation to flow through stochastic nodes by keeping the sampling step non-differentiable.
    2. **Loss Function**: Custom optimization combining:
        - Reconstruction Loss (Binary Cross-Entropy or MSE).
        - Kullback-Leibler (KL) Divergence to measure latent deviation from $\mathcal{N}(0, \mathbf{I})$.
    3. **Training & Generation**: Run the training loop; generate new images by sampling directly from the latent normal distribution $\mathcal{N}(0, \mathbf{I})$.

---

### 3.3 Neural Style Transfer

#### Topic Overview & Description
Neural Style Transfer (NST) blends the semantic layout of a "Content" image with the artistic texture of a "Style" image. Originally proposed by Gatys et al., NST extracts intermediate feature representations from a pretrained CNN (e.g., VGG-19) and optimizes the pixel values of a target image to minimize content and style discrepancies.

#### Associated Python File Breakdown
*   **`03_advanced/03_neural_style_transfer.py`**: An optimization script that loads content/style images, extracts VGG-19 features, and optimizes the target output image using L-BFGS.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn`, `torch.optim`, `torchvision.models`, `torchvision.transforms`, `PIL.Image`
*   **Modules / Classes**:
    *   `class VGGFeatureExtractor(nn.Module)`: Extracts intermediate activations from target layers (e.g., `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1` for style, and `conv4_2` for content).
*   **Execution Flow**:
    1. **Image Initialization**: Clone Content image as starting target canvas ($G$), set `requires_grad=True` on $G$.
    2. **Loss Equations**:
        - Content Loss: Mean squared error of content vs. target feature maps.
        - Style Loss: Gram matrices (outer product of feature vectors to capture spatial correlations) of style vs. target features across multiple layers.
    3. **Optimization Loop**: Use the L-BFGS optimizer (`optim.LBFGS`) because of its high convergence quality in visual optimization. Run forward passes through VGG-19, calculate combined style/content loss, and update $G$'s pixel values directly.

---

### 3.4 Image Captioning (CNN-RNN with Attention)

#### Topic Overview & Description
Image Captioning bridges computer vision and natural language processing. The network extracts visual features using a pretrained CNN backbone (Encoder) and decodes them into a textual caption using a recurrent neural network with attention (Decoder). This architecture allows the decoder to focus on specific spatial regions of the image as it generates each word.

#### Associated Python File Breakdown
*   **`03_advanced/04_image_captioning_model.py`**: Implements the modular architecture, featuring a custom Bahdanau (additive) attention module, a CNN feature projector, and an attention-based LSTM decoder.
*   **`03_advanced/05_image_captioning_train.py`**: Handles vocabulary mapping, dataset tokenization, dynamic seq-mask loaders, and the training loop.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn`, `torch.nn.functional`
*   **Modules / Classes**:
    *   `class EncoderCNN(nn.Module)`: Strips classifier head off of pretrained models (e.g., ResNet-50) and projects spatial feature grids (e.g., $14 \times 14 \times 2048$) to embedding dimensions.
    *   `class Attention(nn.Module)`: Takes visual spatial vectors and hidden decoder state to calculate alignment context coefficients (attention weights).
    *   `class DecoderRNN(nn.Module)`: Combines word embeddings, visual context vectors, and previous hidden states using LSTM cells to predict the next word.
*   **Execution Flow**:
    1. **Feature Extraction**: CNN extracts a spatial grid of visual features.
    2. **Sequence Decoding**: The LSTM decoder receives the starting `<start>` token and the visual features.
    3. **Attention Check**: At each token step, calculate attention scores over the spatial feature grid using the decoder's hidden state, generating a weighted visual context vector.
    4. **Token Generation**: Feed the concatenation of visual context and token embeddings to the LSTM, update hidden states, and predict the next word in the vocabulary.

---

### 3.5 Transformers from Scratch

#### Topic Overview & Description
The Transformer architecture revolutionized sequence modeling by replacing recurrence entirely with self-attention. This module walks through implementing the original encoder-decoder Transformer from scratch (Attention is All You Need). Learners will build and assemble all core components, including scaled dot-product attention, multi-head attention blocks, and positional encoding layers.

#### Associated Python File Breakdown
*   **`03_advanced/06_transformer_model.py`**: Implements positional encoding, attention heads, encoder/decoder layer blocks, and final assembly.
*   **`03_advanced/07_transformer_train.py`**: Translation sequence loading, custom masks generation, and training verification.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn`, `torch.nn.functional`
*   **Modules / Classes**:
    *   `class ScaledDotProductAttention(nn.Module)`: Calculates $\text{Softmax}(\frac{QK^T}{\sqrt{d_k}})V$.
    *   `class MultiHeadAttention(nn.Module)`: Projects inputs to multiple heads, calculates attention outputs, and merges representations.
    *   `class PositionalEncoding(nn.Module)`: Computes sine and cosine frequency wave matrix representations to inject position awareness.
    *   `class EncoderLayer(nn.Module)` & `class DecoderLayer(nn.Module)`: Encapsulates self-attention, cross-attention (for decoder), feed-forward layers, and residual connections with Layer Normalization.
*   **Execution Flow**:
    1. **Mask Generation**: Build causal source padding masks and look-ahead autoregressive target masks.
    2. **Transformer Forward Pass**: Pass source sequence to Encoder; pass shifted target inputs to Decoder alongside encoder outputs; compute final vocabulary logits.
    3. **Verification Loop**: Run a dummy Seq2Seq translation training sequence to confirm shape consistency.

---

### 3.6 Deep Q-Networks (DQN) for Reinforcement Learning

#### Topic Overview & Description
Reinforcement Learning (RL) trains agents to maximize cumulative rewards through environmental interactions. This topic implements a Deep Q-Network (DQN) to solve classic control tasks (e.g., CartPole). It covers Q-learning theory, experience replay buffers to break temporal correlations, and target networks to stabilize learning.

#### Associated Python File Breakdown
*   **`03_advanced/08_dqn_reinforcement_learning.py`**: A complete, self-contained RL pipeline containing environment interfacing, replay memory, target updates, and agent performance tracking.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn`, `torch.optim`, `gymnasium`, `random`, `collections`
*   **Modules / Classes**:
    *   `class QNetwork(nn.Module)`: Simple FNN mapping state vectors to discrete action Q-values.
    *   `class ReplayBuffer`: Buffer class storing transition tuples `(state, action, reward, next_state, done)`.
*   **Execution Flow**:
    1. **Epsilon-Greedy Interaction**: The agent balances exploration (random action selection) and exploitation (using current Q-network predictions).
    2. **Replay Storage**: Store transition tuples in the `ReplayBuffer`.
    3. **Optimization Step**: Sample random mini-batches from replay buffer.
    4. **Loss Calculation**: Minimize Mean Squared Error (or Huber loss) between predicted $Q(s, a)$ and target values $r + \gamma \max_{a'} Q_{\text{target}}(s', a')$.
    5. **Target Network Sync**: Periodically update the target network's weights to match the active training network.

---

# Phase 4: Utilities & Projects

This phase covers standard developer tooling, including model monitoring, learning rate scheduling, distributed scaling, export and serving frameworks, and parameter-efficient fine-tuning (LoRA).

---

### 4.1 TensorBoard Integration & Monitoring

#### Topic Overview & Description
Tracking deep learning experiments using raw console prints is error-prone and hard to interpret. PyTorch's native TensorBoard integration enables real-time monitoring of model runs. This module covers logging scalar metrics (loss, accuracy), weights distribution histograms, model graph architectures, input images, and embedding spaces.

#### Associated Python File Breakdown
*   **`04_utilities/01_tensorboard_visualization.py`**: A clean training pipeline featuring custom logging callbacks that stream metrics to TensorBoard directories.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.utils.tensorboard.SummaryWriter`, `torchvision.utils`
*   **Modules / Classes**: Uses standard CNN classifier architecture for logging.
*   **Execution Flow**:
    1. **SummaryWriter Initialization**: Setup writer pointing to `./runs/experiment_1`.
    2. **Graph Serialization**: Trace model graph with dummy inputs using `writer.add_graph(model, dummy_input)`.
    3. **Image Logging**: Log input sample grids using `writer.add_image("input_images", img_grid)`.
    4. **Metric Logging**: Stream batch/epoch validation loss metrics via `writer.add_scalar("loss/train", loss, global_step)`.
    5. **Weight Histograms**: Monitor gradient distributions over epochs with `writer.add_histogram("weights/layer1", model.conv1.weight, epoch)`.

---

### 4.2 Learning Rate Schedulers & Hyperparameter Tuning

#### Topic Overview & Description
Static learning rates can lead to sub-optimal convergence. Learning rate schedulers dynamically adjust learning rates during training (e.g., decaying after epochs, following cosine cycles, or dropping on plateaus). This topic covers implementing learning rate schedulers and setting up basic grid searches.

#### Associated Python File Breakdown
*   **`04_utilities/02_lr_schedulers.py`**: Demonstrates the impact of different schedulers on loss landscape optimization.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.optim`, `torch.optim.lr_scheduler`
*   **Modules / Classes**: Demonstrates `StepLR`, `CosineAnnealingLR`, and `ReduceLROnPlateau`.
*   **Execution Flow**:
    1. **Instantiate Scheduler**: Define learning rate schedules linked to target optimizers.
    2. **Epoch Stepping**: Call `scheduler.step()` at the end of each epoch (or pass validation metrics for `ReduceLROnPlateau`).
    3. **Hyperparameter Tuning Routine**: Implement a simple, automated nested loop to evaluate and compare multiple scheduler and learning rate combinations.
    4. **Metric Export**: Export training records for comparison.

---

### 4.3 Distributed & Mixed Precision Training (DDP & AMP)

#### Topic Overview & Description
As datasets and models grow, training efficiency becomes critical. Automatic Mixed Precision (AMP) reduces memory footprint and training times by dynamically running computations in mixed FP16/BF16 precision without sacrificing model convergence. Distributed Data Parallel (DDP) scales training across multiple GPUs by duplicating the model and running synchronous gradient updates on partitioned datasets.

#### Associated Python File Breakdown
*   **`04_utilities/03_distributed_amp_training.py`**: High-performance script containing dual GPU DDP wrapping combined with Automatic Mixed Precision.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn.parallel.DistributedDataParallel`, `torch.utils.data.distributed.DistributedSampler`, `torch.cuda.amp.autocast`, `torch.cuda.amp.GradScaler`, `torch.distributed`
*   **Modules / Classes**: Modular training script designed to handle multiple process ranks.
*   **Execution Flow**:
    1. **DDP Setup**: Initialize process groups using `torch.distributed.init_process_group`.
    2. **DDP Wrapping**: Map model to local GPUs and wrap using `DistributedDataParallel(model, device_ids=[local_rank])`.
    3. **Data Partitioning**: Use `DistributedSampler` to ensure process threads receive independent, non-overlapping dataset slices.
    4. **AMP Scaler**: Initialize `GradScaler` to prevent floating-point underflow.
    5. **Training Loop**: Execute model forward pass under the `autocast()` context, backpropagate scaled gradients, and execute step updates via `scaler.step(optimizer)`.

---

### 4.4 Serving PyTorch Models (TorchScript, ONNX, and FastAPI)

#### Topic Overview & Description
Transitioning from research to production requires serving models through high-performance, language-agnostic APIs. This module covers compiling models using TorchScript (tracing and scripting) to bypass Python runtime limitations, exporting to ONNX (Open Neural Network Exchange), and deploying the compiled model as a FastAPI web service endpoint.

#### Associated Python File Breakdown
*   **`04_utilities/04_model_export.py`**: Formats and exports a trained image classifier into TorchScript and ONNX models.
*   **`04_utilities/05_model_serve_api.py`**: A production-ready API endpoint serving sub-millisecond compiled models.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `fastapi`, `uvicorn`, `onnx`, `PIL.Image`, `io`
*   **Modules / Classes**:
    *   FastAPI application framework implementing high-speed endpoint routes.
*   **Execution Flow**:
    1. **Tracing Compilation**: Trace a trained model (`torch.jit.trace`) using a dummy tensor to export a JIT compiler representation.
    2. **ONNX Export**: Call `torch.onnx.export` to export an ONNX representation.
    3. **Web API Setup**: Build a FastAPI web app with an `/predict` POST endpoint.
    4. **Inference Execution**: On request, read image bytes, apply visual transformations, execute inference using the TorchScript/ONNX engine, and return class probabilities as JSON.

---

### 4.5 Parameter-Efficient Fine-Tuning (LoRA from Scratch)

#### Topic Overview & Description
Fine-tuning modern large models with billions of parameters is computationally prohibitive. Parameter-Efficient Fine-Tuning (PEFT) methods freeze base weights and insert a small number of trainable parameters. Low-Rank Adaptation (LoRA) decomposes weight update matrices into two low-rank matrices ($A$ and $B$, where $r \ll d$). This module walks through implementing a custom LoRA layer from scratch and applying it to freeze/fine-tune a neural network.

#### Associated Python File Breakdown
*   **`04_utilities/06_lora_peft_finetuning.py`**: A complete, educational script implementing custom LoRA layers, wrapping neural network blocks, and verifying efficiency.

#### Code Architecture & Implementation Details
*   **Key Imports**: `torch`, `torch.nn`, `math`
*   **Modules / Classes**:
    *   `class LoraLinear(nn.Module)`: A wrapper replacing standard `nn.Linear` layers. It contains:
        - The frozen base linear layer (`nn.Linear`).
        - Two low-rank trainable layers: $A$ (dim $d \times r$, initialized with Gaussian) and $B$ (dim $r \times d$, initialized with zeros).
        - A scale parameter $\alpha / r$.
        - Forward pass: $h = W_{\text{base}}x + \frac{\alpha}{r}(B \times A)x$.
*   **Execution Flow**:
    1. **Adapter Insertion**: Inject custom `LoraLinear` layers into pre-trained classification models.
    2. **Freeze Parameter Logic**: Set `requires_grad = False` across all base layers, leaving only the low-rank $A$ and $B$ adapter parameters trainable.
    3. **Parameter Audit**: Print and compare trainable vs. frozen parameters.
    4. **Verification Step**: Run a sample epoch loop to verify that only LoRA weights update.

---

## Getting Started

To explore or run this cookbook locally:

1. **Clone this repository**:
   ```bash
   git clone https://github.com/your-username/pytorch-cookbook.git
   cd pytorch-cookbook
   ```

2. **Set up environment**:
   Using `pip` or virtualenv:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install torch torchvision torchaudio gymnasium fastapi uvicorn matplotlib pandas scikit-learn
   ```
   ```bash
   uv venv
   uv add torch torchvision matplotlib scikit-learn pandas tensorboard gymnasium onnx onnxscript fastapi
    uv add uvicorn python-multipart
    ```

    To run all 26 educational scripts across the Basics, Intermediate, Advanced, and Utilities phases of this
    PyTorch Cookbook, you will need to add the following libraries.

    Here is the recommended categorized list of libraries:

    1. **Core Deep Learning & Computer Vision**
         - `torch`: The core tensor and Autograd framework.
         - `torchvision`: Specialized computer vision datasets (FashionMNIST), pre-trained models (VGG-19,
             ResNet-18), and image transformations.

    2. **Scientific Computing & Data Processing**
         - `matplotlib`: Data plotting and regression line visualizations.
         - `scikit-learn`: Generating synthetic cluster data (`make_blobs`) for classification tasks.
         - `pandas`: CSV parsing and DataFrame metadata indexing for custom dataset loaders.

    3. **Model Monitoring, Reinforcement Learning & Export**
         - `tensorboard`: Streaming scalars, models, and histograms to TensorBoard.
         - `gymnasium`: Initializing the CartPole-v1 control environment for DQN RL.
         - `onnx` and `onnxscript`: Exporting PyTorch graphs to ONNX models on newer Python interpreters.

    4. **Production Web Serving**
         - `fastapi`: Asynchronous web server framework for API prediction endpoints.
         - `uvicorn`: Production-ready ASGI server to run the FastAPI server.
         - `python-multipart`: Parsing uploaded multipart image file bytes in POST requests.


3. **Browse topics**:
   Follow files chronologically to progress from Phase 1 (`01_basics/`) through Phase 4 (`04_utilities/`).

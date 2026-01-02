
## 🌍 Transformer Architecture from Scratch (English → Spanish NMT)
**A from-scratch PyTorch implementation** of the Transformer model for **English → Spanish neural machine translation**, trained on the **OPUS Books dataset**. This project closely follows the architecture introduced in **“Attention Is All You Need” (Vaswani et al., 2017)**, including multi-head attention, positional encoding, and encoder–decoder layers, providing a clear and practical example of building a Transformer from the ground up. It’s ideal for learning, experimentation, and hands-on practice with sequence-to-sequence models

## ⚙️ Installation
### 1. Clone the repository
```bash
git clone <your-repo-url>
cd project-root
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
**Main libraries used:**
`torch` `datasets` `tokenizers` `tensorboard` `tqdm`

## 📝 Dataset

- **Dataset**: OPUS Books
- **Source Language**: English (`en`)
- **Target Language**: Spanish (`es`)
- Loaded via Hugging Face `datasets`

The dataset is split into:
- **90% Training**
- **10% Validation**

## 🔤 Tokenization

- Word-level tokenization using Hugging Face tokenizers
- Separate tokenizers for source and target languages
- Special tokens:`[PAD]`,`[UNK]`,`[SOS]`,`[EOS]`

Tokenizers are saved as:
```text
tokenizer_en.json
tokenizer_es.json
```

## 🧠 Model Architecture

Transformer is implemented **entirely from scratch**, including:

- Input Embeddings with scaling
- Sinusoidal Positional Encoding
- Multi-Head Self-Attention
- Encoder–Decoder (Cross) Attention
- Feed Forward Networks
- Residual Connections
- Layer Normalization
- Masked Self-Attention in the Decoder

### Model Configuration 
- Encoder layers: 6
- Decoder layers: 6
- Attention heads: 8
- Embedding size (d_model): 512
- Feed-forward size: 2048

## 🚀 Training

```bash
python train.py
```
### Training details
- Batch size: 8
- Epochs: 20
- Optimizer: Adam
- Learning rate: 1e-4
- Loss: CrossEntropyLoss with label smoothing (0.1)
- Maximum sequence length: 350
- Model checkpoints saved after every epoch

## 🔍 Validation & Decoding
- Validation is performed after each epoch
- Uses Greedy Decoding
- Stops generation when `[EOS]` is produced or max length is reached
## ✨ Sample Output
```text
SOURCE:     I am reading a book.
TARGET:     Estoy leyendo un libro.
PREDICTED:  Estoy leyendo un libro.
```

## 🎯 Project Goal
This project is primarily **educational** and aims to:
- Understand the Transformer architecture in depth
- Implement attention mechanisms manually
- Apply the model to a real-world translation task
- Gain hands-on experience with sequence-to-sequence models
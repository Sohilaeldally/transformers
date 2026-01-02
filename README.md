# transformers

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
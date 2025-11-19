CS5760 Natural Language Processing HomeWork-4

Name: PRIYATHAM REDDY SAMA

StudentId: 700773915

Course: CS5760 Natural Language Processing

Question 1: Character-Level RNN (Q1.py)

Components:CharRNN Model: Embedding → LSTM → Linear layer
Architecture:
Embedding dimension: 128
Hidden size: 256
Vocab size: Dynamic (from character set)

Training:
Toy corpus: "hello help heap held world..." repeated (synthetic data)
Sequence length: 50 characters
Train/val split: 80/20
Optimizer: Adam (lr=0.001)
Loss: CrossEntropyLoss


Generation:
Temperature sampling (τ=0.7, 1.0, 1.2)
Stateful hidden state management
250-character generation samples


Key Concepts: Sequence length trade-offs, hidden size capacity, temperature control for diversity

Question 2: Transformer Encoder (Q2.py)

Components: MultiHeadAttention: 4 parallel attention heads with linear projections
PositionalEncoding: Sinusoidal positional embeddings (max_len=100)
TransformerEncoder: Complete encoder block with:

Self-attention layer
Feed-forward network (256→512→256)
Layer normalization
Residual connections

Testing:Toy sentences: "hello world today", "the cat sat down", etc.
Vocabulary: Simple word tokenization
Output: Contextual embeddings (batch × seq_len × d_model)
Visualization: Attention weight heatmap


Key Concepts: Multi-head parallelization, positional awareness, attention distribution


Question 3: Scaled Dot-Product Attention (Q3.py)

Core Mechanism: Formula: Attention(Q,K,V) = softmax(QK^T/√d_k)V
Scaling: Divides by √d_k to prevent softmax saturation
Optional Masking: Supports attention masking for causal/padding scenarios

Testing:Input dimensions: (batch=2, seq_len=4, d_k=64)
Stability checks: Compares scaled vs unscaled scores
Output verification: Row sums ≈ 1.0 (softmax property)

Key Concepts: Numerical stability, dot-product similarity, probabilistic weighting

	HOW TO RUN 
	1. Install Dependencies:
	pip install torch matplotlib numpy

	2. Run Individual Questions
	Python Q1.py #Character-Level RNN Language Model
	Python Q2.py #Mini Transformer Encoder for Sentences
	Python Q3.py #Implement Scaled Dot-Product Attention

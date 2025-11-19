CS5760 Natural Language Processing - HomeWork - 4

Name: PRIYATHAM REDDY SAMA

StudentId: 700773915

Course: CS5760 Natural Language Processing



Q1. Character-Level RNN Language Model (“hello” toy & beyond)
Goal: Train a tiny character-level RNN to predict the next character given previous characters.
Data (toy to start):
•	Start with a small toy corpus you create (e.g., several “hello…”, “help…”, short words/sentences).
•	Then expand to a short plain-text file of ~50–200 KB (any public-domain text of your choice).
Model:
•	Embedding → RNN (Vanilla RNN or GRU or LSTM) → Linear → Softmax over characters.
•	Hidden size 64–256; sequence length 50–100; batch size 64; train 5–20 epochs.
Train:
•	Teacher forcing (use the true previous char as input during training).
•	Cross-entropy loss; Adam optimizer.
Report:
1.	Training/validation loss curves.
2.	Sample 3 temperature-controlled generations (e.g., τ = 0.7, 1.0, 1.2) for 200–400 chars each.
3.	A 3–5 sentence reflection: what changes when you vary sequence length, hidden size, and temperature?
(Connect to slides: embedding, sampling loop, teacher forcing, tradeoffs 


Q2. Mini Transformer Encoder for Sentences
Task: Build a mini Transformer Encoder (NOT full decoder) to process a batch of sentences.
Steps:
1.	Use a small dataset (e.g., 10 short sentences of your choice).
2.	Tokenize and embed the text.
3.	Add sinusoidal positional encoding.
4.	Implement:
o	Self-attention layer
o	Multi-head attention (2 or 4 heads)
o	Feed-forward layer
o	Add & Norm
5.	Show:
o	Input tokens
o	Final contextual embeddings
o	Attention heatmap between words (visual or printed)

Q3. Implement Scaled Dot-Product Attention
Goal: Implement the attention function from your slides:
"Attention"(Q,K,V)="softmax"((QK^T)/√(d_k ))V

Requirements:
	Write a function in PyTorch or TensorFlow to compute attention.
	Test it using random Q, K, V inputs.
	Print:
	Attention weight matrix
	Output vectors
	Softmax stability check (before and after scaling)


	HOW TO RUN 
	1. Install Dependencies:
	pip install torch matplotlib numpy

	2. Run Individual Questions
	Python Q1.py #Character-Level RNN Language Model
	Python Q2.py #Mini Transformer Encoder for Sentences
	Python Q3.py #Implement Scaled Dot-Product Attention

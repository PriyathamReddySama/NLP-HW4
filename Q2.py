 import torch
Import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=256, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention(self.d_k)
    
    def forward(self, Q, K, V):
        batch_size = Q.shape[0]
        
        # Linear projections
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention per head
        output, weights = self.attn(Q, K, V)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.num_heads * self.d_k)
        output = self.W_o(output)
        
        return output, weights

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=256, num_heads=4):
        super().__init__()
        self.embed = nn.Embedding(100, d_model)  # 100 vocab
        self.pos_enc = PositionalEncoding(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embed(x)
        x = self.pos_enc(x)
        
        # Self-attention + Add & Norm
        attn_out, attn_weights = self.attn(x, x, x)
        x = self.ln1(x + attn_out)
        
        # FFN + Add & Norm
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        
        return x, attn_weights

def test_transformer():
    print("\n=== Q2: Mini Transformer Encoder ===\n")
    
    # Toy sentences
    sentences = [
        "hello world today",
        "the cat sat down",
        "birds fly high above"
    ]
    
    # Simple tokenizer (word to idx)
    vocab = {}
    for sent in sentences:
        for word in sent.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    
    print(f"Vocabulary: {vocab}")
    print(f"Vocab size: {len(vocab)}\n")
    
    # Pad to max length
    max_len = max(len(s.split()) for s in sentences)
    token_ids = []
    for sent in sentences:
        ids = [vocab[w] for w in sent.split()]
        ids += [0] * (max_len - len(ids))
        token_ids.append(ids)
    
    x = torch.tensor(token_ids, dtype=torch.long)
    
    print(f"Input tokens:\n{x}\n")
    
    # Model
    model = TransformerEncoder(d_model=64, num_heads=4)
    output, attn_weights = model(x)
    
    print(f"Contextual embeddings shape: {output.shape}")
    print(f"Sample embedding (sent 0, word 0):\n{output[0, 0].detach().numpy()}\n")
    
    # Visualize attention
    avg_attn = attn_weights.mean(dim=1)[0].detach().numpy()  # (seq_len, seq_len)
    
    print("Attention heatmap (averaged over heads):")
    print(avg_attn)
    
    plt.figure(figsize=(6, 5))
    plt.imshow(avg_attn, cmap='viridis')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title('Self-Attention Weights (Sentence 0)')
    plt.savefig('attention_heatmap.png', dpi=100, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    test_scaled_attention()
    test_transformer()


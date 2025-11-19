import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ============ Q1: Character-Level RNN ============

class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_size=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, h=None):
        x = self.embed(x)
        out, h = self.lstm(x, h)
        logits = self.fc(out)
        return logits, h

def char_rnn_train():
    # Toy data + small corpus
    text = "hello help heap heap held hello world hello there " * 20  # 50KB proxy
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    vocab_size = len(chars)
    
    # Encode
    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
    
    # Create sequences
    seq_len = 50
    sequences = [data[i:i+seq_len+1] for i in range(0, len(data)-seq_len-1, 5)]
    X = torch.stack([s[:-1] for s in sequences])
    Y = torch.stack([s[1:] for s in sequences])
    
    # Train/val split
    split = int(0.8 * len(X))
    train_ds = TensorDataset(X[:split], Y[:split])
    val_ds = TensorDataset(X[split:], Y[split:])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CharRNN(vocab_size, embed_dim=128, hidden_size=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    train_losses, val_losses = [], []
    
    for epoch in range(10):
        # Training
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                val_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        print(f"Epoch {epoch+1}: Train={train_losses[-1]:.3f}, Val={val_losses[-1]:.3f}")
    
    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Character-Level RNN Training')
    plt.savefig('rnn_losses.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    # Generate with temperature
    def generate(model, start_char, temp, length=300):
        model.eval()
        input_ids = torch.tensor([char_to_idx[start_char]], dtype=torch.long).to(device)
        h = None
        result = start_char
        
        with torch.no_grad():
            for _ in range(length):
                logits, h = model(input_ids.unsqueeze(0), h)
                logits = logits[0, -1, :] / temp
                probs = torch.softmax(logits, dim=0)
                next_idx = torch.multinomial(probs, 1).item()
                result += idx_to_char[next_idx]
                input_ids = torch.tensor([next_idx], dtype=torch.long).to(device)
        
        return result
    
    print("\n=== GENERATION SAMPLES ===")
    for temp in [0.7, 1.0, 1.2]:
        print(f"\nTemperature {temp}:")
        print(generate(model, 'h', temp, 250))
    
    # Reflection
    print("""
    === REFLECTION ===
    Sequence Length: Longer sequences capture more context but require more computation
    and risk gradient vanishing. Shorter sequences train faster but miss long-range dependencies.
    Hidden Size: Larger hidden sizes (256 vs 64) increase model capacity—better for complex patterns
    but overfit on tiny data. Temperature controls diversity: τ<1 sharpens (repeats), τ>1 smooths
    (creative/noisy). This balances coherence vs exploration, tied to softmax entropy in sampling loops.
    """)

if __name__ == "__main__":
    char_rnn_train()
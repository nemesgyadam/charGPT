import torch
from torch import nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, device="cpu"):
        super().__init__()
        self.vocab_size = vocab_size
        self.device = device

        self.token_embeddings = nn.Embedding(vocab_size, vocab_size)
        self.to(device)
        print(f"Bigram model initialized")

    def forward(self, idx, targets=None):
        # Forward pass
        logits = self.token_embeddings(idx)

        # Calculate loss
        if targets is None:
            # Predict mode
            loss = None
        else:
            # Train mode
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def predict(self, idx, max_new_tokens=10):
        idx = torch.tensor(idx, dtype=torch.long, device=self.device)

        idx = idx.view(1, -1)  # ADD BATCH DIMENSION
       
        for _ in range(max_new_tokens):
            # get the predictions
            logits, _ = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx[0]  # REMOVE BATCH DIMENSION

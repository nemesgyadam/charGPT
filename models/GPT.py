import torch
from torch import nn
from torch.nn import functional as F


# Model Params
N_EMB = 384

HEAD_SIZE = 96 # Paper says N_EMB/N_HEADS
N_HEADS = 6
N_BLOCKS = 6

FF_MULTIPLIER = 4  # Upscale FeedForward layer by this factor

DROPOUT = 0.2


class SelfAttention(nn.Module):

    def __init__(self):
        super().__init__()

        self.key_matrix = nn.Linear(N_EMB, HEAD_SIZE, bias=False)
        self.query_matrix = nn.Linear(N_EMB, HEAD_SIZE, bias=False)
        self.value_matrix = nn.Linear(N_EMB, HEAD_SIZE, bias=False)

        self.register_buffer(
            "tril", torch.tril(torch.ones(CONTEXT_LENGTH, CONTEXT_LENGTH))
        )  # For masking future tokens

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape

        key = self.key_matrix(x)
        query = self.query_matrix(x)
        wei = query @ key.transpose(-1, -2)

        wei = wei * key.shape[-1] ** -0.5  # Normalization
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # Masking
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        value = self.value_matrix(x)
        return wei @ value


class MultiHeadAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention() for _ in range(N_HEADS)])
        self.proj = nn.Linear(HEAD_SIZE * N_HEADS, N_EMB)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        heads = [h(x) for h in self.heads]
        x = torch.cat(heads, dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention()
        self.ffwd = FeedFoward(N_EMB)
        self.ln1 = nn.LayerNorm(N_EMB)
        self.ln2 = nn.LayerNorm(N_EMB)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class FeedFoward(nn.Module):

    def __init__(self, N_EMBd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMB, FF_MULTIPLIER * N_EMB),
            nn.ReLU(),
            nn.Linear(FF_MULTIPLIER * N_EMB, N_EMB),  # Projection
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size=65, context_length=8, device="cpu"):
        super().__init__()
        self.vocab_size = vocab_size
        global CONTEXT_LENGTH
        CONTEXT_LENGTH = context_length
        self.context_length = context_length
        self.device = device

        self.token_embeddings = nn.Embedding(self.vocab_size, N_EMB)
        self.positional_embeddings = nn.Embedding(CONTEXT_LENGTH, N_EMB)

        self.blocks = nn.Sequential(*[Block() for _ in range(N_BLOCKS)])
        self.ln_f = nn.LayerNorm(N_EMB)
        self.lm_head = nn.Linear(N_EMB, self.vocab_size)

        self.to(device)
        self.apply(self._init_weights)

        print(
            f"GPT model initialized with {sum(p.numel() for p in self.parameters())/1e6} M parameters"
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Forward pass
        tok_emb = self.token_embeddings(idx)
        pos_emb = self.positional_embeddings(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)    # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

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
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx[0]         # REMOVE BATCH DIMENSION

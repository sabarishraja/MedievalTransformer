import torch
from torch import nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, head_size, n_embd, context_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(0.2)
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,H)
        q = self.query(x) # (B,T,H)
        # ✅ scale by head size, not embedding size
        wei = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x) # (B,T,H)
        out = wei @ v     # (B, T, H)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, context_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, context_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, context_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, context_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
  def __init__(self, vocab_size, n_embd=32, context_size=8, n_head=4, n_layer=4, n_styles=1):
    super().__init__()
    self.context_size = context_size
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(context_size, n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, context_size=context_size) for _ in range(n_layer)])
    self.style_embedding_table = nn.Embedding(n_styles, n_embd)  # style prefix (mentor style)
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def generate(self, start_idx, style, number_of_tokens):
    assert start_idx.ndim == 2 and style.ndim == 1
    idx = start_idx
    for _ in range(number_of_tokens):
      idx_cond = idx[:, -self.context_size:]
      logits, _ = self(idx_cond, style)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx

  def forward(self, idx, style, targets=None):
    T = min(idx.shape[1] + 1, self.context_size)
    suffix_idx = idx[:, -(self.context_size - 1):]
    emb = self.token_embedding_table(suffix_idx) # (B, T-1, C)

    style_emb = self.style_embedding_table(style)  # (B, C)
    emb = torch.cat((style_emb.unsqueeze(1), emb), dim=1)  # (B, T, C)
    pos_emb = self.position_embedding_table(torch.arange(T, device=suffix_idx.device)) # (T, C)

    x = emb + pos_emb
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x) # (B, T, V)

    loss = None
    if targets is not None:
      B, Tm, C = logits.shape
      loss = F.cross_entropy(logits.view(B*Tm, C), targets.view(B*Tm))
    return logits, loss

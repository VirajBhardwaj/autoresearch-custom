import time 
import torch 
import torch.nn as nn
import torch.nn.functional as F


from dataclasses import dataclass
from prepare import MAX_SEQ_LEN, Tokenizer, make_dataloader, evaluate_bpb

@dataclass 
class GPTConfig:
    sequence_len: int = MAX_SEQ_LEN
    vocab_size: int = 8192
    n_layer: int = 4;
    n_head: int = 4;
    n_embd: int = 128;

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_v = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, cos_sin):
        B, T, C = x.shape
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin):
        x = x + self.attn(norm(x), cos_sin)
        x = x + self.mlp(norm(x))
        return x




    
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(config.sequence_len * 10, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        # Shape: (1, 1, seq_len, head_dim // 2) for (B, n_head, T, d)
        cos = freqs.cos().bfloat16()[None, None, :, :]
        sin = freqs.sin().bfloat16()[None, None, :, :]
        return cos, sin

    def forward(self, idx, targets=None, reduction="mean"):
        B, T = idx.size()
        cos_sin = (self.cos[:, :, :T], self.sin[:, :, :T])

        x = norm(self.wte(idx))
        for block in self.h:
            x = block(x, cos_sin)

        softcap = 15
        logits = self.lm_head(norm(x)).float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=reduction,
            )
        return logits




device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

tokenizer = Tokenizer.from_directory()
config = GPTConfig(vocab_size=tokenizer.get_vocab_size())
model = GPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

BATCH_SIZE = 4
train_loader = make_dataloader(tokenizer, BATCH_SIZE, MAX_SEQ_LEN, "train")

for step in range(50):
    t0 = time.time()
    x, y, epoch = next(train_loader)
    with autocast_ctx:
        loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    print(f"step {step:03d} | loss: {loss.item():.4f} | dt: {(time.time() - t0)*1000:.0f}ms")



print("Running validation eval...")
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, BATCH_SIZE)
print(f"val_bpb: {val_bpb:.6f}")
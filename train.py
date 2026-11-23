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
    return F.normalize(x, dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_v = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.c_q(x).view(B,T,self.n_head,self.head_dim).transpose(1,2)
        k = self.c_k(x).view(B,T,self.n_head,self.head_dim).transpose(1,2)
        v = self.c_v(x).view(B,T,self.n_head,self.head_dim).transpose(1,2)

        y = scaled_dot_product_attention(q, k, v,is_causal = True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

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

    def forward(self, x):
        x = x + self.attn(norm(x))
        x = x + self.mlp(norm(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None, reduction="mean"):
        x = norm(self.wte(idx))
        for block in self.h:
            x = block(x)
        logits = self.lm_head(norm(x)).float()
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
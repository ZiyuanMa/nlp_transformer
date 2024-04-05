import math
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

INIT_STD = 0.02


class RoPE(nn.Module):
    def __init__(self, head_dim, base=10000):
        super().__init__()
        theta = 1. / (base ** (torch.arange(0, head_dim, 2) / head_dim))
        self.register_buffer('theta', theta)

    def forward(self, qk):
        # qk: batch, num_head*2, sequence, head_dim

        s = torch.arange(qk.size(2), device=qk.device)
        
        freqs = torch.outer(s, self.theta) # seq_len, dim // 2
        freqs = torch.cat((freqs, freqs), dim=-1)

        qk1, qk2 = qk.chunk(2, dim=-1)
        qk2 = torch.cat((-qk2, qk1), dim=-1)

        return qk * freqs.cos() + qk2 * freqs.sin()


class RMSNorm(nn.Module):

    def __init__(self, dim_size, eps=1e-6):
        super().__init__()

        self.root_dim = math.sqrt(dim_size)
        self.weight = nn.Parameter(torch.ones(dim_size))
        self.eps = eps
    
    def forward(self, x):

        x = F.normalize(x, dim=-1, eps=self.eps) * self.root_dim * self.weight

        return x

    

class MLP(nn.Module):
    def __init__(self, input_dim, drop_rate):
        super().__init__()

        self.hidden = input_dim*8//3
        self.in_proj = nn.Linear(input_dim, self.hidden*2)
        self.out_proj = nn.Linear(self.hidden, input_dim)
        
        self.drop = nn.Dropout(drop_rate, True)

        torch.nn.init.normal_(self.in_proj.weight, std=INIT_STD)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):

        x = self.in_proj(x)
        x1, x2 = torch.chunk(x, 2, -1)
        x = F.silu(x1) * x2
        x = self.drop(x)
        x = self.out_proj(x)

        return x 
    


class Attention(nn.Module):

    def __init__(self, input_dim, num_heads):
        super().__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        
        self.qkv = nn.Linear(input_dim, input_dim*3, bias=False)
        self.out_proj = nn.Linear(input_dim, input_dim)
        self.rope = RoPE(input_dim//num_heads)

        torch.nn.init.normal_(self.qkv.weight, std=INIT_STD)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):

        qkv = self.qkv(x)

        qkv = rearrange(qkv, 'b s (h d) -> b h s d', h=self.num_heads*3)
        qk, v = qkv.split([self.num_heads*2, self.num_heads], 1)
        qk = self.rope(qk)

        q, k = qk.chunk(2, 1)

        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        x = rearrange(x, 'b h s d -> b s (h d)')
        x = self.out_proj(x)

        return x
    

class Block(nn.Module):

    def __init__(self, input_dim, num_heads, res_drop_rate, h_drop_rate):
        super().__init__()

        self.attn = Attention(input_dim, num_heads)
        self.mlp = MLP(input_dim, h_drop_rate)

        self.norm1 = RMSNorm(input_dim)
        self.norm2 = RMSNorm(input_dim)

        self.drop = nn.Dropout(res_drop_rate, True)

    def forward(self, x):

        x_out = self.attn(self.norm1(x))
        x = self.drop(x_out) + x

        x_out = self.mlp(self.norm2(x))
        x = self.drop(x_out) + x

        return x


class Model(nn.Module):

    def __init__(self, num_embeddings, input_dim, num_heads, num_layers, 
                 embed_drop_rate=0.0, res_drop_rate=0.0, h_drop_rate=0.0, tie=True):
        super().__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_embeddings, input_dim)
        self.layers = nn.ModuleList([Block(input_dim, num_heads,  res_drop_rate, h_drop_rate) for _ in range(num_layers)])
        self.head = nn.Linear(input_dim, num_embeddings, bias=False)

        self.norm = RMSNorm(input_dim) 

        self.drop = nn.Dropout(embed_drop_rate, True)

        # gpt-2 style init method
        nn.init.normal_(self.embedding.weight, std=INIT_STD)
        nn.init.normal_(self.head.weight, std=INIT_STD)

        for n, p in self.named_parameters():
            if 'out_proj' in n and 'weight' in n:
                torch.nn.init.normal_(p, std=INIT_STD/math.sqrt(num_layers*2)) 

        if tie:
            self.head.weight = self.embedding.weight

    def forward(self, x):
        
        x = self.embedding(x)
        x = self.drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.head(x)

        return x

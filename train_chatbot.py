"""Minimal GPT-style Chatbot - Ultra-compact decoder-only architecture."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    D_MODEL = 64
    N_HEADS = 2
    N_LAYERS = 2
    D_FF = 256
    MAX_LEN = 256
    DROPOUT = 0.1
    MODEL_PATH = "best_chatbot_model.pt"


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None, kv_cache=None):
        B, T, C = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if kv_cache is not None:
            k, v = torch.cat([kv_cache[0], k], 2), torch.cat([kv_cache[1], v], 2)
        attn = self.drop(F.softmax((q @ k.transpose(-2, -1)) * self.scale + (mask if mask is not None else 0), dim=-1))
        x = x + self.proj((attn @ v).transpose(1, 2).reshape(B, T, C))
        x = x + self.mlp(self.ln2(x))
        return x, (k, v)


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout=0.1, pad_idx=0):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.drop = nn.Dropout(dropout)
        self.max_len = max_len
        self._init()

    def _init(self):
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.normal_(p, 0, 0.02)

    def forward(self, idx, kv_cache=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device) if kv_cache is None else torch.arange(kv_cache[0][0].size(2), kv_cache[0][0].size(2) + T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        mask = torch.triu(torch.full((T, T), float('-inf'), device=idx.device), 1) if kv_cache is None else None
        new_cache = []
        for i, block in enumerate(self.blocks):
            x, cache = block(x, mask, kv_cache[i] if kv_cache else None)
            new_cache.append(cache)
        return self.head(self.ln_f(x)), new_cache

    def encode(self, src, src_mask=None):
        return src

    def decode(self, tgt, enc_out, src_mask=None, tgt_mask=None, kv_cache=None):
        inp = torch.cat([enc_out, tgt], 1) if kv_cache is None else tgt
        out, cache = self.forward(inp, kv_cache)
        return out[:, -tgt.size(1):, :], cache

    def output_projection(self, x):
        return x

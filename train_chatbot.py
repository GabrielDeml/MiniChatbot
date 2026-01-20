"""Minimal Transformer Chatbot - Optimized for size and efficiency."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    """Model configuration - optimized for small footprint."""
    D_MODEL = 128
    N_HEADS = 4
    N_LAYERS = 2
    D_FF = 512
    MAX_LEN = 128
    DROPOUT = 0.1
    MODEL_PATH = "best_chatbot_model.pt"


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, kv_cache=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out), (k, v)


class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(enc_out).reshape(B, -1, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)[0]
        x = x + self.ff(self.norm2(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = CrossAttention(d_model, n_heads, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None, kv_cache=None):
        normed = self.norm1(x)
        attn_out, new_cache = self.self_attn(normed, tgt_mask, kv_cache)
        x = x + attn_out
        x = x + self.cross_attn(self.norm2(x), enc_out, src_mask)
        x = x + self.ff(self.norm3(x))
        return x, new_cache


class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None, kv_cache=None):
        new_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_cache[i] if kv_cache else None
            x, cache = layer(x, enc_out, src_mask, tgt_mask, layer_cache)
            new_caches.append(cache)
        return self.norm(x), new_caches


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout=0.1, pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = self._create_pos_encoding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.encoder = Encoder(d_model, n_heads, n_layers, d_ff, dropout)
        self.decoder = Decoder(d_model, n_heads, n_layers, d_ff, dropout)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        self.output_projection.weight = self.embedding.weight
        
        self._init_weights()

    def _create_pos_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        x = self.dropout(self.embedding(src) * math.sqrt(self.d_model) + self.pos_encoding[:, :src.size(1)])
        return self.encoder(x, src_mask)

    def decode(self, tgt, enc_out, src_mask=None, tgt_mask=None, kv_cache=None):
        if kv_cache:
            pos_start = kv_cache[0][0].size(2)
            x = self.dropout(self.embedding(tgt) * math.sqrt(self.d_model) + self.pos_encoding[:, pos_start:pos_start + tgt.size(1)])
        else:
            x = self.dropout(self.embedding(tgt) * math.sqrt(self.d_model) + self.pos_encoding[:, :tgt.size(1)])
        return self.decoder(x, enc_out, src_mask, tgt_mask, kv_cache)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encode(src, src_mask)
        dec_out, _ = self.decode(tgt, enc_out, src_mask, tgt_mask)
        return self.output_projection(dec_out)

    def generate_causal_mask(self, size, device):
        return torch.triu(torch.ones(size, size, device=device), diagonal=1)

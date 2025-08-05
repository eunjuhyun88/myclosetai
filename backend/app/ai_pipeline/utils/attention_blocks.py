#!/usr/bin/env python3
"""
🔥 MyCloset AI - Attention Blocks Utility
========================================

Attention 메커니즘 관련 모듈들
- CrossAttentionBlock: Cross-Attention for feature fusion
- SelfAttentionBlock: Self-Attention for feature enhancement
- MultiHeadAttention: Multi-head attention mechanism

Author: MyCloset AI Team
Date: 2025-07-31
Version: 1.0
"""

# Common imports
from app.ai_pipeline.utils.common_imports import (
    torch, nn, F, TORCH_AVAILABLE,
    logging, Dict, Any, Optional
)

if not TORCH_AVAILABLE:
    raise ImportError("PyTorch is required for attention blocks")

class CrossAttentionBlock(nn.Module):
    """Cross-Attention for clothing-person feature fusion"""
    
    def __init__(self, dim, context_dim):
        super().__init__()
        self.norm = nn.GroupNorm(32, dim)
        self.to_q = nn.Conv2d(dim, dim, 1)
        self.to_k = nn.Conv2d(context_dim, dim, 1)
        self.to_v = nn.Conv2d(context_dim, dim, 1)
        self.to_out = nn.Conv2d(dim, dim, 1)
        self.scale = dim ** -0.5
        
    def forward(self, x, context=None):
        if context is None:
            context = x
            
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        
        q = self.to_q(x_norm).view(b, c, h*w).transpose(1, 2)
        k = self.to_k(context).view(b, c, h*w).transpose(1, 2)
        v = self.to_v(context).view(b, c, h*w).transpose(1, 2)
        
        attn = torch.softmax(torch.matmul(q, k.transpose(1, 2)) * self.scale, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).view(b, c, h, w)
        
        return x + self.to_out(out)

class SelfAttentionBlock(nn.Module):
    """Self-Attention for feature enhancement"""
    
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, 1),
            nn.Sigmoid()
        )
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Spatial attention
        spatial_weights = self.spatial_attention(x)
        x_spatial = x * spatial_weights
        
        # Channel attention
        channel_weights = self.channel_attention(x)
        x_channel = x * channel_weights
        
        # Combine both attentions
        return x_spatial + x_channel

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x, context=None, mask=None):
        if context is None:
            context = x
            
        b, seq_len, dim = x.shape
        
        # Linear transformations
        q = self.to_q(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, seq_len, dim)
        out = self.to_out(out)
        
        return out

class CrossModalAttention(nn.Module):
    """Cross-modal attention for multi-modal fusion"""
    
    def __init__(self, query_dim, key_dim, value_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        self.query_proj = nn.Linear(query_dim, query_dim)
        self.key_proj = nn.Linear(key_dim, query_dim)
        self.value_proj = nn.Linear(value_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        
        self.scale = (query_dim // num_heads) ** -0.5
        
    def forward(self, query, key, value):
        b, seq_len, _ = query.shape
        
        # Project to same dimension
        q = self.query_proj(query)
        k = self.key_proj(key)
        v = self.value_proj(value)
        
        # Reshape for multi-head attention
        head_dim = self.query_dim // self.num_heads
        q = q.view(b, seq_len, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(b, -1, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(b, -1, self.num_heads, head_dim).transpose(1, 2)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, seq_len, self.query_dim)
        out = self.out_proj(out)
        
        return out 
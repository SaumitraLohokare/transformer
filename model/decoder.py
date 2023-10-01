import torch
import torch.nn as nn
from torch.nn import functional as F

import math

from common import SelfAttention, CrossAttention, FeedForward, FlashSelfAttention, FlashCrossAttention


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, max_seq_len, n_heads, hidden_size, has_cross_attention=True):
        super().__init__()

        self.has_cross_attention = has_cross_attention

        self.self_attention = SelfAttention(
            embed_dim, max_seq_len, n_heads, is_causal=True)
        if has_cross_attention:
            self.cross_attention = CrossAttention(embed_dim, n_heads)
        self.feed_forward = FeedForward(embed_dim, hidden_size)

    def forward(self, embeddings):

        output = self.self_attention(embeddings)
        if self.has_cross_attention:
            output = self.cross_attention(output)
        output = self.feed_forward(output)

        return output


class FlashDecoderBlock(nn.Module):
    def __init__(self, embed_dim, max_seq_len, n_heads, hidden_size, has_cross_attention=True):
        super().__init__()

        self.has_cross_attention = has_cross_attention

        self.self_attention = FlashSelfAttention(
            embed_dim, n_heads, is_causal=True)
        if has_cross_attention:
            self.cross_attention = FlashCrossAttention(embed_dim, n_heads)
        self.feed_forward = FeedForward(embed_dim, hidden_size)

    def forward(self, embeddings):

        output = self.self_attention(embeddings)
        if self.has_cross_attention:
            output = self.cross_attention(output)
        output = self.feed_forward(output)

        return output

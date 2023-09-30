import torch
import torch.nn as nn
from torch.nn import functional as F

import math

class LayerNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, inputs):
        return F.layer_norm(inputs, self.weight.shape, self.weight, self.bias, 1e-5)

class TransformerEmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        super().__init__()

        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
    
    def forward(self, ids):
        assert len(ids.size()) == 2, "Expected input ids to be of size (batches, sequence_len)"

        _, seq_len = ids.size()

        # Create the tensor for positional embeddings        
        positions = torch.arange(0, seq_len, dtype=torch.long)
        positions = self.pos_embed(positions)

        inputs = self.tok_embed(ids)

        inputs = inputs + positions

        return inputs

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, max_seq_len, n_heads, is_causal=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.is_causal = is_causal

        self.ln = LayerNorm(embed_dim)

        self.qkv_layer = nn.Linear(embed_dim, 3 * embed_dim, bias=None)

        if is_causal: # Will this work?
            self.raw_mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)


    def forward(self, inputs):
        batches, seq_len, _ = inputs.size()

        resid = inputs.clone() # TODO: Is a clone really necessary here?
        inputs = self.ln(inputs)

        q, k, v = self.qkv_layer(inputs).split(self.embed_dim, dim=2)

        q = q.view(batches, seq_len, self.n_heads, self.embed_dim // self.n_heads).transpose(1, 2)
        k = k.view(batches, seq_len, self.n_heads, self.embed_dim // self.n_heads).transpose(1, 2)
        v = v.view(batches, seq_len, self.n_heads, self.embed_dim // self.n_heads).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.embed_dim // self.n_heads)

        if self.is_causal:
            attn = attn.masked_fill(self.raw_mask[:,:,:seq_len,:seq_len] == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)

        outputs = (attn @ v).transpose(1, 2).contiguous().view(batches, seq_len, self.embed_dim)
        outputs = outputs + resid

        return outputs

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads

        self.ln = LayerNorm(embed_dim)

        self.kv_layer = nn.Linear(embed_dim, 2 * embed_dim, bias=None)
        self.q_layer = nn.Linear(embed_dim, embed_dim, bias=None)

    def forward(self, inputs, cross_embedings):
        assert inputs.size()[-1] == cross_embedings.size()[-1], "Both embeddings in cross attention must have the same embedding dimension"

        batches, seq_len, _ = inputs.size()
        resid = inputs.clone() # TODO: Is this clone necessary?

        inputs = self.ln(inputs)
        k, v = self.kv_layer(cross_embedings).split(self.embed_dim, dim=2)
        q = self.q_layer(inputs)

        q = q.view(batches, seq_len, self.n_heads, self.embed_dim // self.n_heads).transpose(1, 2)
        k = k.view(batches, seq_len, self.n_heads, self.embed_dim // self.n_heads).transpose(1, 2)
        v = v.view(batches, seq_len, self.n_heads, self.embed_dim // self.n_heads).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.embed_dim // self.n_heads)
        attn = F.softmax(attn, dim=-1)
        outputs = attn @ v
        outputs = outputs.transpose(1, 2).contiguous().view(batches, seq_len, self.embed_dim)

        outputs = outputs + resid

        return outputs
    
class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size

        self.ln = LayerNorm(embed_dim)
        self.expand_layer = nn.Linear(embed_dim, hidden_size, bias=None)
        self.gelu = nn.GELU()
        self.shrink_layer = nn.Linear(hidden_size, embed_dim, bias=None)
    
    def forward(self, inputs):
        resid = inputs.clone()
        
        outputs = self.ln(inputs)
        outputs = self.expand_layer(outputs)
        outputs = self.gelu(outputs)
        outputs = self.shrink_layer(outputs)

        outputs = outputs + resid

        return outputs
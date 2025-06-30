import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 # Number of heads for the queries
    n_kv_heads: Optional[int] = None # Number of heads for the k and v
    vocab_size: int = -1 # This will be set when we load the tokenizer
    multiple_of: int = 256 
    ffn_dim_multiplier: Optional[int] = None
    norm_eps: float = 1e-5

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # As written in the paper, the dimension of the embedding must be even
    assert head_dim % 2 == 0, "Dimension must be even"

    # BUild the theta parameters
    # According to the formula, theta_i = 10000 ** (2(i-1)/dim) for i = [1, 2, 3, ..., dim/2]
    # Shape: (head_dim/2)

    theta_numerator = torch.arange(0, head_dim/2, 2)
    theta_i = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)

    # construct the positions (the 'm' parameter)
    # Shape: (Seq_len)
    m = torch.arange(seq_len, device=device)

    # Multiply each theta_i by each position m using outer product
    # Shape: Outer_product: (Seq_len) * (Head_dim/2) -> (head_dim/2, Seq_len)
    freqs = torch.outer(m, theta_i).float()

    # We can compute complex numbers in the polar form c = R * exp(i * m * theta), where R = 1 as follows:
    # (Seq_len, head_dim/2) -> (Seq_len, head_dim/2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # (B, Seq_len, H, Head_Dim) -> (B, Seq_len, H, Head_Dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (Seq_len, Head_Dim/2) -> (1, Seq_len, 1, Head_Dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (B, Seq_len, H, Head_Dim/2) * (1, Seq_len, 1, Head_Dim/2) -> (B, Seq_len, H, Head_Dim/2)
    x_rotated = x_complex * freqs_complex
    # (B, Seq_len, H, Head_Dim/2) -> (B, Seq_len, H, Head_Dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, Seq_len, H, Head_Dim/2, 2) -> (B, Seq_len, H, Head_Dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # Compute the root mean square norm
        # (B, Seq_len, Dim) * (B, Seq_len, 1) -> (B, Seq_len, Dim)
        # rsqrt(x): 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        # (Dim) * (B, Seq_len, Dim) -> (B, Seq_len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return (
            # (B, Seq_len, n_kv_heads, 1, Head_Dim)
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        #Indicates the number of heads for the queries
        self.n_heads_q = args.n_heads
        # Indicates the number of heads for the keys and values
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        # The dimension of the model
        self.head_dim = args.dim // args.n_heads
        # Indicates how many times the heads of the keys and values are repeated to match number of the query heads
        self.n_rep = self.n_heads_q // self.n_kv_heads

        self.wq = nn.Linear(args.dim, self.n_heads_q * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape # (B, Seq_len, Dim) or (B, 1, Dim) 
        # because we are doing inference with single tokens so seq_len would be 1
        
        # (B, 1, Dim) -> (B, 1, n_heads_q * head_dim)
        xq = self.wq(x)
        # (B, 1, Dim) -> (B, 1, n_kv_heads * head_dim)
        xk = self.wk(x)
        xv = self.wv(x)

        # Apply rotary embeddings to the queries and keys only
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # Replace the entry in the KV cache with this token
        self.cache_k[:batch_size, start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos+seq_len] = xv

        # Retrieve the keys and values from the KV cache
        # (B, Seq_len_kv, n_kv_heads, head_dim)
        keys = self.cache_k[:batch_size, 0:start_pos + seq_len]
        values = self.cache_v[:batch_size, 0:start_pos + seq_len]

        # So here we are repeating the kv heads to match the number of query heads despite the fact that it reduces
        # speed but we cannot implement grouped query attention because it only works on llama 70B model
        # and we do not have that much powerful GPU so we stick to KIND-OF multi-head attention model
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # (B, 1, n_heads_q, head_dim) -> (B, n_heads_q, 1, head_dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)  # (B, n_kv_heads, Seq_len_kv, head_dim)
        values = values.transpose(1, 2)  # (B, n_kv_heads, Seq_len_kv, head_dim)

        # Compute the attention scores
        # (B, n_heads_q, 1, head_dim) * (B, n_kv_heads, Seq_len_kv, head_dim) -> (B, n_heads_q, 1, Seq_len_kv)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores, dim=-1).type_as(xq)

        # (B, n_heads_q, 1, Seq_len_kv) * (B, n_kv_heads, Seq_len_kv, head_dim) -> (B, n_heads_q, 1, head_dim)
        output = torch.matmul(scores, values)

        # (B, n_heads_q, 1, head_dim) -> (B, 1, n_heads_q, head_dim) -> (B, 1, Dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output) # (B, 1, Dim)

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round hidden_dim to the nearest multiple of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        # Self-attention layer
        self.attention = SelfAttention(args)

        # Feed-forward layer
        self.feed_forward = FeedForward(args)

        # RMS normalization
        # Normalization before self attention
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Normalization before feed-forward block
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, Seq_len, Dim) + (B, Seq_len, Dim) -> (B, Seq_len, Dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        assert args.vocab_size != -1, "vocab_size must be set"
        
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)
        
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, Seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, 'Only single token inputs can be prossessed because of the KV cache, and also because we are doing inference not training'

        # (B, Seq_len) -> (B, Seq_len, Dim)
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the position [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        
        h = self.norm(h)
        output = self.output(h).float()
        return output
import math

import torch
import torch.nn as nn

# TODO: reimplement using einops
class Linear(nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        self.w = nn.Parameter(torch.empty((out_features, in_features),
                                          device=device, 
                                          dtype=dtype))
        
        # initialization
        std = 2 / (in_features + out_features)
        torch.nn.init.trunc_normal_(self.w.data, 
                                    mean=0, 
                                    std=std, 
                                    a=-3*math.sqrt(std), 
                                    b=3*math.sqrt(std))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.w.T)


class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.table = nn.Parameter(torch.empty((num_embeddings, embedding_dim), 
                                              device=device, 
                                              dtype=dtype))
        
        # initialization
        torch.nn.init.trunc_normal_(self.table.data, 
                                    mean=0, 
                                    std=1, 
                                    a=-3, 
                                    b=-3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # take mod to make sure the result is within range
        return self.table[token_ids % self.num_embeddings]


class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.eps = eps
        # directly use ones for initialization
        self.g = nn.Parameter(torch.ones((d_model), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)  # upcast to prevent overflow

        # keep the dimension
        rms_norm = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        x = x / rms_norm * self.g  # element wise matrix multiplication
        return x.to(in_dtype)


class SwiGLU(nn.Module):
    """
    A feed forward network using SwiGLU, which is a combination of 
    SiLU (more smooth at zero) and GLU (reduce the vanish problem by providing 
    a linear path for gradients)
    """
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()

        # up projection
        self.w1 = Linear(d_model, d_ff, device, dtype)
        # gate
        self.w3 = Linear(d_model, d_ff, device, dtype)
        # down projection
        self.w2 = Linear(d_ff, d_model, device, dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x1 = x1 * torch.sigmoid(x1)  # SiLU
        x3 = self.w3(x)  # gate
        x2 = x1 * x3
        return self.w2(x2)


class RotaryPositionEmbedding(nn.Module):
    """
    Apply rotary position embedding

    referred to the impl in gpt-oss:
        https://github.com/openai/gpt-oss/blob/master/gpt_oss/torch/model.py#L63
    
    Actually the oss version does not match to the version described in the assignment

    The rotation is done in the following way:

        1. the input embedding is split as 2d vectors by interleave slicing
        2. the 2d vectors multiple with the rotary matrix
        3. the rotated vectors are concat back to the original dimension
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        self.theta = theta
        # compute the frequencies
        freq = theta ** ((torch.arange(0, d_k, 2, dtype=torch.float32)) / d_k)
        # pre-compute the cos and sin matrix separately, each could be a diagonal matrix
        # the rotary matrix is computed for each position based on `max_seq_len`
        cos = (torch.arange(0, max_seq_len, dtype=torch.float32).view(-1, 1) / freq.view(1, -1)).cos()
        self.register_buffer("cos", cos, persistent=False)  # max_seq_len, d_k // 2
        sin = (torch.arange(0, max_seq_len, dtype=torch.float32).view(-1, 1) / freq.view(1, -1)).sin()
        self.register_buffer("sin", sin, persistent=False)  # max_seq_len, d_k // 2


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions, :].unsqueeze(0)
        sin = self.sin[token_positions, :].unsqueeze(0)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        result = torch.empty_like(x)
        result[..., 0::2] = o1
        result[..., 1::2] = o2
        return result


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    A numeric safe version of softmax
    """
    o, _ = x.max(dim=dim, keepdim=True)
    x = x - o  # subtract the maximum
    s = torch.exp(x).sum(dim=dim, keepdim=True)
    return torch.exp(x) / s


def scale_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
    bsz, seq_len, d_k, d_v = q.shape[0], q.shape[-2], q.shape[-1], v.shape[-1]

    attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        attention_scores.masked_fill_(~mask, float("-inf"))
    attention_scores = softmax(attention_scores, dim=-1)
    return torch.matmul(attention_scores, v)


class CausalMultiHeadAttention(nn.Module):
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len, d_in = x.shape[-2], x.shape[-1]
        q = self.q_proj(x).view(-1, seq_len, self.num_heads, self.d_model // self.num_heads).transpose(-3, -2)  # ..., nh, seq, dh
        k = self.k_proj(x).view(-1, seq_len, self.num_heads, self.d_model // self.num_heads).transpose(-3, -2)
        v = self.v_proj(x).view(-1, seq_len, self.num_heads, self.d_model // self.num_heads).transpose(-3, -2)

        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)).unsqueeze(0)  # 1, seq, seq

        y = scale_dot_product_attention(q, k, v, mask)
        y = y.transpose(-3, -2).contiguous().view(-1, seq_len, self.d_model)
        return self.o_proj(y)


class CausalMultiHeadAttentionWithRope(nn.Module):
    
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)

        self.rope = RotaryPositionEmbedding(theta, self.d_model // self.num_heads, max_seq_len)

    def forward(self, x: torch.Tensor, token_positions=None) -> torch.Tensor:
        seq_len, d_in = x.shape[-2], x.shape[-1]
        q = self.q_proj(x).view(-1, seq_len, self.num_heads, self.d_model // self.num_heads).transpose(-3, -2)  # ..., nh, seq, dh
        k = self.k_proj(x).view(-1, seq_len, self.num_heads, self.d_model // self.num_heads).transpose(-3, -2)
        v = self.v_proj(x).view(-1, seq_len, self.num_heads, self.d_model // self.num_heads).transpose(-3, -2)

        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)).unsqueeze(0)  # 1, seq, seq

        if token_positions is None:
            token_positions = torch.arange(seq_len)

        # apply rotary
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        y = scale_dot_product_attention(q, k, v, mask)
        y = y.transpose(-3, -2).contiguous().view(-1, seq_len, self.d_model)
        return self.o_proj(y)

ATTR = [
    "attn.q_proj.weight",
    "attn.k_proj.weight",
    "attn.v_proj.weight",
    "attn.output_proj.weight",
    "ln1.weight",
    "ln2.weight",
    "ffn.w1.weight",
    "ffn.w2.weight",
    "ffn.w3.weight",
]


class TransformerBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()

        self.ln1 = RMSNorm(d_model)
        self.attn = CausalMultiHeadAttentionWithRope(d_model, num_heads, max_seq_len, theta)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

    def load_weights(self, weights):
        """
        A helper function to load weight from the weights dict
        """
        self.attn.q_proj.w.data = weights["attn.q_proj.weight"]
        self.attn.k_proj.w.data = weights["attn.k_proj.weight"]
        self.attn.v_proj.w.data = weights["attn.v_proj.weight"]
        self.attn.o_proj.w.data = weights["attn.output_proj.weight"]
        self.ln1.g.data = weights["ln1.weight"]
        self.ln2.g.data = weights["ln2.weight"]
        self.ffn.w1.w.data = weights["ffn.w1.weight"]
        self.ffn.w2.w.data = weights["ffn.w2.weight"]
        self.ffn.w3.w.data = weights["ffn.w3.weight"]


class TransformerLM(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, context_length: int, theta: float, vocab_size: int, num_layers: int):
        super().__init__()

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, context_length, theta)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)

    def load_weights(self, weights):
        self.token_embeddings.table.data = weights['token_embeddings.weight']
        for idx, layer in enumerate(self.layers):
            block_weights = {
                attn: weights[f"layers.{idx}.{attn}"] for attn in ATTR
            }
            layer.load_weights(block_weights)
        self.ln_final.g.data = weights['ln_final.weight']
        self.lm_head.w.data = weights['lm_head.weight']

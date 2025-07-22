import math
import torch
import torch.nn as nn
from einops import rearrange, einsum
from jaxtyping import Float, Int
from typing import Literal

from src.utils import softmax
from src.kernels import FlashAttentionTriton


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std: float = 0.5 * (self.weight.size(-1) ** -0.5)
        bound: float = (3 ** 0.5) * std
        torch.nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(
            x,
            self.weight,
            "... in_features, out_features in_features -> ... out_features",
        )


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        torch.nn.init.normal_(self.weight, 0.0, 1.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        else:
            self.register_buffer("weight", torch.tensor(1.0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        RMS = (self.eps + x.square().sum(dim=-1, keepdim=True) / self.d_model).sqrt()
        result = x / RMS * self.weight

        result = result.to(in_dtype)

        return result


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        # d_ff should be 8/3 d_model, while ensuring that
        # the dimensionality of the inner feed-forward layer is a multiple of 64 to make good use of your
        # hardware
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
        torch.nn.init.zeros_(self.w2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu_x = silu(self.w1(x))
        element_product = silu_x * self.w3(x)
        return self.w2(element_product)


class SiLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        # d_ff should be 4 d_model
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        torch.nn.init.zeros_(self.w2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu_x = silu(self.w1(x))
        return self.w2(silu_x)


class ReLU2FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        # d_ff should be 4 d_model
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        torch.nn.init.zeros_(self.w2.weight)  # Initialize weights to zero for the second layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        relu2_x = torch.nn.functional.relu(self.w1(x)).square()
        return self.w2(relu2_x)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        i = torch.arange(max_seq_len, device=device).view(-1, 1)
        k = torch.arange(1, d_k // 2 + 1, device=device)
        theta_i_k = i / theta ** (2 * (k - 1) / d_k)
        sin = theta_i_k.sin()
        cos = theta_i_k.cos()
        R = [cos, -sin, sin, cos]
        R = rearrange(R, "(d1 d2) i k -> i k d1 d2", d1=2, d2=2)
        self.register_buffer("R", R, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        position_matrices = self.R[token_positions]
        x = rearrange(x, "... seq (k d1) -> ... seq k d1", d1=2)
        applied_rotary = einsum(
            x, position_matrices, "... seq k d1, ... seq k d2 d1 -> ... seq k d2"
        )
        return rearrange(applied_rotary, "... seq k d2 -> ... seq (k d2)")


def scaled_dot_product_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    mask: None | torch.Tensor,
) -> torch.Tensor:
    # pre_softmax_values = einsum(
    #     queries,
    #     keys,
    #     "batch_size ... seq_len_q d_k, batch_size ... seq_len_k d_k -> batch_size ... seq_len_q seq_len_k",
    # ) / math.sqrt(queries.shape[-1])

    # if mask is not None:
    #     pre_softmax_values.masked_fill_(~mask, -torch.inf)

    # weights = softmax(pre_softmax_values, -1)

    # return einsum(
    #     weights,
    #     values,
    #     "batch_size ... seq_len_q seq_len_k, batch_size ... seq_len_k d_v -> batch_size ... seq_len_q d_v",
    # )
    
    # Using Fused Kernel for better performance
    return FlashAttentionTriton.apply(
        queries, keys, values, mask
    )


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        qk_norm: bool = True,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        d_head = d_model // num_heads

        super().__init__()

        self.num_heads = num_heads

        self.q_proj: Float[torch.Tensor, " d_model d_model"] = Linear(
            d_model, d_model, device=device, dtype=dtype
        )
        self.k_proj: Float[torch.Tensor, " d_model d_model"] = Linear(
            d_model, d_model, device=device, dtype=dtype
        )
        self.v_proj: Float[torch.Tensor, " d_model d_model"] = Linear(
            d_model, d_model, device=device, dtype=dtype
        )
        # self.proj: Float[torch.Tensor, ""]
        self.output_proj: Float[torch.Tensor, " d_model d_model"] = Linear(
            d_model, d_model, device=device, dtype=dtype
        )
        self.q_norm = (
            RMSNorm(d_head, elementwise_affine=False, device=device, dtype=dtype)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            RMSNorm(d_head, elementwise_affine=False, device=device, dtype=dtype)
            if qk_norm
            else nn.Identity()
        )
        self.rope = rope

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)
        
        queries = rearrange(
            queries,
            "... seq_len (num_heads d_heads) -> (... num_heads) seq_len d_heads",
            num_heads=self.num_heads,
        ).contiguous()
        keys = rearrange(
            keys,
            "... seq_len (num_heads d_heads) -> (... num_heads) seq_len d_heads",
            num_heads=self.num_heads,
        ).contiguous()
        values = rearrange(
            values,
            "... seq_len (num_heads d_heads) -> (... num_heads) seq_len d_heads",
            num_heads=self.num_heads,
        ).contiguous()
        

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        if self.rope is not None:
            queries = self.rope(queries, token_positions)
            keys = self.rope(keys, token_positions)

        values = scaled_dot_product_attention(queries, keys, values, True)

        values = rearrange(
            values,
            "(b num_heads) seq_len d_heads -> b seq_len (num_heads d_heads)",
            num_heads=self.num_heads,
        )

        return self.output_proj(values)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        qk_norm: bool = True,
        rope: RotaryPositionalEmbedding | None = None,
        post_norm: bool | None = False,
        ffn_type: Literal["swiglu", "silu", "relu2"] = "swiglu",
    ) -> None:
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, qk_norm, rope)
        self.ln2 = RMSNorm(d_model)
        self.ffn = (
            SwiGLU(d_model, d_ff)
            if ffn_type == "swiglu"
            else SiLUFFN(d_model, d_ff) if ffn_type == "silu" else ReLU2FFN(d_model, d_ff)
        )
        self.rope = rope is not None
        self.post_norm = post_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        position_encodings = torch.arange(x.shape[-2]) if self.rope else None
        if self.post_norm is None:  # No norm
            x = x + self.attn(x, position_encodings)
            x = x + self.ffn(x)
            return x

        elif self.post_norm:  # Post-norm
            res = x
            x = self.attn(x, position_encodings)
            x = self.ln1(res + x)

            res = x
            x = self.ffn(x)
            x = self.ln2(res + x)
            return x

        else:  # Pre-norm
            res = x

            x = self.ln1(x)
            x = self.attn(x, position_encodings)
            x = res + x

            res = x

            x = self.ln2(x)
            x = self.ffn(x)

            return res + x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        qk_norm: bool = True,
        rope_theta: float | None = None,
        post_norm: bool | None = False,
        ffn_type: Literal["swiglu", "silu", "relu2"] = "swiglu",
    ) -> None:
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.token_embeddings = Embedding(vocab_size, d_model)

        d_k = d_model // num_heads
        rope = (
            RotaryPositionalEmbedding(rope_theta, d_k, context_length)
            if rope_theta is not None
            else None
        )

        self.layers = nn.ModuleList(
            TransformerBlock(
                d_model, num_heads, d_ff, qk_norm, rope, post_norm, ffn_type
            )
            for _ in range(num_layers)
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

        self.scalars = nn.Parameter(torch.ones(num_layers // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)

        # U-net structure as in modded nano-gpt
        n = len(self.layers) // 2
        skip_connections = []
        skip_weights = self.scalars[:n]
        for i, layer in enumerate(self.layers):
            if i >= n:
                x = x + skip_weights[i - n] * skip_connections.pop()
            x = layer(x)
            if i < n:
                skip_connections.append(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1) ** 0.5))
        return logits


if __name__ == "__main__":
    model = TransformerLM(50257, 1024, 48, 1600, 25, 6400, False, 1000)
    # model = TransformerLM(100, 10, 3, 32, 2, 64, False, 1000)
    print(sum(p.numel() for p in model.parameters()))
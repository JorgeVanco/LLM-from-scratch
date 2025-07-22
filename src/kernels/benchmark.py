import torch
import triton
from src.kernels import FlashAttentionTriton

def test_timing_flash_forward_backward() -> None:
    n_heads = 16
    d_head = 64
    sequence_length = 16384
    q, k, v = torch.randn(
        3, n_heads, sequence_length, d_head, device='cuda', dtype=torch.bfloat16, requires_grad=True
    )

    flash = torch.compile(FlashAttentionTriton.apply)

    def flash_forward_backward():
        o = flash(q, k, v, True)
        loss = o.sum()
        loss.backward()


    results = triton.testing.do_bench(flash_forward_backward, rep=10000, warmup=1000)
    print(results)

if __name__ == "__main__":
    test_timing_flash_forward_backward()
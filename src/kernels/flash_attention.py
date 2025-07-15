import torch
import math
from einops import rearrange, einsum
import triton
import triton.language as tl
from typing import NoReturn


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Bq = 16
        Bk = 16
        q_shape = Q.shape

        Q = rearrange(Q, "b ... d -> b (...) d")
        K = rearrange(K, "b ... d -> b (...) d")
        V = rearrange(V, "b ... d -> b (...) d")
        
        b, Nq, D = Q.size()
        _, Nk, _ = K.size()

        Tq = math.ceil(Q.size(1) / Bq)
        Tk = math.ceil(K.size(1) / Bk)

        QQ = torch.chunk(Q, Tq, dim=1)
        KK = torch.chunk(K, Tk, dim=1)
        VV = torch.chunk(V, Tk, dim=1)

        assert len(QQ) == Tq, f"Expected {Tq} chunks for Q, but got {len(QQ)}"
        assert len(KK) == Tk, f"Expected {Tk} chunks for K, but got {len(KK)}"
        assert len(VV) == Tk, f"Expected {Tk} chunks for V, but got {len(VV)}"
        
        assert QQ[0].size(1) == Bq, f"Expected first chunk of Q to have size {Bq}, but got {QQ[0].size(0)}"
        assert KK[0].size(1) == Bk, f"Expected first chunk of K to have size {Bk}, but got {KK[0].size(0)}"
        assert VV[0].size(1) == Bk, f"Expected first chunk of V to have size {Bk}, but got {VV[0].size(0)}"
        assert QQ[0].size(2) == Q.size(2), f"Expected first chunk of Q to have size {Q.size(2)}, but got {QQ[0].size(2)}"
        assert KK[0].size(2) == K.size(2), f"Expected first chunk of K to have size {K.size(2)}, but got {KK[0].size(2)}"
        assert VV[0].size(2) == V.size(2), f"Expected first chunk of V to have size {V.size(2)}, but got {VV[0].size(2)}"


        O = torch.empty((b, Tq, Bq, V.size(2)), device=Q.device)
        L = torch.empty((b, Tq, Bq), device=Q.device)
    
        for i in range(1, Tq+1):
            Qi = QQ[i-1]
            Oi = torch.empty((Tk+1, b, Bq, V.size(-1)), device=Qi.device)
            li = torch.empty((Tk+1, b, Bq), device=Qi.device)
            mi = torch.empty((Tk+1, b, Bq), device=Qi.device)
            
            Oi[0].fill_(0.0)
            li[0].fill_(0.0)
            mi[0].fill_(-torch.inf)

            for j in range(1, Tk+1):
                Kj = KK[j-1]
                Vj = VV[j-1]

                Si_j = (Qi @ Kj.transpose(-2, -1)) / (K.size(-1) ** 0.5)

                assert Si_j.shape == (b, Bq, Bk), f"Expected Si_j to have shape {(b, Bq, Bk)}, but got {Si_j.shape}"

                mi[j] = torch.maximum(mi[j-1], Si_j.amax(-1))

                Pi_j = torch.exp(Si_j - mi[j].unsqueeze(-1))
                assert Pi_j.shape == (b, Bq, Bk), f"Expected Pi_j to have shape {(b, Bq, Bk)}, but got {Pi_j.shape}"
                li[j] = torch.exp(mi[j-1] - mi[j]) * li[j-1] + Pi_j.sum(-1)
                Oi[j] = torch.diag_embed(torch.exp(mi[j-1] - mi[j])) @ Oi[j-1] + Pi_j @ Vj
            Li = mi[Tk] + torch.log(li[Tk])
            O[:, i-1] = torch.diag_embed(1 / li[Tk]) @ Oi[Tk]
            L[:, i-1] = Li

        O = O.view(q_shape[:-1] + (V.size(-1),))
        ctx.save_for_backward(L.view(q_shape[:-1]), Q, K, V, O)

        return O

    @staticmethod
    def backward(ctx, grad_out):
        L, Q, K, V, O = ctx.saved_tensors
        D = (O * grad_out).sum(-1)
        d = (1 / (K.size(-1) ** 0.5))
        S = (Q @ K.transpose(-2, -1)) * d
        P = torch.empty_like(S)
        for i in range(S.shape[1]):
            for j in range(S.shape[2]):
                P[:, i, j] = torch.exp(S[:, i, j] - L[:, i])
                
        dV = P.transpose(-2, -1) @ grad_out
        dP = grad_out @ V.transpose(-2, -1)
        
        dS = torch.empty_like(S)
        for i in range(S.shape[1]):
            for j in range(S.shape[2]):
                dS[:, i, j] = P[:, i, j] * (dP[:, i, j] - D[:, i])
        
        dQ = dS @ K * d 
        dK = dS.transpose(-2, -1) @ Q * d
            
        return dQ.view(Q.shape), dK.view(K.shape), dV.view(V.shape), None
    
    
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    is_causal: tl.constexpr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
) -> NoReturn:
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    m_before = tl.full([Q_TILE_SIZE], float("-inf"), dtype=tl.float32)
    m = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    
    
    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option='zero')
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        Ki = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option='zero')
        Vi = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option='zero')

        S = tl.dot(Qi, Ki.T) * scale
        
        if is_causal:
            k_start = j * K_TILE_SIZE
            k_indices = k_start + tl.arange(0, K_TILE_SIZE)
            q_tile_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            mask = q_tile_indices[:, None] < k_indices[None, :]
            S = tl.where(mask, -1e6, S)
        
        m = tl.maximum(m_before, tl.max(S, axis=-1))

        P = tl.exp(S - m[:, None])
        l = tl.exp(m_before - m) * l + tl.sum(P, axis=-1)
        o = tl.exp(m_before - m)[:, None] * o + tl.dot(P.to(V_block_ptr.type.element_ty), Vi)

        m_before = m

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

        
    o = (1 / l)[:, None] * o
    l = m + tl.log(l)
    
    tl.store(O_block_ptr, o)
    tl.store(L_block_ptr, l)


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Bq = 16
        Bk = 16
        q_shape = Q.shape

        Q = rearrange(Q, "b ... d -> b (...) d")
        K = rearrange(K, "b ... d -> b (...) d")
        V = rearrange(V, "b ... d -> b (...) d")
        
        b, Nq, D = Q.size()
        _, Nk, _ = K.size()

        Tq = math.ceil(Q.size(1) / Bq)
        Tk = math.ceil(K.size(1) / Bk)

        QQ = torch.chunk(Q, Tq, dim=1)
        KK = torch.chunk(K, Tk, dim=1)
        VV = torch.chunk(V, Tk, dim=1)

        assert len(QQ) == Tq, f"Expected {Tq} chunks for Q, but got {len(QQ)}"
        assert len(KK) == Tk, f"Expected {Tk} chunks for K, but got {len(KK)}"
        assert len(VV) == Tk, f"Expected {Tk} chunks for V, but got {len(VV)}"
        
        assert QQ[0].size(1) == Bq, f"Expected first chunk of Q to have size {Bq}, but got {QQ[0].size(0)}"
        assert KK[0].size(1) == Bk, f"Expected first chunk of K to have size {Bk}, but got {KK[0].size(0)}"
        assert VV[0].size(1) == Bk, f"Expected first chunk of V to have size {Bk}, but got {VV[0].size(0)}"
        assert QQ[0].size(2) == Q.size(2), f"Expected first chunk of Q to have size {Q.size(2)}, but got {QQ[0].size(2)}"
        assert KK[0].size(2) == K.size(2), f"Expected first chunk of K to have size {K.size(2)}, but got {KK[0].size(2)}"
        assert VV[0].size(2) == V.size(2), f"Expected first chunk of V to have size {V.size(2)}, but got {VV[0].size(2)}"


        O = torch.empty((b, Tq, Bq, V.size(2)), device=Q.device)
        L = torch.empty((b, Tq, Bq), device=Q.device)
    
        for i in range(1, Tq+1):
            Qi = QQ[i-1]
            Oi = torch.empty((Tk+1, b, Bq, V.size(-1)), device=Qi.device)
            li = torch.empty((Tk+1, b, Bq), device=Qi.device)
            mi = torch.empty((Tk+1, b, Bq), device=Qi.device)
            
            Oi[0].fill_(0.0)
            li[0].fill_(0.0)
            mi[0].fill_(-torch.inf)

            for j in range(1, Tk+1):
                Kj = KK[j-1]
                Vj = VV[j-1]

                Si_j = (Qi @ Kj.transpose(-2, -1)) / (K.size(-1) ** 0.5)

                assert Si_j.shape == (b, Bq, Bk), f"Expected Si_j to have shape {(b, Bq, Bk)}, but got {Si_j.shape}"

                mi[j] = torch.maximum(mi[j-1], Si_j.amax(-1))

                Pi_j = torch.exp(Si_j - mi[j].unsqueeze(-1))
                assert Pi_j.shape == (b, Bq, Bk), f"Expected Pi_j to have shape {(b, Bq, Bk)}, but got {Pi_j.shape}"
                li[j] = torch.exp(mi[j-1] - mi[j]) * li[j-1] + Pi_j.sum(-1)
                Oi[j] = torch.diag_embed(torch.exp(mi[j-1] - mi[j])) @ Oi[j-1] + Pi_j @ Vj
            Li = mi[Tk] + torch.log(li[Tk])
            O[:, i-1] = torch.diag_embed(1 / li[Tk]) @ Oi[Tk]
            L[:, i-1] = Li

        ctx.save_for_backward(L.view(q_shape[:-1]), Q, K, V, O)

        return O.view(q_shape[:-1] + (V.size(-1),))

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError
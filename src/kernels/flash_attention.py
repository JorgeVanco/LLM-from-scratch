import torch
import math
from einops import rearrange, einsum

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
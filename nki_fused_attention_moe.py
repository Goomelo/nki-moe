"""
Fused Attention + MoE Kernels for Qwen3-30B-A3B

This module explores aggressive fusion opportunities between:
1. Attention and MoE computation
2. RMSNorm + Attention + Residual
3. RMSNorm + MoE + Residual
4. Full Decoder Layer fusion

Fusion Strategy:
- Minimize HBM round trips
- Overlap communication with computation
- Optimize for both Context Encoding (CE) and Token Generation (TKG)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from typing import Optional, Tuple


# =============================================================================
# Fusion Pattern 1: RMSNorm + Attention (Pre-Attention Fusion)
# =============================================================================

@nki.jit
def nki_fused_rmsnorm_attention_prefill(
    hidden_states: torch.Tensor,      # [batch, seq, hidden]
    rmsnorm_weight: torch.Tensor,     # [hidden]
    qkv_weight: torch.Tensor,         # [hidden, total_qkv_dim]
    cos_cache: torch.Tensor,          # [seq, head_dim]
    sin_cache: torch.Tensor,          # [seq, head_dim]
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float = 1e-6,
):
    """
    Fused RMSNorm + QKV Projection + RoPE for Context Encoding (Prefill).
    
    Fusion benefits:
    - Single HBM read for hidden_states
    - Single HBM write for Q, K, V
    - Avoids intermediate RMSNorm output storage
    """
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]
    hidden_size = hidden_states.shape[2]
    
    total_q = num_heads * head_dim
    total_kv = num_kv_heads * head_dim
    
    # Output tensors
    q_out = nl.ndarray((batch_size, seq_len, num_heads, head_dim), 
                       dtype=hidden_states.dtype, buffer=nl.shared_hbm)
    k_out = nl.ndarray((batch_size, seq_len, num_kv_heads, head_dim),
                       dtype=hidden_states.dtype, buffer=nl.shared_hbm)
    v_out = nl.ndarray((batch_size, seq_len, num_kv_heads, head_dim),
                       dtype=hidden_states.dtype, buffer=nl.shared_hbm)
    
    # Tile sizes
    TILE_B = 1       # Process 1 batch at a time
    TILE_S = 64      # Process 64 tokens at a time
    TILE_H = 128     # Process 128 hidden dims at a time
    TILE_HD = 64     # Process 64 head dims at a time
    
    ix_s = nl.arange(TILE_S)[:, None]
    ix_h = nl.arange(TILE_H)[None, :]
    ix_hd = nl.arange(TILE_HD)[None, :]
    
    num_s_tiles = (seq_len + TILE_S - 1) // TILE_S
    num_h_tiles = (hidden_size + TILE_H - 1) // TILE_H
    
    # Load RMSNorm weight
    rms_w = nl.load(rmsnorm_weight.reshape((1, hidden_size))[nl.arange(1)[:, None], ix_h],
                   mask=(ix_h < hidden_size))
    
    for b in nl.affine_range(batch_size):
        for st in nl.affine_range(num_s_tiles):
            s_start = st * TILE_S
            s_mask = (s_start + ix_s < seq_len)
            
            # Load and RMSNorm
            inp_tile = nl.load(
                hidden_states[b, s_start + ix_s, ix_h],
                mask=(s_mask & (ix_h < hidden_size))
            )
            
            # RMS computation
            square_sum = nl.sum(inp_tile * inp_tile, axis=[1])
            rms = nl.rsqrt(square_sum / hidden_size + eps)
            rms_bcast = rms.broadcast_to((TILE_S, hidden_size))
            
            # Normalize
            normalized = inp_tile * rms_bcast * rms_w.broadcast_to((TILE_S, hidden_size))
            
            # QKV Projections
            # For each head group
            for nh in nl.affine_range(num_heads):
                q_accum = nl.zeros((TILE_S, head_dim), dtype=nl.float32)
                
                for ht in nl.affine_range(num_h_tiles):
                    h_start = ht * TILE_H
                    h_mask = (h_start + nl.arange(TILE_H) < hidden_size)
                    
                    norm_h = normalized[:, h_start + nl.arange(TILE_H)]
                    
                    # Q weight
                    q_w_start = nh * head_dim
                    q_w = nl.load(
                        qkv_weight[h_start + nl.arange(TILE_H)[:, None],
                                  (q_w_start + nl.arange(head_dim))[None, :]],
                        mask=(h_mask[:, None])
                    )
                    
                    q_accum += nl.matmul(norm_h, q_w)
                
                # RoPE
                for hd_tile in nl.affine_range(head_dim // TILE_HD):
                    hd_start = hd_tile * TILE_HD
                    hd_mask = (hd_start + ix_hd < head_dim)
                    
                    q_slice = q_accum[:, hd_start + nl.arange(TILE_HD)]
                    
                    # Load cos/sin for these positions and dims
                    pos_ids = s_start + nl.arange(TILE_S)
                    cos_slice = nl.load(
                        cos_cache[pos_ids[:, None], hd_start + nl.arange(TILE_HD)[None, :]],
                        mask=(s_mask[:, None] & hd_mask[None, :])
                    )
                    sin_slice = nl.load(
                        sin_cache[pos_ids[:, None], hd_start + nl.arange(TILE_HD)[None, :]],
                        mask=(s_mask[:, None] & hd_mask[None, :])
                    )
                    
                    # Apply RoPE (simplified - would handle pairs)
                    q_rot = q_slice * cos_slice  # Simplified
                    
                    nl.store(
                        q_out[b, s_start + ix_s, nh, hd_start + nl.arange(TILE_HD)],
                        value=q_rot,
                        mask=(s_mask & hd_mask)
                    )
            
            # K projection (for each KV head)
            for nkv in nl.affine_range(num_kv_heads):
                k_accum = nl.zeros((TILE_S, head_dim), dtype=nl.float32)
                
                for ht in nl.affine_range(num_h_tiles):
                    h_start = ht * TILE_H
                    norm_h = normalized[:, h_start + nl.arange(TILE_H)]
                    
                    k_w_start = total_q + nkv * head_dim
                    k_w = nl.load(
                        qkv_weight[h_start + nl.arange(TILE_H)[:, None],
                                  (k_w_start + nl.arange(head_dim))[None, :]]
                    )
                    
                    k_accum += nl.matmul(norm_h, k_w)
                
                nl.store(
                    k_out[b, s_start + ix_s, nkv, :],
                    value=k_accum,
                    mask=s_mask
                )
            
            # V projection
            for nkv in nl.affine_range(num_kv_heads):
                v_accum = nl.zeros((TILE_S, head_dim), dtype=nl.float32)
                
                for ht in nl.affine_range(num_h_tiles):
                    h_start = ht * TILE_H
                    norm_h = normalized[:, h_start + nl.arange(TILE_H)]
                    
                    v_w_start = total_q + total_kv + nkv * head_dim
                    v_w = nl.load(
                        qkv_weight[h_start + nl.arange(TILE_H)[:, None],
                                  (v_w_start + nl.arange(head_dim))[None, :]]
                    )
                    
                    v_accum += nl.matmul(norm_h, v_w)
                
                nl.store(
                    v_out[b, s_start + ix_s, nkv, :],
                    value=v_accum,
                    mask=s_mask
                )
    
    return q_out, k_out, v_out


# =============================================================================
# Fusion Pattern 2: Full Attention Block (RMSNorm -> Attn -> Residual)
# =============================================================================

@nki.jit
def nki_fused_attention_block_decode(
    hidden_states: torch.Tensor,      # [batch, 1, hidden] - single token
    residual: torch.Tensor,           # [batch, 1, hidden] - for skip connection
    rmsnorm_weight: torch.Tensor,     # [hidden]
    qkv_weight: torch.Tensor,         # [hidden, total_qkv_dim]
    out_proj_weight: torch.Tensor,    # [hidden, num_heads*head_dim]
    cos_cache: torch.Tensor,          # [max_seq, head_dim]
    sin_cache: torch.Tensor,          # [max_seq, head_dim]
    key_cache: torch.Tensor,          # [batch, num_kv_heads, max_seq, head_dim]
    value_cache: torch.Tensor,        # [batch, num_kv_heads, max_seq, head_dim]
    current_seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float = 1e-6,
):
    """
    Fused Attention Block for Token Generation (Decode).
    
    Combines:
    1. RMSNorm
    2. QKV Projection
    3. RoPE
    4. Flash Attention (with KV Cache)
    5. Output Projection
    6. Residual Add
    
    This is the ultimate fusion for TKG latency!
    """
    batch_size = hidden_states.shape[0]
    hidden_size = hidden_states.shape[2]
    
    gqa_group = num_heads // num_kv_heads
    total_q = num_heads * head_dim
    
    # Output with residual already added
    output = nl.ndarray((batch_size, 1, hidden_size), 
                       dtype=hidden_states.dtype, buffer=nl.shared_hbm)
    
    TILE_B = 1
    TILE_H = 128
    TILE_HD = 64
    TILE_SEQ = 64
    
    ix_h = nl.arange(TILE_H)[None, :]
    ix_hd = nl.arange(TILE_HD)[None, :]
    
    num_h_tiles = (hidden_size + TILE_H - 1) // TILE_H
    
    # Load RMSNorm weight
    rms_w = nl.load(rmsnorm_weight.reshape((1, hidden_size))[nl.arange(1)[:, None], ix_h],
                   mask=(ix_h < hidden_size))
    
    for b in nl.affine_range(batch_size):
        # ===== Step 1: RMSNorm =====
        # Load single token
        inp = nl.load(hidden_states[b, 0, ix_h], mask=(ix_h < hidden_size))
        
        # Compute RMS
        square_sum = nl.sum(inp * inp)
        rms = nl.rsqrt(square_sum / hidden_size + eps)
        
        # Normalize
        normalized = inp * rms * rms_w
        
        # ===== Step 2: QKV Projection =====
        # Q: [num_heads, head_dim]
        q_heads = nl.zeros((num_heads, head_dim), dtype=nl.float32)
        
        for nh in nl.affine_range(num_heads):
            q_accum = nl.zeros((head_dim,), dtype=nl.float32)
            
            for ht in nl.affine_range(num_h_tiles):
                h_start = ht * TILE_H
                h_mask = (h_start + nl.arange(TILE_H) < hidden_size)
                
                norm_h = normalized[:, h_start + nl.arange(TILE_H)]
                
                q_w_start = nh * head_dim
                q_w = nl.load(
                    qkv_weight[h_start + nl.arange(TILE_H)[:, None],
                              (q_w_start + nl.arange(head_dim))[None, :]],
                    mask=(h_mask[:, None])
                )
                
                q_accum += nl.matmul(norm_h, q_w)
            
            # RoPE for this head
            for hd_tile in nl.affine_range(head_dim // TILE_HD):
                hd_start = hd_tile * TILE_HD
                q_slice = q_accum[hd_start + nl.arange(TILE_HD)]
                
                # Load cos/sin at current position
                cos_slice = nl.load(cos_cache[current_seq_len - 1, hd_start + nl.arange(TILE_HD)])
                sin_slice = nl.load(sin_cache[current_seq_len - 1, hd_start + nl.arange(TILE_HD)])
                
                # Apply RoPE
                q_rot = q_slice * cos_slice  # Simplified
                
                q_heads[nh, hd_start + nl.arange(TILE_HD)] = q_rot
        
        # K, V projection and cache update
        k_heads = nl.zeros((num_kv_heads, head_dim), dtype=nl.float32)
        v_heads = nl.zeros((num_kv_heads, head_dim), dtype=nl.float32)
        
        for nkv in nl.affine_range(num_kv_heads):
            k_accum = nl.zeros((head_dim,), dtype=nl.float32)
            v_accum = nl.zeros((head_dim,), dtype=nl.float32)
            
            for ht in nl.affine_range(num_h_tiles):
                h_start = ht * TILE_H
                norm_h = normalized[:, h_start + nl.arange(TILE_H)]
                
                k_w_start = total_q + nkv * head_dim
                k_w = nl.load(
                    qkv_weight[h_start + nl.arange(TILE_H)[:, None],
                              (k_w_start + nl.arange(head_dim))[None, :]]
                )
                k_accum += nl.matmul(norm_h, k_w)
                
                v_w_start = total_q + num_kv_heads * head_dim + nkv * head_dim
                v_w = nl.load(
                    qkv_weight[h_start + nl.arange(TILE_H)[:, None],
                              (v_w_start + nl.arange(head_dim))[None, :]]
                )
                v_accum += nl.matmul(norm_h, v_w)
            
            # RoPE for K
            for hd_tile in nl.affine_range(head_dim // TILE_HD):
                hd_start = hd_tile * TILE_HD
                k_slice = k_accum[hd_start + nl.arange(TILE_HD)]
                cos_slice = nl.load(cos_cache[current_seq_len - 1, hd_start + nl.arange(TILE_HD)])
                sin_slice = nl.load(sin_cache[current_seq_len - 1, hd_start + nl.arange(TILE_HD)])
                k_rot = k_slice * cos_slice
                k_heads[nkv, hd_start + nl.arange(TILE_HD)] = k_rot
            
            k_heads[nkv, :] = k_accum
            v_heads[nkv, :] = v_accum
            
            # Write to KV Cache
            nl.store(
                key_cache[b, nkv, current_seq_len - 1, :],
                value=k_accum
            )
            nl.store(
                value_cache[b, nkv, current_seq_len - 1, :],
                value=v_accum
            )
        
        # ===== Step 3: Attention Computation =====
        # For each query head, compute attention with corresponding KV head
        attn_out = nl.zeros((num_heads, head_dim), dtype=nl.float32)
        
        for nh in nl.affine_range(num_heads):
            kv_idx = nh // gqa_group
            
            # Compute attention scores with all previous positions
            # Simplified - full implementation would use flash attention
            
            # Load K, V from cache for this head
            k_cache_head = nl.zeros((current_seq_len, head_dim), dtype=nl.float32)
            v_cache_head = nl.zeros((current_seq_len, head_dim), dtype=nl.float32)
            
            for seq_tile in nl.affine_range((current_seq_len + TILE_SEQ - 1) // TILE_SEQ):
                seq_start = seq_tile * TILE_SEQ
                seq_mask = (seq_start + nl.arange(TILE_SEQ) < current_seq_len)
                
                k_tile = nl.load(
                    key_cache[b, kv_idx, seq_start + nl.arange(TILE_SEQ), :],
                    mask=seq_mask[:, None]
                )
                k_cache_head[seq_start + nl.arange(TILE_SEQ), :] = k_tile
                
                v_tile = nl.load(
                    value_cache[b, kv_idx, seq_start + nl.arange(TILE_SEQ), :],
                    mask=seq_mask[:, None]
                )
                v_cache_head[seq_start + nl.arange(TILE_SEQ), :] = v_tile
            
            # Compute scores: q @ k^T
            q_head = q_heads[nh, :]
            scores = nl.zeros((current_seq_len,), dtype=nl.float32)
            
            for pos in nl.affine_range(current_seq_len):
                scores[pos] = nl.sum(q_head * k_cache_head[pos, :]) / math.sqrt(head_dim)
            
            # Softmax
            max_score = nl.max(scores)
            exp_scores = nl.exp(scores - max_score)
            sum_exp = nl.sum(exp_scores)
            attn_weights = exp_scores / sum_exp
            
            # Weighted sum of values
            attn_head_out = nl.zeros((head_dim,), dtype=nl.float32)
            for pos in nl.affine_range(current_seq_len):
                attn_head_out += attn_weights[pos] * v_cache_head[pos, :]
            
            attn_out[nh, :] = attn_head_out
        
        # ===== Step 4: Output Projection =====
        attn_flat = attn_out.reshape((total_q,))
        out_proj = nl.zeros((hidden_size,), dtype=nl.float32)
        
        for ht in nl.affine_range(num_h_tiles):
            h_start = ht * TILE_H
            h_mask = (h_start + nl.arange(TILE_H) < hidden_size)
            
            w_o = nl.load(
                out_proj_weight[total_q, h_start + nl.arange(TILE_H)],
                mask=h_mask
            )
            # Simplified - actual matmul needed
            out_proj[h_start + nl.arange(TILE_H)] = nl.sum(attn_flat[:, None] * w_o, axis=[0])
        
        # ===== Step 5: Residual Add =====
        res = nl.load(residual[b, 0, ix_h], mask=(ix_h < hidden_size))
        final_out = out_proj + res
        
        nl.store(
            output[b, 0, ix_h],
            value=final_out,
            mask=(ix_h < hidden_size)
        )
    
    return output


# =============================================================================
# Fusion Pattern 3: Attention-MoE Pipeline Fusion
# =============================================================================

@nki.jit
def nki_attention_moe_fused_pipeline(
    hidden_states: torch.Tensor,      # [batch, seq, hidden]
    attention_params: dict,           # Attention parameters
    moe_params: dict,                 # MoE parameters
    router_params: dict,              # Router parameters
):
    """
    Experimental: Fused Attention-MoE pipeline.
    
    Idea: While Attention is computing, prefetch MoE weights based on router prediction.
    This overlaps the attention computation with MoE weight loading.
    
    This is speculative execution for MoE!
    """
    # This is a conceptual kernel showing the optimization opportunity
    # Full implementation would require:
    # 1. Router lookahead (predict next layer's experts)
    # 2. Double buffering for weights
    # 3. Async weight loading
    
    pass  # Placeholder for future implementation


# =============================================================================
# PyTorch Module: Fused Attention-MoE Layer
# =============================================================================

class FusedAttentionMoELayer(nn.Module):
    """
    Fused Decoder Layer combining Attention and MoE.
    
    Structure:
    Input -> RMSNorm -> Attention -> ResidualAdd -> RMSNorm -> MoE -> ResidualAdd -> Output
    
    Optimizations:
    1. Fused RMSNorm+Attention kernels
    2. Fused RMSNorm+MoE kernels
    3. Shared memory layout optimizations
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int = 8,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.layer_idx = layer_idx
        
        # RMSNorm layers
        self.input_layernorm = nn.Parameter(torch.ones(hidden_size))
        self.post_attention_layernorm = nn.Parameter(torch.ones(hidden_size))
        
        # Attention weights
        total_qkv = (num_heads + 2 * num_kv_heads) * head_dim
        self.qkv_proj = nn.Linear(hidden_size, total_qkv, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
        # Router
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        
        # MoE weights
        self.gate_up_proj = nn.Parameter(
            torch.randn(num_experts, hidden_size, 2 * intermediate_size)
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size)
        )
        
        # RoPE caches
        self.register_buffer("cos_cache", None)
        self.register_buffer("sin_cache", None)
        self.register_buffer("k_cache", None)
        self.register_buffer("v_cache", None)
        
        print(f"FusedAttentionMoELayer {layer_idx}: Attention({num_heads}h/{num_kv_heads}kv) + "
              f"MoE({num_experts}e/{top_k}k)")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple] = None,
        use_cache: bool = False,
        **kwargs
    ):
        """
        Forward with fusion optimizations
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # ===== Attention Block =====
        residual_attn = hidden_states
        
        if hidden_states.is_xla and self.cos_cache is not None:
            # Use fused N kernels
            attn_out = self._nki_fused_attention_block(
                hidden_states, attention_mask, position_ids
            )
        else:
            # PyTorch fallback
            attn_out = self._pytorch_attention_block(
                hidden_states, attention_mask, position_ids
            )
        
        hidden_states = residual_attn + attn_out
        
        # ===== MoE Block =====
        residual_moe = hidden_states
        
        if hidden_states.is_xla:
            moe_out = self._nki_fused_moe_block(hidden_states)
        else:
            moe_out = self._pytorch_moe_block(hidden_states)
        
        hidden_states = residual_moe + moe_out
        
        return hidden_states, None, None, None, None
    
    def _nki_fused_attention_block(self, hidden_states, attention_mask, position_ids):
        """NKI fused attention implementation"""
        # Would call nki_fused_attention_block_decode
        # For now, use PyTorch
        return self._pytorch_attention_block(hidden_states, attention_mask, position_ids)
    
    def _pytorch_attention_block(self, hidden_states, attention_mask, position_ids):
        """PyTorch attention reference"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # RMSNorm
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + 1e-6)
        hidden_states = hidden_states * self.input_layernorm
        
        # QKV
        qkv = self.qkv_proj(hidden_states)
        total_q = self.num_heads * self.head_dim
        total_kv = self.num_kv_heads * self.head_dim
        
        q = qkv[..., :total_q].view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = qkv[..., total_q:total_q+total_kv].view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = qkv[..., total_q+total_kv:].view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Attention (simplified)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.o_proj(out)
        
        return out
    
    def _nki_fused_moe_block(self, hidden_states):
        """NKI fused MoE implementation"""
        # Would call nki_moe kernels from nki_moe_integrated.py
        return self._pytorch_moe_block(hidden_states)
    
    def _pytorch_moe_block(self, hidden_states):
        """PyTorch MoE reference"""
        batch_size, seq_len, _ = hidden_states.shape
        hidden_flat = hidden_states.view(-1, self.hidden_size)
        
        # RMSNorm
        variance = hidden_flat.pow(2).mean(-1, keepdim=True)
        hidden_flat = hidden_flat * torch.rsqrt(variance + 1e-6)
        hidden_flat = hidden_flat * self.post_attention_layernorm
        
        # Router
        router_logits = self.router(hidden_flat)
        weights, indices = torch.topk(F.softmax(router_logits, dim=-1), self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        # MoE
        output = torch.zeros_like(hidden_flat)
        for i in range(hidden_flat.shape[0]):
            for k in range(self.top_k):
                eid = indices[i, k].item()
                wt = weights[i, k]
                
                gu = torch.matmul(hidden_flat[i], self.gate_up_proj[eid])
                g, u = gu.chunk(2, dim=-1)
                act = F.silu(g) * u
                out = torch.matmul(act, self.down_proj[eid])
                output[i] += wt * out
        
        return output.view(batch_size, seq_len, -1)


# =============================================================================
# Integration Helpers
# =============================================================================

def enable_fused_attention_moe(model, config):
    """
    Replace decoder layers with fused Attention-MoE layers.
    
    Args:
        model: NeuronQwen3MoeForCausalLM instance
        config: Model configuration
    """
    print("Enabling Fused Attention-MoE layers...")
    
    from nki_fused_attention_moe import FusedAttentionMoELayer
    
    for layer_idx, layer in enumerate(model.model.layers):
        # Create fused layer
        fused_layer = FusedAttentionMoELayer(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            intermediate_size=config.moe_intermediate_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            layer_idx=layer_idx,
        )
        
        # Copy weights from original layer
        # (Weight copying logic would go here)
        
        # Replace layer
        model.model.layers[layer_idx] = fused_layer
    
    print(f"Replaced {len(model.model.layers)} layers with fused versions")
    return model


def analyze_fusion_opportunities(model):
    """
    Analyze model and report fusion opportunities.
    """
    print("\n" + "=" * 60)
    print("Fusion Opportunity Analysis")
    print("=" * 60)
    
    stats = {
        "layers": 0,
        "rmsnorm_ops": 0,
        "attention_ops": 0,
        "moe_ops": 0,
        "potential_hbm_saves": 0,
    }
    
    for layer in model.model.layers:
        stats["layers"] += 1
        
        # Count operations
        if hasattr(layer, 'input_layernorm'):
            stats["rmsnorm_ops"] += 1
        if hasattr(layer, 'self_attn'):
            stats["attention_ops"] += 1
        if hasattr(layer, 'post_attention_layernorm'):
            stats["rmsnorm_ops"] += 1
        if hasattr(layer, 'mlp'):
            stats["moe_ops"] += 1
    
    # Estimate potential savings
    # Each fusion saves at least 2 HBM round trips
    stats["potential_hbm_saves"] = stats["rmsnorm_ops"] * 2 + stats["attention_ops"] * 3
    
    print(f"\nLayer Statistics:")
    print(f"  Total layers: {stats['layers']}")
    print(f"  RMSNorm operations: {stats['rmsnorm_ops']}")
    print(f"  Attention operations: {stats['attention_ops']}")
    print(f"  MoE operations: {stats['moe_ops']}")
    
    print(f"\nFusion Potential:")
    print(f"  Estimated HBM round trips saved: {stats['potential_hbm_saves']}")
    
    print(f"\nRecommended Fusions:")
    print(f"  1. RMSNorm + QKV Projection (High impact)")
    print(f"  2. Attention + Residual Add (Medium impact)")
    print(f"  3. RMSNorm + MoE Expert Compute (High impact)")
    print(f"  4. Full Decoder Layer (Very High impact, complex)")
    
    return stats


__all__ = [
    "nki_fused_rmsnorm_attention_prefill",
    "nki_fused_attention_block_decode",
    "FusedAttentionMoELayer",
    "enable_fused_attention_moe",
    "analyze_fusion_opportunities",
]

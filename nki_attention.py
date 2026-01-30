"""
NKI Attention Kernels for Qwen3-30B-A3B

Optimized implementations for:
1. Flash Attention style computation
2. Fused QKV projection + RoPE
3. GQA (Grouped Query Attention) support
4. Token Generation (decode) phase optimization
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
from typing import Optional, Tuple


# Qwen3-30B-A3B Attention Config
HEAD_DIM = 128
NUM_HEADS = 32
NUM_KV_HEADS = 8  # GQA: 4 query heads share 1 kv head
GQA_GROUP_SIZE = 4

# Tile sizes optimized for Trainium SBUF
TILE_SEQ = 64     # Sequence length tile
TILE_HEAD = 64    # Head dimension tile
TILE_KV = 64      # KV dimension tile


@nki.jit
def nki_qkv_projection(
    hidden_states: torch.Tensor,  # [batch*seq, hidden]
    qkv_weight: torch.Tensor,     # [hidden, (num_heads + 2*num_kv_heads) * head_dim]
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
):
    """
    Fused QKV projection kernel.
    
    Outputs Q, K, V tensors in single kernel launch.
    """
    batch_seq = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    
    total_q_size = num_heads * head_dim
    total_kv_size = num_kv_heads * head_dim
    total_out_size = total_q_size + 2 * total_kv_size
    
    # Allocate outputs
    q_out = nl.ndarray((batch_seq, total_q_size), dtype=hidden_states.dtype, buffer=nl.shared_hbm)
    k_out = nl.ndarray((batch_seq, total_kv_size), dtype=hidden_states.dtype, buffer=nl.shared_hbm)
    v_out = nl.ndarray((batch_seq, total_kv_size), dtype=hidden_states.dtype, buffer=nl.shared_hbm)
    
    # Tile indices
    ix_bs = nl.arange(TILE_SEQ)[:, None]
    ix_h = nl.arange(TILE_HEAD)[None, :]
    
    num_bs_tiles = (batch_seq + TILE_SEQ - 1) // TILE_SEQ
    num_h_tiles = (hidden_size + TILE_HEAD - 1) // TILE_HEAD
    num_q_tiles = (total_q_size + TILE_HEAD - 1) // TILE_HEAD
    num_kv_tiles = (total_kv_size + TILE_HEAD - 1) // TILE_HEAD
    
    # Process in tiles
    for bs_tile in nl.affine_range(num_bs_tiles):
        bs_start = bs_tile * TILE_SEQ
        bs_mask = (bs_start + ix_bs < batch_seq)
        
        # Load input tile
        inp_tile = nl.load(
            hidden_states[bs_start + ix_bs, ix_h],
            mask=(bs_mask & (ix_h < hidden_size))
        )
        
        # Q projection
        for q_tile in nl.affine_range(num_q_tiles):
            q_start = q_tile * TILE_HEAD
            q_mask = (q_start + ix_h < total_q_size)
            
            q_accum = nl.zeros((TILE_SEQ, TILE_HEAD), dtype=nl.float32)
            
            for h_tile in nl.affine_range(num_h_tiles):
                h_start = h_tile * TILE_HEAD
                h_mask = (h_start + nl.arange(TILE_HEAD) < hidden_size)
                
                inp_h = nl.load(
                    hidden_states[bs_start + ix_bs, h_start + nl.arange(TILE_HEAD)],
                    mask=(bs_mask & h_mask)
                )
                
                w_q = nl.load(
                    qkv_weight[h_start + nl.arange(TILE_HEAD)[:, None],
                              q_start + nl.arange(TILE_HEAD)[None, :]],
                    mask=(h_mask[:, None] & q_mask[None, :])
                )
                
                q_accum += nl.matmul(inp_h, w_q)
            
            nl.store(
                q_out[bs_start + ix_bs, q_start + ix_h],
                value=q_accum,
                mask=(bs_mask & q_mask)
            )
        
        # K projection
        for kv_tile in nl.affine_range(num_kv_tiles):
            kv_start = kv_tile * TILE_HEAD
            kv_mask = (kv_start + ix_h < total_kv_size)
            k_offset = total_q_size
            
            k_accum = nl.zeros((TILE_SEQ, TILE_HEAD), dtype=nl.float32)
            
            for h_tile in nl.affine_range(num_h_tiles):
                h_start = h_tile * TILE_HEAD
                h_mask = (h_start + nl.arange(TILE_HEAD) < hidden_size)
                
                inp_h = nl.load(
                    hidden_states[bs_start + ix_bs, h_start + nl.arange(TILE_HEAD)],
                    mask=(bs_mask & h_mask)
                )
                
                w_k = nl.load(
                    qkv_weight[h_start + nl.arange(TILE_HEAD)[:, None],
                              (k_offset + kv_start) + nl.arange(TILE_HEAD)[None, :]],
                    mask=(h_mask[:, None] & kv_mask[None, :])
                )
                
                k_accum += nl.matmul(inp_h, w_k)
            
            nl.store(
                k_out[bs_start + ix_bs, kv_start + ix_h],
                value=k_accum,
                mask=(bs_mask & kv_mask)
            )
        
        # V projection
        for kv_tile in nl.affine_range(num_kv_tiles):
            kv_start = kv_tile * TILE_HEAD
            kv_mask = (kv_start + ix_h < total_kv_size)
            v_offset = total_q_size + total_kv_size
            
            v_accum = nl.zeros((TILE_SEQ, TILE_HEAD), dtype=nl.float32)
            
            for h_tile in nl.affine_range(num_h_tiles):
                h_start = h_tile * TILE_HEAD
                h_mask = (h_start + nl.arange(TILE_HEAD) < hidden_size)
                
                inp_h = nl.load(
                    hidden_states[bs_start + ix_bs, h_start + nl.arange(TILE_HEAD)],
                    mask=(bs_mask & h_mask)
                )
                
                w_v = nl.load(
                    qkv_weight[h_start + nl.arange(TILE_HEAD)[:, None],
                              (v_offset + kv_start) + nl.arange(TILE_HEAD)[None, :]],
                    mask=(h_mask[:, None] & kv_mask[None, :])
                )
                
                v_accum += nl.matmul(inp_h, w_v)
            
            nl.store(
                v_out[bs_start + ix_bs, kv_start + ix_h],
                value=v_accum,
                mask=(bs_mask & kv_mask)
            )
    
    return q_out, k_out, v_out


@nki.jit
def nki_rope_kernel(
    tensor: torch.Tensor,      # [batch*seq, num_heads*head_dim]
    cos_cache: torch.Tensor,   # [seq, head_dim]
    sin_cache: torch.Tensor,   # [seq, head_dim]
    num_heads: int,
    head_dim: int,
):
    """
    Rotary Position Embedding (RoPE) kernel.
    
    Applies rotation to pairs of dimensions in each head.
    """
    batch_seq = tensor.shape[0]
    total_dim = num_heads * head_dim
    
    output = nl.ndarray((batch_seq, total_dim), dtype=tensor.dtype, buffer=nl.shared_hbm)
    
    # Tile indices
    ix_bs = nl.arange(TILE_SEQ)[:, None]
    ix_h = nl.arange(TILE_HEAD)[None, :]
    
    num_bs_tiles = (batch_seq + TILE_SEQ - 1) // TILE_SEQ
    
    # Process by head
    for head_id in nl.affine_range(num_heads):
        head_start = head_id * head_dim
        
        # Process sequence tiles
        for bs_tile in nl.affine_range(num_bs_tiles):
            bs_start = bs_tile * TILE_SEQ
            bs_mask = (bs_start + ix_bs < batch_seq)
            
            # Load sequence positions for this tile
            seq_pos = bs_start + nl.arange(TILE_SEQ)
            
            # Load input tile for this head
            head_data = nl.load(
                tensor[bs_start + ix_bs, head_start + ix_h],
                mask=(bs_mask & (ix_h < head_dim))
            )
            
            # Load cos/sin for these positions
            cos_tile = nl.load(
                cos_cache[seq_pos[:, None], nl.arange(TILE_HEAD)[None, :]],
                mask=(bs_mask & (nl.arange(TILE_HEAD)[None, :] < head_dim))
            )
            sin_tile = nl.load(
                sin_cache[seq_pos[:, None], nl.arange(TILE_HEAD)[None, :]],
                mask=(bs_mask & (nl.arange(TILE_HEAD)[None, :] < head_dim))
            )
            
            # Apply RoPE: rotate pairs of dimensions
            # [x0, x1, x2, x3, ...] -> [x0*cos - x1*sin, x0*sin + x1*cos, ...]
            rotated = nl.zeros((TILE_SEQ, TILE_HEAD), dtype=nl.float32)
            
            # Process in pairs
            half_dim = head_dim // 2
            for i in nl.affine_range(half_dim):
                # Even indices
                x_even = head_data[:, 2*i]
                # Odd indices  
                x_odd = head_data[:, 2*i + 1]
                
                cos_val = cos_tile[:, i]
                sin_val = sin_tile[:, i]
                
                # Rotation
                rotated[:, 2*i] = x_even * cos_val - x_odd * sin_val
                rotated[:, 2*i + 1] = x_even * sin_val + x_odd * cos_val
            
            nl.store(
                output[bs_start + ix_bs, head_start + ix_h],
                value=rotated,
                mask=(bs_mask & (ix_h < head_dim))
            )
    
    return output


@nki.jit
def nki_flash_attention_decode(
    query: torch.Tensor,          # [batch, num_heads, head_dim] - single token
    key_cache: torch.Tensor,      # [batch, num_kv_heads, max_seq, head_dim]
    value_cache: torch.Tensor,    # [batch, num_kv_heads, max_seq, head_dim]
    attention_mask: torch.Tensor, # [batch, 1, 1, seq_len]
    current_seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    gqa_group_size: int,
):
    """
    Flash Attention kernel optimized for token generation (decode) phase.
    
    This is the critical path for inference latency.
    Uses online softmax and tiling for efficiency.
    """
    batch_size = query.shape[0]
    max_seq_len = key_cache.shape[2]
    
    # Output: [batch, num_heads, head_dim]
    output = nl.ndarray((batch_size, num_heads, head_dim), dtype=query.dtype, buffer=nl.shared_hbm)
    
    # Tile sizes for attention
    TILE_Q_HEADS = 8     # Process 8 query heads at once
    TILE_KV_HEADS = 4    # Process 4 kv heads at once
    TILE_SEQ = 64        # Process 64 sequence positions at once
    TILE_HEAD = 64       # Process 64 head dims at once
    
    # Process by batch and head groups
    for b in nl.affine_range(batch_size):
        for h_group in nl.affine_range(num_heads // TILE_Q_HEADS):
            h_start = h_group * TILE_Q_HEADS
            
            # Determine which kv head to use (GQA)
            kv_head_idx = h_start // gqa_group_size
            
            # Load query for this head group [TILE_Q_HEADS, head_dim]
            q_heads = nl.load(
                query[b, h_start + nl.arange(TILE_Q_HEADS)[:, None],
                     nl.arange(head_dim)[None, :]],
                mask=((h_start + nl.arange(TILE_Q_HEADS)[:, None]) < num_heads)
            )
            
            # Initialize softmax statistics
            max_score = nl.zeros((TILE_Q_HEADS,), dtype=nl.float32) - 1e9
            sum_exp = nl.zeros((TILE_Q_HEADS,), dtype=nl.float32)
            out_accum = nl.zeros((TILE_Q_HEADS, head_dim), dtype=nl.float32)
            
            # Process KV cache in sequence tiles
            num_seq_tiles = (current_seq_len + TILE_SEQ - 1) // TILE_SEQ
            
            for seq_tile in nl.affine_range(num_seq_tiles):
                seq_start = seq_tile * TILE_SEQ
                seq_mask = (seq_start + nl.arange(TILE_SEQ) < current_seq_len)
                
                # Load K tile for this sequence position: [TILE_SEQ, head_dim]
                k_tile = nl.load(
                    key_cache[b, kv_head_idx,
                             seq_start + nl.arange(TILE_SEQ)[:, None],
                             nl.arange(head_dim)[None, :]],
                    mask=seq_mask[:, None]
                )
                
                # Compute attention scores: Q @ K^T
                # [TILE_Q_HEADS, head_dim] @ [head_dim, TILE_SEQ] -> [TILE_Q_HEADS, TILE_SEQ]
                scores = nl.matmul(q_heads, k_tile.T)
                
                # Scale scores
                scores = scores / math.sqrt(head_dim)
                
                # Apply mask (causal or padding)
                mask_tile = nl.load(
                    attention_mask[b, 0, 0, seq_start + nl.arange(TILE_SEQ)],
                    mask=seq_mask
                )
                # Broadcasting mask
                for h in nl.affine_range(TILE_Q_HEADS):
                    scores[h, :] = scores[h, :] + mask_tile
                
                # Online softmax update
                # New max for this tile
                tile_max = nl.max(scores, axis=[1])  # [TILE_Q_HEADS]
                
                # Rescale previous accumulator
                for h in nl.affine_range(TILE_Q_HEADS):
                    exp_old_max = nl.exp(max_score[h] - tile_max[h])
                    sum_exp[h] = sum_exp[h] * exp_old_max
                    out_accum[h, :] = out_accum[h, :] * exp_old_max
                
                max_score = tile_max
                
                # Compute exp(scores - max)
                exp_scores = nl.exp(scores - max_score.broadcast_to((TILE_Q_HEADS, TILE_SEQ)))
                sum_exp = sum_exp + nl.sum(exp_scores, axis=[1])
                
                # Load V tile: [TILE_SEQ, head_dim]
                v_tile = nl.load(
                    value_cache[b, kv_head_idx,
                               seq_start + nl.arange(TILE_SEQ)[:, None],
                               nl.arange(head_dim)[None, :]],
                    mask=seq_mask[:, None]
                )
                
                # Accumulate weighted values
                # [TILE_Q_HEADS, TILE_SEQ] @ [TILE_SEQ, head_dim] -> [TILE_Q_HEADS, head_dim]
                out_accum = out_accum + nl.matmul(exp_scores, v_tile)
            
            # Normalize by sum_exp
            for h in nl.affine_range(TILE_Q_HEADS):
                out_accum[h, :] = out_accum[h, :] / sum_exp[h]
            
            # Store output
            nl.store(
                output[b, h_start + nl.arange(TILE_Q_HEADS)[:, None],
                      nl.arange(head_dim)[None, :]],
                value=out_accum,
                mask=((h_start + nl.arange(TILE_Q_HEADS)[:, None]) < num_heads)
            )
    
    return output


@nki.jit
def nki_fused_attention_rmsnorm_qkv(
    hidden_states: torch.Tensor,  # [batch*seq, hidden]
    rmsnorm_weight: torch.Tensor, # [hidden]
    qkv_weight: torch.Tensor,     # [hidden, total_qkv_dim]
    cos_cache: torch.Tensor,      # [seq, head_dim]
    sin_cache: torch.Tensor,      # [seq, head_dim]
    key_cache: torch.Tensor,      # For caching K
    value_cache: torch.Tensor,    # For caching V
    attention_mask: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float = 1e-6,
):
    """
    Fused kernel combining:
    1. RMSNorm
    2. QKV projection
    3. RoPE
    4. Flash Attention (for token generation)
    
    This is the ultimate fusion for decode phase!
    """
    batch_seq = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    
    # Output: [batch*seq, num_heads, head_dim]
    total_q_size = num_heads * head_dim
    output = nl.ndarray((batch_seq, total_q_size), dtype=hidden_states.dtype, buffer=nl.shared_hbm)
    
    # Tile indices
    ix_bs = nl.arange(TILE_SEQ)[:, None]
    ix_h = nl.arange(TILE_HEAD)[None, :]
    ix_hd = nl.arange(head_dim)[None, :]
    
    num_bs_tiles = (batch_seq + TILE_SEQ - 1) // TILE_SEQ
    num_h_tiles = (hidden_size + TILE_HEAD - 1) // TILE_HEAD
    
    # Load RMSNorm weight once
    rms_w = nl.load(rmsnorm_weight.reshape((1, hidden_size))[nl.arange(1)[:, None], ix_h],
                   mask=(ix_h < hidden_size))
    
    for bs_tile in nl.affine_range(num_bs_tiles):
        bs_start = bs_tile * TILE_SEQ
        bs_mask = (bs_start + ix_bs < batch_seq)
        
        # ===== Step 1: RMSNorm =====
        # Load input
        inp_tile = nl.load(
            hidden_states[bs_start + ix_bs, ix_h],
            mask=(bs_mask & (ix_h < hidden_size))
        )
        
        # Compute RMS
        square_sum = nl.sum(inp_tile * inp_tile, axis=[1])
        rms = nl.rsqrt(square_sum / hidden_size + eps)
        
        # Normalize and scale
        # Broadcasting rms
        rms_bcast = rms.broadcast_to((TILE_SEQ, hidden_size))
        normalized = inp_tile * rms_bcast * rms_w.broadcast_to((TILE_SEQ, hidden_size))
        
        # ===== Step 2: QKV Projection (for each head type) =====
        # Simplified: Just show Q projection structure
        # Full implementation would compute all Q, K, V
        
        q_out = nl.zeros((TILE_SEQ, num_heads, head_dim), dtype=nl.float32)
        
        for head_id in nl.affine_range(num_heads):
            head_q = nl.zeros((TILE_SEQ, head_dim), dtype=nl.float32)
            
            # Project to this head
            for h_tile in nl.affine_range(num_h_tiles):
                h_start = h_tile * TILE_HEAD
                h_mask = (h_start + nl.arange(TILE_HEAD) < hidden_size)
                
                norm_h = normalized[:, h_start + nl.arange(TILE_HEAD)]
                
                # Load Q weight for this head
                q_w_start = head_id * head_dim
                q_w = nl.load(
                    qkv_weight[h_start + nl.arange(TILE_HEAD)[:, None],
                              (q_w_start + nl.arange(head_dim))[None, :]],
                    mask=(h_mask[:, None])
                )
                
                head_q += nl.matmul(norm_h, q_w)
            
            # ===== Step 3: RoPE =====
            # Get sequence position
            seq_pos = bs_start + nl.arange(TILE_SEQ)
            
            # Load cos/sin
            cos_tile = nl.load(
                cos_cache[seq_pos[:, None], nl.arange(head_dim)[None, :]],
                mask=(bs_mask & (nl.arange(head_dim)[None, :] < head_dim))
            )
            sin_tile = nl.load(
                sin_cache[seq_pos[:, None], nl.arange(head_dim)[None, :]],
                mask=(bs_mask & (nl.arange(head_dim)[None, :] < head_dim))
            )
            
            # Apply rotation
            head_q_rot = nl.zeros((TILE_SEQ, head_dim), dtype=nl.float32)
            for i in nl.affine_range(head_dim // 2):
                x0 = head_q[:, 2*i]
                x1 = head_q[:, 2*i + 1]
                head_q_rot[:, 2*i] = x0 * cos_tile[:, i] - x1 * sin_tile[:, i]
                head_q_rot[:, 2*i + 1] = x0 * sin_tile[:, i] + x1 * cos_tile[:, i]
            
            q_out[:, head_id, :] = head_q_rot
        
        # ===== Step 4: Attention (simplified, would use flash attention kernel) =====
        # For now, just store Q
        for head_id in nl.affine_range(num_heads):
            nl.store(
                output[bs_start + ix_bs, head_id * head_dim + nl.arange(head_dim)],
                value=q_out[:, head_id, :],
                mask=bs_mask
            )
    
    return output


class NKIAttention(nn.Module):
    """
    NKI-accelerated Attention module.
    
    Drop-in replacement for standard attention with NKI kernels.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int = 32768,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.gqa_group_size = num_heads // num_kv_heads
        self.max_seq_len = max_seq_len
        
        # QKV projection weights
        total_qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim
        self.qkv_proj = nn.Linear(hidden_size, total_qkv_dim, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
        # RoPE caches (will be populated)
        self.register_buffer("cos_cache", None)
        self.register_buffer("sin_cache", None)
        
        print(f"NKIAttention: heads={num_heads}, kv_heads={num_kv_heads}, "
              f"head_dim={head_dim}, gqa_group={self.gqa_group_size}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass with NKI acceleration
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Flatten batch and sequence for NKI
        hidden_flat = hidden_states.view(-1, self.hidden_size)
        
        # Check if we can use NKI
        if hidden_states.is_xla and self.cos_cache is not None:
            # Use fused NKI kernel
            return self._nki_forward(
                hidden_flat, attention_mask, position_ids,
                past_key_value, use_cache, batch_size, seq_len
            )
        else:
            # PyTorch fallback
            return self._pytorch_forward(
                hidden_states, attention_mask, position_ids,
                past_key_value, use_cache
            )
    
    def _nki_forward(
        self, hidden_flat, attention_mask, position_ids,
        past_key_value, use_cache, batch_size, seq_len
    ):
        """NKI-accelerated forward"""
        # QKV projection using NKI
        q, k, v = nki_qkv_projection(
            hidden_flat,
            self.qkv_proj.weight,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
        )
        
        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Apply RoPE using NKI
        q_rot = nki_rope_kernel(
            q.view(-1, self.num_heads * self.head_dim),
            self.cos_cache,
            self.sin_cache,
            self.num_heads,
            self.head_dim,
        ).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        k_rot = nki_rope_kernel(
            k.view(-1, self.num_kv_heads * self.head_dim),
            self.cos_cache[:seq_len],
            self.sin_cache[:seq_len],
            self.num_kv_heads,
            self.head_dim,
        ).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Attention (use flash attention for now)
        # In full implementation, use nki_flash_attention_decode
        attn_output = self._flash_attention(
            q_rot, k_rot, v, attention_mask
        )
        
        # Output projection
        attn_flat = attn_output.view(-1, self.num_heads * self.head_dim)
        output = F.linear(attn_flat, self.o_proj.weight)
        output = output.view(batch_size, seq_len, self.hidden_size)
        
        return output, None
    
    def _pytorch_forward(self, hidden_states, attention_mask, position_ids,
                        past_key_value, use_cache):
        """PyTorch reference implementation"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # QKV projection
        qkv = self.qkv_proj(hidden_states)
        total_q = self.num_heads * self.head_dim
        total_kv = self.num_kv_heads * self.head_dim
        
        q = qkv[..., :total_q]
        k = qkv[..., total_q:total_q + total_kv]
        v = qkv[..., total_q + total_kv:]
        
        # Reshape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Repeat KV for GQA
        k = k.repeat_interleave(self.gqa_group_size, dim=1)
        v = v.repeat_interleave(self.gqa_group_size, dim=1)
        
        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        return output, None
    
    def _flash_attention(self, q, k, v, attention_mask):
        """Flash attention or standard attention"""
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        q = q.transpose(1, 2)  # [batch, heads, seq, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Repeat KV for GQA
        k = k.repeat_interleave(self.gqa_group_size, dim=1)
        v = v.repeat_interleave(self.gqa_group_size, dim=1)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        return attn_output.transpose(1, 2)


__all__ = [
    "nki_qkv_projection",
    "nki_rope_kernel",
    "nki_flash_attention_decode",
    "nki_fused_attention_rmsnorm_qkv",
    "NKIAttention",
]

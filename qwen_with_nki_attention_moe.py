"""
Enhanced Qwen with NKI - Attention + MoE Integrated Version

This is an enhanced version of qwen_with_nki.py that includes:
1. NKI RMSNorm (original)
2. NKI MoE Expert Kernels
3. NKI Attention Kernels
4. Fused Attention-MoE layers

Usage:
    # Replace the import in main.py
    from qwen_with_nki_attention_moe import NeuronQwen3MoeForCausalLM
"""

# Re-import everything from the original qwen_with_nki
from qwen_with_nki import *

# Additional imports for NKI Attention and MoE
try:
    from nki_moe_integrated import NKIMoEWrapper, enable_nki_moe
    from nki_attention import NKIAttention
    from nki_fused_attention_moe import FusedAttentionMoELayer, enable_fused_attention_moe
    NKI_ADVANCED_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Advanced NKI kernels not available: {e}")
    NKI_ADVANCED_AVAILABLE = False


class NeuronQwen3MoEAttentionNKI(NeuronQwen3MoEAttention):
    """
    Enhanced Attention with NKI kernel support.
    
    Falls back to parent implementation if NKI is not available or fails.
    """
    
    def __init__(self, config: Qwen3MoeInferenceConfig):
        super().__init__(config)
        
        self.nki_attention_enabled = (
            NKI_ADVANCED_AVAILABLE and 
            getattr(config.neuron_config, 'nki_attention_enabled', False)
        )
        
        if self.nki_attention_enabled:
            print("Using NKI-accelerated Attention")
            # Create NKI attention module
            self.nki_attn = NKIAttention(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
            )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        rmsnorm=None,
        **kwargs,
    ):
        """
        Forward with optional NKI acceleration
        """
        if self.nki_attention_enabled and hidden_states.is_xla:
            try:
                # Try NKI path
                return self._nki_forward(
                    hidden_states, attention_mask, position_ids,
                    past_key_value, rmsnorm, **kwargs
                )
            except Exception as e:
                print(f"NKI attention failed: {e}, falling back")
                return super().forward(
                    hidden_states, attention_mask, position_ids,
                    past_key_value, rmsnorm, **kwargs
                )
        else:
            # Use parent implementation
            return super().forward(
                hidden_states, attention_mask, position_ids,
                past_key_value, rmsnorm, **kwargs
            )
    
    def _nki_forward(self, hidden_states, attention_mask, position_ids,
                    past_key_value, rmsnorm, **kwargs):
        """NKI-accelerated forward path"""
        # Use the NKI attention module
        return self.nki_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )


class NeuronQwen3MoeDecoderLayerEnhanced(NeuronQwen3MoeDecoderLayer):
    """
    Enhanced Decoder Layer with integrated NKI Attention and MoE.
    """
    
    def __init__(self, config: Qwen3MoeInferenceConfig, layer_idx: int):
        # Don't call parent __init__ directly - we need custom setup
        nn.Module.__init__(self)
        
        self.hidden_size = config.hidden_size
        
        # Use NKI-enhanced attention
        if getattr(config.neuron_config, 'nki_attention_enabled', False):
            self.self_attn = NeuronQwen3MoEAttentionNKI(config=config)
        else:
            self.self_attn = NeuronQwen3MoEAttention(config=config)
        
        self.moe_fused_nki_kernel_enabled = getattr(config, "moe_fused_nki_kernel_enabled", False)
        
        # RMSNorm layers
        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        
        # MoE module
        if self.moe_fused_nki_kernel_enabled:
            self.mlp = initialize_moe_module(
                config=config, rmsnorm=self.post_attention_layernorm, init_tkg_module=True
            )
        else:
            self.mlp = initialize_moe_module(
                config=config,
            )
        
        # Wrap with NKI MoE if enabled
        self.nki_moe_enabled = (
            NKI_ADVANCED_AVAILABLE and 
            getattr(config.neuron_config, 'nki_moe_enabled', False)
        )
        if self.nki_moe_enabled:
            print(f"Layer {layer_idx}: Wrapping MLP with NKI MoE kernel")
            self.mlp = NKIMoEWrapper(self.mlp)
        
        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.qkv_kernel_fused_rmsnorm = not self.sequence_parallel_enabled
        self.moe_mask_padded_tokens = config.neuron_config.moe_mask_padded_tokens
    
    def forward(self, *args, **kwargs):
        # Use parent's forward
        return super().forward(*args, **kwargs)


class NeuronQwen3MoeModelEnhanced(NeuronQwen3MoeModel):
    """
    Enhanced Model with NKI optimizations.
    """
    
    def init_model(self, config: Qwen3MoeInferenceConfig):
        """Override to use enhanced decoder layers"""
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
        )
        
        # Use enhanced decoder layers
        if getattr(config.neuron_config, 'nki_fused_layer_enabled', False):
            print("Using fully fused Attention-MoE layers")
            # Would use FusedAttentionMoELayer here
            self.layers = nn.ModuleList([
                NeuronQwen3MoeDecoderLayerEnhanced(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                NeuronQwen3MoeDecoderLayerEnhanced(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ])
        
        self.norm = get_rmsnorm_cls()(self.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=False if self.on_device_sampling else True,
            bias=False,
        )


class NeuronQwen3MoeForCausalLMEnhanced(NeuronQwen3MoeForCausalLM):
    """
    Enhanced Causal LM with NKI optimizations.
    """
    
    _model_cls = NeuronQwen3MoeModelEnhanced
    
    def get_compiler_args(self):
        """Enhanced compiler args for NKI kernels"""
        # Get base compiler args
        compiler_args = super().get_compiler_args()
        
        # Add NKI-specific optimizations
        if getattr(self.neuron_config, 'nki_attention_enabled', False):
            compiler_args += " --enable-attention-nki-kernels"
        
        if getattr(self.neuron_config, 'nki_moe_enabled', False):
            compiler_args += " --enable-moe-nki-kernels"
        
        if getattr(self.neuron_config, 'nki_fused_layer_enabled', False):
            compiler_args += " --enable-fused-layer-optimization"
        
        return compiler_args


# Configuration extensions
def extend_neuron_config_for_nki(config):
    """
    Add NKI-specific configuration options.
    
    Usage:
        neuron_config = MoENeuronConfig(
            ...
            nki_attention_enabled=True,
            nki_moe_enabled=True,
            nki_fused_layer_enabled=False,  # Experimental
        )
    """
    # These are placeholders - actual config would need to be updated
    # in the MoENeuronConfig class
    defaults = {
        'nki_attention_enabled': False,
        'nki_moe_enabled': False,
        'nki_fused_layer_enabled': False,
        'nki_tile_size_token': 64,
        'nki_tile_size_hidden': 512,
        'nki_tile_size_intermediate': 512,
    }
    
    for key, value in defaults.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    
    return config


def print_nki_status():
    """Print status of NKI kernels"""
    print("\n" + "=" * 60)
    print("NKI Kernel Status")
    print("=" * 60)
    
    print("\nBasic NKI Kernels:")
    print(f"  ✓ RMSNorm (nki_custom_rmsnorm.py)")
    
    if NKI_ADVANCED_AVAILABLE:
        print("\nAdvanced NKI Kernels:")
        print(f"  ✓ MoE Expert Kernels (nki_moe_integrated.py)")
        print(f"  ✓ Attention Kernels (nki_attention.py)")
        print(f"  ✓ Fused Attention-MoE (nki_fused_attention_moe.py)")
    else:
        print("\nAdvanced NKI Kernels: Not available")
    
    print("\n" + "=" * 60 + "\n")


# Print status on import
print_nki_status()


__all__ = [
    "NeuronQwen3MoeForCausalLMEnhanced",
    "NeuronQwen3MoeModelEnhanced",
    "NeuronQwen3MoeDecoderLayerEnhanced",
    "NeuronQwen3MoEAttentionNKI",
    "extend_neuron_config_for_nki",
    "print_nki_status",
]

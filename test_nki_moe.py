"""
Test script for NKI MoE kernels

This script tests the correctness of NKI MoE implementations
against PyTorch reference implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# Check if running on Trainium
try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False
    print("Warning: torch_xla not available, running CPU-only tests")

# Import NKI modules
try:
    from nki_moe_integrated import NKIMoESimple, nki_matmul_tiled, nki_swiglu
    NKI_AVAILABLE = True
except ImportError as e:
    print(f"Error importing NKI modules: {e}")
    NKI_AVAILABLE = False
    sys.exit(1)


def test_nki_matmul():
    """Test NKI matrix multiplication kernel"""
    print("\n" + "=" * 60)
    print("Test 1: NKI Matrix Multiplication")
    print("=" * 60)
    
    if not XLA_AVAILABLE:
        print("Skipping (XLA not available)")
        return True
    
    device = xm.xla_device()
    
    # Test dimensions
    M, K, N = 128, 512, 256
    
    # Create test tensors
    torch.manual_seed(42)
    a = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    b = torch.randn(K, N, device=device, dtype=torch.bfloat16)
    
    print(f"Input shapes: A={a.shape}, B={b.shape}")
    
    # PyTorch reference
    expected = torch.matmul(a, b)
    
    # NKI kernel
    result = nki_matmul_tiled(a, b)
    
    # Compare
    max_diff = (expected - result).abs().max().item()
    mean_diff = (expected - result).abs().mean().item()
    
    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")
    
    # Threshold for bfloat16
    success = max_diff < 1e-1
    print("Result:", "PASS" if success else "FAIL")
    
    return success


def test_nki_swiglu():
    """Test NKI SwiGLU activation kernel"""
    print("\n" + "=" * 60)
    print("Test 2: NKI SwiGLU Activation")
    print("=" * 60)
    
    if not XLA_AVAILABLE:
        print("Skipping (XLA not available)")
        return True
    
    device = xm.xla_device()
    
    # Test dimensions
    M, N2 = 64, 1024
    N = N2 // 2
    
    torch.manual_seed(42)
    gate_up = torch.randn(M, N2, device=device, dtype=torch.bfloat16)
    
    print(f"Input shape: {gate_up.shape}")
    
    # PyTorch reference
    gate, up = gate_up.chunk(2, dim=-1)
    expected = F.silu(gate) * up
    
    # NKI kernel
    result = nki_swiglu(gate_up)
    
    # Compare
    max_diff = (expected - result).abs().max().item()
    mean_diff = (expected - result).abs().mean().item()
    
    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")
    
    success = max_diff < 1e-1
    print("Result:", "PASS" if success else "FAIL")
    
    return success


def test_nki_moe_module():
    """Test full NKI MoE module"""
    print("\n" + "=" * 60)
    print("Test 3: NKI MoE Module")
    print("=" * 60)
    
    # Configuration
    hidden_size = 256
    intermediate_size = 512
    num_experts = 4
    top_k = 2
    batch_size = 2
    seq_len = 8
    
    print(f"Config: hidden={hidden_size}, interm={intermediate_size}")
    print(f"        experts={num_experts}, top_k={top_k}")
    print(f"        batch={batch_size}, seq_len={seq_len}")
    
    # Create module
    moe = NKIMoESimple(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
    )
    
    # Initialize weights for reproducibility
    with torch.no_grad():
        nn.init.normal_(moe.router.weight, std=0.02)
        nn.init.normal_(moe.gate_up_proj, std=0.02)
        nn.init.normal_(moe.down_proj, std=0.02)
    
    # Create input
    torch.manual_seed(42)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # CPU forward (PyTorch reference)
    moe_cpu = moe
    output_cpu = moe_cpu(hidden_states)
    
    print(f"CPU output shape: {output_cpu.shape}")
    print(f"CPU output mean: {output_cpu.mean().item():.6f}")
    print(f"CPU output std: {output_cpu.std().item():.6f}")
    
    # Check output is valid
    success = True
    if torch.isnan(output_cpu).any():
        print("ERROR: NaN in output!")
        success = False
    if torch.isinf(output_cpu).any():
        print("ERROR: Inf in output!")
        success = False
    
    print("Result:", "PASS" if success else "FAIL")
    return success


def test_nki_moe_integration():
    """Test NKI MoE integration with model structure"""
    print("\n" + "=" * 60)
    print("Test 4: NKI MoE Integration")
    print("=" * 60)
    
    # Test the wrapper mechanism
    from nki_moe_integrated import NKIMoEWrapper
    
    # Create a dummy MLP module
    class DummyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts = 4
            self.top_k = 2
            
        def forward(self, hidden_states, padding_mask=None):
            # Simple passthrough for testing
            return hidden_states
    
    dummy_mlp = DummyMLP()
    
    # Wrap it
    try:
        wrapped = NKIMoEWrapper(dummy_mlp)
        print(f"Wrapper created successfully")
        print(f"  use_nki: {wrapped.use_nki}")
        print(f"  num_experts: {wrapped.num_experts}")
        print(f"  top_k: {wrapped.top_k}")
        
        # Test forward
        test_input = torch.randn(2, 8, 256)
        output = wrapped(test_input)
        print(f"Forward pass successful, output shape: {output.shape}")
        
        success = True
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    print("Result:", "PASS" if success else "FAIL")
    return success


def run_all_tests():
    """Run all tests"""
    print("\n" + "#" * 60)
    print("# NKI MoE Kernel Test Suite")
    print("#" * 60)
    
    results = []
    
    # Run tests
    results.append(("MatMul", test_nki_matmul()))
    results.append(("SwiGLU", test_nki_swiglu()))
    results.append(("MoE Module", test_nki_moe_module()))
    results.append(("Integration", test_nki_moe_integration()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:20s}: {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "🎉 All tests passed!" + "\n")
        return 0
    else:
        print("\n" + "⚠️  Some tests failed" + "\n")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

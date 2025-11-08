import torch
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class BenchmarkConfig:
    batch_size: int
    seq_len: int
    num_heads: int
    head_dim: int
    dtype: torch.dtype
    causal: bool = False
    dropout_p: float = 0.0

def check_installations():
    """Check if required packages are installed"""
    print("=" * 80)
    print("SYSTEM CHECK")
    print("=" * 80)
    
    components = {}
    
    try:
        import triton
        components['triton'] = triton.__version__
        print(f"✓ Triton: {triton.__version__}")
    except ImportError:
        components['triton'] = None
        print("✗ Triton not installed")
    
    try:
        from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
        components['flash_attn'] = True
        print("✓ Flash Attention 2: installed")
    except ImportError:
        components['flash_attn'] = None
        print("✗ Flash Attention 2 not installed")
    
    if torch.cuda.is_available():
        components['cuda'] = True
        print(f"✓ CUDA: {torch.version.cuda}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ Compute Capability: {torch.cuda.get_device_capability(0)}")
        
        # Memory info
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓ GPU Memory: {total_mem:.2f} GB")
    else:
        components['cuda'] = None
        print("✗ CUDA not available")
    
    print(f"✓ PyTorch: {torch.__version__}")
    
    return components

def pytorch_attention(q, k, v, causal=False, dropout_p=0.0):
    """Standard PyTorch attention with optional causal masking"""
    scale = 1.0 / math.sqrt(q.size(-1))
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    if causal:
        seq_len = q.size(2)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
    
    attn = torch.softmax(attn, dim=-1)
    
    if dropout_p > 0.0:
        attn = torch.nn.functional.dropout(attn, p=dropout_p)
    
    return torch.matmul(attn, v)

# Custom Triton Attention Kernel
def create_triton_attention():
    """Create a Triton attention kernel"""
    try:
        import triton
        import triton.language as tl
        
        @triton.jit
        def _fwd_kernel(
            Q, K, V, Out,
            stride_qz, stride_qh, stride_qm, stride_qk,
            stride_kz, stride_kh, stride_kn, stride_kk,
            stride_vz, stride_vh, stride_vn, stride_vk,
            stride_oz, stride_oh, stride_om, stride_ok,
            Z, H, M, N, D,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_D: tl.constexpr,
        ):
            start_m = tl.program_id(0)
            off_hz = tl.program_id(1)
            
            qk_scale = 1.0 / tl.sqrt(D.to(tl.float32))
            
            off_z = off_hz // H
            off_h = off_hz % H
            
            q_offset = off_z * stride_qz + off_h * stride_qh
            k_offset = off_z * stride_kz + off_h * stride_kh
            v_offset = off_z * stride_vz + off_h * stride_vh
            o_offset = off_z * stride_oz + off_h * stride_oh
            
            offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = tl.arange(0, BLOCK_N)
            offs_d = tl.arange(0, BLOCK_D)
            
            q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
            k_ptrs = K + k_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
            v_ptrs = V + v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
            
            q = tl.load(q_ptrs, mask=offs_m[:, None] < M, other=0.0)
            
            acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
            
            for start_n in range(0, N, BLOCK_N):
                k = tl.load(k_ptrs, mask=(start_n + offs_n[None, :]) < N, other=0.0)
                v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < N, other=0.0)
                
                qk = tl.dot(q, k) * qk_scale
                qk = tl.softmax(qk, axis=1)
                
                acc += tl.dot(qk.to(v.dtype), v)
                
                k_ptrs += BLOCK_N * stride_kn
                v_ptrs += BLOCK_N * stride_vn
            
            o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
            tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < M)
        
        return _fwd_kernel
    except Exception as e:
        print(f"Warning: Could not create Triton kernel: {e}")
        return None

def benchmark_implementation(name, func, q, k, v, config, num_runs=50, warmup=10):
    """Benchmark a single attention implementation"""
    # Warmup
    for _ in range(warmup):
        _ = func(q, k, v)
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated()
    
    start = time.time()
    for _ in range(num_runs):
        out = func(q, k, v)
        torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / num_runs
    peak_mem = torch.cuda.max_memory_allocated()
    mem_used = (peak_mem - start_mem) / 1024**2
    
    return {
        'name': name,
        'time_ms': avg_time * 1000,
        'memory_mb': mem_used,
        'output': out
    }

def run_comprehensive_benchmark(config: BenchmarkConfig, num_runs=50):
    """Run comprehensive benchmarks across all implementations"""
    print(f"\n{'=' * 80}")
    print(f"BENCHMARK: batch={config.batch_size}, seq={config.seq_len}, "
          f"heads={config.num_heads}, dim={config.head_dim}, "
          f"causal={config.causal}, dtype={config.dtype}")
    print(f"{'=' * 80}\n")
    
    device = torch.device("cuda")
    
    # Create input tensors
    q = torch.randn(config.batch_size, config.num_heads, config.seq_len, 
                    config.head_dim, device=device, dtype=config.dtype)
    k = torch.randn(config.batch_size, config.num_heads, config.seq_len, 
                    config.head_dim, device=device, dtype=config.dtype)
    v = torch.randn(config.batch_size, config.num_heads, config.seq_len, 
                    config.head_dim, device=device, dtype=config.dtype)
    
    results = []
    
    # 1. PyTorch Native
    print("1. PyTorch Native Attention...")
    try:
        result = benchmark_implementation(
            "PyTorch Native",
            lambda q, k, v: pytorch_attention(q, k, v, config.causal, config.dropout_p),
            q, k, v, config, num_runs
        )
        results.append(result)
        print(f"   ✓ Time: {result['time_ms']:.3f} ms | Memory: {result['memory_mb']:.2f} MB")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # 2. PyTorch SDPA
    print("\n2. PyTorch SDPA (optimized)...")
    try:
        result = benchmark_implementation(
            "PyTorch SDPA",
            lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=config.causal, dropout_p=config.dropout_p
            ),
            q, k, v, config, num_runs
        )
        results.append(result)
        speedup = results[0]['time_ms'] / result['time_ms']
        print(f"   ✓ Time: {result['time_ms']:.3f} ms | Memory: {result['memory_mb']:.2f} MB | "
              f"Speedup: {speedup:.2f}x")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # 3. Flash Attention 2
    print("\n3. Flash Attention 2...")
    try:
        from flash_attn import flash_attn_func
        
        # Flash attention format: (batch, seqlen, nheads, headdim)
        q_fa = q.transpose(1, 2).contiguous()
        k_fa = k.transpose(1, 2).contiguous()
        v_fa = v.transpose(1, 2).contiguous()
        
        result = benchmark_implementation(
            "Flash Attention 2",
            lambda q, k, v: flash_attn_func(q, k, v, causal=config.causal, dropout_p=config.dropout_p),
            q_fa, k_fa, v_fa, config, num_runs
        )
        results.append(result)
        speedup = results[0]['time_ms'] / result['time_ms']
        print(f"   ✓ Time: {result['time_ms']:.3f} ms | Memory: {result['memory_mb']:.2f} MB | "
              f"Speedup: {speedup:.2f}x")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # 4. Flash Attention 2 with QKV packed (more efficient)
    print("\n4. Flash Attention 2 (QKV Packed)...")
    try:
        from flash_attn import flash_attn_qkvpacked_func
        
        # Create packed QKV tensor
        qkv = torch.stack([q, k, v], dim=2).transpose(1, 2).contiguous()
        
        result = benchmark_implementation(
            "Flash Attention 2 Packed",
            lambda qkv, _, __: flash_attn_qkvpacked_func(qkv, causal=config.causal, dropout_p=config.dropout_p),
            qkv, None, None, config, num_runs
        )
        results.append(result)
        speedup = results[0]['time_ms'] / result['time_ms']
        print(f"   ✓ Time: {result['time_ms']:.3f} ms | Memory: {result['memory_mb']:.2f} MB | "
              f"Speedup: {speedup:.2f}x")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # 5. Verify numerical accuracy
    print("\n5. Numerical Accuracy Check...")
    if len(results) > 1:
        baseline = results[0]['output']
        for i, result in enumerate(results[1:], 1):
            try:
                # Transpose back if needed for comparison
                out = result['output']
                if out.dim() == 4 and out.shape[1] != baseline.shape[1]:
                    out = out.transpose(1, 2)
                
                # Handle packed QKV case - output shape is different
                if baseline.shape != out.shape:
                    print(f"   {result['name']}: Shape mismatch (baseline: {baseline.shape}, output: {out.shape}) - skipping comparison")
                    continue
                
                max_diff = (baseline - out).abs().max().item()
                mean_diff = (baseline - out).abs().mean().item()
                print(f"   {result['name']}:")
                print(f"     Max diff: {max_diff:.6e} | Mean diff: {mean_diff:.6e}")
            except Exception as e:
                print(f"   {result['name']}: Error comparing - {e}")
    
    return results

def profile_memory_scaling():
    """Profile memory usage across different sequence lengths"""
    print(f"\n{'=' * 80}")
    print("MEMORY SCALING ANALYSIS")
    print(f"{'=' * 80}\n")
    
    seq_lengths = [256, 512, 1024, 2048, 4096]
    configs = {
        'Native': [],
        'Flash Attn 2': []
    }
    
    for seq_len in seq_lengths:
        print(f"Testing sequence length: {seq_len}")
        config = BenchmarkConfig(
            batch_size=2,
            seq_len=seq_len,
            num_heads=8,
            head_dim=64,
            dtype=torch.float16
        )
        
        device = torch.device("cuda")
        q = torch.randn(config.batch_size, config.num_heads, seq_len, 
                       config.head_dim, device=device, dtype=config.dtype)
        k = torch.randn(config.batch_size, config.num_heads, seq_len, 
                       config.head_dim, device=device, dtype=config.dtype)
        v = torch.randn(config.batch_size, config.num_heads, seq_len, 
                       config.head_dim, device=device, dtype=config.dtype)
        
        # Test native
        try:
            torch.cuda.reset_peak_memory_stats()
            _ = pytorch_attention(q, k, v)
            torch.cuda.synchronize()
            mem = torch.cuda.max_memory_allocated() / 1024**2
            configs['Native'].append(mem)
            print(f"  Native: {mem:.2f} MB")
        except:
            configs['Native'].append(None)
        
        # Test Flash Attention
        try:
            from flash_attn import flash_attn_func
            q_fa = q.transpose(1, 2).contiguous()
            k_fa = k.transpose(1, 2).contiguous()
            v_fa = v.transpose(1, 2).contiguous()
            
            torch.cuda.reset_peak_memory_stats()
            _ = flash_attn_func(q_fa, k_fa, v_fa)
            torch.cuda.synchronize()
            mem = torch.cuda.max_memory_allocated() / 1024**2
            configs['Flash Attn 2'].append(mem)
            print(f"  Flash Attn 2: {mem:.2f} MB")
        except:
            configs['Flash Attn 2'].append(None)
    
    # Plot results
    try:
        plt.figure(figsize=(10, 6))
        for name, mems in configs.items():
            valid_data = [(s, m) for s, m in zip(seq_lengths, mems) if m is not None]
            if valid_data:
                seqs, mems_valid = zip(*valid_data)
                plt.plot(seqs, mems_valid, marker='o', label=name, linewidth=2)
        
        plt.xlabel('Sequence Length', fontsize=12)
        plt.ylabel('Peak Memory (MB)', fontsize=12)
        plt.title('Memory Scaling: Flash Attention vs Native', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('memory_scaling.png', dpi=150)
        print(f"\n✓ Memory scaling plot saved to 'memory_scaling.png'")
    except Exception as e:
        print(f"Could not create plot: {e}")

def test_gradient_computation():
    """Test backward pass and gradient computation"""
    print(f"\n{'=' * 80}")
    print("GRADIENT COMPUTATION TEST")
    print(f"{'=' * 80}\n")
    
    config = BenchmarkConfig(
        batch_size=2,
        seq_len=512,
        num_heads=8,
        head_dim=64,
        dtype=torch.float16
    )
    
    device = torch.device("cuda")
    
    q = torch.randn(config.batch_size, config.num_heads, config.seq_len, 
                    config.head_dim, device=device, dtype=config.dtype, requires_grad=True)
    k = torch.randn(config.batch_size, config.num_heads, config.seq_len, 
                    config.head_dim, device=device, dtype=config.dtype, requires_grad=True)
    v = torch.randn(config.batch_size, config.num_heads, config.seq_len, 
                    config.head_dim, device=device, dtype=config.dtype, requires_grad=True)
    
    # Test PyTorch SDPA backward
    print("1. PyTorch SDPA (forward + backward)...")
    try:
        start = time.time()
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        print(f"   ✓ Time: {(time.time() - start) * 1000:.3f} ms")
        print(f"   ✓ Gradient shapes: q={q.grad.shape}, k={k.grad.shape}, v={v.grad.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test Flash Attention backward
    print("\n2. Flash Attention 2 (forward + backward)...")
    try:
        from flash_attn import flash_attn_func
        
        # Reset gradients and recreate tensors with requires_grad
        q2 = torch.randn(config.batch_size, config.num_heads, config.seq_len, 
                        config.head_dim, device=device, dtype=config.dtype, requires_grad=True)
        k2 = torch.randn(config.batch_size, config.num_heads, config.seq_len, 
                        config.head_dim, device=device, dtype=config.dtype, requires_grad=True)
        v2 = torch.randn(config.batch_size, config.num_heads, config.seq_len, 
                        config.head_dim, device=device, dtype=config.dtype, requires_grad=True)
        
        q_fa = q2.transpose(1, 2).contiguous()
        k_fa = k2.transpose(1, 2).contiguous()
        v_fa = v2.transpose(1, 2).contiguous()
        
        start = time.time()
        out = flash_attn_func(q_fa, k_fa, v_fa)
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        print(f"   ✓ Time: {(time.time() - start) * 1000:.3f} ms")
    except Exception as e:
        print(f"   ✗ Error: {e}")

def main():
    """Main test suite"""
    components = check_installations()
    
    if not components['cuda']:
        print("\n✗ CUDA required. Exiting.")
        return
    
    # Test configurations
    test_configs = [
        BenchmarkConfig(2, 512, 8, 64, torch.float16, causal=False),
        BenchmarkConfig(4, 1024, 8, 64, torch.float16, causal=False),
        BenchmarkConfig(2, 2048, 8, 64, torch.float16, causal=True),
        BenchmarkConfig(1, 4096, 16, 64, torch.float16, causal=False),
        BenchmarkConfig(2, 1024, 8, 128, torch.bfloat16, causal=True),
    ]
    
    all_results = []
    
    for config in test_configs:
        try:
            results = run_comprehensive_benchmark(config, num_runs=50)
            all_results.append((config, results))
        except Exception as e:
            print(f"Benchmark failed: {e}")
    
    # Memory scaling test
    try:
        profile_memory_scaling()
    except Exception as e:
        print(f"Memory profiling failed: {e}")
    
    # Gradient test
    try:
        test_gradient_computation()
    except Exception as e:
        print(f"Gradient test failed: {e}")
    
    # Summary
    print(f"\n{'=' * 80}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 80}\n")
    
    for config, results in all_results:
        print(f"Seq Length: {config.seq_len}, Causal: {config.causal}")
        for result in results:
            speedup = results[0]['time_ms'] / result['time_ms'] if results else 1.0
            print(f"  {result['name']:25s}: {result['time_ms']:7.3f} ms ({speedup:.2f}x)")
        print()

if __name__ == "__main__":
    main()
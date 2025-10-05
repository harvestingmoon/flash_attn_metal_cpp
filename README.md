# Flash Attention Metal for Apple Silicon

A high-performance Flash Attention implementation optimized for Apple Silicon using metal-cpp. Inspiration of this project was taken from the original metal-flash-attention implemented in Swift but there is a growing need for bindings.


## Performance

The Metal implementation provides significant speedups over PyTorch on Apple Silicon:

- **2-4x faster** than MLX for most configurations
- **14-82x faster** than PyTorch CPU for large sequences
- **Linear memory scaling** with sequence length (vs quadratic for naive attention)
- **Optimized for Apple Silicon** with tuned tile sizes and memory access patterns

### Benchmark Results

The following benchmarks were conducted on Apple Silicon using half precision (fp16):

```
Configuration                       Metal        MLX    PyTorch     vs MLX   vs PyTorch
--------------------------------------------------------------------------------------------
Small: 256 tokens                   0.20ms      0.52ms      2.90ms      2.59x       14.30x
Medium: 512 tokens                  0.31ms      0.98ms      8.28ms      3.13x       26.54x
Large: 1024 tokens                  0.48ms      1.74ms     31.09ms      3.59x       64.33x
XLarge: 2048 tokens                 1.05ms      3.27ms     86.08ms      3.12x       82.05x
384 tokens, d=128                   0.39ms      1.15ms      6.33ms      2.96x       16.29x
768 tokens, d=128                   0.65ms      2.12ms     20.77ms      3.27x       32.01x
Batch=1: 1024 tokens                0.19ms      0.53ms      7.04ms      2.79x       36.98x
Batch=2: 1024 tokens                0.30ms      0.98ms     14.93ms      3.22x       49.06x
Batch=8: 1024 tokens                0.85ms      3.28ms     55.61ms      3.85x       65.45x
```

Performance varies by configuration, but typical speedups are observed for:
- Sequence lengths: 256-2048 tokens
- Head dimensions: 64-128  
- Batch sizes: 1-8project provides a native Metal backend for Flash Attention with significant performance improvements over traditional implementations.

## Overview
Flash Attention Metal implements the FlashAttention-2 algorithm using Apple's Metal Performance Shaders framework, specifically optimized for Apple Silicon GPUs. The implementation provides:

- **Memory-efficient attention** following FlashAttention-2 principles
- **Native Metal compute shaders** for optimal Apple Silicon performance  
- **Python bindings** for easy integration with existing ML workflows
- **Comprehensive testing** with correctness verification against PyTorch reference
- **Support for various attention patterns** including causal, windowed, and grouped-query attention

## Features

- **FlashAttention-2 Algorithm**: Memory-efficient attention with linear memory complexity
- **Apple Silicon Optimized**: Native Metal shaders tuned for M1/M2/M3 architectures
- **Attention Types**: 
  - Standard multi-head attention
  - Causal (masked) attention with optimized early termination
  - Half precision (fp16) computation for optimal performance
- **Advanced Optimizations**:
  - Vectorized operations using float4 SIMD
  - Coalesced memory access patterns
  - Aggressive 32x32 tiling for Apple Silicon
  - Online softmax with numerical stability
- **Python Integration**: Easy-to-use Python bindings via pybind11
- **Correctness Verified**: Comprehensive test suite against PyTorch reference
- **Performance Benchmarks**: Comparisons with PyTorch and MLX implementations

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3)
- **Python 3.8+**
- **CMake 3.20+** 
- **Xcode Command Line Tools**

### Python Dependencies
```bash
pip install numpy torch
pip install mlx-py  # Optional, for performance comparisons
```

## Quick Start

### 1. Build the Project

```bash
./build.sh
```

This will:
- Configure the project with CMake
- Compile the Metal shaders and C++ core
- Build Python bindings
- Create the `_flash_attn_metal` module

### 2. Verify Installation

```bash
# Check correctness against PyTorch reference
./build.sh verify

# Run comprehensive unit tests  
./build.sh test
```

### 3. Use in Python

```python
import _flash_attn_metal
import numpy as np

# Initialize Metal backend
_flash_attn_metal.initialize()

# Load and compile shaders
with open('kernels/common.metal') as f:
    common_src = f.read()
with open('kernels/flash_attention_fwd_optimized.metal') as f:
    opt_src = f.read()

_flash_attn_metal.compile_shaders(common_src + '\n' + opt_src)

# Create sample data (batch=1, seqlen=512, heads=8, headdim=64)
q = np.random.randn(1, 512, 8, 64).astype(np.float32)
k = np.random.randn(1, 512, 8, 64).astype(np.float32)  
v = np.random.randn(1, 512, 8, 64).astype(np.float32)

# Run flash attention
output = _flash_attn_metal.forward(
    q, k, v, 
    softmax_scale=1.0/8.0,  # 1/sqrt(head_dim)
    is_causal=True  # Causal masking supported
)

print(f"Output shape: {output.shape}")
```

## Build Commands

The `build.sh` script supports several commands:

```bash
./build.sh              # Build the project (default)
./build.sh clean        # Clean build artifacts  
./build.sh rebuild      # Clean and build
./build.sh test         # Build and run unit tests
./build.sh verify       # Build and verify correctness 
./build.sh all          # Clean, build, verify, and test
./build.sh help         # Show help message
```

## Testing & Verification

### Correctness Verification
`verify_correctness.py` compares the Metal implementation against PyTorch reference:

```bash
python3 verify_correctness.py
```

This script:
- Tests various matrix sizes and attention configurations
- Measures numerical precision (relative and absolute error)
- Ensures outputs match PyTorch within acceptable tolerances

### Unit Testing  
`test.py` provides comprehensive unit tests covering:

```bash
python3 test.py
```

Tests include:
- Different batch sizes, sequence lengths, head counts
- Causal vs non-causal attention patterns
- Half precision (fp16) numerical accuracy
- Performance benchmarks vs PyTorch and MLX
- Edge cases and error handling

## Performance

The Metal implementation provides significant speedups over PyTorch on Apple Silicon:

- **2-4x faster** than PyTorch CUDA emulation
- **1.5-2x faster** than MLX for most configurations
- **Linear memory scaling** with sequence length (vs quadratic for naive attention)
- **Optimized for Apple Silicon** with tuned tile sizes and memory access patterns

Performance varies by configuration, but typical speedups are observed for:
- Sequence lengths: 512-8192 tokens
- Head dimensions: 64-128  
- Batch sizes: 1-32

## Project Structure

```
├── build.sh                    # Build script
├── CMakeLists.txt             # CMake configuration
├── include/
│   └── flash_attn_core.hpp    # C++ header definitions
├── src/
│   └── flash_attn_core.cpp    # C++ implementation
├── python/
│   └── bindings.cpp           # Python bindings (pybind11)
├── kernels/
│   ├── common.metal           # Common Metal utilities
│   ├── flash_attention_fwd.metal          # Standard kernel
│   └── flash_attention_fwd_optimized.metal # Optimized kernel
├── external/
│   └── metal-cpp/             # Metal C++ headers
├── test.py                    # Unit tests
├── verify_correctness.py      # Correctness verification
└── build/                     # Build output directory
```

## Technical Details

### Metal Shaders
- **Tiled computation** with optimized block sizes for Apple Silicon
- **Shared memory usage** to minimize global memory access
- **SIMD group optimizations** for parallel computation
- **Function constants** for compile-time optimization

### Memory Layout
- **Row-major tensors** with shape `[batch, seqlen, num_heads, head_dim]`
- **Contiguous memory** requirements for optimal performance
- **32KB threadgroup memory** utilization for maximum tile sizes

### Supported Operations
- Forward pass with optional causal masking
- Half precision (fp16) computation
- Vectorized SIMD operations (float4)
- Online softmax with numerical stability
- Optimized memory access patterns

### Implementation Status

**Currently Active in Optimized Kernel:**
- Standard multi-head attention
- Causal masking
- Half precision (fp16) computation
- Vectorized SIMD operations
- Memory-efficient tiling

**Infrastructure Available (common.metal) - Needs Integration:**
- **Rotary Position Embedding (RoPE)**: Full implementation with cos/sin tables
- **Windowed Attention**: Left/right window masking functions
- **ALiBi Support**: Linear bias computation for positions
- **Dropout**: Training-time dropout with pseudo-random generation
- **GQA/MQA Support**: Head mapping utilities for grouped/multi-query attention
- **Advanced Utilities**: SIMD reductions, vectorized loads, numerical stability helpers

**Missing Infrastructure:**
- **Variable Length Sequences (VarLen)**: No padding mask support implemented

## TODO / Future Improvements

### Integration Tasks (Infrastructure Exists)
The following features have complete implementations in `kernels/common.metal` but need integration into the optimized kernel:

- **Rotary Position Embedding (RoPE)**: Integrate existing `apply_rotary_embedding()` functions
- **Windowed Attention**: Integrate `is_window_masked()` for sliding window patterns  
- **Grouped-Query Attention (GQA)**: Use `gqa_kv_head_idx()` for head mapping
- **Multi-Query Attention (MQA)**: Leverage existing GQA infrastructure
- **ALiBi Support**: Integrate `compute_alibi_bias()` for positional biases
- **Dropout**: Integrate `apply_dropout()` for training support
- **Advanced Optimizations**: Use existing SIMD utilities and vectorized operations

### New Development Tasks
- **Variable Length Sequences (VarLen)**: Implement padding mask support for batched sequences
- **Backward Pass**: Implement gradient computation for training
- **Multi-Latent Attention (MLA)**: Research and implement advanced attention mechanisms
- **Mixed Precision**: Support for different precision modes beyond fp16
- **Dynamic Batching**: Support for variable batch sizes during inference

### Performance Optimizations
- **Kernel Fusion**: Combine multiple operations for better throughput
- **Memory Bandwidth Optimization**: Further reduce global memory access
- **Multi-GPU Support**: Scale across multiple Apple Silicon devices

Contributions towards these goals are welcome!

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Verify correctness: `./build.sh verify`
5. Run full test suite: `./build.sh test`
6. Submit a pull request

## License

This project is open source. Please check the license file for details.

## Acknowledgments

- **FlashAttention-2** paper and original implementation
- **Apple Metal Performance Shaders** documentation
- **metal-cpp** C++ wrapper for Metal APIs
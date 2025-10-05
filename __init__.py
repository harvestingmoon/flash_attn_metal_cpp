"""
Flash Attention for Apple Silicon Metal

High-performance Flash Attention implementation optimized for Apple Silicon GPUs.
"""

__version__ = "0.1.0"

import numpy as np

try:
    # Try relative import first (when imported as a package)
    try:
        from . import _flash_attn_metal
    except ImportError:
        # Fall back to direct import (when run from the directory)
        import _flash_attn_metal
    
    # Initialize on import
    if not _flash_attn_metal.initialize():
        raise RuntimeError("Failed to initialize Metal device")
    
    # Load shaders
    import os
    shader_dir = os.path.join(os.path.dirname(__file__), 'kernels')
    
    # Try to load compiled metallib first
    metallib_path = os.path.join(os.path.dirname(__file__), 'build', 'flash_attention.metallib')
    if os.path.exists(metallib_path):
        _flash_attn_metal.load_shaders(metallib_path)
    else:
        # Fall back to compiling from source
        common_metal = os.path.join(shader_dir, 'common.metal')
        fwd_metal = os.path.join(shader_dir, 'flash_attention_fwd.metal')
        
        if os.path.exists(common_metal) and os.path.exists(fwd_metal):
            with open(common_metal, 'r') as f:
                common_src = f.read()
            with open(fwd_metal, 'r') as f:
                fwd_src = f.read()
            
            # Combine sources
            combined_src = common_src + "\n\n" + fwd_src
            _flash_attn_metal.compile_shaders(combined_src)
        else:
            raise RuntimeError("Metal shaders not found. Please ensure kernels are available.")
    
    def metal_attention(q, k, v, causal=False, softmax_scale=None, dropout_p=0.0):
        """
        Compute flash attention on Apple Silicon Metal.
        
        Args:
            q: Query tensor, shape (batch, seqlen_q, num_heads, head_dim), dtype float16
            k: Key tensor, shape (batch, seqlen_k, num_heads_k, head_dim), dtype float16
            v: Value tensor, shape (batch, seqlen_k, num_heads_k, head_dim), dtype float16
            causal: Whether to apply causal masking
            softmax_scale: Scale for softmax. If None, uses 1/sqrt(head_dim)
            dropout_p: Dropout probability (currently not supported)
        
        Returns:
            output: Output tensor, shape (batch, seqlen_q, num_heads, head_dim), dtype float16
        """
        # Convert to numpy if needed
        if not isinstance(q, np.ndarray):
            q = np.array(q)
        if not isinstance(k, np.ndarray):
            k = np.array(k)
        if not isinstance(v, np.ndarray):
            v = np.array(v)
        
        # Ensure float16
        q = q.astype(np.float16)
        k = k.astype(np.float16)
        v = v.astype(np.float16)
        
        # Auto-compute scale if not provided
        if softmax_scale is None:
            head_dim = q.shape[-1]
            softmax_scale = 1.0 / np.sqrt(head_dim)
        
        # Call the C++ backend
        output = _flash_attn_metal.forward(
            q, k, v,
            softmax_scale=float(softmax_scale),
            is_causal=bool(causal),
            dropout_p=float(dropout_p)
        )
        
        return output
    
    # Export main functions
    initialize = _flash_attn_metal.initialize
    get_device_info = _flash_attn_metal.get_device_info
    
    __all__ = ['metal_attention', 'initialize', 'get_device_info']
    
except ImportError as e:
    import sys
    import os
    
    # Try to provide helpful error message
    error_msg = f"""
    Failed to import Metal Flash Attention module: {e}
    
    Please ensure the module is built correctly:
      cd {os.path.dirname(__file__)}
      mkdir -p build && cd build
      cmake ..
      make -j
      cd ..
    
    Then install with:
      pip install -e .
    """
    
    print(error_msg, file=sys.stderr)
    raise

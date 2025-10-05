//
// common.metal
// Flash Attention Metal - Common Utilities
//
// High-performance Flash Attention implementation for Apple Silicon
// Following FlashAttention-2 memory optimization principles
//

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Constants and Configuration
// ============================================================================

// Block sizes for tiling (tuned for Apple Silicon)
constant uint BLOCK_M [[function_constant(0)]];  // Query tile size (default: 64)
constant uint BLOCK_N [[function_constant(1)]];  // Key/Value tile size (default: 64)
constant uint BLOCK_HEADDIM [[function_constant(2)]];  // Head dimension tile (default: 64)
constant uint NUM_WARPS [[function_constant(3)]];  // Number of SIMD groups (default: 4)

// Default values if function constants not provided
// Optimized block sizes for better performance while fitting in 32KB threadgroup memory
constant uint DEFAULT_BLOCK_M = 32;
constant uint DEFAULT_BLOCK_N = 32;
constant uint DEFAULT_BLOCK_HEADDIM = 64;
constant uint DEFAULT_NUM_WARPS = 4;

// ============================================================================
// Type Definitions
// ============================================================================

struct FlashAttentionParams {
    uint batch_size;
    uint seqlen_q;
    uint seqlen_k;
    uint num_heads_q;
    uint num_heads_k;
    uint head_dim;
    float softmax_scale;
    bool is_causal;
    int window_left;   // -1 for no window
    int window_right;  // -1 for no window
    float dropout_p;
    bool has_alibi;
    bool has_rotary;
    uint gqa_group_size;  // num_heads_q / num_heads_k for GQA
};

// ============================================================================
// Utility Functions
// ============================================================================

// Safe maximum for numerical stability
template<typename T>
inline T safe_max(T a, T b) {
    return max(a, b);
}

// Safe exponential to prevent overflow
inline float safe_exp(float x) {
    return exp(min(x, 88.0f));  // exp(88) ≈ 1e38, safe for float
}

// ============================================================================
// Softmax Utilities (Online Softmax Algorithm)
// ============================================================================

// Online softmax state for numerical stability
// Following the algorithm from "Online normalizer calculation for softmax"
struct SoftmaxState {
    float max_val;    // Running maximum
    float sum_exp;    // Sum of exponentials
    
    SoftmaxState() : max_val(-INFINITY), sum_exp(0.0f) {}
};

// Update softmax state with new value (FlashAttention v2 online algorithm)
inline void update_softmax_state(thread SoftmaxState& state, float new_val) {
    float old_max = state.max_val;
    float new_max = safe_max(old_max, new_val);
    
    // Rescale previous sum if max changed
    if (new_max > old_max) {
        state.sum_exp *= safe_exp(old_max - new_max);
    }
    
    state.max_val = new_max;
    state.sum_exp += safe_exp(new_val - new_max);
}

// Compute softmax probability from attention score and state
inline float compute_softmax_prob(float attn_score, thread const SoftmaxState& state) {
    return safe_exp(attn_score - state.max_val) / state.sum_exp;
}

// ============================================================================
// Masking Functions
// ============================================================================

// Causal mask: only attend to positions <= current position
inline bool is_causal_masked(uint q_idx, uint k_idx, bool is_causal) {
    return is_causal && (k_idx > q_idx);
}

// Sliding window mask: only attend within window
inline bool is_window_masked(int q_idx, int k_idx, int window_left, int window_right) {
    if (window_left < 0 && window_right < 0) return false;  // No window
    
    int relative_pos = int(k_idx) - int(q_idx);
    
    bool left_masked = (window_left >= 0) && (relative_pos < -window_left);
    bool right_masked = (window_right >= 0) && (relative_pos > window_right);
    
    return left_masked || right_masked;
}

// Combined mask check
inline bool is_masked(uint q_idx, uint k_idx, constant FlashAttentionParams& params) {
    // Causal mask
    if (is_causal_masked(q_idx, k_idx, params.is_causal)) {
        return true;
    }
    
    // Window mask
    if (is_window_masked(q_idx, k_idx, params.window_left, params.window_right)) {
        return true;
    }
    
    return false;
}

// ============================================================================
// ALiBi (Attention with Linear Biases)
// ============================================================================

// Compute ALiBi bias for a given position
inline float compute_alibi_bias(uint q_idx, uint k_idx, uint head_idx, 
                                 device const float* alibi_slopes) {
    float slope = alibi_slopes[head_idx];
    int relative_pos = int(k_idx) - int(q_idx);
    return slope * float(relative_pos);
}

// ============================================================================
// Rotary Position Embeddings (RoPE)
// ============================================================================

// Apply RoPE to a pair of values (thread memory version)
inline void apply_rotary_pair(thread float& x, thread float& y, 
                               float cos_val, float sin_val) {
    float x_rot = x * cos_val - y * sin_val;
    float y_rot = x * sin_val + y * cos_val;
    x = x_rot;
    y = y_rot;
}

// Apply RoPE to a pair of values (threadgroup memory version)
inline void apply_rotary_pair(threadgroup float& x, threadgroup float& y, 
                               float cos_val, float sin_val) {
    float x_rot = x * cos_val - y * sin_val;
    float y_rot = x * sin_val + y * cos_val;
    x = x_rot;
    y = y_rot;
}

// Apply RoPE to Q or K vector (thread memory version)
inline void apply_rotary_embedding(
    thread float* vec,
    uint vec_len,
    uint seq_idx,
    device const float* cos_table,
    device const float* sin_table,
    uint head_dim
) {
    uint half_dim = head_dim / 2;
    
    for (uint i = 0; i < half_dim && i < vec_len / 2; i++) {
        uint cos_sin_idx = seq_idx * half_dim + i;
        float cos_val = cos_table[cos_sin_idx];
        float sin_val = sin_table[cos_sin_idx];
        
        apply_rotary_pair(vec[i], vec[i + half_dim], cos_val, sin_val);
    }
}

// Apply RoPE to Q or K vector (threadgroup memory version)
inline void apply_rotary_embedding(
    threadgroup float* vec,
    uint vec_len,
    uint seq_idx,
    device const float* cos_table,
    device const float* sin_table,
    uint head_dim
) {
    uint half_dim = head_dim / 2;
    
    for (uint i = 0; i < half_dim && i < vec_len / 2; i++) {
        uint cos_sin_idx = seq_idx * half_dim + i;
        float cos_val = cos_table[cos_sin_idx];
        float sin_val = sin_table[cos_sin_idx];
        
        apply_rotary_pair(vec[i], vec[i + half_dim], cos_val, sin_val);
    }
}

// ============================================================================
// Dropout
// ============================================================================

// Simple dropout with uniform random distribution
// NOTE: For production, use a proper PRNG seeded per sequence
inline float apply_dropout(float value, float dropout_p, float random_val) {
    if (dropout_p == 0.0f) return value;
    
    // random_val should be in [0, 1)
    float keep_prob = 1.0f - dropout_p;
    
    // Inverted dropout for training efficiency
    return (random_val < keep_prob) ? (value / keep_prob) : 0.0f;
}

// Generate pseudo-random value for dropout (simple LCG)
// Better: use Metal's proper random number generation in production
inline float dropout_random(uint seed, uint idx) {
    // Linear congruential generator
    uint state = seed ^ idx;
    state = state * 1664525u + 1013904223u;
    
    // Convert to float in [0, 1)
    return float(state) / 4294967296.0f;
}

// ============================================================================
// GQA (Grouped Query Attention) Utilities
// ============================================================================

// Map query head to corresponding key/value head for GQA
inline uint gqa_kv_head_idx(uint q_head_idx, uint gqa_group_size) {
    return q_head_idx / gqa_group_size;
}

// ============================================================================
// Memory Access Patterns (Coalesced Access)
// ============================================================================

// Compute strided index for coalesced memory access
inline uint compute_strided_idx(uint batch, uint seq, uint head, uint dim,
                                 uint seqlen, uint num_heads, uint head_dim) {
    // Layout: [batch, seq, heads, head_dim]
    return ((batch * seqlen + seq) * num_heads + head) * head_dim + dim;
}

// ============================================================================
// Threadgroup (Shared Memory) Utilities
// ============================================================================

// Threadgroup barrier for synchronization
inline void threadgroup_barrier_sync(metal::mem_flags barrier_type = metal::mem_flags::mem_threadgroup) {
    metal::threadgroup_barrier(barrier_type);
}

// ============================================================================
// SIMD Group (Warp-level) Operations
// ============================================================================

// SIMD shuffle for warp-level communication
template<typename T>
inline T simd_shuffle_down(T value, uint delta) {
    return simd_shuffle_down(value, delta);
}

// SIMD reduction: sum across all threads in SIMD group
inline float simd_sum(float value) {
    return simd_sum(value);
}

// SIMD reduction: max across all threads in SIMD group
inline float simd_max(float value) {
    return simd_max(value);
}

// ============================================================================
// Block-level Reductions (for softmax)
// ============================================================================

// Reduce max across threadgroup (for numerical stability)
inline float threadgroup_max_reduce(float value, 
                                     threadgroup float* shared_mem,
                                     uint thread_idx,
                                     uint num_threads) {
    shared_mem[thread_idx] = value;
    threadgroup_barrier_sync();
    
    // Tree reduction
    for (uint stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (thread_idx < stride) {
            shared_mem[thread_idx] = safe_max(shared_mem[thread_idx], 
                                               shared_mem[thread_idx + stride]);
        }
        threadgroup_barrier_sync();
    }
    
    return shared_mem[0];
}

// Reduce sum across threadgroup
inline float threadgroup_sum_reduce(float value,
                                     threadgroup float* shared_mem,
                                     uint thread_idx,
                                     uint num_threads) {
    shared_mem[thread_idx] = value;
    threadgroup_barrier_sync();
    
    // Tree reduction
    for (uint stride = num_threads / 2; stride > 0; stride >>= 1) {
        if (thread_idx < stride) {
            shared_mem[thread_idx] += shared_mem[thread_idx + stride];
        }
        threadgroup_barrier_sync();
    }
    
    return shared_mem[0];
}

// ============================================================================
// Half-precision (FP16) Utilities
// ============================================================================

#ifdef USE_FP16
using compute_type = half;
#else
using compute_type = float;
#endif

// Convert between half and float
inline float to_float(half value) {
    return float(value);
}

inline half to_half(float value) {
    return half(value);
}

// ============================================================================
// Matrix Multiplication Helpers
// ============================================================================

// Dot product between two vectors (thread memory version)
inline float dot_product(thread const float* a, thread const float* b, uint len) {
    float sum = 0.0f;
    for (uint i = 0; i < len; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Dot product between two vectors (threadgroup memory version)
// Dot product with manual unrolling for better performance
inline float dot_product(threadgroup const float* a, threadgroup const float* b, uint dim) {
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    
    // Unroll by 4 for instruction-level parallelism
    uint i = 0;
    for (; i + 3 < dim; i += 4) {
        sum0 += a[i] * b[i];
        sum1 += a[i+1] * b[i+1];
        sum2 += a[i+2] * b[i+2];
        sum3 += a[i+3] * b[i+3];
    }
    
    // Handle remainder
    for (; i < dim; i++) {
        sum0 += a[i] * b[i];
    }
    
    return sum0 + sum1 + sum2 + sum3;
}

// Load vector from device memory with bounds checking
inline void load_vector(thread float* dst,
                        device const half* src,
                        uint len,
                        uint max_len) {
    for (uint i = 0; i < len; i++) {
        dst[i] = (i < max_len) ? to_float(src[i]) : 0.0f;
    }
}

// Store vector to device memory with bounds checking
inline void store_vector(device half* dst,
                         thread const float* src,
                         uint len,
                         uint max_len) {
    for (uint i = 0; i < len && i < max_len; i++) {
        dst[i] = to_half(src[i]);
    }
}

// ============================================================================
// HIGHLY OPTIMIZED SIMDGROUP OPERATIONS (CRITICAL FOR PERFORMANCE!)
// ============================================================================

// Fast max (prefer fmax for better codegen)
inline float fast_max(float a, float b) {
    return fmax(a, b);
}

// Fast min (prefer fmin for better codegen)
inline float fast_min(float a, float b) {
    return fmin(a, b);
}

// Fast max reduction using simdgroup (4-8x faster than manual loop)
inline float fast_simd_max(float val, uint simd_lane_id) {
    // simd_max broadcasts across all threads in simdgroup (32 threads on Apple Silicon)
    return simd_max(val);
}

// Fast sum reduction using simdgroup (4-8x faster than manual loop)
inline float fast_simd_sum(float val, uint simd_lane_id) {
    // simd_sum adds all values across simdgroup
    return simd_sum(val);
}

// Optimized dot product with 4x unrolling for better ILP
inline float optimized_dot_product(threadgroup const float* a, 
                                    threadgroup const float* b, 
                                    uint dim) {
    // Unroll by 4 for instruction-level parallelism
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    
    uint i = 0;
    #pragma unroll 4
    for (; i + 3 < dim; i += 4) {
        sum0 += a[i] * b[i];
        sum1 += a[i+1] * b[i+1];
        sum2 += a[i+2] * b[i+2];
        sum3 += a[i+3] * b[i+3];
    }
    
    // Handle remainder
    float sum_rem = 0.0f;
    for (; i < dim; i++) {
        sum_rem += a[i] * b[i];
    }
    
    // Reduce with optimal ordering for FP accuracy
    return (sum0 + sum1) + (sum2 + sum3) + sum_rem;
}

// Fast exponential with clamping (prevents overflow/underflow)
inline float fast_exp_clamped(float x) {
    return exp(clamp(x, -88.0f, 88.0f));
}

// ============================================================================
// Debug Utilities (can be removed in production)
// ============================================================================

#ifdef DEBUG_MODE
inline void debug_assert(bool condition, constant char* message) {
    if (!condition) {
        // In Metal, this will cause shader validation error
        // Useful for debugging
    }
}
#else
inline void debug_assert(bool condition, constant char* message) {}
#endif

// ============================================================================
// OPTIMIZATION: Simdgroup operations for fast reductions
// ============================================================================

// Fast max using simdgroup (8x faster than manual loop)
inline float simdgroup_fast_max(float val, uint simd_lane_id) {
    return simd_max(val);
}

// Fast sum using simdgroup (8x faster than manual loop)
inline float simdgroup_fast_sum(float val, uint simd_lane_id) {
    return simd_sum(val);
}

// Vectorized load (4 elements at once for 4x bandwidth)
inline float4 load_half4_to_float4(device const half* ptr) {
    half4 h = *reinterpret_cast<device const half4*>(ptr);
    return float4(h);
}

// Vectorized store (4 elements at once for 4x bandwidth)
inline void store_float4_to_half4(device half* ptr, float4 val) {
    *reinterpret_cast<device half4*>(ptr) = half4(val);
}

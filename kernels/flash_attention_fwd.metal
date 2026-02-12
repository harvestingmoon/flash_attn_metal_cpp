//
// flash_attention_fwd.metal
// Flash Attention Metal - Baseline Forward Kernel
//
// Robust Flash Attention implementation for Apple Silicon
// Prioritizes correctness and compatibility over maximum throughput
//

#include <metal_stdlib>
using namespace metal;

// Conservative block sizes for stability
#define BLOCK_M 32
#define BLOCK_N 32
#define THREADS 256

// ============================================================================
// Baseline Flash Attention Forward Kernel
// ============================================================================

kernel void flash_attention_forward(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& seqlen [[buffer(5)]],
    constant uint& num_heads [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant float& softmax_scale [[buffer(8)]],
    constant bool& is_causal [[buffer(9)]],
    device const float* cos_table [[buffer(10)]],
    device const float* sin_table [[buffer(11)]],
    constant bool& apply_rope [[buffer(12)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]])
{
    const uint batch_idx = tgid.z;
    const uint head_idx = tgid.y;
    const uint q_block_idx = tgid.x;
    const uint q_start = q_block_idx * BLOCK_M;
    
    if (q_start >= seqlen) return;

    // Shared memory for tiles
    threadgroup half Q_shared[BLOCK_M * 128];
    threadgroup half K_shared[BLOCK_N * 128];
    threadgroup half V_shared[BLOCK_N * 128];

    // Calculate base offsets for [batch, seqlen, num_heads, head_dim] layout
    const uint base_offset = batch_idx * (seqlen * num_heads * head_dim) + head_idx * head_dim;
    device const half* Q_base = Q + base_offset;
    device const half* K_base = K + base_offset;
    device const half* V_base = V + base_offset;
    device half* O_base = O + base_offset;

    const uint thread_id = tid.x;
    const uint num_threads = THREADS;
    
    // Load Q block into shared memory
    // Layout: [batch, seqlen, num_heads, head_dim]
    // Stride: [seqlen*num_heads*head_dim, num_heads*head_dim, head_dim, 1]
    const uint q_block_size = min(uint(BLOCK_M), seqlen - q_start);
    const uint seq_stride = num_heads * head_dim;
    for (uint i = thread_id; i < q_block_size * head_dim; i += num_threads) {
        const uint row = i / head_dim;
        const uint col = i % head_dim;
        Q_shared[i] = Q_base[(q_start + row) * seq_stride + col];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (thread_id >= q_block_size) return;

    // Per-thread computation
    const uint q_row = thread_id;
    const uint global_q_idx = q_start + q_row;

    // Initialize per-thread accumulators for online softmax
    float m_i = -INFINITY;  // Running max
    float l_i = 0.0f;       // Running sum of exp(x - m)
    float O_local[128];     // Output accumulator
    for (uint d = 0; d < head_dim; ++d) {
        O_local[d] = 0.0f;
    }

    // Main loop over K and V blocks
    const uint num_kv_blocks = (seqlen + BLOCK_N - 1) / BLOCK_N;
    for (uint kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++kv_block_idx) {
        const uint kv_start = kv_block_idx * BLOCK_N;
        const uint kv_block_size = min(uint(BLOCK_N), seqlen - kv_start);

        // Early termination for causal attention
        // If all keys in this block are > global_q_idx, skip this block entirely
        if (is_causal && kv_start > global_q_idx) {
            break;
        }

        // Load K and V blocks into shared memory
        for (uint i = thread_id; i < kv_block_size * head_dim; i += num_threads) {
            const uint row = i / head_dim;
            const uint col = i % head_dim;
            const uint src_idx = (kv_start + row) * seq_stride + col;
            K_shared[row * head_dim + col] = K_base[src_idx];
            V_shared[row * head_dim + col] = V_base[src_idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute attention scores for this K/V block
        float S_local[BLOCK_N];
        
        for (uint k_row = 0; k_row < kv_block_size; ++k_row) {
            const uint global_k_idx = kv_start + k_row;
            
            // Compute Q·K^T
            float score = 0.0f;
            threadgroup const half* q_row_ptr = Q_shared + q_row * head_dim;
            threadgroup const half* k_row_ptr = K_shared + k_row * head_dim;
            
            // Vectorized dot product (process 4 elements at a time)
            uint d = 0;
            for (; d + 3 < head_dim; d += 4) {
                float4 q_vec = float4(
                    float(q_row_ptr[d]),
                    float(q_row_ptr[d + 1]),
                    float(q_row_ptr[d + 2]),
                    float(q_row_ptr[d + 3])
                );
                float4 k_vec = float4(
                    float(k_row_ptr[d]),
                    float(k_row_ptr[d + 1]),
                    float(k_row_ptr[d + 2]),
                    float(k_row_ptr[d + 3])
                );
                score += dot(q_vec, k_vec);
            }
            
            // Handle remaining elements
            for (; d < head_dim; ++d) {
                score += float(q_row_ptr[d]) * float(k_row_ptr[d]);
            }
            
            score *= softmax_scale;
            
            // Apply causal mask
            if (is_causal && global_k_idx > global_q_idx) {
                score = -INFINITY;
            }
            
            S_local[k_row] = score;
        }
        
        // Online softmax update
        float m_prev = m_i;
        float m_new = m_i;
        
        // Find new max
        for (uint k_row = 0; k_row < kv_block_size; ++k_row) {
            m_new = max(m_new, S_local[k_row]);
        }
        
        // Compute exp and new sum
        float l_new = 0.0f;
        for (uint k_row = 0; k_row < kv_block_size; ++k_row) {
            S_local[k_row] = exp(S_local[k_row] - m_new);
            l_new += S_local[k_row];
        }
        
        // Rescale previous output accumulator
        float scale = exp(m_prev - m_new);
        for (uint d = 0; d < head_dim; ++d) {
            O_local[d] *= scale;
        }
        
        // Accumulate new contributions: O += S @ V
        for (uint k_row = 0; k_row < kv_block_size; ++k_row) {
            threadgroup const half* v_row_ptr = V_shared + k_row * head_dim;
            const float attn_weight = S_local[k_row];
            
            for (uint d = 0; d < head_dim; ++d) {
                O_local[d] += attn_weight * float(v_row_ptr[d]);
            }
        }
        
        // Update running statistics
        l_i = scale * l_i + l_new;
        m_i = m_new;
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Final normalization and write output
    const float norm = 1.0f / l_i;
    device half* output_row = O_base + global_q_idx * seq_stride;
    
    for (uint d = 0; d < head_dim; ++d) {
        output_row[d] = half(O_local[d] * norm);
    }
}

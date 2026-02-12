#include <metal_stdlib>
using namespace metal;

#define OPT_BLOCK_M 32
#define OPT_BLOCK_N 32
#define OPT_THREADS 256

inline void apply_rope_to_registers_vec4(
    thread float* data,                  // Operates on a local float array
    device const float* cos_ptr,
    device const float* sin_ptr,
    uint head_dim)
{
    const uint half_dim = head_dim / 2;
    const uint num_vec4 = half_dim / 4;

    // Vectorized processing: 4 pairs (8 elements) per iteration
    for (uint vec_idx = 0; vec_idx < num_vec4; ++vec_idx) {
        const uint base_pair = vec_idx * 4;
        
        // Load 4 pairs (8 elements)
        float4 x0_vec = float4(
            data[base_pair * 2 + 0], data[base_pair * 2 + 2],
            data[base_pair * 2 + 4], data[base_pair * 2 + 6]
        );
        float4 x1_vec = float4(
            data[base_pair * 2 + 1], data[base_pair * 2 + 3],
            data[base_pair * 2 + 5], data[base_pair * 2 + 7]
        );
        float4 cos_vec = float4(
            cos_ptr[base_pair + 0], cos_ptr[base_pair + 1],
            cos_ptr[base_pair + 2], cos_ptr[base_pair + 3]
        );
        float4 sin_vec = float4(
            sin_ptr[base_pair + 0], sin_ptr[base_pair + 1],
            sin_ptr[base_pair + 2], sin_ptr[base_pair + 3]
        );
        
        // Apply RoPE rotation to all 4 pairs simultaneously
        float4 rotated_0 = x0_vec * cos_vec - x1_vec * sin_vec;
        float4 rotated_1 = x0_vec * sin_vec + x1_vec * cos_vec;
        
        // Store back
        data[base_pair * 2 + 0] = rotated_0.x;
        data[base_pair * 2 + 1] = rotated_1.x;
        data[base_pair * 2 + 2] = rotated_0.y;
        data[base_pair * 2 + 3] = rotated_1.y;
        data[base_pair * 2 + 4] = rotated_0.z;
        data[base_pair * 2 + 5] = rotated_1.z;
        data[base_pair * 2 + 6] = rotated_0.w;
        data[base_pair * 2 + 7] = rotated_1.w;
    }
    
    // Handle remaining pairs (if head_dim/2 is not a multiple of 4)
    const uint remaining_start = num_vec4 * 4;
    for (uint pair_idx = remaining_start; pair_idx < half_dim; ++pair_idx) {
        const uint idx0 = pair_idx * 2;
        const uint idx1 = idx0 + 1;
        
        const float x0 = data[idx0];
        const float x1 = data[idx1];
        const float cos_val = cos_ptr[pair_idx];
        const float sin_val = sin_ptr[pair_idx];
        
        data[idx0] = x0 * cos_val - x1 * sin_val;
        data[idx1] = x0 * sin_val + x1 * cos_val;
    }
}

// ============================================================================
// NEW: Dot Product Helper for Register Data (float*)
// ============================================================================

inline float vectorized_dot_product_registers(
    thread const float* a,
    thread const float* b,
    uint head_dim)
{
    float sum = 0.0f;
    
    // Process 8 elements at a time with two float4 ops
    uint d = 0;
    for (; d + 7 < head_dim; d += 8) {
        float4 a0 = float4(a[d], a[d+1], a[d+2], a[d+3]);
        float4 b0 = float4(b[d], b[d+1], b[d+2], b[d+3]);
        float4 a1 = float4(a[d+4], a[d+5], a[d+6], a[d+7]);
        float4 b1 = float4(b[d+4], b[d+5], b[d+6], b[d+7]);
        
        sum += dot(a0, b0) + dot(a1, b1);
    }
    
    // Handle remaining elements
    for (; d < head_dim; ++d) {
        sum += a[d] * b[d];
    }
    
    return sum;
}


// ============================================================================
// Final Kernel
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
    const uint q_start = q_block_idx * OPT_BLOCK_M;
    if (q_start >= seqlen) return;

    // Shared memory for tiles
    threadgroup half Q_shared[OPT_BLOCK_M * 128]; // Max head_dim = 128
    threadgroup half K_shared[OPT_BLOCK_N * 128];
    threadgroup half V_shared[OPT_BLOCK_N * 128];

    // Calculate base offsets for [batch, seqlen, num_heads, head_dim] layout
    const uint base_offset = batch_idx * (seqlen * num_heads * head_dim) + head_idx * head_dim;
    device const half* Q_base = Q + base_offset;
    device const half* K_base = K + base_offset;
    device const half* V_base = V + base_offset;
    device half* O_base = O + base_offset;

    const uint thread_id = tid.x;
    const uint num_threads = OPT_THREADS;
    const uint seq_stride = num_heads * head_dim;
    
    // Load Q block into shared memory
    // Layout: [batch, seqlen, num_heads, head_dim]
    const uint q_block_size = min(uint(OPT_BLOCK_M), seqlen - q_start);
    for (uint i = thread_id; i < q_block_size * head_dim; i += num_threads) {
        Q_shared[i] = Q_base[(q_start + i / head_dim) * seq_stride + (i % head_dim)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (thread_id >= q_block_size) return;

    // --- OPTIMIZATION: Load Q to registers and apply RoPE ---
    const uint q_row = thread_id;
    const uint global_q_idx = q_start + q_row;

    // 1. Load this thread's Q-vector from shared memory into private registers (local array)
    float Q_local[128];
    for(uint d = 0; d < head_dim; ++d) {
        Q_local[d] = float(Q_shared[q_row * head_dim + d]);
    }

    // 2. Apply RoPE directly to the registers (if enabled)
    if (apply_rope) {
        const uint global_pos = global_q_idx;
        device const float* cos_row = cos_table + global_pos * (head_dim / 2);
        device const float* sin_row = sin_table + global_pos * (head_dim / 2);
        apply_rope_to_registers_vec4(Q_local, cos_row, sin_row, head_dim);
    }

    // Initialize per-thread accumulators
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float O_local[128];
    for (uint d = 0; d < head_dim; ++d) O_local[d] = 0.0f;

    // Main loop over K and V blocks
    const uint num_kv_blocks = (seqlen + OPT_BLOCK_N - 1) / OPT_BLOCK_N;
    for (uint kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++kv_block_idx) {
        const uint kv_start = kv_block_idx * OPT_BLOCK_N;
        const uint kv_block_size = min(uint(OPT_BLOCK_N), seqlen - kv_start);

        // Load K and V blocks into shared memory
        for (uint i = thread_id; i < kv_block_size * head_dim; i += num_threads) {
            const uint row = i / head_dim;
            const uint col = i % head_dim;
            const uint src_idx = (kv_start + row) * seq_stride + col;
            K_shared[row * head_dim + col] = K_base[src_idx];
            V_shared[row * head_dim + col] = V_base[src_idx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // REMOVED: In-place RoPE on K_shared and its barrier are no longer needed here.

        float S_local[OPT_BLOCK_N];

        // Compute scores for this Q row against the K block
        for (uint k_row = 0; k_row < kv_block_size; ++k_row) {
            // --- OPTIMIZATION: Load K to registers and apply RoPE ---
            // 1. Load K-vector for this dot product into private registers
            float K_local[128];
            threadgroup const half* k_row_ptr_shared = K_shared + k_row * head_dim;
            for(uint d = 0; d < head_dim; ++d) {
                K_local[d] = float(k_row_ptr_shared[d]);
            }

            // 2. Apply RoPE directly to K-vector in registers
            const uint global_k_idx = kv_start + k_row;
            if (apply_rope) {
                device const float* cos_row = cos_table + global_k_idx * (head_dim / 2);
                device const float* sin_row = sin_table + global_k_idx * (head_dim / 2);
                apply_rope_to_registers_vec4(K_local, cos_row, sin_row, head_dim);
            }
            
            // 3. Compute dot product using register data
            float score = vectorized_dot_product_registers(Q_local, K_local, head_dim);
            score *= softmax_scale;

            if (is_causal && global_k_idx > global_q_idx) {
                score = -INFINITY;
            }
            S_local[k_row] = score;
        }
        
        // --- Online Softmax and Value Accumulation (no changes needed here) ---
        float m_prev = m_i;
        float m_new = m_i;
        for (uint k_row = 0; k_row < kv_block_size; ++k_row) m_new = max(m_new, S_local[k_row]);
        
        float l_new = 0.0f;
        for (uint k_row = 0; k_row < kv_block_size; ++k_row) {
            S_local[k_row] = exp(S_local[k_row] - m_new);
            l_new += S_local[k_row];
        }
        
        float scale = exp(m_prev - m_new);
        for (uint d = 0; d < head_dim; ++d) O_local[d] *= scale;
        
        for (uint k_row = 0; k_row < kv_block_size; ++k_row) {
            threadgroup const half* v_row_ptr = V_shared + k_row * head_dim;
            const float attn_weight = S_local[k_row];
            for (uint d = 0; d < head_dim; ++d) {
                O_local[d] += attn_weight * float(v_row_ptr[d]);
            }
        }
        
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
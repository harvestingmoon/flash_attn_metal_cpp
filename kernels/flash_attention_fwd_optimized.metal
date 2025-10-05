
#include <metal_stdlib>
using namespace metal;

#define OPT_BLOCK_M 32   
#define OPT_BLOCK_N 32   
#define OPT_THREADS 256  

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
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    
    const uint batch_idx = tgid.z;
    const uint head_idx = tgid.y;
    const uint q_block_idx = tgid.x;
    const uint q_start = q_block_idx * OPT_BLOCK_M;
    if (q_start >= seqlen) return;
    
    threadgroup half Q_shared[OPT_BLOCK_M * 128];  // Max head_dim = 128
    threadgroup half K_shared[OPT_BLOCK_N * 128];
    threadgroup half V_shared[OPT_BLOCK_N * 128];
    
    const uint base_offset = (batch_idx * num_heads + head_idx) * seqlen * head_dim;
    device const half* Q_base = Q + base_offset;
    device const half* K_base = K + base_offset;
    device const half* V_base = V + base_offset;
    device half* O_base = O + base_offset;
    
    // Thread configuration - each thread processes one Q row
    const uint thread_id = tid.x;
    const uint num_threads = OPT_THREADS;
    
    
    const uint q_block_size = min(uint(OPT_BLOCK_M), seqlen - q_start);
    
    for (uint i = thread_id; i < q_block_size * head_dim; i += num_threads) {
        const uint row = i / head_dim;
        const uint col = i % head_dim;
        const uint src_idx = (q_start + row) * head_dim + col;
        if (q_start + row < seqlen) {
            Q_shared[row * head_dim + col] = Q_base[src_idx];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (thread_id >= q_block_size) {
        return;
    }
    
    const uint q_row = thread_id;
    const uint global_q_idx = q_start + q_row;
    
    
    float m_i = -INFINITY;  
    float l_i = 0.0f;       
    float O_local[128];     
    
    for (uint d = 0; d < head_dim; ++d) {
        O_local[d] = 0.0f;
    }
    
    // Get pointer to Q row in shared memory
    threadgroup const half* q_row_ptr = Q_shared + q_row * head_dim;
    
    // Process K,V in blocks 
    const uint num_kv_blocks = (seqlen + OPT_BLOCK_N - 1) / OPT_BLOCK_N;
    for (uint kv_block_idx = 0; kv_block_idx < num_kv_blocks; ++kv_block_idx) {
        const uint kv_start = kv_block_idx * OPT_BLOCK_N;
        const uint kv_block_size = min(uint(OPT_BLOCK_N), seqlen - kv_start);

        // Load K block cooperatively with coalesced access
        for (uint i = thread_id; i < kv_block_size * head_dim; i += num_threads) {
            const uint row = i / head_dim;
            const uint col = i % head_dim;
            const uint src_idx = (kv_start + row) * head_dim + col;
            if (kv_start + row < seqlen) {
                K_shared[row * head_dim + col] = K_base[src_idx];
            }
        }
        
        // Load V block cooperatively
        for (uint i = thread_id; i < kv_block_size * head_dim; i += num_threads) {
            const uint row = i / head_dim;
            const uint col = i % head_dim;
            const uint src_idx = (kv_start + row) * head_dim + col;
            if (kv_start + row < seqlen) {
                V_shared[row * head_dim + col] = V_base[src_idx];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute attention scores for this Q row against K block
        float S_local[32]; 
        
        // Compute S = Q @ K^T * scale with vectorized operations
        for (uint k_row = 0; k_row < kv_block_size; ++k_row) {
            threadgroup const half* k_row_ptr = K_shared + k_row * head_dim;
            
            // Vectorized dot product using float4
            float score = 0.0f;
            
            // Process 4 elements at a time for better performance
            uint d = 0;
            for (; d + 3 < head_dim; d += 4) {
                float4 q_vec = float4(
                    float(q_row_ptr[d]),
                    float(q_row_ptr[d+1]),
                    float(q_row_ptr[d+2]),
                    float(q_row_ptr[d+3])
                );
                float4 k_vec = float4(
                    float(k_row_ptr[d]),
                    float(k_row_ptr[d+1]),
                    float(k_row_ptr[d+2]),
                    float(k_row_ptr[d+3])
                );
                score += dot(q_vec, k_vec);
            }
            
            // Handle remaining elements
            for (; d < head_dim; ++d) {
                score += float(q_row_ptr[d]) * float(k_row_ptr[d]);
            }
            
            score *= softmax_scale;
            
            const uint global_k_idx = kv_start + k_row;
            if (is_causal && global_k_idx > global_q_idx) {
                score = -INFINITY;
            }
            
            S_local[k_row] = score;
        }
        
        // Online softmax update
        float m_prev = m_i;
        float m_new = m_i;
        
        // Find max in current block
        for (uint k_row = 0; k_row < kv_block_size; ++k_row) {
            m_new = max(m_new, S_local[k_row]);
        }
        
        // Compute exp and sum
        float l_new = 0.0f;
        for (uint k_row = 0; k_row < kv_block_size; ++k_row) {
            S_local[k_row] = exp(S_local[k_row] - m_new);
            l_new += S_local[k_row];
        }
        
        // Rescale previous output
        float scale = exp(m_prev - m_new);
        for (uint d = 0; d < head_dim; ++d) {
            O_local[d] *= scale;
        }
        
        // Accumulate new contribution: O += S @ V using vectorized operations
        for (uint k_row = 0; k_row < kv_block_size; ++k_row) {
            threadgroup const half* v_row_ptr = V_shared + k_row * head_dim;
            const float attn_weight = S_local[k_row];
            
            uint d = 0;
            for (; d + 3 < head_dim; d += 4) {
                float4 v_vec = float4(
                    float(v_row_ptr[d]),
                    float(v_row_ptr[d+1]),
                    float(v_row_ptr[d+2]),
                    float(v_row_ptr[d+3])
                );
                float4 o_vec = float4(
                    O_local[d],
                    O_local[d+1],
                    O_local[d+2],
                    O_local[d+3]
                );
                o_vec += attn_weight * v_vec;
                O_local[d] = o_vec.x;
                O_local[d+1] = o_vec.y;
                O_local[d+2] = o_vec.z;
                O_local[d+3] = o_vec.w;
            }
            
            for (; d < head_dim; ++d) {
                O_local[d] += attn_weight * float(v_row_ptr[d]);
            }
        }
        
        l_i = scale * l_i + l_new;
        m_i = m_new;
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Final normalization and write output
    const float norm = 1.0f / l_i;
    device half* output_row = O_base + global_q_idx * head_dim;
    
    uint d = 0;
    for (; d + 3 < head_dim; d += 4) {
        output_row[d] = half(O_local[d] * norm);
        output_row[d+1] = half(O_local[d+1] * norm);
        output_row[d+2] = half(O_local[d+2] * norm);
        output_row[d+3] = half(O_local[d+3] * norm);
    }
    
    for (; d < head_dim; ++d) {
        output_row[d] = half(O_local[d] * norm);
    }
}

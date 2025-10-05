//
// flash_attn_core.hpp
// Core Flash Attention Metal Implementation
//

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

// Forward declarations for Metal types
namespace MTL {
    class Device;
    class CommandQueue;
    class Library;
    class ComputePipelineState;
    class Buffer;
    class CommandBuffer;
    enum GPUFamily : long;
}

namespace flash_attn_metal {

// ============================================================================
// Flash Attention Parameters
// ============================================================================

struct FlashAttentionParams {
    uint32_t batch_size;
    uint32_t seqlen_q;
    uint32_t seqlen_k;
    uint32_t num_heads_q;
    uint32_t num_heads_k;
    uint32_t head_dim;
    float softmax_scale;
    bool is_causal;
    int window_left;
    int window_right;
    float dropout_p;
    bool has_alibi;
    bool has_rotary;
    uint32_t gqa_group_size;
};

// ============================================================================
// Metal Device Manager
// ============================================================================

class MetalDevice {
public:
    MetalDevice();
    ~MetalDevice();
    
    // Initialize device and resources
    bool initialize();
    bool is_initialized() const;
    
    // Shader management
    bool load_metallib(const std::string& path);
    bool compile_from_source(const std::string& source);
    bool create_pipeline(const std::string& kernel_name, const std::string& pipeline_id);
    
    // Accessors
    MTL::Device* device() const;
    MTL::CommandQueue* queue() const;
    MTL::ComputePipelineState* get_pipeline(const std::string& pipeline_id) const;
    
    // Device info
    std::string device_name() const;
    uint64_t max_buffer_length() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// ============================================================================
// Buffer Pool for Memory Optimization (CRITICAL for performance!)
// ============================================================================

class BufferPool {
public:
    explicit BufferPool(MetalDevice& device);
    ~BufferPool();
    
    // Get or create a buffer of specified size
    MTL::Buffer* get_buffer(size_t size, const std::string& name = "");
    
    // Return buffer to pool for reuse
    void return_buffer(MTL::Buffer* buf);
    
    // Clear all cached buffers
    void clear();
    
    // Statistics
    size_t cache_hit_count() const { return cache_hits_; }
    size_t cache_miss_count() const { return cache_misses_; }
    size_t total_allocated_bytes() const { return total_allocated_; }
    
private:
    MetalDevice& device_;
    std::unordered_map<size_t, std::vector<MTL::Buffer*>> buffer_cache_;
    size_t cache_hits_ = 0;
    size_t cache_misses_ = 0;
    size_t total_allocated_ = 0;
};

// ============================================================================
// Flash Attention Forward Pass
// ============================================================================

class FlashAttentionForward {
public:
    explicit FlashAttentionForward(MetalDevice& device);
    ~FlashAttentionForward();
    
    // Execute forward pass
    // Inputs: Q, K, V as FP16 data pointers
    // Output: O as FP16 data pointer (pre-allocated)
    // Optional: LSE (log-sum-exp) for backward pass
    bool execute(
        const void* Q_data,
        const void* K_data,
        const void* V_data,
        void* O_data,
        const FlashAttentionParams& params,
        void* LSE_data = nullptr,
        const float* alibi_slopes = nullptr,
        const float* cos_table = nullptr,
        const float* sin_table = nullptr
    );
    
    // Get buffer pool statistics
    void print_stats() const;
    
private:
    MetalDevice& device_;
    std::unique_ptr<BufferPool> buffer_pool_;
};

// ============================================================================
// Helper Functions
// ============================================================================

// Read file contents
std::string read_file(const std::string& path);

// Check if file exists
bool file_exists(const std::string& path);

// Find metallib file (searches common locations)
std::string find_metallib();

// Compute buffer size for tensors
size_t compute_tensor_size(uint32_t batch, uint32_t seq, uint32_t heads, uint32_t dim, bool is_fp16 = true);

} // namespace flash_attn_metal

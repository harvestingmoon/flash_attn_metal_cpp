//
// flash_attn_core.cpp
// Core Flash Attention Metal Implementation
//

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION  
#define MTL_PRIVATE_IMPLEMENTATION

#include "flash_attn_core.hpp"
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <map>
#include <stdexcept>
#include <cstring>

namespace flash_attn_metal {

// ============================================================================
// MetalDevice Implementation
// ============================================================================

class MetalDevice::Impl {
public:
    MTL::Device* device = nullptr;
    MTL::CommandQueue* queue = nullptr;
    MTL::Library* library = nullptr;
    std::map<std::string, MTL::ComputePipelineState*> pipelines;
    
    ~Impl() {
        for (auto& pair : pipelines) {
            if (pair.second) pair.second->release();
        }
        if (library) library->release();
        if (queue) queue->release();
        if (device) device->release();
    }
};

MetalDevice::MetalDevice() : pImpl(std::make_unique<Impl>()) {}

MetalDevice::~MetalDevice() = default;

bool MetalDevice::initialize() {
    pImpl->device = MTL::CreateSystemDefaultDevice();
    if (!pImpl->device) {
        std::cerr << "Failed to create Metal device\n";
        return false;
    }
    
    pImpl->queue = pImpl->device->newCommandQueue();
    if (!pImpl->queue) {
        std::cerr << "Failed to create command queue\n";
        return false;
    }
    
    return true;
}

bool MetalDevice::is_initialized() const {
    return pImpl->device != nullptr && pImpl->queue != nullptr;
}

MTL::Device* MetalDevice::device() const {
    return pImpl->device;
}

MTL::CommandQueue* MetalDevice::queue() const {
    return pImpl->queue;
}

MTL::ComputePipelineState* MetalDevice::get_pipeline(const std::string& pipeline_id) const {
    auto it = pImpl->pipelines.find(pipeline_id);
    return (it != pImpl->pipelines.end()) ? it->second : nullptr;
}

std::string MetalDevice::device_name() const {
    if (!pImpl->device) return "No device";
    return pImpl->device->name()->utf8String();
}

uint64_t MetalDevice::max_buffer_length() const {
    if (!pImpl->device) return 0;
    return pImpl->device->maxBufferLength();
}

bool MetalDevice::load_metallib(const std::string& path) {
    if (!pImpl->device) {
        std::cerr << "Device not initialized\n";
        return false;
    }
    
    NS::Error* error = nullptr;
    NS::String* filePath = NS::String::string(path.c_str(), NS::UTF8StringEncoding);
    pImpl->library = pImpl->device->newLibrary(filePath, &error);
    
    if (!pImpl->library || error) {
        if (error) {
            std::cerr << "Failed to load metallib: " 
                     << error->localizedDescription()->utf8String() << "\n";
        }
        return false;
    }
    
    return true;
}

bool MetalDevice::compile_from_source(const std::string& source) {
    if (!pImpl->device) {
        std::cerr << "Device not initialized\n";
        return false;
    }
    
    NS::Error* error = nullptr;
    NS::String* sourceStr = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
    
    MTL::CompileOptions* options = MTL::CompileOptions::alloc()->init();
    options->setFastMathEnabled(true);
    
    pImpl->library = pImpl->device->newLibrary(sourceStr, options, &error);
    options->release();
    
    if (!pImpl->library || error) {
        if (error) {
            std::cerr << "Failed to compile Metal source: " 
                     << error->localizedDescription()->utf8String() << "\n";
        }
        return false;
    }
    
    return true;
}

bool MetalDevice::create_pipeline(const std::string& kernel_name, const std::string& pipeline_id) {
    if (!pImpl->library) {
        std::cerr << "Library not loaded\n";
        return false;
    }
    
    NS::Error* error = nullptr;
    NS::String* funcName = NS::String::string(kernel_name.c_str(), NS::UTF8StringEncoding);
    
    // Create function constants to match kernel expectations
    MTL::FunctionConstantValues* constantValues = MTL::FunctionConstantValues::alloc()->init();
    
    // Set block sizes (must match DEFAULT_BLOCK_M/N/HEADDIM in kernel)
    uint32_t block_m = 32;
    uint32_t block_n = 32;
    uint32_t block_headdim = 64;
    uint32_t num_warps = 4;
    
    constantValues->setConstantValue(&block_m, MTL::DataTypeUInt, NS::UInteger(0));
    constantValues->setConstantValue(&block_n, MTL::DataTypeUInt, NS::UInteger(1));
    constantValues->setConstantValue(&block_headdim, MTL::DataTypeUInt, NS::UInteger(2));
    constantValues->setConstantValue(&num_warps, MTL::DataTypeUInt, NS::UInteger(3));
    
    MTL::Function* function = pImpl->library->newFunction(funcName, constantValues, &error);
    constantValues->release();
    
    if (!function || error) {
        if (error) {
            std::cerr << "Failed to create function with constants: " 
                     << error->localizedDescription()->utf8String() << "\n";
        }
        return false;
    }
    
    MTL::ComputePipelineState* pipeline = pImpl->device->newComputePipelineState(function, &error);
    function->release();
    
    if (!pipeline || error) {
        if (error) {
            std::cerr << "Failed to create pipeline: " 
                     << error->localizedDescription()->utf8String() << "\n";
        }
        return false;
    }
    
    pImpl->pipelines[pipeline_id] = pipeline;
    return true;
}

// ============================================================================
// Buffer Pool Implementation
// ============================================================================

BufferPool::BufferPool(MetalDevice& device) : device_(device) {}

BufferPool::~BufferPool() {
    clear();
}

MTL::Buffer* BufferPool::get_buffer(size_t size, const std::string& name) {
    // Round up to nearest power of 2 for better cache hit rate
    size_t rounded_size = 1;
    while (rounded_size < size) {
        rounded_size <<= 1;
    }
    
    // Try to get cached buffer
    auto it = buffer_cache_.find(rounded_size);
    if (it != buffer_cache_.end() && !it->second.empty()) {
        MTL::Buffer* buf = it->second.back();
        it->second.pop_back();
        cache_hits_++;
        return buf;
    }
    
    // Allocate new buffer
    MTL::Buffer* buf = device_.device()->newBuffer(rounded_size,
                                                     MTL::ResourceStorageModeShared);
    if (!buf) {
        std::cerr << "[BufferPool] Failed to allocate " << rounded_size << " bytes\n";
        return nullptr;
    }
    
    cache_misses_++;
    total_allocated_ += rounded_size;
    
    return buf;
}

void BufferPool::return_buffer(MTL::Buffer* buf) {
    if (!buf) return;
    
    size_t size = buf->length();
    buffer_cache_[size].push_back(buf);
}

void BufferPool::clear() {
    for (auto& pair : buffer_cache_) {
        for (MTL::Buffer* buf : pair.second) {
            buf->release();
        }
    }
    buffer_cache_.clear();
    cache_hits_ = 0;
    cache_misses_ = 0;
    total_allocated_ = 0;
}

// ============================================================================
// Flash Attention Forward Pass
// ============================================================================

FlashAttentionForward::FlashAttentionForward(MetalDevice& device) 
    : device_(device),
      buffer_pool_(std::make_unique<BufferPool>(device))
{
}

FlashAttentionForward::~FlashAttentionForward() = default;

void FlashAttentionForward::print_stats() const {
    size_t total_calls = buffer_pool_->cache_hit_count() + buffer_pool_->cache_miss_count();
    if (total_calls == 0) {
        std::cout << "[BufferPool] No calls yet\n";
        return;
    }
    
    float hit_rate = 100.0f * buffer_pool_->cache_hit_count() / total_calls;
    std::cout << "[BufferPool Stats]\n";
    std::cout << "  Cache hits: " << buffer_pool_->cache_hit_count() << "\n";
    std::cout << "  Cache misses: " << buffer_pool_->cache_miss_count() << "\n";
    std::cout << "  Hit rate: " << hit_rate << "%\n";
    std::cout << "  Total allocated: " << (buffer_pool_->total_allocated_bytes() / 1024.0 / 1024.0) 
              << " MB\n";
}

bool FlashAttentionForward::execute(
    const void* Q_data,
    const void* K_data,
    const void* V_data,
    void* O_data,
    const FlashAttentionParams& params,
    void* LSE_data,
    const float* alibi_slopes,
    const float* cos_table,
    const float* sin_table
) {
    if (!device_.is_initialized()) {
        std::cerr << "Device not initialized\n";
        return false;
    }
    
    MTL::ComputePipelineState* pipeline = device_.get_pipeline("forward");
    if (!pipeline) {
        std::cerr << "Forward pipeline not found\n";
        return false;
    }
    
    // Calculate buffer sizes (FP16 = 2 bytes)
    size_t q_size = compute_tensor_size(params.batch_size, params.seqlen_q, 
                                         params.num_heads_q, params.head_dim);
    size_t k_size = compute_tensor_size(params.batch_size, params.seqlen_k,
                                         params.num_heads_k, params.head_dim);
    size_t v_size = k_size;
    size_t o_size = q_size;
    size_t lse_size = params.batch_size * params.num_heads_q * params.seqlen_q * sizeof(float);
    
    // OPTIMIZATION: Use buffer pool instead of allocating new buffers
    MTL::Buffer* Q_buf = buffer_pool_->get_buffer(q_size, "Q");
    MTL::Buffer* K_buf = buffer_pool_->get_buffer(k_size, "K");
    MTL::Buffer* V_buf = buffer_pool_->get_buffer(v_size, "V");
    MTL::Buffer* O_buf = buffer_pool_->get_buffer(o_size, "O");
    MTL::Buffer* LSE_buf = buffer_pool_->get_buffer(lse_size, "LSE");
    
    if (!Q_buf || !K_buf || !V_buf || !O_buf || !LSE_buf) {
        std::cerr << "Failed to allocate buffers from pool\n";
        return false;
    }
    
    // Copy input data to GPU buffers
    std::memcpy(Q_buf->contents(), Q_data, q_size);
    std::memcpy(K_buf->contents(), K_data, k_size);
    std::memcpy(V_buf->contents(), V_data, v_size);
    
    // Initialize output buffer to zero (important for buffer pool reuse)
    std::memset(O_buf->contents(), 0, o_size);
    std::memset(LSE_buf->contents(), 0, lse_size);
    
    // Create params buffer (small, not worth pooling)
    MTL::Buffer* params_buf = device_.device()->newBuffer(&params, sizeof(FlashAttentionParams),
                                                           MTL::ResourceStorageModeShared);
    
    // Optional buffers
    MTL::Buffer* alibi_buf = nullptr;
    MTL::Buffer* cos_buf = nullptr;
    MTL::Buffer* sin_buf = nullptr;
    
    if (alibi_slopes && params.has_alibi) {
        size_t alibi_size = params.num_heads_q * sizeof(float);
        alibi_buf = device_.device()->newBuffer(alibi_slopes, alibi_size,
                                                 MTL::ResourceStorageModeShared);
    }
    
    if (cos_table && sin_table && params.has_rotary) {
        size_t rope_size = params.seqlen_q * (params.head_dim / 2) * sizeof(float);
        cos_buf = device_.device()->newBuffer(cos_table, rope_size,
                                               MTL::ResourceStorageModeShared);
        sin_buf = device_.device()->newBuffer(sin_table, rope_size,
                                               MTL::ResourceStorageModeShared);
    }
    
    // Create command buffer
    MTL::CommandBuffer* cmd_buf = device_.queue()->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = cmd_buf->computeCommandEncoder();
    
    // Set pipeline
    encoder->setComputePipelineState(pipeline);
    
    // Set buffers (matching Metal kernel signature)
    encoder->setBuffer(Q_buf, 0, 0);
    encoder->setBuffer(K_buf, 0, 1);
    encoder->setBuffer(V_buf, 0, 2);
    encoder->setBuffer(O_buf, 0, 3);
    encoder->setBuffer(LSE_buf, 0, 4);
    encoder->setBuffer(cos_buf, 0, 5);
    encoder->setBuffer(sin_buf, 0, 6);
    encoder->setBuffer(alibi_buf, 0, 7);
    encoder->setBuffer(params_buf, 0, 8);
    
    // Calculate dispatch dimensions for tiled kernel
    uint32_t block_m = 32;  // DEFAULT_BLOCK_M in kernel
    uint32_t num_q_blocks = (params.seqlen_q + block_m - 1) / block_m;
    
    // Grid: (num_q_blocks, num_heads, batch)
    MTL::Size grid_size = MTL::Size(num_q_blocks, params.num_heads_q, params.batch_size);
    
    // Simple per-thread version: 256 threads (32x8) is sufficient
    MTL::Size threadgroup_size = MTL::Size(32, 8, 1);  // 256 threads
    
    // Dispatch compute kernel
    encoder->dispatchThreadgroups(grid_size, threadgroup_size);
    encoder->endEncoding();
    
    // Execute
    cmd_buf->commit();
    cmd_buf->waitUntilCompleted();
    
    // Copy output back
    std::memcpy(O_data, O_buf->contents(), o_size);
    if (LSE_data) {
        std::memcpy(LSE_data, LSE_buf->contents(), lse_size);
    }
    
    // OPTIMIZATION: Return buffers to pool instead of releasing
    buffer_pool_->return_buffer(Q_buf);
    buffer_pool_->return_buffer(K_buf);
    buffer_pool_->return_buffer(V_buf);
    buffer_pool_->return_buffer(O_buf);
    buffer_pool_->return_buffer(LSE_buf);
    
    // Release small buffers (not worth caching)
    params_buf->release();
    if (alibi_buf) alibi_buf->release();
    if (cos_buf) cos_buf->release();
    if (sin_buf) sin_buf->release();
    
    return true;
}

// ============================================================================
// Helper Functions
// ============================================================================

std::string read_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

bool file_exists(const std::string& path) {
    std::ifstream file(path);
    return file.good();
}

std::string find_metallib() {
    std::vector<std::string> search_paths = {
        "./flash_attention.metallib",
        "../kernels/flash_attention.metallib",
        "../../kernels/flash_attention.metallib",
    };
    
    for (const auto& path : search_paths) {
        if (file_exists(path)) {
            return path;
        }
    }
    
    return "";
}

size_t compute_tensor_size(uint32_t batch, uint32_t seq, uint32_t heads, uint32_t dim, bool is_fp16) {
    size_t element_size = is_fp16 ? 2 : 4;
    return static_cast<size_t>(batch) * seq * heads * dim * element_size;
}

} // namespace flash_attn_metal

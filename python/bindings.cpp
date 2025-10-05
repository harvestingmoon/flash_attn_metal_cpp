//
// python_bindings.cpp
// Python bindings for Flash Attention Metal using pybind11
//

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "flash_attn_core.hpp"

#include <iostream>
#include <stdexcept>
#include <memory>

namespace py = pybind11;
using namespace flash_attn_metal;

static std::unique_ptr<MetalDevice> g_device = nullptr;
static std::unique_ptr<FlashAttentionForward> g_forward = nullptr;

void ensure_device_initialized() {
    if (!g_device) {
        g_device = std::make_unique<MetalDevice>();
        if (!g_device->initialize()) {
            throw std::runtime_error("Failed to initialize Metal device");
        }
    }
}

void ensure_forward_initialized() {
    ensure_device_initialized();
    if (!g_forward) {
        g_forward = std::make_unique<FlashAttentionForward>(*g_device);
    }
}

// ============================================================================
// Python API Functions
// ============================================================================

bool py_initialize() {
    try {
        ensure_device_initialized();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Initialization error: " << e.what() << "\n";
        return false;
    }
}

bool py_load_shaders(const std::string& metallib_path) {
    try {
        ensure_device_initialized();
        
        if (!g_device->load_metallib(metallib_path)) {
            return false;
        }
        
        if (!g_device->create_pipeline("flash_attention_forward", "forward")) {
            return false;
        }
        
        ensure_forward_initialized();
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Shader loading error: " << e.what() << "\n";
        return false;
    }
}

bool py_compile_shaders(const std::string& shader_source) {
    try {
        ensure_device_initialized();
        
        if (!g_device->compile_from_source(shader_source)) {
            return false;
        }
        
        if (!g_device->create_pipeline("flash_attention_forward", "forward")) {
            return false;
        }
        
        ensure_forward_initialized();
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Shader compilation error: " << e.what() << "\n";
        return false;
    }
}

py::dict py_get_device_info() {
    try {
        ensure_device_initialized();
        
        py::dict info;
        info["device_name"] = g_device->device_name();
        info["max_buffer_length"] = g_device->max_buffer_length();
        info["initialized"] = g_device->is_initialized();
        
        return info;
        
    } catch (const std::exception& e) {
        std::cerr << "Device info error: " << e.what() << "\n";
        return py::dict();
    }
}

// FIXED: Accept py::array instead of py::array_t<uint16_t> to avoid pybind11's
// automatic conversion from float16 to uint16 (which converts values instead of preserving bits)
py::array py_flash_attention_forward(
    py::array Q,  // Generic array - will be float16
    py::array K,
    py::array V,
    float softmax_scale,
    bool is_causal,
    float dropout_p,
    py::object alibi_slopes,
    py::object window_size,
    py::object cos,
    py::object sin
) {
    try {
        ensure_forward_initialized();
        
        // Get array buffer info
        auto Q_buf = Q.request();
        auto K_buf = K.request();
        auto V_buf = V.request();
        
        // Verify dtype is float16 (2 bytes per element) 
        // NOTE: The memory layout of float16 is identical to uint16, so we can
        // safely reinterpret the pointer without conversion
        if (Q_buf.itemsize != 2 || K_buf.itemsize != 2 || V_buf.itemsize != 2) {
            throw std::runtime_error("Input arrays must be 16-bit (float16)");
        }
        
        // Validate shapes
        if (Q_buf.ndim != 4 || K_buf.ndim != 4 || V_buf.ndim != 4) {
            throw std::runtime_error("Input arrays must be 4D [batch, seqlen, num_heads, head_dim]");
        }
        
        uint32_t batch = Q_buf.shape[0];
        uint32_t seqlen_q = Q_buf.shape[1];
        uint32_t num_heads_q = Q_buf.shape[2];
        uint32_t head_dim = Q_buf.shape[3];
        
        uint32_t seqlen_k = K_buf.shape[1];
        uint32_t num_heads_k = K_buf.shape[2];
        
        // Validate K and V match
        if (K_buf.shape[0] != batch || V_buf.shape[0] != batch ||
            K_buf.shape[1] != seqlen_k || V_buf.shape[1] != seqlen_k ||
            K_buf.shape[2] != num_heads_k || V_buf.shape[2] != num_heads_k ||
            K_buf.shape[3] != head_dim || V_buf.shape[3] != head_dim) {
            throw std::runtime_error("K and V shapes must match");
        }
        
        // Setup parameters
        FlashAttentionParams params;
        params.batch_size = batch;
        params.seqlen_q = seqlen_q;
        params.seqlen_k = seqlen_k;
        params.num_heads_q = num_heads_q;
        params.num_heads_k = num_heads_k;
        params.head_dim = head_dim;
        params.softmax_scale = softmax_scale;
        params.is_causal = is_causal;
        params.dropout_p = dropout_p;
        params.gqa_group_size = num_heads_q / num_heads_k;
        
        // Window size
        params.window_left = -1;
        params.window_right = -1;
        if (!window_size.is_none()) {
            auto ws = window_size.cast<py::tuple>();
            if (py::len(ws) == 2) {
                params.window_left = ws[0].cast<int>();
                params.window_right = ws[1].cast<int>();
            }
        }
        
        // ALiBi
        params.has_alibi = !alibi_slopes.is_none();
        
        // RoPE
        params.has_rotary = !cos.is_none() && !sin.is_none();
        
        // Allocate output with same dtype as Q (float16)
        auto O = py::array(py::dtype("float16"), {batch, seqlen_q, num_heads_q, head_dim});
        auto O_buf = O.request();
        
        // Execute - cast pointers to uint16_t* (safe because float16 has same memory layout)
        float* alibi_ptr = nullptr;
        if (params.has_alibi) {
            auto alibi_array = alibi_slopes.cast<py::array_t<float>>();
            alibi_ptr = static_cast<float*>(alibi_array.request().ptr);
        }
        
        // RoPE tables (float32)
        float* cos_ptr = nullptr;
        float* sin_ptr = nullptr;
        if (params.has_rotary) {
            auto cos_array = cos.cast<py::array_t<float>>();
            auto sin_array = sin.cast<py::array_t<float>>();
            cos_ptr = static_cast<float*>(cos_array.request().ptr);
            sin_ptr = static_cast<float*>(sin_array.request().ptr);
        }
        
        bool success = g_forward->execute(
            static_cast<const uint16_t*>(Q_buf.ptr),  // Reinterpret float16* as uint16_t*
            static_cast<const uint16_t*>(K_buf.ptr),
            static_cast<const uint16_t*>(V_buf.ptr),
            static_cast<uint16_t*>(O_buf.ptr),
            params,
            nullptr,  // LSE
            alibi_ptr,
            cos_ptr,  // cos_table
            sin_ptr   // sin_table
        );
        
        if (!success) {
            throw std::runtime_error("Forward pass error: Forward pass failed");
        }
        
        return O;
        
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Forward pass error: ") + e.what());
    }
}

// ============================================================================
// Pybind11 Module Definition
// ============================================================================

PYBIND11_MODULE(_flash_attn_metal, m) {
    m.doc() = "Flash Attention Metal backend for Apple Silicon";
    
    // Functions
    m.def("initialize", &py_initialize,
          "Initialize Metal device");
    
    m.def("load_shaders", &py_load_shaders,
          py::arg("metallib_path"),
          "Load compiled Metal shaders from metallib file");
    
    m.def("compile_shaders", &py_compile_shaders,
          py::arg("shader_source"),
          "Compile Metal shaders from source code at runtime");
    
    m.def("get_device_info", &py_get_device_info,
          "Get Metal device information");
    
    m.def("forward", &py_flash_attention_forward,
          py::arg("Q"),
          py::arg("K"),
          py::arg("V"),
          py::arg("softmax_scale") = 0.0f,
          py::arg("is_causal") = false,
          py::arg("dropout_p") = 0.0f,
          py::arg("alibi_slopes") = py::none(),
          py::arg("window_size") = py::none(),
          py::arg("cos") = py::none(),
          py::arg("sin") = py::none(),
          "Flash Attention forward pass");
}

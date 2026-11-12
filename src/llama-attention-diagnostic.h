#pragma once

#include <cstdio>
#include <cstdlib>

// Diagnostic output is deliberately separate from phase timing and is not a benchmark trace.
inline bool llama_attn_diagnostic_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_ATTN_DIAGNOSTIC");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

inline bool llama_cpu_attn_timing_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_CPU_ATTN_TIMING");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

inline bool llama_kv_page_coverage_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_KV_PAGE_COVERAGE_DIAGNOSTIC");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}

inline bool llama_attn_linking_enabled() {
    return llama_attn_diagnostic_enabled() || llama_cpu_attn_timing_enabled() || llama_kv_page_coverage_enabled();
}

inline bool llama_kvb_diagnostic_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_KVB_DIAGNOSTIC");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}


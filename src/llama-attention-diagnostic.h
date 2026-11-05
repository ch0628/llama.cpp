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

inline bool llama_kvb_diagnostic_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("LLAMA_KVB_DIAGNOSTIC");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}


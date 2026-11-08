#pragma once

#include "llama.h"

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <vector>
#include <ctime>

enum class phase_kind { START, END, CALL_START, CALL_RETURN, POINT };

struct phase_fields {
    int64_t task_id     = -1;
    int64_t slot_id     = -1;
    int64_t item_idx    = -1;
    int64_t sub_idx     = -1;
    int64_t token_count = -1;
    int64_t token_id    = -1;
    int64_t pos_start   = -1;
    int64_t pos_end     = -1;
    int64_t image_w     = -1;
    int64_t image_h     = -1;
    int64_t status      = -1;
    int64_t call_id     = -1;
    int64_t ubatch_idx  = -1;
    int64_t decode_step_idx = -1;
    int64_t seq_id      = -1;
    int64_t stream_id   = -1;
    int64_t kv_used_cells = -1;
    int64_t kv_used_max_p1 = -1;
    int64_t attention_n_kv = -1;
    int64_t valid_count = -1;
    int64_t masked_count = -1;
    int64_t graph_reused = -1;
    int64_t prev_n_tokens = -1;
    int64_t prev_n_kv = -1;
    int64_t prev_n_stream = -1;
    int64_t n_stream = -1;
    int64_t query_idx = -1;
    int64_t key_start = -1;
    int64_t key_end = -1;
    int64_t request_id = -1;
    int64_t q_len = -1;
    int64_t duration_ns = -1;
    int64_t kv_capacity_cells = -1;
    int64_t kv_k_logical_bytes = -1;
    int64_t kv_v_logical_bytes = -1;
    int64_t kv_backend_bytes = -1;
    const char * input_kind = "";
    const char * note   = "";
};

inline uint64_t phase_now_ns() {
#ifdef CLOCK_MONOTONIC_RAW
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return uint64_t(ts.tv_sec) * 1000000000ULL + uint64_t(ts.tv_nsec);
#else
    return 0;
#endif
}

struct phase_record {
    uint64_t run_id;
    uint64_t seq;
    uint64_t ts_ns;
    const char * component;
    const char * name;
    phase_kind kind;
    uint64_t span_id;
    phase_fields fields;
};

struct phase_log {
    std::mutex mutex;
    std::vector<phase_record> records;
    uint64_t run_id = 0;
    uint64_t next_seq = 1;
    uint64_t next_span = 1;
};

LLAMA_API phase_log & phase_logger();

struct phase_call_context {
    uint64_t call_id = 0;
    uint64_t link_call_id = 0;
    int64_t decode_step_idx = -1;
    int64_t ubatch_idx = -1;
    int64_t request_id = -1;
    int64_t task_id = -1;
    int64_t slot_id = -1;
    const char * stage = "UNKNOWN";
    const char * input_kind = "UNKNOWN";
};

LLAMA_API phase_call_context & phase_active_call();
LLAMA_API uint64_t & phase_active_graph_generation();

struct phase_request_context {
    int64_t request_id = -1;
    int64_t task_id = -1;
    int64_t slot_id = -1;
    int64_t item_idx = -1;
};

LLAMA_API phase_request_context & phase_active_request();

LLAMA_API void phase_kv_snapshot(const llama_context * ctx, const char * name, phase_fields fields = {});
LLAMA_API void kv_residency_snapshot(
        const llama_context * ctx,
        const char * event,
        const char * stage,
        phase_fields fields = {});

inline void phase_set_call(uint64_t call_id, int64_t decode_step_idx = -1) {
    auto & call = phase_active_call();
    call = {};
    call.call_id = call_id;
    call.decode_step_idx = decode_step_idx;
    call.ubatch_idx = -1;
}

inline void phase_set_link(const phase_fields & fields, const char * stage) {
    auto & call = phase_active_call();
    call.link_call_id = fields.call_id > 0 ? fields.call_id : 0;
    call.request_id = fields.request_id;
    call.task_id = fields.task_id;
    call.slot_id = fields.slot_id;
    call.stage = stage;
    call.input_kind = fields.input_kind;
}

inline const char * phase_events_path() {
#ifdef CLOCK_MONOTONIC_RAW
    static const char * path = std::getenv("LLAMA_PHASE_EVENTS_PATH");
    return path && path[0] ? path : nullptr;
#else
    return nullptr;
#endif
}

inline bool phase_diagnostic() {
    static const char * value = std::getenv("LLAMA_PHASE_DIAGNOSTIC");
    return phase_events_path() && value && value[0] == '1' && value[1] == '\0';
}

inline bool phase_sync_validate() {
    static const char * value = std::getenv("LLAMA_PHASE_SYNC_VALIDATE");
    return phase_events_path() && value && value[0] == '1' && value[1] == '\0';
}

inline uint64_t phase_event(const char * component, const char * name, phase_kind kind, uint64_t span_id = 0, phase_fields fields = {}) {
    if (!phase_events_path()) {
        return 0;
    }
    const uint64_t ts_ns = phase_now_ns();
    const auto & call = phase_active_call();
    if (fields.call_id == -1 && call.call_id) {
        fields.call_id = call.call_id;
    }
    if (fields.ubatch_idx == -1) {
        fields.ubatch_idx = call.ubatch_idx;
    }
    if (fields.decode_step_idx == -1) {
        fields.decode_step_idx = call.decode_step_idx;
    }
    auto & log = phase_logger();
    std::lock_guard<std::mutex> lock(log.mutex);
    if (!log.run_id) {
        log.run_id = ts_ns;
    }
    if (kind == phase_kind::START || kind == phase_kind::CALL_START) {
        span_id = log.next_span++;
        if (kind == phase_kind::CALL_START && fields.call_id == -1) {
            fields.call_id = span_id;
        }
    }
    log.records.push_back({ log.run_id, log.next_seq++, ts_ns, component, name, kind, span_id, fields });
    return span_id;
}

inline int64_t phase_last_prefill_call(int64_t request_id) {
    if (!phase_events_path()) {
        return -1;
    }
    auto & log = phase_logger();
    std::lock_guard<std::mutex> lock(log.mutex);
    for (auto it = log.records.rbegin(); it != log.records.rend(); ++it) {
        if (it->fields.request_id == request_id && it->kind == phase_kind::CALL_RETURN &&
            std::strcmp(it->name, "LLM_PREFILL_CALL_RETURN") == 0) {
            return it->fields.call_id;
        }
    }
    return -1;
}

inline bool phase_flush() {
    if (!phase_events_path()) {
        return true;
    }
    auto & log = phase_logger();
    std::lock_guard<std::mutex> lock(log.mutex);
    if (log.records.empty()) {
        return true;
    }
    std::ofstream out(phase_events_path(), std::ios::app);
    if (!out) {
        return false;
    }
    if (out.tellp() == 0) {
        out << "schema_version,run_id,seq,ts_ns,clock_domain,component,event_name,event_kind,span_id,task_id,slot_id,item_idx,sub_idx,token_count,token_id,pos_start,pos_end,image_w,image_h,status,call_id,ubatch_idx,decode_step_idx,seq_id,stream_id,kv_used_cells,kv_used_max_p1,attention_n_kv,valid_count,masked_count,graph_reused,prev_n_tokens,prev_n_kv,prev_n_stream,n_stream,query_idx,key_start,key_end,note,request_id,input_kind,q_len,duration_ns,kv_capacity_cells,kv_k_logical_bytes,kv_v_logical_bytes,kv_backend_bytes\n";
    }
    auto optional = [&](int64_t value) {
        if (value != -1) {
            out << value;
        }
    };
    for (const auto & event : log.records) {
        out << "2," << event.run_id << ',' << event.seq << ',' << event.ts_ns << ",pi_monotonic_raw," << event.component << ',' << event.name << ',';
        const char * kind_name = nullptr;
        switch (event.kind) {
            case phase_kind::START:       kind_name = "START";       break;
            case phase_kind::END:         kind_name = "END";         break;
            case phase_kind::CALL_START:  kind_name = "CALL_START";  break;
            case phase_kind::CALL_RETURN: kind_name = "CALL_RETURN"; break;
            case phase_kind::POINT:       kind_name = "POINT";       break;
        }
        out << kind_name << ',';
        if (event.span_id) {
            out << event.span_id;
        }
        out << ','; optional(event.fields.task_id);
        out << ','; optional(event.fields.slot_id);
        out << ','; optional(event.fields.item_idx);
        out << ','; optional(event.fields.sub_idx);
        out << ','; optional(event.fields.token_count);
        out << ','; optional(event.fields.token_id);
        out << ','; optional(event.fields.pos_start);
        out << ','; optional(event.fields.pos_end);
        out << ','; optional(event.fields.image_w);
        out << ','; optional(event.fields.image_h);
        out << ','; optional(event.fields.status);
        out << ','; optional(event.fields.call_id);
        out << ','; optional(event.fields.ubatch_idx);
        out << ','; optional(event.fields.decode_step_idx);
        out << ','; optional(event.fields.seq_id);
        out << ','; optional(event.fields.stream_id);
        out << ','; optional(event.fields.kv_used_cells);
        out << ','; optional(event.fields.kv_used_max_p1);
        out << ','; optional(event.fields.attention_n_kv);
        out << ','; optional(event.fields.valid_count);
        out << ','; optional(event.fields.masked_count);
        out << ','; optional(event.fields.graph_reused);
        out << ','; optional(event.fields.prev_n_tokens);
        out << ','; optional(event.fields.prev_n_kv);
        out << ','; optional(event.fields.prev_n_stream);
        out << ','; optional(event.fields.n_stream);
        out << ','; optional(event.fields.query_idx);
        out << ','; optional(event.fields.key_start);
        out << ','; optional(event.fields.key_end);
        out << ',' << event.fields.note;
        out << ','; optional(event.fields.request_id);
        out << ',' << event.fields.input_kind;
        out << ','; optional(event.fields.q_len);
        out << ','; optional(event.fields.duration_ns);
        out << ','; optional(event.fields.kv_capacity_cells);
        out << ','; optional(event.fields.kv_k_logical_bytes);
        out << ','; optional(event.fields.kv_v_logical_bytes);
        out << ','; optional(event.fields.kv_backend_bytes);
        out << '\n';
    }
    out.close();
    if (!out) {
        return false;
    }
    log.records.clear();
    log.run_id = 0;
    return true;
}

#include "llama-page-activity.h"
#include "llama-phase-events.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace llama_page_activity;

static void require(bool condition) {
    if (!condition) {
        std::abort();
    }
}

static void test_process_status_parser() {
    std::istringstream input(
        "Name:\tllama-server\n"
        "VmRSS:\t  123456 kB\n"
        "VmSwap:\t789 kB\n");
    const auto result = parse_process_status(input);
    require(result.vm_rss_kib.available && result.vm_rss_kib.value == 123456);
    require(result.vm_swap_kib.available && result.vm_swap_kib.value == 789);

    std::istringstream malformed("VmRSS: invalid kB\nVmSwap: 0 bytes\n");
    const auto missing = parse_process_status(malformed);
    require(!missing.vm_rss_kib.available);
    require(!missing.vm_swap_kib.available);
}

static void test_vmstat_parser() {
    std::istringstream input(
        "nr_free_pages 99\n"
        "pgscan_kswapd 100\n"
        "pgscan_direct 11\n"
        "pgsteal_kswapd 80\n"
        "pgsteal_direct 7\n"
        "allocstall_normal 3\n"
        "pswpin 5\n"
        "pswpout 6\n");
    const auto result = parse_vmstat(input);
    require(result.pgscan_kswapd_pages.available && result.pgscan_kswapd_pages.value == 100);
    require(result.pgscan_direct_pages.available && result.pgscan_direct_pages.value == 11);
    require(result.pgsteal_kswapd_pages.available && result.pgsteal_kswapd_pages.value == 80);
    require(result.pgsteal_direct_pages.available && result.pgsteal_direct_pages.value == 7);
    require(result.allocstall_normal.available && result.allocstall_normal.value == 3);
    require(result.pswpin_pages.available && result.pswpin_pages.value == 5);
    require(result.pswpout_pages.available && result.pswpout_pages.value == 6);

    std::istringstream partial("pswpin nope\npswpout 9 extra\npgscan_direct 2\n");
    const auto missing = parse_vmstat(partial);
    require(!missing.pswpin_pages.available);
    require(!missing.pswpout_pages.available);
    require(missing.pgscan_direct_pages.available && missing.pgscan_direct_pages.value == 2);
}

static void test_deltas_and_gate() {
    uint64_t counter = 0;
    require(counter_delta({ true, 10 }, { true, 14 }, counter) && counter == 4);
    require(!counter_delta({ true, 14 }, { true, 10 }, counter));
    require(!counter_delta({}, { true, 10 }, counter));

    int64_t gauge = 0;
    require(signed_delta({ true, 10 }, { true, 14 }, gauge) && gauge == 4);
    require(signed_delta({ true, 14 }, { true, 10 }, gauge) && gauge == -4);
    require(!signed_delta({}, { true, 10 }, gauge));

    require(enabled_value("1"));
    require(!enabled_value(nullptr));
    require(!enabled_value("0"));
    require(!enabled_value("1 "));
}

static void test_runtime_gate_and_csv() {
#if defined(__linux__)
    const std::string path = "/tmp/llama-page-activity-test-" + std::to_string(getpid()) + ".csv";
    std::remove(path.c_str());
    setenv("LLAMA_PAGE_ACTIVITY_LOG_PATH", path.c_str(), 1);
    unsetenv("LLAMA_PAGE_ACTIVITY_DIAGNOSTIC");

    phase_fields fields;
    fields.request_id = 42;
    fields.task_id = 42;
    fields.slot_id = 3;
    page_activity_snapshot("KV_PRE_REQUEST", "REQUEST_START", fields);
    require(std::ifstream(path).good() == false);

    setenv("LLAMA_PAGE_ACTIVITY_DIAGNOSTIC", "1", 1);
    page_activity_snapshot("KV_PRE_REQUEST", "REQUEST_START", fields);
    page_activity_snapshot("SEQ_RM_COMPAT_CHECK_START", "SEQ_RM_CHECK_START");
    page_activity_snapshot("SEQ_RM_LLAMA_DECODE_START", "SEQ_RM_DECODE_START");
    page_activity_snapshot("SEQ_RM_LLAMA_DECODE_RETURN", "SEQ_RM_DECODE_END");
    page_activity_snapshot("SEQ_RM_COMPAT_CHECK_END", "SEQ_RM_CHECK_END");
    page_activity_snapshot("SERVER_READY", "SERVER_READY");
    std::ifstream input(path);
    require(input.good());
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) {
        lines.push_back(line);
    }
    require(lines.size() == 7);
    require(lines[0].find("process_minor_faults") != std::string::npos);
    require(lines[1].find(",KV_PRE_REQUEST,REQUEST_START,42,42,3,") != std::string::npos);
    require(lines[2].find(",SEQ_RM_COMPAT_CHECK_START,SEQ_RM_CHECK_START,,,,") != std::string::npos);
    require(lines[3].find(",SEQ_RM_LLAMA_DECODE_START,SEQ_RM_DECODE_START,,,,") != std::string::npos);
    require(lines[4].find(",SEQ_RM_LLAMA_DECODE_RETURN,SEQ_RM_DECODE_END,,,,") != std::string::npos);
    require(lines[5].find(",SEQ_RM_COMPAT_CHECK_END,SEQ_RM_CHECK_END,,,,") != std::string::npos);
    require(lines[6].find(",SERVER_READY,SERVER_READY,,,,") != std::string::npos);

    unsetenv("LLAMA_PAGE_ACTIVITY_DIAGNOSTIC");
    page_activity_snapshot("KV_PREFILL_END", "PREFILL_COMPLETE", fields);
    std::ifstream after(path);
    size_t line_count = 0;
    while (std::getline(after, line)) {
        ++line_count;
    }
    require(line_count == 7);
    std::remove(path.c_str());
#endif
}

static std::vector<std::string> split_csv(const std::string & line) {
    std::vector<std::string> result;
    std::istringstream input(line);
    std::string field;
    while (std::getline(input, field, ',')) {
        result.push_back(field);
    }
    if (!line.empty() && line.back() == ',') {
        result.emplace_back();
    }
    return result;
}

static void test_phase_mode_gates_and_total() {
#if defined(__linux__)
    auto run_phase_off_child = [](const char * linking_gate) {
        const pid_t child = fork();
        require(child >= 0);
        if (child == 0) {
            unsetenv("LLAMA_PHASE_EVENTS_PATH");
            setenv("LLAMA_GENERIC_CALL_TIMING", "0", 1);
            setenv("LLAMA_KV_PHASE_SNAPSHOT", "0", 1);
            unsetenv("LLAMA_ATTN_DIAGNOSTIC");
            unsetenv("LLAMA_CPU_ATTN_TIMING");
            unsetenv("LLAMA_KV_PAGE_COVERAGE_DIAGNOSTIC");
            if (linking_gate) {
                setenv(linking_gate, "1", 1);
            }
            const bool gates_off = !phase_generic_call_timing_enabled() && !phase_kv_snapshot_enabled();
            const uint64_t link_call_id = phase_resolve_call_id(0, 0);
            const bool link_matches = linking_gate ? link_call_id != 0 : link_call_id == 0;
            _exit(gates_off && link_matches ? 0 : 1);
        }
        int status = 0;
        require(waitpid(child, &status, 0) == child);
        require(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    };
    run_phase_off_child(nullptr);
    run_phase_off_child("LLAMA_ATTN_DIAGNOSTIC");
    run_phase_off_child("LLAMA_CPU_ATTN_TIMING");
    run_phase_off_child("LLAMA_KV_PAGE_COVERAGE_DIAGNOSTIC");

    const std::string path = "/tmp/llama-phase-mode-test-" + std::to_string(getpid()) + ".csv";
    std::remove(path.c_str());
    setenv("LLAMA_PHASE_EVENTS_PATH", path.c_str(), 1);
    setenv("LLAMA_GENERIC_CALL_TIMING", "false", 1);
    setenv("LLAMA_KV_PHASE_SNAPSHOT", "00", 1);
    require(phase_generic_call_timing_enabled());
    require(phase_kv_snapshot_enabled());

    auto set_gate = [](const char * name, const char * value) {
        if (value) {
            setenv(name, value, 1);
        } else {
            unsetenv(name);
        }
    };
    auto record_case = [&](int64_t request_id, const char * generic, const char * kv_snapshot) {
        set_gate("LLAMA_GENERIC_CALL_TIMING", generic);
        set_gate("LLAMA_KV_PHASE_SNAPSHOT", kv_snapshot);

        phase_fields fields;
        fields.request_id = request_id;
        fields.task_id = request_id;
        fields.slot_id = 3;
        fields.status = 0;
        phase_prefill_total_reset(request_id);
        phase_event("test", "PHASE_MARKER", phase_kind::POINT, 0, fields);

        uint64_t generic_span = 0;
        if (phase_generic_call_timing_enabled()) {
            generic_span = phase_event(
                "test", "LLM_PREFILL_CALL_START", phase_kind::CALL_START, 0, fields);
            require(phase_resolve_call_id(generic_span, generic_span + 1) == generic_span);
        }
        if (phase_kv_snapshot_enabled()) {
            phase_event("test", "KV_TEST_SNAPSHOT", phase_kind::POINT, 0, fields);
        }

        fields.duration_ns = 10;
        fields.q_len = 4;
        phase_prefill_total_add(fields);
        fields.duration_ns = 20;
        fields.q_len = 6;
        phase_prefill_total_add(fields);
        if (generic_span) {
            phase_event("test", "LLM_PREFILL_CALL_RETURN", phase_kind::CALL_RETURN, generic_span, fields);
        } else {
            const uint64_t link_call_id = phase_resolve_call_id(0, 0);
            require(link_call_id != 0);
            fields.call_id = static_cast<int64_t>(link_call_id);
            phase_set_call(link_call_id, 5);
            phase_set_link(fields, "PREFILL");
            phase_active_call().ubatch_idx = 2;
            require(phase_active_call().call_id == link_call_id);
            require(phase_active_call().link_call_id == link_call_id);
            require(phase_active_call().request_id == request_id);
            require(phase_active_call().ubatch_idx == 2);
            require(std::string(phase_active_call().stage) == "PREFILL");
            phase_set_call(0);
            phase_note_prefill_call(request_id, link_call_id);
            require(phase_last_prefill_call(request_id) == static_cast<int64_t>(link_call_id));
        }
        phase_prefill_total_emit(fields);
        require(phase_flush());
    };

    record_case(10, nullptr, nullptr); // E: legacy Generic + KV-A
    record_case(20, nullptr, "0");  // C: Generic only
    record_case(30, "0", nullptr);  // D: KV-A only
    record_case(40, "0", "0");    // B: Phase only

    std::ifstream input(path);
    require(input.good());
    std::string line;
    require(bool(std::getline(input, line)));
    const auto header = split_csv(line);
    std::unordered_map<std::string, size_t> columns;
    for (size_t i = 0; i < header.size(); ++i) {
        columns[header[i]] = i;
    }
    struct observed_case {
        int phase = 0;
        int generic = 0;
        int kv_snapshot = 0;
        int total = 0;
        bool valid_total = false;
    };
    std::unordered_map<int64_t, observed_case> observed;
    while (std::getline(input, line)) {
        const auto row = split_csv(line);
        require(row.size() == header.size());
        const auto request = row[columns["request_id"]];
        require(!request.empty());
        auto & current = observed[std::stoll(request)];
        const auto & name = row[columns["event_name"]];
        if (name == "PHASE_MARKER") {
            current.phase++;
        }
        if (name == "LLM_PREFILL_CALL_RETURN") {
            current.generic++;
        }
        if (name == "KV_TEST_SNAPSHOT") {
            current.kv_snapshot++;
        }
        if (name == "LLM_PREFILL_EVAL_TOTAL") {
            current.total++;
            current.valid_total = row[columns["duration_ns"]] == "30" &&
                row[columns["q_len"]] == "10" &&
                row[columns["note"]] == "sum_prompt_llama_decode_calls";
        }
    }
    require(observed[10].phase == 1 && observed[10].generic == 1 && observed[10].kv_snapshot == 1 && observed[10].total == 0);
    require(observed[20].phase == 1 && observed[20].generic == 1 && observed[20].kv_snapshot == 0 && observed[20].total == 0);
    require(observed[30].phase == 1 && observed[30].generic == 0 && observed[30].kv_snapshot == 1 && observed[30].total == 1 && observed[30].valid_total);
    require(observed[40].phase == 1 && observed[40].generic == 0 && observed[40].kv_snapshot == 0 && observed[40].total == 1 && observed[40].valid_total);
    std::remove(path.c_str());
#endif
}

int main() {
    test_process_status_parser();
    test_vmstat_parser();
    test_deltas_and_gate();
    test_runtime_gate_and_csv();
    test_phase_mode_gates_and_total();
    return 0;
}

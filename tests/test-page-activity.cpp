#include "llama-page-activity.h"
#include "llama-phase-events.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#if defined(__linux__)
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

int main() {
    test_process_status_parser();
    test_vmstat_parser();
    test_deltas_and_gate();
    test_runtime_gate_and_csv();
    return 0;
}

#include "llama-kv-repeated-activity.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

// The configured Raspberry Pi build is Release, so keep synthetic checks active.
#undef assert
#define assert(condition) do { if (!(condition)) std::abort(); } while (0)

using namespace llama_kv_repeated_activity;

static llama_kv_residency::bitmap make_bitmap(std::initializer_list<bool> pages) {
    llama_kv_residency::bitmap result;
    result.resize(pages.size());
    uint64_t page = 0;
    for (bool resident : pages) {
        result.set(page++, resident);
    }
    return result;
}

static scan_result make_scan(uint64_t timestamp_ns, std::initializer_list<bool> pages) {
    scan_result result;
    result.timestamp_ns = timestamp_ns;
    result.mincore_elapsed_ns = 123;
    result.bitmap = make_bitmap(pages);
    result.valid = true;
    result.status = "ok";
    return result;
}

static void test_entered_exited_and_repetition() {
    accumulator observations(4);
    observations.observe("REQUEST_START", "PREFILL", make_scan(1000000, { false, true, false, true }));
    observations.observe("PERIODIC", "PREFILL", make_scan(3000000, { true, true, false, false }));
    observations.observe("PREFILL_COMPLETE", "PREFILL", make_scan(6500000, { false, true, true, false }));

    const auto & rows = observations.samples();
    assert(rows.size() == 3);
    assert(!rows[0].has_interval && !rows[0].has_transition);
    assert(rows[1].has_interval && rows[1].actual_interval_ns == 2000000);
    assert(rows[1].transition.entered_pages == 1);
    assert(rows[1].transition.exited_pages == 1);
    assert(rows[2].actual_interval_ns == 3500000);
    assert(rows[2].transition.entered_pages == 1);
    assert(rows[2].transition.exited_pages == 1);

    const auto totals = observations.calculate_summary();
    assert(totals.total_entered_pages == 2);
    assert(totals.total_exited_pages == 2);
    assert(totals.total_observed_transitions == 4);
    assert(totals.pages_with_observed_transition == 3);
    assert(totals.pages_with_repeated_transition == 1);
    assert(totals.repeat_transition_excess == 1);
    assert(totals.max_observed_transitions_per_page == 2);
    assert(observations.page_transitions()[0].entered == 1);
    assert(observations.page_transitions()[0].exited == 1);
}

static void test_failed_sample_breaks_adjacent_comparison() {
    accumulator observations(1);
    observations.observe("REQUEST_START", "PREFILL", make_scan(1, { false }));
    scan_result failure;
    failure.timestamp_ns = 2;
    failure.status = "synthetic_failure";
    observations.observe("PERIODIC", "PREFILL", std::move(failure));
    observations.observe("PERIODIC", "PREFILL", make_scan(3, { true }));
    assert(!observations.samples()[1].valid);
    assert(!observations.samples()[2].has_transition);
    assert(observations.calculate_summary().total_observed_transitions == 0);
}

static void test_interval_configuration() {
    assert(parse_interval_ms(nullptr) == 1000);
    assert(parse_interval_ms("") == 1000);
    assert(parse_interval_ms("0") == 1000);
    assert(parse_interval_ms("invalid") == 1000);
    assert(parse_interval_ms("250") == 250);
}

static void test_worker_periodic_sampling_and_shutdown() {
    std::atomic<uint64_t> calls { 0 };
    {
        sampling_worker worker(1, 5, [&] {
            const uint64_t call = ++calls;
            return make_scan(call * 1000000, { (call % 2) != 0 });
        });
        assert(worker.start("REQUEST_START", "PREFILL"));
        std::this_thread::sleep_for(std::chrono::milliseconds(18));
        assert(worker.boundary("PREFILL_COMPLETE", "PREFILL", "DECODE"));
        assert(worker.stop("DECODE_COMPLETE", "DECODE"));

        const auto & rows = worker.observations().samples();
        assert(rows.size() >= 4);
        assert(rows.front().event == "REQUEST_START");
        assert(rows.front().phase == "PREFILL");
        assert(rows.back().event == "DECODE_COMPLETE");
        assert(rows.back().phase == "DECODE");
        bool saw_periodic = false;
        bool saw_prefill_complete = false;
        for (const auto & row : rows) {
            saw_periodic |= row.event == "PERIODIC";
            saw_prefill_complete |= row.event == "PREFILL_COMPLETE";
        }
        assert(saw_periodic);
        assert(saw_prefill_complete);
    }
    const uint64_t stopped_calls = calls.load();
    std::this_thread::sleep_for(std::chrono::milliseconds(12));
    assert(calls.load() == stopped_calls);
}

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--off") == 0) {
        assert(!llama_kv_residency::enabled_value(nullptr));
        assert(!llama_kv_residency::enabled_value("0"));
        return 0;
    }
    test_entered_exited_and_repetition();
    test_failed_sample_breaks_adjacent_comparison();
    test_interval_configuration();
    test_worker_periodic_sampling_and_shutdown();
    return 0;
}

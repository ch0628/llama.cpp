#include "llama-kv-residency.h"

#include <cassert>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

using namespace llama_kv_residency;

static bitmap make_bitmap(std::initializer_list<bool> pages) {
    bitmap result;
    result.resize(pages.size());
    uint64_t page = 0;
    for (bool resident : pages) {
        result.set(page++, resident);
    }
    return result;
}

static void test_transition_counts() {
    const bitmap previous = make_bitmap({ false, true, true, false });
    const bitmap current = make_bitmap({ true, false, true, false });
    transition_counts counts;
    assert(calculate_transition(previous, current, counts));
    assert(counts.entered_pages == 1);
    assert(counts.exited_pages == 1);
    assert(counts.stayed_resident_pages == 1);
    assert(counts.stayed_nonresident_pages == 1);
    assert(counts.previous_resident_pages == 2);
    assert(counts.current_resident_pages == 2);
    assert(counts.compared_pages == 4);
    assert(counts.valid());

    bitmap wrong_size;
    wrong_size.resize(3);
    assert(!calculate_transition(previous, wrong_size, counts));
}

static void test_range_alignment() {
    uintptr_t start = 0;
    uintptr_t end = 0;
    assert(align_range(0x1003, 4096, 4096, start, end) == range_status::ok);
    assert(start == 0x1000);
    assert(end == 0x3000);
    assert(align_range(std::numeric_limits<uintptr_t>::max() - 1, 4, 4096, start, end) ==
        range_status::address_overflow);
    assert(align_range(0x1000, 0, 4096, start, end) == range_status::empty);
}

static void test_overlap_deduplication() {
    const bitmap previous_a = make_bitmap({ false, true });
    const bitmap current_a = make_bitmap({ true, true });
    const bitmap previous_b = make_bitmap({ true, false });
    const bitmap current_b = make_bitmap({ true, false });
    const std::vector<transition_bitmap_range> ranges = {
        { 0x1000, 0x3000, 4096, &previous_a, &current_a },
        { 0x2000, 0x4000, 4096, &previous_b, &current_b },
    };
    transition_counts counts;
    assert(calculate_unique_transition(ranges, counts));
    assert(counts.compared_pages == 3);
    assert(counts.entered_pages == 1);
    assert(counts.exited_pages == 0);
    assert(counts.stayed_resident_pages == 1);
    assert(counts.stayed_nonresident_pages == 1);
    assert(counts.valid());
}

static void test_snapshot_status_and_gate() {
    assert(enabled_value("1"));
    assert(!enabled_value(nullptr));
    assert(!enabled_value("0"));
    assert(!enabled_value("1 "));
    assert(std::string(snapshot_status(0, 0)) == "no_scannable_buffers");
    assert(std::string(snapshot_status(2, 0)) == "no_scannable_buffers");
    assert(std::string(snapshot_status(2, 1)) == "partial_failure");
    assert(std::string(snapshot_status(2, 2)) == "ok");
}

int main() {
    test_transition_counts();
    test_range_alignment();
    test_overlap_deduplication();
    test_snapshot_status_and_gate();
    return 0;
}

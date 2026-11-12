#include "llama-kv-residency.h"
#include "llama-kv-vma.h"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace llama_kv_vma;

static void require(bool condition) {
    if (!condition) {
        std::abort();
    }
}

static void test_maps_parser() {
    std::istringstream input(
        "1000-3000 rw-p 00000000 00:00 0\n"
        "4000-5000 r--p 00001000 08:01 7 /tmp/model file.gguf\n"
        "6000-7000 rw-p 00000000 00:00 0 [heap]\n");
    bool complete = false;
    const auto mappings = parse_maps(input, complete);
    require(complete);
    require(mappings.size() == 3);
    require(mappings[0].start == 0x1000 && mappings[0].end == 0x3000);
    require(mappings[0].permissions == "rw-p" && mappings[0].name.empty());
    require(std::string(mapping_kind(mappings[0].name)) == "anonymous");
    require(mappings[1].name == "/tmp/model file.gguf");
    require(std::string(mapping_kind(mappings[1].name)) == "file");
    require(std::string(mapping_kind(mappings[2].name)) == "heap");

    std::istringstream malformed("bad maps line\n8000-9000 rw-p 00000000 00:00 0\n");
    const auto partial = parse_maps(malformed, complete);
    require(!complete && partial.size() == 1);
}

static void test_single_vma_intersection() {
    const std::vector<mapping> mappings = {
        { 0x1000, 0x3000, "rw-p", "" },
    };
    const auto result = calculate_intersections(0x1800, 0x2800, mappings);
    require(result.intersections.size() == 1);
    require(result.intersections[0].start == 0x1800);
    require(result.intersections[0].end == 0x2800);
    require(result.intersections[0].size() == 0x1000);
    require(result.intersections[0].vma_size() == 0x2000);
    require(result.intersections[0].outside_buffer_bytes() == 0x1000);
    require(result.covered_bytes == 0x1000);
    require(result.uncovered_bytes == 0);
    require(result.total_vma_outside_buffer_bytes == 0x1000);
}

static void test_multiple_vmas() {
    const std::vector<mapping> mappings = {
        { 0x4000, 0x5000, "rw-p", "" },
        { 0x1000, 0x2000, "rw-p", "" },
        { 0x2000, 0x4000, "rw-p", "" },
    };
    const auto result = calculate_intersections(0x1800, 0x4800, mappings);
    require(result.intersections.size() == 3);
    require(result.intersections[0].start == 0x1800 && result.intersections[0].end == 0x2000);
    require(result.intersections[1].start == 0x2000 && result.intersections[1].end == 0x4000);
    require(result.intersections[2].start == 0x4000 && result.intersections[2].end == 0x4800);
    require(result.covered_bytes == 0x3000);
    require(result.uncovered_bytes == 0);
    require(result.total_vma_outside_buffer_bytes == 0x1000);
}

static void test_uncovered_range() {
    const std::vector<mapping> mappings = {
        { 0x1000, 0x2000, "rw-p", "" },
        { 0x3000, 0x5000, "rw-p", "" },
    };
    const auto result = calculate_intersections(0x1000, 0x5000, mappings);
    require(result.intersections.size() == 2);
    require(result.covered_bytes == 0x3000);
    require(result.uncovered_bytes == 0x1000);
}

static void test_alignment_and_boundary_pages() {
    uintptr_t aligned_start = 0;
    uintptr_t aligned_end = 0;
    require(llama_kv_residency::align_range(
            0x1040, 0x1000, 0x1000, aligned_start, aligned_end) ==
        llama_kv_residency::range_status::ok);
    require(aligned_start == 0x1000 && aligned_end == 0x3000);

    const std::vector<mapping> mappings = {
        { 0x1000, 0x3000, "rw-p", "" },
    };
    const auto exact = calculate_intersections(0x1040, 0x2040, mappings);
    require(exact.covered_bytes == 0x1000);
    require(exact.intersections[0].outside_buffer_bytes() == 0x1000);
}

static void test_gate() {
    require(enabled_value("1"));
    require(!enabled_value(nullptr));
    require(!enabled_value("0"));
    require(!enabled_value("1 "));

    unsetenv("LLAMA_KV_VMA_DIAGNOSTIC");
    require(!enabled_value(std::getenv("LLAMA_KV_VMA_DIAGNOSTIC")));
    require(setenv("LLAMA_KV_VMA_DIAGNOSTIC", "1", 1) == 0);
    require(enabled_value(std::getenv("LLAMA_KV_VMA_DIAGNOSTIC")));
    unsetenv("LLAMA_KV_VMA_DIAGNOSTIC");
}

static void test_live_maps_coverage() {
#if defined(__linux__)
    int value = 0;
    const uintptr_t start = reinterpret_cast<uintptr_t>(&value);
    std::ifstream input("/proc/self/maps");
    require(input.good());
    bool complete = false;
    const auto mappings = parse_maps(input, complete);
    const auto result = calculate_intersections(start, start + sizeof(value), mappings);
    require(complete);
    require(result.intersections.size() == 1);
    require(result.covered_bytes == sizeof(value));
    require(result.uncovered_bytes == 0);
#endif
}

int main() {
    test_maps_parser();
    test_single_vma_intersection();
    test_multiple_vmas();
    test_uncovered_range();
    test_alignment_and_boundary_pages();
    test_gate();
    test_live_maps_coverage();
    return 0;
}

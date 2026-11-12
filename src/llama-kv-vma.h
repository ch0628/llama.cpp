#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <istream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace llama_kv_vma {

struct mapping {
    uintptr_t start = 0;
    uintptr_t end = 0;
    std::string permissions;
    std::string name;
};

struct intersection {
    mapping vma;
    uintptr_t start = 0;
    uintptr_t end = 0;

    uint64_t size() const {
        return static_cast<uint64_t>(end - start);
    }

    uint64_t vma_size() const {
        return static_cast<uint64_t>(vma.end - vma.start);
    }

    uint64_t outside_buffer_bytes() const {
        return vma_size() - size();
    }
};

struct report {
    std::vector<intersection> intersections;
    uint64_t covered_bytes = 0;
    uint64_t uncovered_bytes = 0;
    uint64_t total_vma_outside_buffer_bytes = 0;
};

inline bool enabled_value(const char * value) {
    return value && value[0] == '1' && value[1] == '\0';
}

inline bool parse_hex_address(const std::string & text, uintptr_t & value) {
    if (text.empty() || text[0] == '-') {
        return false;
    }
    char * end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text.c_str(), &end, 16);
    if (errno == ERANGE || end == text.c_str() || *end != '\0' ||
        parsed > static_cast<unsigned long long>(std::numeric_limits<uintptr_t>::max())) {
        return false;
    }
    value = static_cast<uintptr_t>(parsed);
    return true;
}

inline bool parse_maps_line(const std::string & line, mapping & result) {
    std::istringstream fields(line);
    std::string range;
    std::string offset;
    std::string device;
    std::string inode;
    if (!(fields >> range >> result.permissions >> offset >> device >> inode)) {
        return false;
    }

    const auto dash = range.find('-');
    if (dash == std::string::npos || range.find('-', dash + 1) != std::string::npos ||
        !parse_hex_address(range.substr(0, dash), result.start) ||
        !parse_hex_address(range.substr(dash + 1), result.end) || result.start >= result.end) {
        return false;
    }

    std::getline(fields, result.name);
    const auto first = result.name.find_first_not_of(" \t");
    result.name = first == std::string::npos ? "" : result.name.substr(first);
    return true;
}

inline std::vector<mapping> parse_maps(std::istream & input, bool & complete) {
    std::vector<mapping> result;
    complete = true;
    std::string line;
    while (std::getline(input, line)) {
        mapping item;
        if (!parse_maps_line(line, item)) {
            complete = false;
            continue;
        }
        result.push_back(std::move(item));
    }
    return result;
}

inline const char * mapping_kind(const std::string & name) {
    if (name.empty()) {
        return "anonymous";
    }
    if (name == "[heap]") {
        return "heap";
    }
    if (name.compare(0, 6, "[stack") == 0) {
        return "stack";
    }
    if (name.compare(0, 6, "[anon:") == 0 || name.compare(0, 12, "[anon_shmem:") == 0) {
        return "named_anonymous";
    }
    if (name.front() == '[') {
        return "special";
    }
    return "file";
}

inline report calculate_intersections(
        uintptr_t buffer_start,
        uintptr_t buffer_end,
        const std::vector<mapping> & mappings) {
    report result;
    if (buffer_start >= buffer_end) {
        return result;
    }

    for (const auto & vma : mappings) {
        const uintptr_t start = std::max(buffer_start, vma.start);
        const uintptr_t end = std::min(buffer_end, vma.end);
        if (start < end) {
            result.intersections.push_back({ vma, start, end });
        }
    }
    std::sort(result.intersections.begin(), result.intersections.end(),
            [](const intersection & left, const intersection & right) {
                return left.start < right.start || (left.start == right.start && left.end < right.end);
            });

    uintptr_t covered_end = buffer_start;
    for (const auto & item : result.intersections) {
        if (item.end > covered_end) {
            const uintptr_t start = std::max(item.start, covered_end);
            result.covered_bytes += static_cast<uint64_t>(item.end - start);
            covered_end = item.end;
        }
        result.total_vma_outside_buffer_bytes += item.outside_buffer_bytes();
    }
    const uint64_t buffer_bytes = static_cast<uint64_t>(buffer_end - buffer_start);
    result.uncovered_bytes = buffer_bytes - result.covered_bytes;
    return result;
}

} // namespace llama_kv_vma

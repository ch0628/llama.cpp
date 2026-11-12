#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace llama_kv_residency {

struct bitmap {
    uint64_t page_count = 0;
    std::vector<uint8_t> data;

    void resize(uint64_t pages) {
        page_count = pages;
        data.assign(static_cast<size_t>((pages + 7) / 8), 0);
    }

    bool get(uint64_t page) const {
        return page < page_count && (data[static_cast<size_t>(page / 8)] & (uint8_t(1) << (page % 8))) != 0;
    }

    void set(uint64_t page, bool resident) {
        if (page >= page_count) {
            return;
        }
        uint8_t & value = data[static_cast<size_t>(page / 8)];
        const uint8_t mask = uint8_t(1) << (page % 8);
        value = resident ? value | mask : value & ~mask;
    }
};

struct transition_counts {
    uint64_t entered_pages = 0;
    uint64_t exited_pages = 0;
    uint64_t stayed_resident_pages = 0;
    uint64_t stayed_nonresident_pages = 0;
    uint64_t previous_resident_pages = 0;
    uint64_t current_resident_pages = 0;
    uint64_t compared_pages = 0;

    bool valid() const {
        return int64_t(previous_resident_pages) + int64_t(entered_pages) - int64_t(exited_pages) ==
                int64_t(current_resident_pages) &&
            entered_pages + exited_pages + stayed_resident_pages + stayed_nonresident_pages == compared_pages &&
            int64_t(entered_pages) - int64_t(exited_pages) ==
                int64_t(current_resident_pages) - int64_t(previous_resident_pages);
    }

    void add(bool previous, bool current) {
        previous_resident_pages += previous;
        current_resident_pages += current;
        compared_pages++;
        if (previous && current) {
            stayed_resident_pages++;
        } else if (previous) {
            exited_pages++;
        } else if (current) {
            entered_pages++;
        } else {
            stayed_nonresident_pages++;
        }
    }
};

inline bool enabled_value(const char * value) {
    return value && value[0] == '1' && value[1] == '\0';
}

inline const char * snapshot_status(size_t buffer_count, size_t complete_buffer_count) {
    if (buffer_count == 0 || complete_buffer_count == 0) {
        return "no_scannable_buffers";
    }
    return buffer_count == complete_buffer_count ? "ok" : "partial_failure";
}

inline bool calculate_transition(
        const bitmap & previous,
        const bitmap & current,
        transition_counts & counts,
        uint64_t first_page = 0,
        uint64_t page_count = std::numeric_limits<uint64_t>::max()) {
    if (previous.page_count != current.page_count || first_page > previous.page_count) {
        return false;
    }
    const uint64_t available = previous.page_count - first_page;
    page_count = std::min(page_count, available);
    for (uint64_t page = first_page; page < first_page + page_count; ++page) {
        counts.add(previous.get(page), current.get(page));
    }
    return true;
}

using page_range = std::pair<uintptr_t, uintptr_t>;

inline void add_covered_range(std::vector<page_range> & covered, uintptr_t start, uintptr_t end) {
    std::vector<page_range> merged;
    merged.reserve(covered.size() + 1);
    bool inserted = false;
    for (const auto & range : covered) {
        if (range.second < start) {
            merged.push_back(range);
        } else if (end < range.first) {
            if (!inserted) {
                merged.emplace_back(start, end);
                inserted = true;
            }
            merged.push_back(range);
        } else {
            start = std::min(start, range.first);
            end = std::max(end, range.second);
        }
    }
    if (!inserted) {
        merged.emplace_back(start, end);
    }
    covered.swap(merged);
}

inline std::vector<page_range> uncovered_ranges(
        const std::vector<page_range> & covered,
        uintptr_t start,
        uintptr_t end) {
    std::vector<page_range> result;
    uintptr_t cursor = start;
    for (const auto & range : covered) {
        if (range.second <= cursor) {
            continue;
        }
        if (range.first >= end) {
            break;
        }
        if (range.first > cursor) {
            result.emplace_back(cursor, std::min(range.first, end));
        }
        cursor = std::max(cursor, range.second);
        if (cursor >= end) {
            break;
        }
    }
    if (cursor < end) {
        result.emplace_back(cursor, end);
    }
    return result;
}

enum class range_status {
    ok,
    address_overflow,
    alignment_overflow,
    empty,
};

inline range_status align_range(
        uintptr_t base,
        uint64_t allocated_bytes,
        uintptr_t page_size,
        uintptr_t & aligned_start,
        uintptr_t & aligned_end) {
    const uintptr_t max_addr = std::numeric_limits<uintptr_t>::max();
    if (page_size == 0 || allocated_bytes > max_addr - base) {
        return range_status::address_overflow;
    }
    const uintptr_t allocation_end = base + static_cast<uintptr_t>(allocated_bytes);
    if (allocation_end > max_addr - (page_size - 1)) {
        return range_status::alignment_overflow;
    }
    aligned_start = (base / page_size) * page_size;
    aligned_end = ((allocation_end + page_size - 1) / page_size) * page_size;
    return aligned_end > aligned_start ? range_status::ok : range_status::empty;
}

struct transition_bitmap_range {
    uintptr_t start = 0;
    uintptr_t end = 0;
    uint64_t page_size = 0;
    const bitmap * previous = nullptr;
    const bitmap * current = nullptr;
};

inline bool calculate_unique_transition(
        const std::vector<transition_bitmap_range> & ranges,
        transition_counts & counts) {
    std::vector<page_range> covered;
    for (const auto & range : ranges) {
        if (!range.previous || !range.current || range.page_size == 0 || range.start >= range.end ||
            range.start % range.page_size != 0 || range.end % range.page_size != 0 ||
            range.previous->page_count != (range.end - range.start) / range.page_size ||
            range.current->page_count != range.previous->page_count) {
            return false;
        }
        for (const auto & uncovered : uncovered_ranges(covered, range.start, range.end)) {
            const uint64_t first_page = (uncovered.first - range.start) / range.page_size;
            const uint64_t page_count = (uncovered.second - uncovered.first) / range.page_size;
            if (!calculate_transition(*range.previous, *range.current, counts, first_page, page_count)) {
                return false;
            }
        }
        add_covered_range(covered, range.start, range.end);
    }
    return counts.valid();
}

} // namespace llama_kv_residency

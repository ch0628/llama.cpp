#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <istream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>

namespace llama_page_activity {

struct optional_u64 {
    bool available = false;
    uint64_t value = 0;
};

struct process_status {
    optional_u64 vm_rss_kib;
    optional_u64 vm_swap_kib;
};

struct vmstat_counters {
    optional_u64 pgscan_kswapd_pages;
    optional_u64 pgscan_direct_pages;
    optional_u64 pgsteal_kswapd_pages;
    optional_u64 pgsteal_direct_pages;
    optional_u64 allocstall_normal;
    optional_u64 pswpin_pages;
    optional_u64 pswpout_pages;
};

inline bool parse_u64(const std::string & text, uint64_t & value) {
    if (text.empty() || text[0] == '-') {
        return false;
    }
    char * end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
    if (errno == ERANGE || end == text.c_str() || *end != '\0') {
        return false;
    }
    value = static_cast<uint64_t>(parsed);
    return true;
}

inline process_status parse_process_status(std::istream & input) {
    process_status result;
    std::string line;
    while (std::getline(input, line)) {
        const auto colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        const std::string key = line.substr(0, colon);
        if (key != "VmRSS" && key != "VmSwap") {
            continue;
        }
        std::istringstream values(line.substr(colon + 1));
        std::string raw;
        std::string unit;
        std::string extra;
        uint64_t parsed = 0;
        if (!(values >> raw >> unit) || unit != "kB" || (values >> extra) || !parse_u64(raw, parsed)) {
            continue;
        }
        optional_u64 & destination = key == "VmRSS" ? result.vm_rss_kib : result.vm_swap_kib;
        if (!destination.available) {
            destination = { true, parsed };
        }
    }
    return result;
}

inline vmstat_counters parse_vmstat(std::istream & input) {
    vmstat_counters result;
    std::unordered_map<std::string, optional_u64 *> fields = {
        { "pgscan_kswapd",   &result.pgscan_kswapd_pages },
        { "pgscan_direct",   &result.pgscan_direct_pages },
        { "pgsteal_kswapd",  &result.pgsteal_kswapd_pages },
        { "pgsteal_direct",  &result.pgsteal_direct_pages },
        { "allocstall_normal", &result.allocstall_normal },
        { "pswpin",          &result.pswpin_pages },
        { "pswpout",         &result.pswpout_pages },
    };

    std::string line;
    while (std::getline(input, line)) {
        std::istringstream values(line);
        std::string key;
        std::string raw;
        std::string extra;
        if (!(values >> key >> raw) || (values >> extra)) {
            continue;
        }
        const auto found = fields.find(key);
        uint64_t parsed = 0;
        if (found != fields.end() && !found->second->available && parse_u64(raw, parsed)) {
            *found->second = { true, parsed };
        }
    }
    return result;
}

inline bool counter_delta(const optional_u64 & before, const optional_u64 & after, uint64_t & delta) {
    if (!before.available || !after.available || after.value < before.value) {
        return false;
    }
    delta = after.value - before.value;
    return true;
}

inline bool signed_delta(const optional_u64 & before, const optional_u64 & after, int64_t & delta) {
    if (!before.available || !after.available) {
        return false;
    }
    if (after.value >= before.value) {
        const uint64_t magnitude = after.value - before.value;
        if (magnitude > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            return false;
        }
        delta = static_cast<int64_t>(magnitude);
        return true;
    }
    const uint64_t magnitude = before.value - after.value;
    if (magnitude > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return false;
    }
    delta = -static_cast<int64_t>(magnitude);
    return true;
}

inline bool enabled_value(const char * value) {
    return value && value[0] == '1' && value[1] == '\0';
}

} // namespace llama_page_activity

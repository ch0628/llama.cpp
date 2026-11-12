#pragma once

#include "llama-kv-residency.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace llama_kv_repeated_activity {

struct scan_result {
    uint64_t timestamp_ns = 0;
    uint64_t mincore_elapsed_ns = 0;
    llama_kv_residency::bitmap bitmap;
    bool valid = false;
    std::string status;
    int error_no = 0;
};

struct sample {
    uint64_t sample_id = 0;
    uint64_t timestamp_ns = 0;
    uint64_t actual_interval_ns = 0;
    uint64_t mincore_elapsed_ns = 0;
    uint64_t resident_pages = 0;
    llama_kv_residency::transition_counts transition;
    std::string event;
    std::string phase;
    bool has_interval = false;
    bool has_transition = false;
    bool valid = false;
    std::string status;
    int error_no = 0;
};

struct page_transition {
    uint32_t entered = 0;
    uint32_t exited = 0;

    uint64_t total() const {
        return uint64_t(entered) + uint64_t(exited);
    }
};

struct summary {
    uint64_t sample_count = 0;
    uint64_t valid_sample_count = 0;
    uint64_t failed_sample_count = 0;
    uint64_t total_entered_pages = 0;
    uint64_t total_exited_pages = 0;
    uint64_t total_observed_transitions = 0;
    uint64_t pages_with_observed_transition = 0;
    uint64_t pages_with_repeated_transition = 0;
    uint64_t repeat_transition_excess = 0;
    uint64_t max_observed_transitions_per_page = 0;
    uint64_t mincore_elapsed_ns_total = 0;
    uint64_t max_mincore_elapsed_ns = 0;
};

inline uint64_t parse_interval_ms(const char * value, uint64_t fallback = 1000) {
    if (!value || !value[0] || value[0] == '-') {
        return fallback;
    }
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (!end || *end != '\0' || parsed == 0 || parsed > std::numeric_limits<uint32_t>::max()) {
        return fallback;
    }
    return static_cast<uint64_t>(parsed);
}

class accumulator {
public:
    explicit accumulator(uint64_t page_count) : page_count_(page_count), transitions_(page_count) {
    }

    void observe(const char * event, const char * phase, scan_result result) {
        sample current;
        current.sample_id = samples_.size() + 1;
        current.timestamp_ns = result.timestamp_ns;
        current.mincore_elapsed_ns = result.mincore_elapsed_ns;
        current.event = event ? event : "";
        current.phase = phase ? phase : "";
        current.valid = result.valid && result.bitmap.page_count == page_count_;
        current.status = result.status;
        current.error_no = result.error_no;
        if (last_timestamp_ns_ > 0 && current.timestamp_ns >= last_timestamp_ns_) {
            current.has_interval = true;
            current.actual_interval_ns = current.timestamp_ns - last_timestamp_ns_;
        }
        last_timestamp_ns_ = current.timestamp_ns;

        if (!current.valid) {
            if (current.status.empty()) {
                current.status = result.valid ? "bitmap_size_mismatch" : "scan_failed";
            }
            previous_valid_ = false;
            samples_.push_back(std::move(current));
            return;
        }

        current.resident_pages = 0;
        for (uint64_t page = 0; page < page_count_; ++page) {
            current.resident_pages += result.bitmap.get(page);
        }
        if (!have_initial_) {
            initial_ = result.bitmap;
            have_initial_ = true;
        }
        final_ = result.bitmap;

        if (previous_valid_) {
            current.has_transition = llama_kv_residency::calculate_transition(
                    previous_, result.bitmap, current.transition) && current.transition.valid();
            if (!current.has_transition) {
                current.valid = false;
                current.status = "transition_invariant_failed";
                previous_valid_ = false;
                samples_.push_back(std::move(current));
                return;
            }
            for (uint64_t page = 0; page < page_count_; ++page) {
                const bool before = previous_.get(page);
                const bool after = result.bitmap.get(page);
                if (!before && after) {
                    transitions_[static_cast<size_t>(page)].entered++;
                } else if (before && !after) {
                    transitions_[static_cast<size_t>(page)].exited++;
                }
            }
        }
        previous_ = std::move(result.bitmap);
        previous_valid_ = true;
        samples_.push_back(std::move(current));
    }

    const std::vector<sample> & samples() const {
        return samples_;
    }

    const std::vector<page_transition> & page_transitions() const {
        return transitions_;
    }

    const llama_kv_residency::bitmap & initial_bitmap() const {
        return initial_;
    }

    const llama_kv_residency::bitmap & final_bitmap() const {
        return final_;
    }

    bool has_initial_bitmap() const {
        return have_initial_;
    }

    summary calculate_summary() const {
        summary result;
        result.sample_count = samples_.size();
        for (const auto & item : samples_) {
            result.valid_sample_count += item.valid;
            result.failed_sample_count += !item.valid;
            result.mincore_elapsed_ns_total += item.mincore_elapsed_ns;
            result.max_mincore_elapsed_ns = std::max(result.max_mincore_elapsed_ns, item.mincore_elapsed_ns);
            if (item.has_transition) {
                result.total_entered_pages += item.transition.entered_pages;
                result.total_exited_pages += item.transition.exited_pages;
            }
        }
        result.total_observed_transitions = result.total_entered_pages + result.total_exited_pages;
        for (const auto & page : transitions_) {
            const uint64_t count = page.total();
            result.pages_with_observed_transition += count > 0;
            result.pages_with_repeated_transition += count >= 2;
            result.repeat_transition_excess += count > 1 ? count - 1 : 0;
            result.max_observed_transitions_per_page = std::max(
                    result.max_observed_transitions_per_page, count);
        }
        return result;
    }

private:
    uint64_t page_count_ = 0;
    uint64_t last_timestamp_ns_ = 0;
    bool previous_valid_ = false;
    bool have_initial_ = false;
    llama_kv_residency::bitmap previous_;
    llama_kv_residency::bitmap initial_;
    llama_kv_residency::bitmap final_;
    std::vector<sample> samples_;
    std::vector<page_transition> transitions_;
};

class sampling_worker {
public:
    using scan_callback = std::function<scan_result()>;

    sampling_worker(uint64_t page_count, uint64_t interval_ms, scan_callback scan) :
        interval_ms_(interval_ms), scan_(std::move(scan)), observations_(page_count) {
    }

    ~sampling_worker() {
        stop_without_sample();
    }

    sampling_worker(const sampling_worker &) = delete;
    sampling_worker & operator=(const sampling_worker &) = delete;

    bool start(const char * event, const char * phase) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (thread_.joinable()) {
            return false;
        }
        pending_ = { event ? event : "", phase ? phase : "", phase ? phase : "", false, ++requested_generation_ };
        running_ = true;
        thread_ = std::thread([this] { run(); });
        completed_.wait(lock, [&] { return completed_generation_ >= requested_generation_ || !running_; });
        return completed_generation_ >= requested_generation_;
    }

    bool boundary(const char * event, const char * sample_phase, const char * following_phase) {
        return request_sample(event, sample_phase, following_phase, false);
    }

    bool stop(const char * event, const char * phase) {
        const bool sampled = request_sample(event, phase, phase, true);
        if (thread_.joinable()) {
            thread_.join();
        }
        return sampled;
    }

    void stop_without_sample() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!thread_.joinable()) {
                return;
            }
            abort_ = true;
            wake_.notify_all();
        }
        thread_.join();
    }

    const accumulator & observations() const {
        return observations_;
    }

private:
    struct command {
        std::string event;
        std::string sample_phase;
        std::string following_phase;
        bool stop = false;
        uint64_t generation = 0;
    };

    bool request_sample(const char * event, const char * sample_phase, const char * following_phase, bool stop) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!thread_.joinable() || !running_) {
            return false;
        }
        const uint64_t generation = ++requested_generation_;
        pending_ = { event ? event : "", sample_phase ? sample_phase : "",
            following_phase ? following_phase : "", stop, generation };
        wake_.notify_all();
        completed_.wait(lock, [&] { return completed_generation_ >= generation || !running_; });
        return completed_generation_ >= generation;
    }

    void run() {
        std::unique_lock<std::mutex> lock(mutex_);
        auto next_periodic = std::chrono::steady_clock::now() + std::chrono::milliseconds(interval_ms_);
        while (!abort_) {
            command current;
            if (pending_.generation > completed_generation_) {
                current = pending_;
            } else if (wake_.wait_until(lock, next_periodic, [&] {
                    return abort_ || pending_.generation > completed_generation_;
                })) {
                if (abort_) {
                    break;
                }
                current = pending_;
            } else {
                current.event = "PERIODIC";
                current.sample_phase = current_phase_;
            }

            lock.unlock();
            scan_result result = scan_();
            observations_.observe(current.event.c_str(), current.sample_phase.c_str(), std::move(result));
            lock.lock();

            if (current.generation > 0) {
                current_phase_ = current.following_phase;
                completed_generation_ = current.generation;
                completed_.notify_all();
            }
            next_periodic = std::chrono::steady_clock::now() + std::chrono::milliseconds(interval_ms_);
            if (current.stop) {
                break;
            }
        }
        running_ = false;
        completed_.notify_all();
    }

    uint64_t interval_ms_ = 1000;
    scan_callback scan_;
    accumulator observations_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable completed_;
    std::thread thread_;
    command pending_;
    std::string current_phase_;
    uint64_t requested_generation_ = 0;
    uint64_t completed_generation_ = 0;
    bool running_ = false;
    bool abort_ = false;
};

} // namespace llama_kv_repeated_activity

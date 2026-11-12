#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

static void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct csv_table {
    std::vector<std::map<std::string, std::string>> rows;
};

static std::vector<std::string> split_csv(const std::string & line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                field.push_back('"');
                ++i;
            } else {
                quoted = !quoted;
            }
        } else if (c == ',' && !quoted) {
            fields.push_back(field);
            field.clear();
        } else {
            field.push_back(c);
        }
    }
    fields.push_back(field);
    return fields;
}

static csv_table read_csv(const fs::path & path) {
    std::ifstream input(path);
    require(input.good(), "missing CSV: " + path.string());
    std::string line;
    require(bool(std::getline(input, line)), "empty CSV: " + path.string());
    const auto header = split_csv(line);
    csv_table table;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const auto values = split_csv(line);
        require(values.size() == header.size(), "malformed CSV row in " + path.string());
        std::map<std::string, std::string> row;
        for (size_t i = 0; i < header.size(); ++i) {
            row.emplace(header[i], values[i]);
        }
        table.rows.push_back(std::move(row));
    }
    return table;
}

static const std::map<std::string, std::string> & find_row(
        const csv_table & table, uint64_t execution_id, const char * access) {
    const std::string id = std::to_string(execution_id);
    for (const auto & row : table.rows) {
        if (row.at("execution_id") == id && row.at("access_kind") == access) {
            return row;
        }
    }
    throw std::runtime_error("missing row execution=" + id + " access=" + std::string(access));
}

struct tensor_fixture {
    static constexpr size_t buffer_size = 16*1024*1024;

    size_t page_size = 4096;
    void * allocation = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_context * ctx = nullptr;

    tensor_fixture() {
        const long system_page_size = sysconf(_SC_PAGESIZE);
        if (system_page_size > 0) {
            page_size = size_t(system_page_size);
        }
        require(posix_memalign(&allocation, page_size, buffer_size + page_size) == 0, "posix_memalign failed");
        void * const base = static_cast<char *>(allocation) + 64;
        buffer = ggml_backend_cpu_buffer_from_ptr(base, buffer_size);
        require(buffer != nullptr, "external CPU buffer allocation failed");

        ggml_init_params params = {
            /* .mem_size   = */ 16*1024*1024,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ctx = ggml_init(params);
        require(ctx != nullptr, "ggml_init failed");
    }

    ~tensor_fixture() {
        if (ctx) {
            ggml_free(ctx);
        }
        if (buffer) {
            ggml_backend_buffer_free(buffer);
        }
        std::free(allocation);
    }

    void allocate_all() {
        ggml_tallocr allocator = ggml_tallocr_new(buffer);
        for (ggml_tensor * tensor = ggml_get_first_tensor(ctx); tensor; tensor = ggml_get_next_tensor(ctx, tensor)) {
            if (tensor->buffer) {
                continue;
            }
            const ggml_status status = tensor->view_src ?
                ggml_backend_view_init(tensor) : ggml_tallocr_alloc(&allocator, tensor);
            require(status == GGML_STATUS_SUCCESS, std::string("tensor allocation failed: ") + tensor->name);
        }
    }
};

static ggml_cpu_attn_diag_link make_link(
        uint64_t generation, uint64_t execution, uint64_t call, int64_t ubatch,
        const char * stage = "PREFILL", int64_t request = 17) {
    ggml_cpu_attn_diag_link link = {};
    link.graph_generation = generation;
    link.execution_id = execution;
    link.call_id = call;
    link.request_id = request;
    link.task_id = 23;
    link.slot_id = 0;
    link.ubatch_idx = ubatch;
    link.layer = -1;
    link.stage = stage;
    link.input_kind = "TOKENS";
    return link;
}

static void compute_graph(ggml_cgraph * graph, int n_threads = 4) {
    ggml_threadpool_params params = ggml_threadpool_params_default(n_threads);
    ggml_threadpool * pool = ggml_threadpool_new(&params);
    require(pool != nullptr, "threadpool allocation failed");
    ggml_cplan plan = ggml_graph_plan(graph, n_threads, pool);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.empty() ? nullptr : work.data();
    const ggml_status status = ggml_graph_compute(graph, &plan);
    ggml_threadpool_free(pool);
    require(status == GGML_STATUS_SUCCESS, "graph compute failed");
}

static void fill_f16(ggml_tensor * tensor, float value) {
    std::vector<float> source(size_t(ggml_nelements(tensor)), value);
    ggml_cpu_fp32_to_fp16(source.data(), static_cast<ggml_fp16_t *>(tensor->data), int64_t(source.size()));
}

static void fill_f32(ggml_tensor * tensor, float value) {
    std::fill_n(static_cast<float *>(tensor->data), size_t(ggml_nelements(tensor)), value);
}

static void run_set_rows(
        uint64_t generation, uint64_t execution, const char * stage = "PREFILL", int64_t request = 17,
        bool expect_binding = true, bool use_v = false) {
    tensor_fixture fixture;
    constexpr int64_t row_elements = 2048;
    constexpr int64_t cells = 4;
    constexpr int64_t writes = 4;

    ggml_tensor * cache = ggml_new_tensor_2d(fixture.ctx, GGML_TYPE_F16, row_elements, cells);
    ggml_tensor * source = ggml_new_tensor_2d(fixture.ctx, GGML_TYPE_F16, row_elements, writes);
    ggml_tensor * indices = ggml_new_tensor_1d(fixture.ctx, GGML_TYPE_I32, writes);
    ggml_tensor * out = ggml_set_rows(fixture.ctx, cache, source, indices);
    ggml_cgraph * graph = ggml_new_graph(fixture.ctx);
    ggml_build_forward_expand(graph, out);
    fixture.allocate_all();

    require(reinterpret_cast<uintptr_t>(cache->data)%fixture.page_size == 64,
            "synthetic KV tensor is not 64-byte page-misaligned");
    fill_f16(cache, 0.0f);
    fill_f16(source, 1.0f);
    const int32_t cell_indices[writes] = { 0, 2, 2, 3 };
    std::memcpy(indices->data, cell_indices, sizeof(cell_indices));

    ggml_cpu_kv_coverage_register_tensor(
        cache, 7, use_v ? GGML_CPU_KV_COVERAGE_TENSOR_V : GGML_CPU_KV_COVERAGE_TENSOR_K);
    ggml_cpu_kv_coverage_register_node(
        out, generation, 7, use_v ? GGML_CPU_KV_COVERAGE_NODE_V_WRITE : GGML_CPU_KV_COVERAGE_NODE_K_WRITE);
    const int bound = ggml_cpu_attn_diag_bind_graph(graph, make_link(generation, execution, 71, 2, stage, request));
    require(bound == (expect_binding ? 1 : 0), "unexpected SET_ROWS binding count");
    compute_graph(graph);
    ggml_cpu_kv_coverage_unregister_tensor(cache);
}

static void initialize_mask(ggml_tensor * mask, const std::set<std::pair<int64_t, int64_t>> & valid) {
    std::vector<float> values(size_t(ggml_nelements(mask)), -INFINITY);
    for (const auto & [key, query] : valid) {
        values[size_t(query*mask->ne[0] + key)] = 0.0f;
    }
    ggml_cpu_fp32_to_fp16(values.data(), static_cast<ggml_fp16_t *>(mask->data), int64_t(values.size()));
}

static void run_flash(
        uint64_t generation, uint64_t execution, int64_t n_query, int64_t n_kv,
        int64_t n_query_heads, int64_t n_kv_heads,
        const std::set<std::pair<int64_t, int64_t>> & valid, const char * stage) {
    tensor_fixture fixture;
    constexpr int64_t head_size = 64;

    ggml_tensor * q = ggml_new_tensor_4d(fixture.ctx, GGML_TYPE_F32, head_size, n_query, n_query_heads, 1);
    ggml_tensor * k = ggml_new_tensor_4d(fixture.ctx, GGML_TYPE_F16, head_size, n_kv, n_kv_heads, 1);
    ggml_tensor * v = ggml_new_tensor_4d(fixture.ctx, GGML_TYPE_F16, head_size, n_kv, n_kv_heads, 1);
    ggml_tensor * mask = ggml_new_tensor_4d(fixture.ctx, GGML_TYPE_F16, n_kv, n_query, 1, 1);
    ggml_tensor * out = ggml_flash_attn_ext(fixture.ctx, q, k, v, mask, 1.0f/std::sqrt(float(head_size)), 0.0f, 0.0f);
    ggml_cgraph * graph = ggml_new_graph(fixture.ctx);
    ggml_build_forward_expand(graph, out);
    fixture.allocate_all();

    fill_f32(q, 0.25f);
    fill_f16(k, 0.5f);
    fill_f16(v, 0.75f);
    initialize_mask(mask, valid);

    ggml_cpu_kv_coverage_register_tensor(k, 11, GGML_CPU_KV_COVERAGE_TENSOR_K);
    ggml_cpu_kv_coverage_register_tensor(v, 11, GGML_CPU_KV_COVERAGE_TENSOR_V);
    ggml_cpu_attn_diag_register_node(out, generation, 11);
    require(ggml_cpu_attn_diag_bind_graph(graph, make_link(generation, execution, 111, 3, stage)) == 1,
            "Flash Attention node was not bound");
    compute_graph(graph);
    ggml_cpu_kv_coverage_unregister_tensor(k);
    ggml_cpu_kv_coverage_unregister_tensor(v);
}

static void run_general_unsupported(uint64_t generation, uint64_t execution) {
    tensor_fixture fixture;
    ggml_tensor * a = ggml_new_tensor_2d(fixture.ctx, GGML_TYPE_F32, 16, 16);
    ggml_tensor * b = ggml_new_tensor_2d(fixture.ctx, GGML_TYPE_F32, 16, 1);
    ggml_tensor * out = ggml_mul_mat(fixture.ctx, a, b);
    ggml_cgraph * graph = ggml_new_graph(fixture.ctx);
    ggml_build_forward_expand(graph, out);
    fixture.allocate_all();
    fill_f32(a, 0.5f);
    fill_f32(b, 0.25f);
    ggml_cpu_kv_coverage_register_node(out, generation, 13, GGML_CPU_KV_COVERAGE_NODE_UNSUPPORTED_GENERAL);
    require(ggml_cpu_attn_diag_bind_graph(graph, make_link(generation, execution, 131, 4, "DECODE")) == 1,
            "GENERAL node was not bound");
    compute_graph(graph);
}

static void run_backend_unsupported(uint64_t generation, uint64_t execution) {
    tensor_fixture fixture;
    ggml_tensor * cache = ggml_new_tensor_2d(fixture.ctx, GGML_TYPE_F16, 64, 4);
    ggml_tensor * source = ggml_new_tensor_2d(fixture.ctx, GGML_TYPE_F16, 64, 1);
    ggml_tensor * indices = ggml_new_tensor_1d(fixture.ctx, GGML_TYPE_I32, 1);
    ggml_tensor * out = ggml_set_rows(fixture.ctx, cache, source, indices);
    ggml_cgraph * graph = ggml_new_graph(fixture.ctx);
    ggml_build_forward_expand(graph, out);
    fixture.allocate_all();
    fill_f16(cache, 0.0f);
    fill_f16(source, 1.0f);
    *static_cast<int32_t *>(indices->data) = 2;

    ggml_cpu_kv_coverage_register_tensor(cache, 14, GGML_CPU_KV_COVERAGE_TENSOR_K);
    ggml_cpu_kv_coverage_register_node(out, generation, 14, GGML_CPU_KV_COVERAGE_NODE_K_WRITE);
    ggml_cpu_kv_coverage_mark_unsupported_backend(out, "SYNTHETIC");
    require(ggml_cpu_attn_diag_bind_graph(graph, make_link(generation, execution, 141, 5, "PREFILL")) == 1,
            "unsupported backend node was not bound");
    compute_graph(graph);
    ggml_cpu_kv_coverage_unregister_tensor(cache);
}

static void set_output_environment(const fs::path & directory, bool enabled) {
    const fs::path layout = directory/"layout.csv";
    const fs::path summary = directory/"summary.csv";
    const fs::path intervals = directory/"intervals.csv";
    setenv("LLAMA_KV_PAGE_COVERAGE_LAYOUT_PATH", layout.c_str(), 1);
    setenv("LLAMA_KV_PAGE_COVERAGE_SUMMARY_PATH", summary.c_str(), 1);
    setenv("LLAMA_KV_PAGE_COVERAGE_INTERVALS_PATH", intervals.c_str(), 1);
    if (enabled) {
        setenv("LLAMA_KV_PAGE_COVERAGE_DIAGNOSTIC", "1", 1);
    } else {
        unsetenv("LLAMA_KV_PAGE_COVERAGE_DIAGNOSTIC");
    }
}

static void test_off(const fs::path & directory) {
    set_output_environment(directory, false);
    ggml_cpu_kv_coverage_reset_for_test();
    run_set_rows(1, 100, "PREFILL", 17, false);
    ggml_cpu_kv_coverage_flush();
    require(!fs::exists(directory/"layout.csv"), "OFF mode wrote a layout CSV");
    require(!fs::exists(directory/"summary.csv"), "OFF mode wrote a summary CSV");
    require(!fs::exists(directory/"intervals.csv"), "OFF mode wrote an intervals CSV");
}

static void test_on(const fs::path & directory) {
    set_output_environment(directory, true);
    ggml_cpu_kv_coverage_reset_for_test();

    run_set_rows(1, 101);
    ggml_cpu_kv_coverage_flush();

    run_set_rows(8, 108, "PREFILL", 17, true, true);
    ggml_cpu_kv_coverage_flush();

    run_flash(2, 102, 1, 16, 4, 2, { { 3, 0 }, { 7, 0 } }, "DECODE");
    ggml_cpu_kv_coverage_flush();

    run_flash(3, 103, 1, 512, 4, 2, { { 0, 0 }, { 127, 0 }, { 128, 0 }, { 511, 0 } }, "DECODE");
    ggml_cpu_kv_coverage_flush();

    run_flash(4, 104, 64, 128, 1, 1, { { 65, 0 } }, "PREFILL");
    ggml_cpu_kv_coverage_flush();

    run_general_unsupported(5, 105);
    ggml_cpu_kv_coverage_flush();

    run_backend_unsupported(7, 107);
    ggml_cpu_kv_coverage_flush();

    // A compatibility/probe graph may execute the same CPU store, but an invalid
    // request/stage link must never be attributed to user Prefill or Decode.
    run_set_rows(6, 106, "UNKNOWN", -1);
    ggml_cpu_kv_coverage_flush();

    const csv_table summary = read_csv(directory/"summary.csv");
    const csv_table layout = read_csv(directory/"layout.csv");
    const csv_table intervals = read_csv(directory/"intervals.csv");
    require(!layout.rows.empty(), "no accessed KV layouts were emitted");
    require(!intervals.rows.empty(), "no page intervals were emitted");

    const auto & write = find_row(summary, 101, "K_WRITE");
    require(write.at("kernel_path") == "SET_ROWS", "wrong SET_ROWS path");
    require(write.at("source_range_count") == "4", "SET_ROWS did not retain repeated cell stores");
    require(write.at("source_bytes") == std::to_string(4*4096), "wrong SET_ROWS source byte count");
    require(write.at("unique_pages") == "5", "SET_ROWS page union did not handle +64 boundary sharing");
    require(write.at("key_index_rle") == "0;2-3", "SET_ROWS ignored actual destination cell indices");

    const auto & v_write = find_row(summary, 108, "V_WRITE");
    require(v_write.at("kernel_path") == "SET_ROWS" && v_write.at("source_bytes") == std::to_string(4*4096) &&
            v_write.at("unique_pages") == "5" && v_write.at("key_index_rle") == "0;2-3",
            "V_WRITE was not separated from K_WRITE");

    const auto & one_k = find_row(summary, 102, "K_READ");
    const auto & one_v = find_row(summary, 102, "V_READ");
    require(one_k.at("kernel_path") == "ONE_CHUNK" && one_v.at("kernel_path") == "ONE_CHUNK",
            "small decode did not use ONE_CHUNK");
    require(one_k.at("key_index_rle") == "3;7" && one_v.at("key_index_rle") == "3;7",
            "ONE_CHUNK recorded mask-skipped keys");
    require(one_k.at("source_bytes") == std::to_string(4*2*128) &&
            one_v.at("source_bytes") == std::to_string(4*2*128),
            "ONE_CHUNK source byte count is wrong");
    require(one_k.at("kv_head_mask") == "3", "ONE_CHUNK did not merge both KV heads");

    const auto & split_k = find_row(summary, 103, "K_READ");
    const auto & split_v = find_row(summary, 103, "V_READ");
    require(split_k.at("kernel_path") == "SPLIT_KV" && split_v.at("kernel_path") == "SPLIT_KV",
            "large decode did not use SPLIT_KV");
    require(split_k.at("key_index_rle") == "0;127-128;511" &&
            split_v.at("key_index_rle") == "0;127-128;511",
            "SPLIT_KV recorded mask-skipped keys");
    require(split_k.at("source_bytes") == std::to_string(4*4*128) &&
            split_v.at("source_bytes") == std::to_string(4*4*128),
            "SPLIT_KV worker records were not merged correctly");

    const auto & tiled_k = find_row(summary, 104, "K_READ");
    const auto & tiled_v = find_row(summary, 104, "V_READ");
    require(tiled_k.at("kernel_path") == "TILED" && tiled_v.at("kernel_path") == "TILED",
            "prefill did not use TILED");
    require(tiled_k.at("key_index_rle") == "64-127" && tiled_v.at("key_index_rle") == "64-127",
            "TILED did not include the packed masked portion of a partial tile");
    require(tiled_k.at("source_bytes") == std::to_string(64*128) &&
            tiled_v.at("source_bytes") == std::to_string(64*128),
            "TILED source byte count is wrong");
    require(tiled_k.at("tile_coordinates").find(":k64+64") != std::string::npos,
            "TILED coordinates were not emitted");

    const auto & unsupported = find_row(summary, 105, "UNSUPPORTED");
    require(unsupported.at("kernel_path") == "GENERAL" && unsupported.at("status") == "unsupported_path",
            "GENERAL was presented as measured coverage");
    require(unsupported.at("source_bytes") == "0" && unsupported.at("unique_pages") == "0" &&
            unsupported.at("dram_traffic") == "not_measured",
            "unsupported/DRAM semantics are incorrect");

    const auto & unsupported_backend = find_row(summary, 107, "UNSUPPORTED");
    require(unsupported_backend.at("kernel_path") == "BACKEND:SYNTHETIC" &&
            unsupported_backend.at("status") == "unsupported_path" &&
            unsupported_backend.at("source_bytes") == "0" && unsupported_backend.at("unique_pages") == "0",
            "unverified backend was presented as measured coverage");

    for (const auto & row : summary.rows) {
        require(row.at("execution_id") != "106", "INIT/compatibility access was attributed to the user request");
        require(row.at("dram_traffic") == "not_measured", "DRAM traffic was accidentally claimed");
    }
}

int main(int argc, char ** argv) {
    try {
        const bool off = argc == 2 && std::string(argv[1]) == "--off";
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        const fs::path directory = fs::temp_directory_path()/
            (std::string("llama-kv-page-coverage-") + std::to_string(getpid()) + "-" +
             std::to_string(nonce) + (off ? "-off" : "-on"));
        fs::create_directories(directory);
        if (off) {
            test_off(directory);
        } else {
            test_on(directory);
        }
        fs::remove_all(directory);
        return 0;
    } catch (const std::exception & error) {
        fprintf(stderr, "test-kv-page-coverage: %s\n", error.what());
        return 1;
    }
}

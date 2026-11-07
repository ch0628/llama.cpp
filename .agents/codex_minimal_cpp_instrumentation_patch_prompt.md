# Codex Prompt — Minimal llama.cpp Phase Instrumentation Patch

You are modifying my local `llama.cpp` checkout for a Raspberry Pi 4 CPU-only VLM profiling experiment.

## Hard constraints

- Target commit: `f3f1a8f27`
- Model path: Qwen3-VL through MTMD
- Pilot assumptions:
  - batch = 1
  - one active request
  - speculative decoding OFF
  - CPU backend only
- Goal is measurement instrumentation only.
- Do NOT change model behavior, batching policy, thread counts, sampling behavior, graph construction, preprocessing semantics, KV-cache behavior, or output content.
- Normal instrumentation mode must NOT insert `llama_synchronize()` into the normal execution path.
- Sync validation must be opt-in only.
- Keep the patch minimal and easy to revert.
- Do not refactor unrelated llama.cpp code.
- Do not add external dependencies.

Before editing:
1. Confirm `git rev-parse HEAD`.
2. Locate the exact functions listed below in this checkout.
3. If any function/path differs from the expected commit, stop and report the mismatch instead of guessing.

---

## 1. Timestamp source

Use Linux `CLOCK_MONOTONIC_RAW`, nanoseconds.

Minimal helper behavior:

```cpp
uint64_t phase_now_ns();
```

Requirements:
- timestamp acquisition only
- no file I/O inside the timestamp helper
- no flush inside hot paths

All phase events belong to:

```text
clock_domain = pi_monotonic_raw
```

---

## 2. Event logger

Implement the smallest practical in-process event logger suitable for Pilot A.

Requirements:
- disabled unless explicitly enabled
- lightweight in-memory buffering
- one record per event
- write/flush outside the timed hot path when practical
- no dependency on wall-clock time
- preserve event ordering with a monotonically increasing `seq`
- support repeated span pairing with `span_id`
- single active request is sufficient for Pilot A; do not over-engineer multi-tenant support

Prefer an environment-variable switch rather than adding CLI surface unless the repository already has a clearly better mechanism.

Suggested switches:

```text
LLAMA_PHASE_EVENTS_PATH=/path/to/phase_events.csv
LLAMA_PHASE_SYNC_VALIDATE=0|1
```

If there is a cleaner existing llama.cpp configuration pattern, use it, but explain the choice.

### Raw record fields

The physical serialization may be CSV or JSONL. Prefer the simplest implementation that does not require an external library.

Required logical fields:

```text
schema_version
run_id
seq
ts_ns
clock_domain
component
event_name
event_kind
span_id
task_id
slot_id
item_idx
sub_idx
token_count
token_id
pos_start
pos_end
image_w
image_h
status
note
```

Optional fields may be empty.

For Pilot A, if a globally unique per-request `run_id` is not naturally available in deep MTMD helpers without invasive plumbing:
- do not redesign the request stack;
- use the least invasive existing request/task identifier where available;
- document which fields are unavailable in deep helper events;
- preserve `seq` and one-request Pilot correctness.

---

## 3. Canonical event names to instrument

### Request
- `SERVER_INFERENCE_START`

### Image
- `IMAGE_INPUT_START`
- `IMAGE_INPUT_END`
- `IMAGE_DECODE_START`
- `IMAGE_DECODE_END`

### Text tokenization
- `TEXT_TOKENIZE_START`
- `TEXT_TOKENIZE_END`

### Vision preprocessing
- `VISION_PREPROCESS_START`
- `VISION_PREPROCESS_END`

### Vision encoder / projector
- `VISION_INTERNAL_WARMUP_START`
- `VISION_INTERNAL_WARMUP_END`
- `VISION_GRAPH_PREP_START`
- `VISION_GRAPH_PREP_END`
- `VISION_INPUT_SETUP_START`
- `VISION_INPUT_SETUP_END`
- `VISION_COMPUTE_START`
- `VISION_COMPUTE_END`
- `VISION_EMBED_COPY_START`
- `VISION_EMBED_COPY_END`
- `VISUAL_EMBEDDING_READY`

### Prefill
- `VISUAL_PREFILL_CALL_START`
- `VISUAL_PREFILL_CALL_RETURN`
- `TEXT_PREFILL_CALL_START`
- `TEXT_PREFILL_CALL_RETURN`
- `PREFILL_SUBMISSION_COMPLETE`

### Decode
- `DECODE_START`
- `TOKEN_READY`
- `GENERATED_TOKEN_EVAL_CALL_START`
- `GENERATED_TOKEN_EVAL_CALL_RETURN`
- `DECODE_END`

### Response
- `SERVER_FINAL_RESPONSE_START`
- `SERVER_FINAL_RESPONSE_RETURN`

### Validation-only
- `LLM_SYNC_START`
- `LLM_SYNC_END`

Do NOT emit a fake `PREFILL_COMPUTE_COMPLETE` unless compute completion is actually guaranteed.

---

## 4. Exact intended insertion points

Use the real source in this checkout as the authority. Expected locations:

### A. Server inference start

File:
```text
tools/server/server-context.cpp
```

Function:
```text
launch_slot_with_task(...)
```

Emit `SERVER_INFERENCE_START` after the normal non-child request transitions to `SLOT_STATE_STARTED`.

---

### B. Image input

File:
```text
tools/server/server-common.cpp
```

Function:
```text
handle_media(...)
```

Emit:
- `IMAGE_INPUT_START` immediately before actual acquisition/normalization branch work
- `IMAGE_INPUT_END` after encoded bytes are ready and pushed into `out_files`

The semantic output is encoded image bytes, not pixels.

---

### C. Image decode

File:
```text
tools/mtmd/mtmd-helper.cpp
```

Function:
```text
mtmd_helper_bitmap_init_from_buf(...)
```

Image path only.

Start immediately before:

```cpp
stbi_load_from_memory(...)
```

End after successful bitmap creation / temporary RGB buffer cleanup and before returning the bitmap.

---

### D. Text tokenization

File:
```text
tools/mtmd/mtmd.cpp
```

Function:
```text
mtmd_tokenize_text_internal(...)
```

Emit:
- start before token buffer sizing / first `llama_tokenize()`
- end after final token vector size is known and before return

Record final token count on END.

---

### E. Vision preprocessing

File:
```text
tools/mtmd/mtmd.cpp
```

Function:
```text
mtmd_tokenizer::add_media(...)
```

Wrap the actual:

```cpp
ctx->image_preproc->preprocess(img_u8)
```

call with `VISION_PREPROCESS_START/END`.

Do not time the whole `mtmd_tokenize()` as Vision preprocessing.

---

### F. Vision encoder + projector

Find `clip_encode(...)`.

Expected boundaries:

1. `VISION_INTERNAL_WARMUP_START/END`
   around first-use:
   ```cpp
   clip_model_loader::warmup(...)
   ```

2. `VISION_GRAPH_PREP_START`
   before scheduler reset / graph build

3. `VISION_GRAPH_PREP_END`
   after graph allocation

4. `VISION_INPUT_SETUP_START`
   immediately after graph prep

5. `VISION_INPUT_SETUP_END`
   immediately before:
   ```cpp
   ggml_backend_sched_graph_compute(...)
   ```

6. `VISION_COMPUTE_START/END`
   directly around:
   ```cpp
   ggml_backend_sched_graph_compute(...)
   ```

7. `VISION_EMBED_COPY_START/END`
   around:
   ```cpp
   ggml_backend_tensor_get(...)
   ```
   for the output embedding

8. `VISUAL_EMBEDDING_READY`
   immediately after output embedding is available in the user-side buffer

Do not split Encoder and Projector internally for Pilot A.

---

### G. Visual Prefill

Function:
```text
mtmd_helper_decode_image_chunk(...)
```

For each visual embedding sub-batch, wrap:

```cpp
llama_decode(lctx, batch_embd_view)
```

with:

```text
VISUAL_PREFILL_CALL_START
VISUAL_PREFILL_CALL_RETURN
```

Record if readily available:
- sub-batch index
- token count
- position range
- status

Do not label RETURN as compute-complete.

---

### H. Text Prefill / generated-token evaluation

File:
```text
tools/server/server-context.cpp
```

At the normal server `llama_decode(ctx_tgt, batch_view)` call:

- if the batch corresponds to prompt input (`is_prompt == true`):
  - `TEXT_PREFILL_CALL_START`
  - `TEXT_PREFILL_CALL_RETURN`

- if the batch corresponds to generated-token evaluation (`is_prompt == false`):
  - `GENERATED_TOKEN_EVAL_CALL_START`
  - `GENERATED_TOKEN_EVAL_CALL_RETURN`

Pilot A is single active request / non-speculative, so prefer the simplest correct classification for that case.

Do not silently classify a mixed batch as one phase. If mixed classification is possible at this location, report it and implement a Pilot-safe guard.

---

### I. Prefill semantic transition

In `post_decode(...)`, when:

```cpp
slot.state = SLOT_STATE_GENERATING;
```

is reached after prompt evaluation, emit:

```text
PREFILL_SUBMISSION_COMPLETE
```

This is a semantic/state marker, not guaranteed compute-complete.

---

### J. Decode start

Immediately before the first normal:

```cpp
common_sampler_sample(...)
```

for the request, emit `DECODE_START` once.

---

### K. Token ready

Immediately after:

```cpp
common_sampler_sample(...)
```

returns, emit:

```text
TOKEN_READY
```

Record:
- generated token index
- token id
- slot id where available

---

### L. Decode end

In:

```text
process_token(...)
```

after all stop conditions have been evaluated, when:

```text
slot.has_next_token == false
```

emit `DECODE_END` before return.

---

### M. Final response

In the normal stop path around:

```cpp
send_final_response(slot);
```

emit:

```text
SERVER_FINAL_RESPONSE_START
SERVER_FINAL_RESPONSE_RETURN
```

Do not claim `SERVER_FINAL_RESPONSE_RETURN` is network transmission complete.

---

## 5. Sync-validation mode

Normal mode:
```text
NO explicit llama_synchronize()
```

Validation mode only, controlled by the opt-in switch:

After target LLM `llama_decode()` calls:

```text
CALL_RETURN
→ LLM_SYNC_START
→ llama_synchronize(...)
→ LLM_SYNC_END
```

This is diagnostic only.

Do not add this around Vision `ggml_backend_sched_graph_compute()` for this validation.

---

## 6. Flush behavior

For Pilot A:
- buffer events in memory
- flush at a safe request-completion point, preferably after final-response server work and before request state is irreversibly discarded
- avoid `fflush()` or file open/write on every event

If server lifecycle makes per-request flush unsafe, use the smallest robust alternative and explain it.

---

## 7. Build safety

After editing:

1. Format only the code you touched if the repository has a local formatter convention.
2. Run the narrowest relevant build that produces `llama-server`.
3. Do not modify optimization flags or backend configuration.
4. Do not proceed to performance conclusions.

---

## 8. Required output to me

When finished, report only:

### A. Modified files
Exact paths.

### B. Design choices
- logger location
- enable switch
- serialization
- buffer/flush strategy
- how `run_id`, `task_id`, `slot_id`, `span_id`, `seq` are populated
- any schema field intentionally unavailable in a deep helper

### C. Event coverage
A checklist of every requested event:
```text
implemented / not implemented / reason
```

### D. Sync validation
Where `LLM_SYNC_START/END` was inserted and how it is enabled.

### E. Build result
Exact command and success/failure.

### F. Diff
Run and include:

```bash
git diff --stat
git diff --check
```

Do not hide warnings or failed insertions.

If any requested insertion point cannot be implemented correctly without invasive plumbing, leave that event out and explicitly report it rather than guessing or changing unrelated architecture.

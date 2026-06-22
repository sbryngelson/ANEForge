// ane_e5rt_dispatch.mm - single-op dispatch wrapper over Espresso's e5rt_*
// family, compiled as Objective-C++. Exposes the C ABI that aneforge's
// runtime (aneforge/_runtime.py) binds via ctypes:
//
//   ane_e5rt_program_compile      - compile a MIL into a precompiled ANE op
//   ane_e5rt_program_set_input_fp16 / get_output_fp16
//   ane_e5rt_program_input_buffer / output_buffer  - zero-copy port buffers
//   ane_e5rt_program_execute      - synchronous execute on the op's stream
//   ane_e5rt_program_share_buffer - alias one op's output buffer onto another
//                                   op's input port (resident-state chaining)
//   ane_e5rt_program_release
//   ane_e5rt_compile_check        - compile-only cross-target validation
//
// One compiled op owns one e5rt_execution_stream; encode + prepare are
// deferred to the first execute (see finalize_program).
//
// Build:  sh aneforge/_lib/build.sh
//   (compiles this file into libane_e5rt_dispatch.dylib next to it; must be
//    built on the target Mac since it links Apple's private Espresso runtime.)

#import <Foundation/Foundation.h>
#include <dispatch/dispatch.h>

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sys/stat.h>
#include <errno.h>

extern "C" {
#include "e5rt_api.h"
}

// ---- one-shot symbol loading -------------------------------------------------

static void *fw = NULL;

#define DECL(name) static name##_fn name = NULL
#define LOAD(name)                                                            \
  do {                                                                        \
    if (!name) {                                                              \
      name = (name##_fn)dlsym(fw, #name);                                     \
      if (!name) name = (name##_fn)dlsym(fw, "_" #name);                      \
      if (!name) { fprintf(stderr, "missing sym: %s\n", #name); return -1; }  \
    }                                                                         \
  } while (0)

DECL(e5rt_e5_compiler_config_options_create);
DECL(e5rt_e5_compiler_config_options_release);
DECL(e5rt_e5_compiler_config_options_set_cache_bundle_location);
DECL(e5rt_e5_compiler_create_with_config);
DECL(e5rt_e5_compiler_release);
DECL(e5rt_e5_compiler_options_create);
DECL(e5rt_e5_compiler_options_release);
DECL(e5rt_e5_compiler_options_set_compute_device_types_mask);
DECL(e5rt_e5_compiler_options_set_force_recompilation);
DECL(e5rt_e5_compiler_options_set_segmenter);
DECL(e5rt_e5_compiler_options_set_custom_ane_compiler_options);
DECL(e5rt_e5_compiler_compile);
DECL(e5rt_program_library_create);
DECL(e5rt_program_library_release);
DECL(e5rt_program_library_retain_program_function);
DECL(e5rt_program_function_release);
DECL(e5rt_precompiled_compute_op_create_options_create_with_program_function);
DECL(e5rt_precompiled_compute_op_create_options_release);
DECL(e5rt_precompiled_compute_op_create_options_set_operation_name);
DECL(e5rt_precompiled_compute_op_create_options_set_allocate_intermediate_buffers);
DECL(e5rt_execution_stream_operation_create_precompiled_compute_operation_with_options);
DECL(e5rt_execution_stream_operation_release);
DECL(e5rt_execution_stream_operation_retain_input_port);
DECL(e5rt_execution_stream_operation_retain_output_port);
DECL(e5rt_execution_stream_operation_prepare_op_for_encode);
DECL(e5rt_execution_stream_operation_bind_completion_event);
DECL(e5rt_execution_stream_operation_bind_dependent_events);
DECL(e5rt_io_port_release);
DECL(e5rt_io_port_bind_buffer_object);
DECL(e5rt_buffer_object_alloc);
DECL(e5rt_buffer_object_release);
DECL(e5rt_buffer_object_get_data_ptr);
DECL(e5rt_execution_stream_create);
DECL(e5rt_execution_stream_release);
DECL(e5rt_execution_stream_encode_operation);
DECL(e5rt_execution_stream_execute_sync);
// Espresso provides two stream-submission entrypoints:
//   _e5rt_execution_stream_async_submit  - older, returns "Use submit_async".
//   _e5rt_execution_stream_submit_async  - current; takes (stream, ObjC block).
// We load the newer one. Block signature is `e5rt_error_code_t (^)(void)`.
typedef e5rt_error_code_t (*e5rt_execution_stream_submit_async_fn)(void *stream, id block);
static e5rt_execution_stream_submit_async_fn e5rt_execution_stream_submit_async = NULL;
DECL(e5rt_async_event_create);
DECL(e5rt_async_event_release);
DECL(e5rt_async_event_sync_wait);
DECL(e5rt_async_event_get_last_signaled_value);
DECL(e5rt_async_event_set_active_future_value);

typedef e5rt_error_code_t (*e5rt_execution_stream_reset_fn)(void *stream);
DECL(e5rt_execution_stream_reset);

// Profiling: compiles the program with per-TD perf-event collection so the firmware emits per-TD
// stats at execute (surfaced via the com.apple.ane signposts' perfStatsMask). Env-gated so normal
// runs are unaffected; only ANEFORGE_PROFILE=1 turns it on.
typedef e5rt_error_code_t (*e5rt_e5_compiler_options_set_enable_profiling_fn)(void *obj, int value);
DECL(e5rt_e5_compiler_options_set_enable_profiling);

static int load_all_symbols(void) {
  if (!fw) {
    fw = dlopen("/System/Library/PrivateFrameworks/Espresso.framework/Espresso",
                RTLD_NOW | RTLD_LOCAL);
    if (!fw) { fprintf(stderr, "dlopen Espresso failed: %s\n", dlerror()); return -1; }
  }
  LOAD(e5rt_e5_compiler_config_options_create);
  LOAD(e5rt_e5_compiler_config_options_release);
  LOAD(e5rt_e5_compiler_config_options_set_cache_bundle_location);
  LOAD(e5rt_e5_compiler_create_with_config);
  LOAD(e5rt_e5_compiler_release);
  LOAD(e5rt_e5_compiler_options_create);
  LOAD(e5rt_e5_compiler_options_release);
  LOAD(e5rt_e5_compiler_options_set_compute_device_types_mask);
  LOAD(e5rt_e5_compiler_options_set_force_recompilation);
  LOAD(e5rt_e5_compiler_options_set_segmenter);
  LOAD(e5rt_e5_compiler_compile);
  LOAD(e5rt_program_library_create);
  LOAD(e5rt_program_library_release);
  LOAD(e5rt_program_library_retain_program_function);
  LOAD(e5rt_program_function_release);
  LOAD(e5rt_precompiled_compute_op_create_options_create_with_program_function);
  LOAD(e5rt_precompiled_compute_op_create_options_release);
  LOAD(e5rt_precompiled_compute_op_create_options_set_operation_name);
  LOAD(e5rt_precompiled_compute_op_create_options_set_allocate_intermediate_buffers);
  LOAD(e5rt_execution_stream_operation_create_precompiled_compute_operation_with_options);
  LOAD(e5rt_execution_stream_operation_release);
  LOAD(e5rt_execution_stream_operation_retain_input_port);
  LOAD(e5rt_execution_stream_operation_retain_output_port);
  LOAD(e5rt_execution_stream_operation_prepare_op_for_encode);
  LOAD(e5rt_execution_stream_operation_bind_completion_event);
  LOAD(e5rt_execution_stream_operation_bind_dependent_events);
  LOAD(e5rt_io_port_release);
  LOAD(e5rt_io_port_bind_buffer_object);
  LOAD(e5rt_buffer_object_alloc);
  LOAD(e5rt_buffer_object_release);
  LOAD(e5rt_buffer_object_get_data_ptr);
  LOAD(e5rt_execution_stream_create);
  LOAD(e5rt_execution_stream_release);
  LOAD(e5rt_execution_stream_encode_operation);
  LOAD(e5rt_execution_stream_execute_sync);
  if (!e5rt_execution_stream_submit_async) {
    e5rt_execution_stream_submit_async =
      (e5rt_execution_stream_submit_async_fn)dlsym(fw, "e5rt_execution_stream_submit_async");
    if (!e5rt_execution_stream_submit_async)
      e5rt_execution_stream_submit_async =
        (e5rt_execution_stream_submit_async_fn)dlsym(fw, "_e5rt_execution_stream_submit_async");
    if (!e5rt_execution_stream_submit_async) {
      fprintf(stderr, "missing sym: e5rt_execution_stream_submit_async\n"); return -1;
    }
  }
  LOAD(e5rt_async_event_create);
  LOAD(e5rt_async_event_release);
  LOAD(e5rt_async_event_sync_wait);
  LOAD(e5rt_async_event_get_last_signaled_value);
  LOAD(e5rt_async_event_set_active_future_value);
  LOAD(e5rt_execution_stream_reset);
  // Optional symbols (soft-fail: absence degrades gracefully).
  // NOTE: these symbols resolve WITHOUT the leading underscore on macOS 15+.
  #define SOFTLOAD(name, sym_str) \
    do { if (!name) { \
      name = (name##_fn)dlsym(fw, sym_str); \
      if (!name) name = (name##_fn)dlsym(fw, "_" sym_str); \
    } } while(0)
  SOFTLOAD(e5rt_e5_compiler_options_set_enable_profiling,
           "e5rt_e5_compiler_options_set_enable_profiling");
  #undef SOFTLOAD
  return 0;
}

// ---- internal data model ----------------------------------------------------

typedef struct {
  char *name;
  void *port;
  void *buffer;
  void *data_ptr;
  size_t size_bytes;
} ane_e5rt_port_t;

typedef struct {
  void *library;
  void *function;
  void *op_options;
  void *operation;
  ane_e5rt_port_t *inputs;
  size_t n_inputs;
  ane_e5rt_port_t *outputs;
  size_t n_outputs;
  // optional per-op completion event (bound before encode)
  void *completion_event;     // NULL if none bound
  int encoded_once;           // set true after the op has been encoded at least once
} ane_e5rt_op_t;

// Async-state tracking for the experimental execute_async / wait_for_completion
// surface. The completion block (built in execute_async) only flips `completed`
// when the submit body returns; ANE-strict callers wait on `final_event`.
typedef struct {
  uint64_t final_event_value_before; // last_signaled_value before submission
  pthread_mutex_t mutex;
  pthread_cond_t  cond;
  int submitted;              // 1 = submit_async issued, 0 otherwise
  int completed;              // 1 = completion block fired
  int completion_rc;
} ane_e5rt_async_state_t;

struct ane_e5rt_program {
  // The single compiled op (ops[0]). Held as a small vector for allocation
  // symmetry; the bound surface only ever uses ops[0].
  ane_e5rt_op_t *ops;
  size_t n_ops;
  size_t cap_ops;

  // Compile-time defaults so the experimental add_op() can re-use them.
  char cache_dir[1024];
  char mil_path[1024];
  uint64_t device_mask;

  void *stream;

  // Alias to the op's completion_event (set at finalize; not owned here).
  void *final_event;

  // Set on first encode; blocks share_buffer once the graph is committed.
  int finalized;

  // Async state (experimental execute_async / wait_for_completion surface).
  ane_e5rt_async_state_t async;

  // Port specs saved from compile (names strdup'd); freed on release/error.
  char **input_port_names;
  size_t *input_port_sizes;
  size_t n_input_specs;
  char **output_port_names;
  size_t *output_port_sizes;
  size_t n_output_specs;
};

typedef struct ane_e5rt_program ane_e5rt_program_t;

static ane_e5rt_op_t *prog_push_op(ane_e5rt_program_t *prog) {
  if (prog->n_ops + 1 > prog->cap_ops) {
    size_t new_cap = prog->cap_ops ? prog->cap_ops * 2 : 4;
    void *p = realloc(prog->ops, new_cap * sizeof(ane_e5rt_op_t));
    if (!p) return NULL;
    prog->ops = (ane_e5rt_op_t*)p;
    memset(prog->ops + prog->cap_ops, 0,
           (new_cap - prog->cap_ops) * sizeof(ane_e5rt_op_t));
    prog->cap_ops = new_cap;
  }
  return &prog->ops[prog->n_ops++];
}

static int port_alloc_and_bind(void *operation, const char *name,
                               size_t size_bytes, int is_input,
                               ane_e5rt_port_t *out_port) {
  e5rt_error_code_t err;
  void *port = NULL;
  if (is_input) {
    err = e5rt_execution_stream_operation_retain_input_port(operation, name, &port);
  } else {
    err = e5rt_execution_stream_operation_retain_output_port(operation, name, &port);
  }
  if (err || !port) {
    fprintf(stderr, "retain_%s_port(%s) err=%lld\n",
            is_input ? "input" : "output", name, (long long)err);
    return -1;
  }
  size_t buf_bytes = size_bytes < 64 ? 64 : ((size_bytes + 63) & ~(size_t)63);
  void *buffer = NULL;
  err = e5rt_buffer_object_alloc(&buffer, buf_bytes, 0);
  if (err || !buffer) {
    fprintf(stderr, "buffer_alloc(%zu, type=0) err=%lld\n",
            buf_bytes, (long long)err);
    e5rt_io_port_release(&port);
    return -1;
  }
  void *data_ptr = NULL;
  err = e5rt_buffer_object_get_data_ptr(buffer, &data_ptr);
  if (err || !data_ptr) {
    fprintf(stderr, "get_data_ptr err=%lld\n", (long long)err);
    e5rt_buffer_object_release(&buffer);
    e5rt_io_port_release(&port);
    return -1;
  }
  memset(data_ptr, 0, buf_bytes);
  err = e5rt_io_port_bind_buffer_object(port, buffer);
  if (err) {
    fprintf(stderr, "bind_buffer err=%lld\n", (long long)err);
    e5rt_buffer_object_release(&buffer);
    e5rt_io_port_release(&port);
    return -1;
  }
  out_port->name = strdup(name);
  out_port->port = port;
  out_port->buffer = buffer;
  out_port->data_ptr = data_ptr;
  out_port->size_bytes = size_bytes;
  return 0;
}

static void op_free_ports_and_op(ane_e5rt_op_t *op) {
  for (size_t i = 0; i < op->n_inputs; i++) {
    free(op->inputs[i].name);
    if (op->inputs[i].buffer) e5rt_buffer_object_release(&op->inputs[i].buffer);
    if (op->inputs[i].port) e5rt_io_port_release(&op->inputs[i].port);
  }
  free(op->inputs);
  op->inputs = NULL;
  op->n_inputs = 0;
  for (size_t i = 0; i < op->n_outputs; i++) {
    free(op->outputs[i].name);
    if (op->outputs[i].buffer) e5rt_buffer_object_release(&op->outputs[i].buffer);
    if (op->outputs[i].port) e5rt_io_port_release(&op->outputs[i].port);
  }
  free(op->outputs);
  op->outputs = NULL;
  op->n_outputs = 0;
  if (op->completion_event) e5rt_async_event_release(&op->completion_event);
  op->completion_event = NULL;
  if (op->operation) e5rt_execution_stream_operation_release(&op->operation);
  op->operation = NULL;
  if (op->op_options) e5rt_precompiled_compute_op_create_options_release(&op->op_options);
  op->op_options = NULL;
  if (op->function) e5rt_program_function_release(&op->function);
  op->function = NULL;
  if (op->library) e5rt_program_library_release(&op->library);
  op->library = NULL;
}

// Compile a MIL using the saved compile-time defaults; produces library+function+
// op_options+operation. Caller owns ownership transfer into ane_e5rt_op_t.
static int compile_and_build_op(const char *mil_path, const char *cache_dir,
                                uint64_t device_mask, ane_e5rt_op_t *op_out) {
  void *config = NULL, *compiler = NULL, *options = NULL;
  void *library = NULL, *function = NULL;
  void *op_options = NULL, *operation = NULL;
  e5rt_error_code_t err;

  err = e5rt_e5_compiler_config_options_create(&config);
  if (err || !config) { fprintf(stderr, "config_create err=%lld\n", (long long)err); goto fail; }
  e5rt_e5_compiler_config_options_set_cache_bundle_location(config, cache_dir);

  err = e5rt_e5_compiler_create_with_config(&compiler, config);
  if (err || !compiler) { fprintf(stderr, "compiler_create err=%lld\n", (long long)err); goto fail; }

  err = e5rt_e5_compiler_options_create(&options);
  if (err || !options) goto fail;
  e5rt_e5_compiler_options_set_compute_device_types_mask(options, device_mask);
  e5rt_e5_compiler_options_set_force_recompilation(options, 1);
  e5rt_e5_compiler_options_set_segmenter(options, "graph");
  // opt-in per-TD profiling (firmware perf events), gated by ANEFORGE_PROFILE=1
  if (getenv("ANEFORGE_PROFILE") && e5rt_e5_compiler_options_set_enable_profiling)
    e5rt_e5_compiler_options_set_enable_profiling(options, 1);

  err = e5rt_e5_compiler_compile(compiler, mil_path, options, &library);
  if (err || !library) {
    fprintf(stderr, "compile(mask=0x%llx) err=%lld\n",
            (unsigned long long)device_mask, (long long)err);
    goto fail;
  }

  err = e5rt_program_library_retain_program_function(library, "main", &function);
  if (err || !function) { fprintf(stderr, "retain_function err=%lld\n", (long long)err); goto fail; }

  err = e5rt_precompiled_compute_op_create_options_create_with_program_function(&op_options, function);
  if (err || !op_options) { fprintf(stderr, "op_options_create err=%lld\n", (long long)err); goto fail; }
  e5rt_precompiled_compute_op_create_options_set_operation_name(op_options, "main");
  e5rt_precompiled_compute_op_create_options_set_allocate_intermediate_buffers(op_options, 1);

  err = e5rt_execution_stream_operation_create_precompiled_compute_operation_with_options(&operation, op_options);
  if (err || !operation) { fprintf(stderr, "op_create err=%lld\n", (long long)err); goto fail; }

  op_out->library = library;
  op_out->function = function;
  op_out->op_options = op_options;
  op_out->operation = operation;
  // The compiler/config/options are not needed past here.
  e5rt_e5_compiler_options_release(&options);
  e5rt_e5_compiler_release(&compiler);
  e5rt_e5_compiler_config_options_release(&config);
  return 0;

fail:
  if (operation) e5rt_execution_stream_operation_release(&operation);
  if (op_options) e5rt_precompiled_compute_op_create_options_release(&op_options);
  if (function) e5rt_program_function_release(&function);
  if (library) e5rt_program_library_release(&library);
  if (options) e5rt_e5_compiler_options_release(&options);
  if (compiler) e5rt_e5_compiler_release(&compiler);
  if (config) e5rt_e5_compiler_config_options_release(&config);
  return -1;
}

// ---- compile-only cross-target check (additive; does NOT touch the hot path) ----
// Compiles a MIL to a library for a given TargetArchitecture WITHOUT creating the
// device-bound precompiled operation. This is how we validate that a graph compiles
// for another ANE family (e.g. h13/M1) from this host: the compile step (produces the
// library) is separable from the device-load step (create_precompiled_compute_operation),
// which would fail for a cross-target library on the host silicon. Returns 0 if a library
// was produced for the requested target, nonzero otherwise. custom_ane_options is an
// optional ANE-compiler option string (e.g. "TargetArchitecture=h13"); NULL = host arch.
extern "C" int ane_e5rt_compile_check(const char *mil_path, const char *cache_dir,
                                      uint64_t device_mask, const char *custom_ane_options) {
  if (load_all_symbols() != 0) return -1;
  void *config = NULL, *compiler = NULL, *options = NULL, *library = NULL;
  e5rt_error_code_t err;
  int rc = -1;

  err = e5rt_e5_compiler_config_options_create(&config);
  if (err || !config) goto done;
  e5rt_e5_compiler_config_options_set_cache_bundle_location(config, cache_dir);
  err = e5rt_e5_compiler_create_with_config(&compiler, config);
  if (err || !compiler) goto done;
  err = e5rt_e5_compiler_options_create(&options);
  if (err || !options) goto done;
  e5rt_e5_compiler_options_set_compute_device_types_mask(options, device_mask);
  e5rt_e5_compiler_options_set_force_recompilation(options, 1);
  e5rt_e5_compiler_options_set_segmenter(options, "graph");

  // Optionally set the custom ANE compiler options (e.g. TargetArchitecture). Loaded
  // lazily so a missing symbol degrades to "host arch only" instead of breaking compile.
  if (custom_ane_options && custom_ane_options[0]) {
    if (!e5rt_e5_compiler_options_set_custom_ane_compiler_options) {
      e5rt_e5_compiler_options_set_custom_ane_compiler_options =
        (e5rt_e5_compiler_options_set_custom_ane_compiler_options_fn)
        dlsym(fw, "e5rt_e5_compiler_options_set_custom_ane_compiler_options");
      if (!e5rt_e5_compiler_options_set_custom_ane_compiler_options)
        e5rt_e5_compiler_options_set_custom_ane_compiler_options =
          (e5rt_e5_compiler_options_set_custom_ane_compiler_options_fn)
          dlsym(fw, "_e5rt_e5_compiler_options_set_custom_ane_compiler_options");
    }
    if (!e5rt_e5_compiler_options_set_custom_ane_compiler_options) { rc = -2; goto done; }
    e5rt_e5_compiler_options_set_custom_ane_compiler_options(options, custom_ane_options);
  }

  err = e5rt_e5_compiler_compile(compiler, mil_path, options, &library);
  rc = (err || !library) ? (int)(err ? err : -1) : 0;

done:
  if (library) e5rt_program_library_release(&library);
  if (options) e5rt_e5_compiler_options_release(&options);
  if (compiler) e5rt_e5_compiler_release(&compiler);
  if (config) e5rt_e5_compiler_config_options_release(&config);
  return rc;
}

// Finalize the program: prepare each op (only if previously encoded), reset
// the stream (only if previously used), then encode every op in order.
// Idempotent - after the first call sets prog->finalized=1; subsequent calls
// short-circuit. Returns 0 on success.
//
// Important detail: `e5rt_execution_stream_operation_prepare_op_for_encode`
// rejects ops that have never been encoded (error string:
// "Op has not been encoded and hence cannot be reset to 'ReadyForEncode'").
// So we only call prepare on ops we've already encoded once.
// Similarly `e5rt_execution_stream_reset` rejects fresh streams.
static int any_op_encoded(ane_e5rt_program_t *prog) {
  for (size_t i = 0; i < prog->n_ops; i++) if (prog->ops[i].encoded_once) return 1;
  return 0;
}

static int finalize_program(ane_e5rt_program_t *prog) {
  if (!prog || !prog->stream) return -1;
  if (prog->finalized) return 0;

  int need_reset = any_op_encoded(prog);
  if (need_reset) {
    // Reset stream FIRST - this drops the encoded-op references inside the
    // stream, letting prepare_op_for_encode legally transition encoded ops
    // back to "ReadyForEncode".
    e5rt_error_code_t err = e5rt_execution_stream_reset(prog->stream);
    if (err) { fprintf(stderr, "stream_reset err=%lld\n", (long long)err); return -1; }
    for (size_t i = 0; i < prog->n_ops; i++) {
      if (!prog->ops[i].encoded_once) continue;
      err = e5rt_execution_stream_operation_prepare_op_for_encode(prog->ops[i].operation);
      if (err) { fprintf(stderr, "prepare(op%zu) err=%lld\n", i, (long long)err); return -1; }
    }
  }
  for (size_t i = 0; i < prog->n_ops; i++) {
    e5rt_error_code_t err = e5rt_execution_stream_encode_operation(prog->stream, prog->ops[i].operation);
    if (err) { fprintf(stderr, "encode(op%zu) err=%lld\n", i, (long long)err); return -1; }
    prog->ops[i].encoded_once = 1;
  }
  if (prog->n_ops > 0) prog->final_event = prog->ops[prog->n_ops - 1].completion_event;
  prog->finalized = 1;
  return 0;
}

// Full teardown of a program: stream, ops + their ports, and the saved port
// spec arrays (strdup'd names + size arrays). Used by both the compile error
// ladder and ane_e5rt_program_release so every exit path frees identically.
// Safe on a partially-constructed program: calloc zeroes all fields and
// free(NULL) is a no-op.
static void prog_destroy(ane_e5rt_program_t *prog) {
  if (!prog) return;
  // final_event is an ALIAS of the op's completion_event - not owned here.
  prog->final_event = NULL;
  if (prog->stream) e5rt_execution_stream_release(&prog->stream);
  for (size_t k = 0; k < prog->n_ops; k++) op_free_ports_and_op(&prog->ops[k]);
  free(prog->ops);
  for (size_t i = 0; i < prog->n_input_specs; i++) free(prog->input_port_names[i]);
  free(prog->input_port_names);
  free(prog->input_port_sizes);
  for (size_t i = 0; i < prog->n_output_specs; i++) free(prog->output_port_names[i]);
  free(prog->output_port_names);
  free(prog->output_port_sizes);
  pthread_mutex_destroy(&prog->async.mutex);
  pthread_cond_destroy(&prog->async.cond);
  free(prog);
}

// ---- public API: single-op surface ------------------------------------------

extern "C" ane_e5rt_program_t *ane_e5rt_program_compile(
    const char *mil_path,
    const char *cache_dir,
    uint64_t device_mask,
    const char *const *input_names, const size_t *input_sizes, size_t n_inputs,
    const char *const *output_names, const size_t *output_sizes, size_t n_outputs)
{
  if (load_all_symbols() != 0) return NULL;

  ane_e5rt_program_t *prog = (ane_e5rt_program_t*)calloc(1, sizeof(*prog));
  if (!prog) return NULL;
  pthread_mutex_init(&prog->async.mutex, NULL);
  pthread_cond_init(&prog->async.cond, NULL);
  strncpy(prog->cache_dir, cache_dir ? cache_dir : "", sizeof(prog->cache_dir) - 1);
  strncpy(prog->mil_path, mil_path ? mil_path : "", sizeof(prog->mil_path) - 1);
  prog->device_mask = device_mask;

  // Save port specs (names strdup'd) so the program owns them for its lifetime.
  prog->n_input_specs = n_inputs;
  prog->input_port_names = (char**)calloc(n_inputs, sizeof(char*));
  prog->input_port_sizes = (size_t*)calloc(n_inputs, sizeof(size_t));
  for (size_t i = 0; i < n_inputs; i++) {
    prog->input_port_names[i] = strdup(input_names[i]);
    prog->input_port_sizes[i] = input_sizes[i];
  }
  prog->n_output_specs = n_outputs;
  prog->output_port_names = (char**)calloc(n_outputs, sizeof(char*));
  prog->output_port_sizes = (size_t*)calloc(n_outputs, sizeof(size_t));
  for (size_t i = 0; i < n_outputs; i++) {
    prog->output_port_names[i] = strdup(output_names[i]);
    prog->output_port_sizes[i] = output_sizes[i];
  }

  ane_e5rt_op_t *op = prog_push_op(prog);
  if (!op) { prog_destroy(prog); return NULL; }

  if (compile_and_build_op(mil_path, cache_dir, device_mask, op) != 0) {
    prog_destroy(prog);
    return NULL;
  }

  // Ports.
  op->inputs = (ane_e5rt_port_t*)calloc(n_inputs, sizeof(ane_e5rt_port_t));
  op->n_inputs = n_inputs;
  for (size_t i = 0; i < n_inputs; i++) {
    if (port_alloc_and_bind(op->operation, input_names[i], input_sizes[i], 1, &op->inputs[i]) != 0) {
      prog_destroy(prog); return NULL;
    }
  }
  op->outputs = (ane_e5rt_port_t*)calloc(n_outputs, sizeof(ane_e5rt_port_t));
  op->n_outputs = n_outputs;
  for (size_t i = 0; i < n_outputs; i++) {
    if (port_alloc_and_bind(op->operation, output_names[i], output_sizes[i], 0, &op->outputs[i]) != 0) {
      prog_destroy(prog); return NULL;
    }
  }

  // Create a completion event and bind it BEFORE encode (Espresso refuses to
  // change bind-state once an op is encoded/in-use).
  {
    void *evt = NULL;
    char name[64]; snprintf(name, sizeof(name), "op%zu_complete", prog->n_ops - 1);
    e5rt_error_code_t e = e5rt_async_event_create(&evt, name, 0);
    if (!e && evt) {
      e = e5rt_execution_stream_operation_bind_completion_event(op->operation, evt);
      if (e) { e5rt_async_event_release(&evt); }
      else { op->completion_event = evt; }
    }
  }

  e5rt_error_code_t err = e5rt_execution_stream_create(&prog->stream);
  if (err || !prog->stream) {
    fprintf(stderr, "stream_create err=%lld\n", (long long)err);
    prog_destroy(prog);
    return NULL;
  }

  // Encode + prepare are deferred to finalize_program(), called lazily by
  // execute(), so share_buffer() can rebind ports before the graph commits.
  return prog;
}

static ane_e5rt_port_t *find_port(ane_e5rt_port_t *ports, size_t n, const char *name) {
  for (size_t i = 0; i < n; i++) {
    if (strcmp(ports[i].name, name) == 0) return &ports[i];
  }
  return NULL;
}

extern "C" int ane_e5rt_program_set_input_fp16(ane_e5rt_program_t *prog, const char *port_name,
                                               const uint16_t *data, size_t n_elems) {
  if (!prog || prog->n_ops == 0) return -1;
  ane_e5rt_op_t *op = &prog->ops[0];
  ane_e5rt_port_t *p = find_port(op->inputs, op->n_inputs, port_name);
  if (!p) { fprintf(stderr, "no input port: %s\n", port_name); return -1; }
  size_t bytes = n_elems * sizeof(uint16_t);
  if (bytes > p->size_bytes) {
    fprintf(stderr, "input %s overflow (%zu > %zu)\n", port_name, bytes, p->size_bytes);
    return -1;
  }
  memcpy(p->data_ptr, data, bytes);
  return 0;
}

extern "C" int ane_e5rt_program_get_output_fp16(ane_e5rt_program_t *prog, const char *port_name,
                                                uint16_t *dest, size_t n_elems) {
  if (!prog || prog->n_ops == 0) return -1;
  ane_e5rt_op_t *op = &prog->ops[0];
  ane_e5rt_port_t *p = find_port(op->outputs, op->n_outputs, port_name);
  if (!p) { fprintf(stderr, "no output port: %s\n", port_name); return -1; }
  size_t bytes = n_elems * sizeof(uint16_t);
  if (bytes > p->size_bytes) bytes = p->size_bytes;
  memcpy(dest, p->data_ptr, bytes);
  return 0;
}

// Zero-copy accessors: hand back the persistent CPU+ANE-visible buffer for a port so the
// host can read/write it directly (np.frombuffer) and skip the per-call memcpy that
// set_input_fp16 / get_output_fp16 do. The pointer is valid for the program's lifetime.
extern "C" int ane_e5rt_program_input_buffer(ane_e5rt_program_t *prog, const char *port_name,
                                             void **out_ptr, size_t *out_bytes) {
  if (!prog || prog->n_ops == 0) return -1;
  ane_e5rt_port_t *p = find_port(prog->ops[0].inputs, prog->ops[0].n_inputs, port_name);
  if (!p || !p->data_ptr) return -1;
  if (out_ptr) *out_ptr = p->data_ptr;
  if (out_bytes) *out_bytes = p->size_bytes;
  return 0;
}

extern "C" int ane_e5rt_program_output_buffer(ane_e5rt_program_t *prog, const char *port_name,
                                              void **out_ptr, size_t *out_bytes) {
  if (!prog || prog->n_ops == 0) return -1;
  ane_e5rt_port_t *p = find_port(prog->ops[0].outputs, prog->ops[0].n_outputs, port_name);
  if (!p || !p->data_ptr) return -1;
  if (out_ptr) *out_ptr = p->data_ptr;
  if (out_bytes) *out_bytes = p->size_bytes;
  return 0;
}

extern "C" int ane_e5rt_program_execute(ane_e5rt_program_t *prog) {
  if (!prog || !prog->stream) return -1;
  if (finalize_program(prog) != 0) return -1;
  e5rt_error_code_t err = e5rt_execution_stream_execute_sync(prog->stream);
  if (err) { fprintf(stderr, "execute_sync err=%lld\n", (long long)err); return -1; }
  return 0;
}

extern "C" void ane_e5rt_program_release(ane_e5rt_program_t *prog) {
  prog_destroy(prog);
}

// Rebinds dst_op_idx's input port `dst_in_port` to the SAME e5rt_buffer_object
// that src_op_idx's output port `src_out_port` is bound to, so dst reads from
// src's output buffer (resident-state chaining). Must be called BEFORE the
// first execute (i.e. before finalize_program commits the graph).
//
// Ownership rule: after a successful rebind, src owns the shared buffer and
// dst->buffer is cleared so op_free_ports_and_op releases it exactly once (via
// src). On a failed rebind, dst is left fully intact and -1 is returned.
extern "C" int ane_e5rt_program_share_buffer(ane_e5rt_program_t *prog,
                                             size_t src_op_idx, const char *src_out_port,
                                             size_t dst_op_idx, const char *dst_in_port) {
  if (!prog || src_op_idx >= prog->n_ops || dst_op_idx >= prog->n_ops) return -1;
  if (prog->finalized) {
    fprintf(stderr, "share_buffer: program already finalized\n"); return -1;
  }
  ane_e5rt_port_t *src = find_port(prog->ops[src_op_idx].outputs,
                                   prog->ops[src_op_idx].n_outputs, src_out_port);
  if (!src) { fprintf(stderr, "src out port %s not found\n", src_out_port); return -1; }
  ane_e5rt_port_t *dst = find_port(prog->ops[dst_op_idx].inputs,
                                   prog->ops[dst_op_idx].n_inputs, dst_in_port);
  if (!dst) { fprintf(stderr, "dst in port %s not found\n", dst_in_port); return -1; }

  // Bind FIRST so a failed rebind cannot leave dst pointing into freed memory.
  e5rt_error_code_t err = e5rt_io_port_bind_buffer_object(dst->port, src->buffer);
  if (err) {
    fprintf(stderr, "rebind err=%lld\n", (long long)err);
    return -1;  // dst left fully intact
  }
  // Bind succeeded: release dst's old buffer, then alias src's.
  if (dst->buffer) e5rt_buffer_object_release(&dst->buffer);
  dst->data_ptr = src->data_ptr;
  dst->size_bytes = src->size_bytes;
  dst->buffer = NULL;  // src owns the shared buffer; clear dst to avoid double-release
  return 0;
}

// =============================================================================
// EXPERIMENTAL SURFACE - not bound by the Python frontend (aneforge/_runtime.py).
//
// The functions below are a directly-callable C research surface for multi-op
// streams and async completion. The 9 production functions above are the only
// ones the ctypes layer wires up; everything from here down is exercised by
// hand from native test harnesses. Kept building (and in sync with the docs)
// even though it is not part of the supported Python API.
// =============================================================================

// Recursive directory create (replaces the old shell-out to mkdir -p, which
// invoked a subshell). Creates each level via NSFileManager with intermediate
// directories. Pre-existing directories are not an error. Returns 0 on success.
static int mkdir_recursive(const char *path) {
  if (!path || !path[0]) return -1;
  NSString *p = [NSString stringWithUTF8String:path];
  NSError *err = nil;
  BOOL ok = [[NSFileManager defaultManager] createDirectoryAtPath:p
                                       withIntermediateDirectories:YES
                                                        attributes:nil
                                                             error:&err];
  return ok ? 0 : -1;
}

// ---- multi-op stream extension ----------------------------------------------
//
// add_op() compiles MIL_PATH and adds a precompiled op onto the same stream.
// It uses the same cache_dir and device_mask captured at compile time, plus
// a unique cache subdirectory per op so the per-op bundle paths don't clash.
//
// Returns the new op index (>= 0), or -1 on error.
extern "C" int ane_e5rt_program_add_op(ane_e5rt_program_t *prog,
                                       const char *mil_path,
                                       const char *input_name, size_t input_size,
                                       const char *output_name, size_t output_size) {
  if (!prog || !prog->stream) return -1;
  // Per-op sub-cache so we don't overwrite the existing bundle. Created
  // programmatically (no shell-out) via mkdir_recursive.
  char per_op_cache[1280];
  snprintf(per_op_cache, sizeof(per_op_cache), "%s/op%zu", prog->cache_dir, prog->n_ops);
  mkdir_recursive(per_op_cache);  // best-effort; compile reports a hard failure

  ane_e5rt_op_t *op = prog_push_op(prog);
  if (!op) return -1;

  if (compile_and_build_op(mil_path, per_op_cache, prog->device_mask, op) != 0) {
    prog->n_ops--;
    return -1;
  }

  // Bind one input + one output for the op.
  op->inputs = (ane_e5rt_port_t*)calloc(1, sizeof(ane_e5rt_port_t));
  op->n_inputs = 1;
  if (port_alloc_and_bind(op->operation, input_name, input_size, 1, &op->inputs[0]) != 0) {
    op_free_ports_and_op(op);
    prog->n_ops--;
    return -1;
  }
  op->outputs = (ane_e5rt_port_t*)calloc(1, sizeof(ane_e5rt_port_t));
  op->n_outputs = 1;
  if (port_alloc_and_bind(op->operation, output_name, output_size, 0, &op->outputs[0]) != 0) {
    op_free_ports_and_op(op);
    prog->n_ops--;
    return -1;
  }

  // Bind a per-op completion event BEFORE we mark the op for encode. This is
  // necessary so that bind-state mutations stay legal - once an op gets
  // encoded onto a stream Espresso refuses to change its bind state.
  {
    void *evt = NULL;
    char name[64]; snprintf(name, sizeof(name), "op%zu_complete", prog->n_ops - 1);
    e5rt_error_code_t e = e5rt_async_event_create(&evt, name, 0);
    if (!e && evt) {
      e = e5rt_execution_stream_operation_bind_completion_event(op->operation, evt);
      if (e) { e5rt_async_event_release(&evt); }
      else { op->completion_event = evt; }
    }
  }

  // Defer encode to finalize_program(). Mark unfinalized so the next execute
  // re-encodes (including this new op).
  prog->finalized = 0;
  prog->final_event = NULL;
  return (int)(prog->n_ops - 1);
}

extern "C" int ane_e5rt_program_set_input_fp16_op(ane_e5rt_program_t *prog, size_t op_idx,
                                                  const char *port_name,
                                                  const uint16_t *data, size_t n_elems) {
  if (!prog || op_idx >= prog->n_ops) return -1;
  ane_e5rt_op_t *op = &prog->ops[op_idx];
  ane_e5rt_port_t *p = find_port(op->inputs, op->n_inputs, port_name);
  if (!p) { fprintf(stderr, "no input port on op[%zu]: %s\n", op_idx, port_name); return -1; }
  size_t bytes = n_elems * sizeof(uint16_t);
  if (bytes > p->size_bytes) { fprintf(stderr, "input overflow\n"); return -1; }
  memcpy(p->data_ptr, data, bytes);
  return 0;
}

extern "C" int ane_e5rt_program_get_output_fp16_op(ane_e5rt_program_t *prog, size_t op_idx,
                                                   const char *port_name,
                                                   uint16_t *dest, size_t n_elems) {
  if (!prog || op_idx >= prog->n_ops) return -1;
  ane_e5rt_op_t *op = &prog->ops[op_idx];
  ane_e5rt_port_t *p = find_port(op->outputs, op->n_outputs, port_name);
  if (!p) { fprintf(stderr, "no output port on op[%zu]: %s\n", op_idx, port_name); return -1; }
  size_t bytes = n_elems * sizeof(uint16_t);
  if (bytes > p->size_bytes) bytes = p->size_bytes;
  memcpy(dest, p->data_ptr, bytes);
  return 0;
}

extern "C" int ane_e5rt_program_execute_multi(ane_e5rt_program_t *prog) {
  if (!prog || !prog->stream) return -1;
  if (finalize_program(prog) != 0) return -1;
  e5rt_error_code_t err = e5rt_execution_stream_execute_sync(prog->stream);
  if (err) { fprintf(stderr, "execute_sync(multi) err=%lld\n", (long long)err); return -1; }
  return 0;
}

extern "C" size_t ane_e5rt_program_get_op_count(ane_e5rt_program_t *prog) {
  return prog ? prog->n_ops : 0;
}

// ---- chain ops via async event ---------------------------------------------
//
// Binds the same async-event to both src_op_idx (as completion) and dst_op_idx
// (as a single-element dependent-events array). Models a "src happens-before dst"
// edge. Must be called BEFORE finalize_program() commits the topology; ordering
// on a single stream is also FIFO regardless, so on one stream this is mostly a
// happens-before probe.
extern "C" int ane_e5rt_program_chain_ops(ane_e5rt_program_t *prog,
                                          size_t src_op_idx, size_t dst_op_idx,
                                          const char *event_name) {
  if (!prog || src_op_idx >= prog->n_ops || dst_op_idx >= prog->n_ops) return -1;
  if (prog->finalized) {
    fprintf(stderr, "chain_ops: program already finalized; cannot mutate event topology\n");
    return -1;
  }
  ane_e5rt_op_t *src = &prog->ops[src_op_idx];
  ane_e5rt_op_t *dst = &prog->ops[dst_op_idx];

  // src may already have a completion event from auto-binding at compile/add_op.
  // Reuse if present; otherwise create one now.
  void *event = src->completion_event;
  if (!event) {
    e5rt_error_code_t err = e5rt_async_event_create(&event, event_name ? event_name : "chain", 0);
    if (err || !event) { fprintf(stderr, "event_create err=%lld\n", (long long)err); return -1; }
    err = e5rt_execution_stream_operation_bind_completion_event(src->operation, event);
    if (err) { fprintf(stderr, "bind_completion err=%lld\n", (long long)err);
               e5rt_async_event_release(&event); return -1; }
    src->completion_event = event;
  }

  void *events[1] = { event };
  e5rt_error_code_t err = e5rt_execution_stream_operation_bind_dependent_events(dst->operation, events, 1);
  if (err) { fprintf(stderr, "bind_dependent_events err=%lld\n", (long long)err); return -1; }
  return 0;
}

extern "C" int ane_e5rt_program_get_chain_event_last_signaled(ane_e5rt_program_t *prog,
                                                              size_t op_idx,
                                                              uint64_t *out) {
  if (!prog || op_idx >= prog->n_ops || !out) return -1;
  ane_e5rt_op_t *op = &prog->ops[op_idx];
  if (!op->completion_event) return -1;
  e5rt_error_code_t err = e5rt_async_event_get_last_signaled_value(op->completion_event, out);
  if (err) return -1;
  return 0;
}

// Probe an op's completion-event last-signaled value by op index. Same payload
// as get_chain_event_last_signaled; kept as the documented event-probe name.
extern "C" int ane_e5rt_program_get_event_last_signaled(ane_e5rt_program_t *prog,
                                                        size_t op_idx,
                                                        uint64_t *out) {
  return ane_e5rt_program_get_chain_event_last_signaled(prog, op_idx, out);
}

// ---- async submit + completion block ---------------------------------------
//
// The Espresso disassembly is unambiguous: `_e5rt_execution_stream_submit_async`
// does `mov x0, x1 ; bl _objc_retain` on its second argument. So we must hand
// it a real ObjC block, not a raw function pointer. We construct one with the
// `^{ ... }` syntax which is a Clang language extension that generates a
// `_NSConcreteStackBlock` with the captured `prog` pointer; copying it with
// `[block copy]` (or `Block_copy`) moves it to the heap so it survives past
// the stack frame.

static void async_complete_unlock(ane_e5rt_program_t *prog, int rc) {
  if (!prog) return;
  pthread_mutex_lock(&prog->async.mutex);
  prog->async.completed = 1;
  prog->async.completion_rc = rc;
  pthread_cond_broadcast(&prog->async.cond);
  pthread_mutex_unlock(&prog->async.mutex);
}

// Returns 0 on success, -1 on error. Each op carries a completion event that
// was bound at compile/add_op time. The last-op event is used as the program's
// "final" event for sync_wait.
//
// CONTRACT: the submit block captures the program pointer; the caller MUST call
// ane_e5rt_program_wait_for_completion() before ane_e5rt_program_release() so
// the block never runs against a freed program.
extern "C" int ane_e5rt_program_execute_async(ane_e5rt_program_t *prog) {
  if (!prog || !prog->stream) return -1;
  if (prog->n_ops == 0) return -1;
  if (finalize_program(prog) != 0) return -1;

  // Reset async state.
  pthread_mutex_lock(&prog->async.mutex);
  prog->async.submitted = 1;
  prog->async.completed = 0;
  prog->async.completion_rc = 0;
  pthread_mutex_unlock(&prog->async.mutex);

  prog->final_event = prog->ops[prog->n_ops - 1].completion_event;
  if (prog->final_event) {
    e5rt_async_event_get_last_signaled_value(prog->final_event,
                                             &prog->async.final_event_value_before);
  }

  // submit_async takes a block typed `e5rt_error_code_t (^)(void)`. Per the
  // disassembly the block is invoked inside `ExceptionSafeExecute` - the
  // returned error code becomes the std::function's result. The submit
  // *itself* is what's async (kicked off internally), so the block returning
  // success is enough; the per-op completion events fire when ANE finishes.
  ane_e5rt_program_t *captured = prog;
  e5rt_error_code_t (^submit_block)(void) = ^e5rt_error_code_t (void) {
    // We mark "completion" right after the submit body returns; the actual
    // ANE work runs asynchronously and signals the completion event. Anyone
    // who needs ANE-completion-strict semantics should wait on the final
    // event with e5rt_async_event_sync_wait instead of relying on this.
    async_complete_unlock(captured, 0);
    return (e5rt_error_code_t)0;
  };
  e5rt_error_code_t (^heap_block)(void) = [submit_block copy];

  e5rt_error_code_t err = e5rt_execution_stream_submit_async(prog->stream, heap_block);
  if (err) {
    fprintf(stderr, "submit_async err=%lld\n", (long long)err);
    pthread_mutex_lock(&prog->async.mutex);
    prog->async.submitted = 0;
    prog->async.completed = 1;
    prog->async.completion_rc = (int)err;
    pthread_cond_broadcast(&prog->async.cond);
    pthread_mutex_unlock(&prog->async.mutex);
    return -1;
  }
  return 0;
}

// Blocks until ANE-side completion (waits on the final op's completion event).
// Returns 0 on success, -1 on event error, -2 if never submitted.
extern "C" int ane_e5rt_program_wait_for_completion(ane_e5rt_program_t *prog) {
  if (!prog) return -2;
  pthread_mutex_lock(&prog->async.mutex);
  int submitted = prog->async.submitted;
  pthread_mutex_unlock(&prog->async.mutex);
  if (!submitted) return -2;

  // Strong wait: block on the ANE completion event (last op).
  if (prog->final_event) {
    e5rt_error_code_t err = e5rt_async_event_sync_wait(prog->final_event);
    if (err) {
      fprintf(stderr, "sync_wait err=%lld\n", (long long)err);
      pthread_mutex_lock(&prog->async.mutex);
      prog->async.submitted = 0;
      pthread_mutex_unlock(&prog->async.mutex);
      return -1;
    }
  }
  // Also drain the submit-side condvar (might already be signaled).
  pthread_mutex_lock(&prog->async.mutex);
  while (!prog->async.completed) {
    pthread_cond_wait(&prog->async.cond, &prog->async.mutex);
  }
  int rc = prog->async.completion_rc;
  prog->async.submitted = 0;
  pthread_mutex_unlock(&prog->async.mutex);
  return rc == 0 ? 0 : -1;
}

extern "C" int ane_e5rt_program_get_final_event_signaled(ane_e5rt_program_t *prog,
                                                         uint64_t *before, uint64_t *after) {
  if (!prog || !prog->final_event) return -1;
  if (before) *before = prog->async.final_event_value_before;
  if (after) {
    e5rt_error_code_t err = e5rt_async_event_get_last_signaled_value(prog->final_event, after);
    if (err) return -1;
  }
  return 0;
}

// ---- standalone completion-block factory (for users who want a block they
//      hand off to other e5rt entry points) -----------------------------------

typedef void (*ane_e5rt_completion_cb_t)(void *ctx, int rc);

extern "C" void *ane_e5rt_make_completion_block(ane_e5rt_completion_cb_t cb, void *ctx) {
  if (!cb) return NULL;
  void *captured_ctx = ctx;
  ane_e5rt_completion_cb_t captured_cb = cb;
  void (^stack_block)(void) = ^{
    captured_cb(captured_ctx, 0);
  };
  // Move to heap and balance ARC by handing back a +1 retained pointer.
  void *heap = (__bridge_retained void *)[stack_block copy];
  return heap;
}

extern "C" void ane_e5rt_free_completion_block(void *block) {
  if (!block) return;
  // Balance the +1 retain we took in make_completion_block.
  __unused id _x = (__bridge_transfer id)block;
}

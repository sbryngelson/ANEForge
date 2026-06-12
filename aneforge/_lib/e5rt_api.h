// e5rt_api.h - Recovered C-API signatures for Espresso's e5rt_* family.
//
// Symbols come from:
//   /System/Library/PrivateFrameworks/Espresso.framework/Espresso
//
// Signatures recovered via:
//   - dyld_info -exports / nm symbol scan
//   - ipsw disassembly artifacts in
//     ane_artifacts/runtime/espresso_e5rt_latest/ipsw_disass_*.txt
//   - empirical probing in ane_e5rt_*_probe.c (return code 0 = success)
//
// All entry points return int64_t e5rt_error_code_t. Convention: code==0 success;
// nonzero codes carry a string via e5rt_error_code_get_string (but that stub
// returns a C++ std::string under the hood, not const char* — do not call from
// pure C). Inspect failure mode via stderr Espresso log.
//
// ABI: arm64 AAPCS. All "create" entry points return the new object via the
// last out-parameter (out goes LAST). Verified empirically.

#ifndef ANE_E5RT_API_H
#define ANE_E5RT_API_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int64_t e5rt_error_code_t;

// --- Compiler config + compiler ---

typedef e5rt_error_code_t (*e5rt_e5_compiler_config_options_create_fn)(void **out);
typedef e5rt_error_code_t (*e5rt_e5_compiler_config_options_release_fn)(void **obj);
typedef e5rt_error_code_t (*e5rt_e5_compiler_config_options_set_cache_bundle_location_fn)(void *obj, const char *path);
typedef e5rt_error_code_t (*e5rt_e5_compiler_config_options_set_bundle_cache_apfs_purgeable_fn)(void *obj, int value);

typedef e5rt_error_code_t (*e5rt_e5_compiler_create_fn)(void **out_compiler);
typedef e5rt_error_code_t (*e5rt_e5_compiler_create_with_config_fn)(void **out_compiler, void *config);
typedef e5rt_error_code_t (*e5rt_e5_compiler_release_fn)(void **obj);
typedef e5rt_error_code_t (*e5rt_e5_compiler_compile_fn)(void *compiler, const char *mil_or_mlmodelc_path, void *options, void **out_library);

// --- Compiler options ---
//
// Bitmask values for set_compute_device_types_mask (empirically determined):
//   0x1 = BNNS  (CPU)
//   0x2 = MPSGraph (GPU)
//   0x4 = ANE
//   0x7 = auto (planner chooses; ANE preferred when supported)

typedef e5rt_error_code_t (*e5rt_e5_compiler_options_create_fn)(void **out);
typedef e5rt_error_code_t (*e5rt_e5_compiler_options_release_fn)(void **obj);
typedef e5rt_error_code_t (*e5rt_e5_compiler_options_set_compute_device_types_mask_fn)(void *obj, uint64_t mask);
typedef e5rt_error_code_t (*e5rt_e5_compiler_options_get_compute_device_types_mask_fn)(void *obj, uint64_t *out);
typedef e5rt_error_code_t (*e5rt_e5_compiler_options_set_custom_ane_compiler_options_fn)(void *obj, const char *value);
typedef e5rt_error_code_t (*e5rt_e5_compiler_options_set_force_recompilation_fn)(void *obj, int value);
typedef e5rt_error_code_t (*e5rt_e5_compiler_options_set_force_bnns_graph_fn)(void *obj, int value);
typedef e5rt_error_code_t (*e5rt_e5_compiler_options_set_segmenter_fn)(void *obj, const char *value);

// --- Program library + function ---

// Empirically wants (out, path), not (path, out). Reversed order returns
// "Invalid E5 path specified. @ GetE5PathFromCompositeBundle".
typedef e5rt_error_code_t (*e5rt_program_library_create_fn)(void **out_library, const char *bundle_path);
typedef e5rt_error_code_t (*e5rt_program_library_release_fn)(void **obj);
typedef e5rt_error_code_t (*e5rt_program_library_get_num_functions_fn)(void *library, uint64_t *out);
typedef e5rt_error_code_t (*e5rt_program_library_get_function_names_fn)(void *library, uint64_t count, const char **names);
typedef e5rt_error_code_t (*e5rt_program_library_retain_program_function_fn)(void *library, const char *name, void **out);
typedef e5rt_error_code_t (*e5rt_program_function_release_fn)(void **obj);
typedef e5rt_error_code_t (*e5rt_program_function_get_name_fn)(void *function, const char **out);
typedef e5rt_error_code_t (*e5rt_program_function_load_for_execution_fn)(void *function);

// --- Precompiled compute op options ---

// Empirically: (out, function), out FIRST. Reversed order = nullptr error.
typedef e5rt_error_code_t (*e5rt_precompiled_compute_op_create_options_create_with_program_function_fn)(void **out_options, void *function);
typedef e5rt_error_code_t (*e5rt_precompiled_compute_op_create_options_release_fn)(void **obj);
typedef e5rt_error_code_t (*e5rt_precompiled_compute_op_create_options_set_operation_name_fn)(void *obj, const char *name);
typedef e5rt_error_code_t (*e5rt_precompiled_compute_op_create_options_set_allocate_intermediate_buffers_fn)(void *obj, int value);

// --- Execution stream operation ---

// Empirically: (out, options), out FIRST.
typedef e5rt_error_code_t (*e5rt_execution_stream_operation_create_precompiled_compute_operation_with_options_fn)(void **out_operation, void *options);
typedef e5rt_error_code_t (*e5rt_execution_stream_operation_release_fn)(void **obj);
typedef e5rt_error_code_t (*e5rt_execution_stream_operation_retain_input_port_fn)(void *operation, const char *name, void **out_port);
typedef e5rt_error_code_t (*e5rt_execution_stream_operation_retain_output_port_fn)(void *operation, const char *name, void **out_port);
typedef e5rt_error_code_t (*e5rt_execution_stream_operation_prepare_op_for_encode_fn)(void *operation);
typedef e5rt_error_code_t (*e5rt_execution_stream_operation_bind_completion_event_fn)(void *operation, void *event);
typedef e5rt_error_code_t (*e5rt_execution_stream_operation_bind_dependent_events_fn)(void *operation, void **events, uint64_t count);

// --- IO port ---

typedef e5rt_error_code_t (*e5rt_io_port_release_fn)(void **obj);
typedef e5rt_error_code_t (*e5rt_io_port_bind_buffer_object_fn)(void *port, void *buffer);

// --- Buffer object ---
//
// alloc: type 0 = CPU-only buffer (host data ptr)
//        type 1 = ANE-mapped (?)
//        type 2 = IOSurface-backed
//        type >=3 -> "Invalid BufferType @ AllocMemory"
typedef e5rt_error_code_t (*e5rt_buffer_object_alloc_fn)(void **out_buffer, uint64_t size, uint32_t type);
typedef e5rt_error_code_t (*e5rt_buffer_object_release_fn)(void **obj);
typedef e5rt_error_code_t (*e5rt_buffer_object_get_data_ptr_fn)(void *buffer, void **out);
typedef e5rt_error_code_t (*e5rt_buffer_object_get_size_fn)(void *buffer, uint64_t *out);

// --- Execution stream ---

typedef e5rt_error_code_t (*e5rt_execution_stream_create_fn)(void **out_stream);
typedef e5rt_error_code_t (*e5rt_execution_stream_release_fn)(void **obj);
typedef e5rt_error_code_t (*e5rt_execution_stream_encode_operation_fn)(void *stream, void *operation);
typedef e5rt_error_code_t (*e5rt_execution_stream_execute_sync_fn)(void *stream);
typedef e5rt_error_code_t (*e5rt_execution_stream_async_submit_fn)(void *stream);
typedef e5rt_error_code_t (*e5rt_execution_stream_set_quality_of_service_fn)(void *stream, uint64_t qos);
typedef e5rt_error_code_t (*e5rt_execution_stream_set_ane_execution_priority_fn)(void *stream, uint64_t priority);

// --- Async event ---

// Empirical: (out, name, initial_value). NULL name -> "eventName is NULL".
typedef e5rt_error_code_t (*e5rt_async_event_create_fn)(void **out_event, const char *name, uint32_t initial_value);
typedef e5rt_error_code_t (*e5rt_async_event_release_fn)(void **obj);
typedef e5rt_error_code_t (*e5rt_async_event_signal_fn)(void *event);
typedef e5rt_error_code_t (*e5rt_async_event_sync_wait_fn)(void *event);
typedef e5rt_error_code_t (*e5rt_async_event_get_last_signaled_value_fn)(void *event, uint64_t *out);
typedef e5rt_error_code_t (*e5rt_async_event_set_active_future_value_fn)(void *event, uint64_t value);

#ifdef __cplusplus
}
#endif

#endif // ANE_E5RT_API_H

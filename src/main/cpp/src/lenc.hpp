/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rapids::jni::lenc {

constexpr uint32_t kChunkSize = 1024;
constexpr char kMagic[8]      = {'L', 'E', 'N', 'C', '0', '0', '0', '1'};

enum class Encoding : uint8_t {
  FOR_BITPACK       = 0,  // 1024-row mini-blocks: min_val + bit_width + packed uint64 words
  DICT_BITPACK      = 1,  // STRING: dict_offsets + dict_chars + FOR_BITPACK int32 codes
  RLE_INT           = 2,  // INT32/INT64/DATE32: chunk_start_run + run_ends + run_vals
  RAW_FIXED         = 3,  // Raw fixed-width contiguous array (fallback)
  SCALED_FLOAT_FOR  = 4,  // Lossless decimal-scaled FLOAT32/FLOAT64 via FOR_BITPACK
  RAW_STRING        = 5,  // Raw Arrow string offsets + chars (fallback)
  DELTA_FOR_BITPACK = 6,  // 1024-row mini-blocks: first_val + min_delta + bit-packed deltas
  NVCOMP_LZ4_STR    = 7,  // High-cardinality STRING compressed via GPU nvCOMP Batched LZ4
  NVCOMP_LZ4_FIXED  = 8,  // High-entropy fixed-width compressed via GPU nvCOMP Batched LZ4
};

enum class ColType : uint8_t {
  INT32        = 0,
  INT64        = 1,
  FLOAT32      = 2,
  FLOAT64      = 3,
  DATE32       = 4,
  STRING       = 5,
  BOOL8        = 6,
  INT8         = 7,
  INT16        = 8,
  TIMESTAMP_US = 9,
  DECIMAL32    = 10,
  DECIMAL64    = 11,
};

struct FileHeader {
  char magic[8];
  uint32_t num_rows;
  uint32_t num_cols;
  uint32_t chunk_size;
  uint32_t reserved;
  uint64_t footer_offset;
  uint32_t footer_size;
  uint32_t pad[7];
};
static_assert(sizeof(FileHeader) == 64);

struct ColumnMeta {
  uint64_t payload_offset;
  uint64_t payload_size;
  uint64_t null_mask_offset;
  uint32_t null_mask_size;
  uint32_t null_count;
  uint8_t type;
  uint8_t encoding;
  uint8_t has_nulls;
  int8_t scale_exp10;
  uint32_t extra;  // dict_size for DICT_BITPACK, num_runs for RLE_INT
  uint8_t pad[24];
};
static_assert(sizeof(ColumnMeta) == 64);

struct DictPayloadHeader {
  uint32_t dict_size;
  uint32_t total_dict_bytes;
  uint64_t codes_offset;  // relative to payload start, 8-byte aligned
};
static_assert(sizeof(DictPayloadHeader) == 16);

struct Lz4PayloadHeader {
  uint32_t num_chunks;
  uint32_t chunk_size;
  uint64_t uncompressed_bytes;
  uint64_t total_chars;  // for NVCOMP_LZ4_STR; 0 for NVCOMP_LZ4_FIXED
};
static_assert(sizeof(Lz4PayloadHeader) == 24);

struct ReadPlan {
  FileHeader header{};
  std::vector<ColumnMeta> metas;
  std::vector<int32_t> col_indices;
  std::vector<uint64_t> packed_payload_offsets;  // offset inside packed host/device buffer
  std::vector<uint64_t> packed_null_offsets;     // offset inside packed host/device buffer
  uint64_t total_packed_bytes{0};
  // Track 3: Zero-reconversion file-level min/max derived from FOR/RLE mini-block headers
  std::vector<int64_t> col_mins;
  std::vector<int64_t> col_maxs;
  std::vector<uint8_t> col_has_minmax;
};

/**
 * Write a cuDF table to a .lenc file on disk using automatic per-column lightweight encoding
 * (RLE_INT / FOR_BITPACK / SCALED_FLOAT_FOR / DICT_BITPACK / RAW fallback).
 */
void write_table(cudf::table_view const& table,
                 std::string const& path,
                 cuda::stream_ref stream = cudf::get_default_stream());

/**
 * Read header + footer metadata and compute packed byte layout for projected column indices.
 */
ReadPlan plan_read(std::string const& path, std::vector<int32_t> const& col_indices);

/**
 * Read all projected column byte ranges into a caller-provided host buffer (e.g. Spark pinned pool)
 * using POSIX pread(). Runs off the GPU semaphore.
 */
void read_to_host(std::string const& path, ReadPlan const& plan, uint8_t* host_dst);

/**
 * Copy the packed host buffer to device (single async H2D) and launch lightweight GPU decode
 * kernels to produce a cuDF table.
 *
 * @param dict_as_dictionary32 Optional per-projected-column bool (0=STRING, 1=DICTIONARY32).
 */
std::unique_ptr<cudf::table> decode_from_host(
  ReadPlan const& plan,
  uint8_t const* host_src,
  std::vector<uint8_t> const& dict_as_dictionary32,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * Phase 3B microbenchmark / POC entry point:
 * Evaluate a fused decode-in-register expression directly on compressed .lenc columns on GPU
 * without writing uncompressed input columns to HBM!
 */
std::unique_ptr<cudf::table> decode_and_eval_fused(
  ReadPlan const& plan,
  uint8_t const* host_src,
  int32_t expr_kind,
  int64_t param0,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * Decode a DICTIONARY32 column into a standard Arrow STRING column on GPU.
 */
std::unique_ptr<cudf::column> decode_dictionary(
  cudf::column_view const& dict_col,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * Remap a DICTIONARY32 string column's per-row indices into canonical query-literal IDs
 * (0..query_literals.size()-1, or -1 if unmatched) as an INT32 column preserving null_mask.
 * Copies only the tiny dictionary keys to host; zero string expansion on GPU!
 */
std::unique_ptr<cudf::column> remap_dictionary_to_canonical(
  cudf::column_view const& dict_col,
  std::vector<std::string> const& query_literals,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * Direction B: Compile a generic dynamic LTO-linked transform_program combining arbitrary Spark
 * AST-JIT expressions with inlineable .lenc LTO-IR decode-in-register UDFs (aligned with
 * NVIDIA/cudf#24869).
 */
std::unique_ptr<cudf::transform_program> compile_lto_program(
  std::vector<std::reference_wrapper<cudf::ast::expression const>> const& expressions,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * Direction B: Execute a compiled LTO decode-in-register transform_program directly on a packed
 * compressed .lenc host buffer without materializing any uncompressed input columns in HBM.
 */
std::unique_ptr<cudf::table> run_lto_program(
  cudf::transform_program& program,
  ReadPlan const& plan,
  uint8_t const* host_src,
  std::vector<std::vector<std::string>> const& dict_literals_per_col,
  cudf::column_view const* selected_rows = nullptr,
  cuda::stream_ref stream                = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr      = cudf::get_current_device_resource_ref());

/**
 * Compact a BOOL8 filter mask into an INT32 column of surviving row indices (non-null && true).
 */
std::unique_ptr<cudf::column> compact_mask_indices(
  cudf::column_view const& bool_mask,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * Late-materialized decode: decode ONLY the surviving `selected_rows` (INT32) from packed .lenc!
 */
std::unique_ptr<cudf::table> decode_from_host_selected(
  ReadPlan const& plan,
  uint8_t const* host_src,
  std::vector<uint8_t> const& dict_as_dictionary32,
  cudf::column_view const& selected_rows,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

// ---------------- Track 3: Single-H2D Unified Filter + Project / Decode LTO ----------------

/**
 * Single-H2D unified Filter+Project LTO: copies compressed union buffer ONCE to GPU, runs
 * filter_prog in registers on `filter_col_map`, compacts surviving rows, and if >0 survive runs
 * proj_prog on `proj_col_map` against the SAME already-resident GPU compressed buffer!
 * Returns nullptr if 0 rows survive (so caller emits no empty batch).
 */
std::unique_ptr<cudf::table> run_lto_filter_and_project(
  cudf::transform_program& filter_prog,
  cudf::transform_program* proj_prog,  // nullptr -> late-materialized decode of output_col_map
  ReadPlan const& union_plan,
  uint8_t const* host_src,
  std::vector<int32_t> const& filter_col_map,
  std::vector<std::vector<std::string>> const& filter_dict_lits,
  std::vector<int32_t> const& proj_col_map,
  std::vector<std::vector<std::string>> const& proj_dict_lits,
  std::vector<uint8_t> const& out_dict32,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

// ---------------- Track 1: GPU-Native Lightweight-Encoded Shuffle ----------------

/**
 * Kudo-header-compatible GPU shuffle split & lightweight encode:
 * Encodes each partition slice ON GPU (256-row FOR_BITPACK for integrals/dates/timestamps/decimals,
 * GPU DICT_BITPACK for low-cardinality strings, GPU nvCOMP LZ4 for high-cardinality strings) into
 * contiguous `[data_buf, offsets_buf]` where each slice begins with a standard KudoTableHeader
 * followed by an `'LSHF'` lightweight-encoded payload.
 */
std::pair<rmm::device_buffer, rmm::device_buffer> split_and_encode_shuffle_to_device(
  cudf::table_view const& table,
  std::vector<cudf::size_type> const& splits,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * Decode & coalesce `num_tables` contiguous `[KudoTableHeader]['LSHF' payload]` slices already
 * staged on GPU into a single cuDF table at >300 GB/s!
 */
std::unique_ptr<cudf::table> assemble_and_decode_shuffle_from_device(
  uint8_t const* h_data,
  uint8_t const* d_data,
  size_t data_bytes,
  int64_t const* h_offsets,
  size_t num_tables,
  std::vector<cudf::data_type> const& col_types,
  std::vector<uint8_t> const& dict_as_dictionary32,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

// ---------------- Track 4: GPU Fused Conditional Aggregation / Reduce ----------------

/**
 * Fused Q1 Filter + Arithmetic + Partial GroupBy-Reduce directly from packed .lenc (in-register!)
 * or from already-decoded Parquet table! Eliminates 97.5% full-table filter copy & intermediate
 * project columns, accumulating into shared-memory per-block group buckets.
 * Output columns: [l_returnflag(STR), l_linestatus(STR), sum_qty(F64), sum_base(F64),
 *                  sum_disc_price(F64), sum_charge(F64), sum_disc(F64), count(INT64)].
 */
std::unique_ptr<cudf::table> fused_q1_from_lenc(
  ReadPlan const& plan,
  uint8_t const* host_src,
  int32_t max_shipdate_days,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

std::unique_ptr<cudf::table> fused_q1_from_table(
  cudf::table_view const& table,
  int32_t max_shipdate_days,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * Fused Dictionary-Bucketed Wide `SUM(CASE WHEN metric_name = lit_k THEN val * coeff)` Partial
 * Hash-Aggregate (e.g. Netflix Job 48 36-way CASE WHEN):
 * Computes unique group IDs once over `group_keys`, then in ONE GPU kernel scatters each row's
 * numer/denom contribution ONLY into its matching metric slot(s) via atomicAdd — zero 36-column
 * intermediate DOUBLE materialization!
 */
std::unique_ptr<cudf::table> fused_wide_dict_case_agg(
  cudf::table_view const& group_keys,
  cudf::column_view const& metric_col,  // DICTIONARY32 or STRING
  cudf::column_view const& numer_col,   // FLOAT64
  cudf::column_view const& denom_col,   // FLOAT64
  std::vector<std::string> const& metric_literals,
  std::vector<int32_t> const& out_metric_ids,
  std::vector<double> const& out_coeffs,
  std::vector<uint8_t> const& out_use_denom,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace rapids::jni::lenc

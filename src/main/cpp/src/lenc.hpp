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

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
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
  FOR_BITPACK      = 0,  // 1024-row mini-blocks: min_val + bit_width + packed uint64 words
  DICT_BITPACK     = 1,  // STRING: dict_offsets + dict_chars + FOR_BITPACK int32 codes
  RLE_INT          = 2,  // INT32/INT64/DATE32: chunk_start_run + run_ends + run_vals
  RAW_FIXED        = 3,  // Raw fixed-width contiguous array (fallback)
  SCALED_FLOAT_FOR = 4,  // Lossless decimal-scaled FLOAT32/FLOAT64 via FOR_BITPACK
  RAW_STRING       = 5,  // Raw Arrow string offsets + chars (fallback for >65536 distinct)
};

enum class ColType : uint8_t {
  INT32   = 0,
  INT64   = 1,
  FLOAT32 = 2,
  FLOAT64 = 3,
  DATE32  = 4,
  STRING  = 5,
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

struct ReadPlan {
  FileHeader header{};
  std::vector<ColumnMeta> metas;
  std::vector<int32_t> col_indices;
  std::vector<uint64_t> packed_payload_offsets;  // offset inside packed host/device buffer
  std::vector<uint64_t> packed_null_offsets;     // offset inside packed host/device buffer
  uint64_t total_packed_bytes{0};
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

}  // namespace rapids::jni::lenc

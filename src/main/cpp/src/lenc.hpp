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

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream_ref>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace rapids::jni::lenc {

// ---------------- Dictionary helpers (shared across Parquet, Vortex, and Shuffle) ----------------

std::unique_ptr<cudf::column> decode_dictionary(
  cudf::column_view const& dict_col,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

std::unique_ptr<cudf::column> remap_dictionary_to_canonical(
  cudf::column_view const& dict_col,
  std::vector<std::string> const& query_literals,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

// ---------------- GPU-Native Lightweight Shuffle (internal Spark Exchange codec) ----------------

std::pair<rmm::device_buffer, rmm::device_buffer> split_and_encode_shuffle_to_device(
  cudf::table_view const& table,
  std::vector<cudf::size_type> const& splits,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

std::unique_ptr<cudf::table> assemble_and_decode_shuffle_from_device(
  uint8_t const* h_data,
  uint8_t const* d_data,
  size_t total_data_len,
  int64_t const* host_offsets,
  size_t num_tables,
  std::vector<cudf::data_type> const& dtypes,
  std::vector<uint8_t> const& dict_as_dictionary32 = {},
  cuda::stream_ref stream                          = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr                = cudf::get_current_device_resource_ref());

// Column-pruned GPU decode for LightweightCachedBatchSerializer:
// copies ONLY the selected columns' compressed byte slices H2D and decodes ONLY those columns!
std::unique_ptr<cudf::table> decode_cached_lshf_projected(
  uint8_t const* h_lshf,
  size_t lshf_bytes,
  std::vector<int32_t> const& selected_cols,
  std::vector<cudf::data_type> const& selected_dtypes,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

// ---------------- Generic Compiler-Level NVRTC Reduce & Conditional GroupBy ----------------

struct JitReduceProgram;

JitReduceProgram* compile_jit_reduce_program(std::string const& cuda_src,
                                             std::string const& kernel_name);

std::string compile_jit_reduce_to_ptx_cpu_only(std::string const& cuda_src,
                                               std::string const& kernel_name,
                                               std::string const& compute_arch = "compute_75");

JitReduceProgram* load_jit_reduce_program_from_ptx(std::string const& ptx,
                                                   std::string const& kernel_name);

void destroy_jit_reduce_program(JitReduceProgram* prog);

std::unique_ptr<cudf::table> run_jit_reduce_table(
  JitReduceProgram& prog,
  cudf::table_view const& table,
  std::vector<std::vector<std::string>> const& dict_lits,
  std::vector<int32_t> const& group_key_slots,
  int32_t max_groups,
  std::vector<uint8_t> const& agg_kinds,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

struct JitReduceTaskAccumulator;

JitReduceTaskAccumulator* open_jit_reduce_task(
  JitReduceProgram& prog,
  std::vector<std::vector<std::string>> const& dict_lits,
  std::vector<int32_t> const& group_key_slots,
  int32_t max_groups,
  std::vector<uint8_t> const& agg_kinds,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

void add_batch_jit_reduce_task(
  JitReduceTaskAccumulator& acc,
  cudf::table_view const& table,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

std::unique_ptr<cudf::table> finish_jit_reduce_task(
  JitReduceTaskAccumulator& acc,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

void destroy_jit_reduce_task(JitReduceTaskAccumulator* acc);

std::unique_ptr<cudf::table> run_jit_reduce_high_card(
  JitReduceProgram& prog,
  cudf::table_view const& key_table,
  cudf::table_view const& val_table,
  std::vector<std::vector<std::string>> const& dict_lits,
  std::vector<uint8_t> const& agg_kinds,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

// ---------------- Path C: Fused Multi-Way Star Broadcast Equi-Join (L2 LUT / Hash) ----------------

struct StarJoinBuildTable;

StarJoinBuildTable* build_star_join_table(
  cudf::column_view const& key_col,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

void destroy_star_join_table(StarJoinBuildTable* bt);

int64_t run_star_join_count(
  cudf::table_view const& fact_table,
  std::vector<int32_t> const& probe_key_ords,
  std::vector<StarJoinBuildTable const*> const& build_tables,
  std::vector<int32_t> const& extra_nn_ords,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

std::unique_ptr<cudf::table> run_star_join_materialize(
  cudf::table_view const& fact_table,
  int32_t row_start,
  int32_t row_end,
  std::vector<int32_t> const& probe_key_ords,
  std::vector<StarJoinBuildTable const*> const& build_tables,
  std::vector<cudf::table_view> const& dim_payload_tables,
  std::vector<int32_t> const& extra_nn_ords,
  std::vector<int32_t> const& out_src_table,
  std::vector<int32_t> const& out_src_col,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace rapids::jni::lenc

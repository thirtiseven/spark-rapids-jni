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

#include "lenc.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/groupby.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvrtc.h>
#include <thrust/sequence.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rapids::jni::lenc {

namespace {

// Must match the exact C struct emitted into NVRTC-compiled generic reduction kernels!
constexpr int kMaxJitCols = 16;
constexpr int kMaxJitAggs = 64;

struct JitColDesc {
  // Mode: 0 = RAW fixed / Parquet flat, 1 = LENC FOR/SCALED_FLOAT, 2 = LENC RLE,
  //       3 = LENC DICT_BITPACK (codes via FOR + dict_remap)
  uint8_t mode;
  uint8_t has_nulls;
  uint8_t pad[6];
  void const* raw_ptr;
  uint32_t const* null_mask;
  int64_t const* min_vals;
  uint8_t const* bit_widths;
  uint32_t const* chunk_word_offs;
  uint64_t const* packed_words;
  int32_t const* dict_remap;  // maps local dict code -> canonical int32 (-1 if unmatched)
  uint32_t const* rle_chunk_start;
  uint32_t const* rle_run_ends;
  int64_t const* rle_run_vals;
  uint32_t rle_num_runs;
  uint32_t dict_size;
  double inv_scale;
};

struct JitReduceArgs {
  JitColDesc cols[kMaxJitCols];
  uint32_t num_rows;
  int32_t max_groups;
  int32_t num_aggs;
  uint32_t const* active_chunk_mask;  // optional 1,024-row chunk Min-Max prune bit/byte mask (nullptr = all active)
  // Low-card / Global output buffers: length = max_groups * num_aggs
  double* out_f64;
  long long* out_i64;
  unsigned long long* out_cnt;        // non-null count per (group, agg) -> determines SQL NULL & COUNT/AVG!
  // High-card GroupBy (Mode 2) inputs & outputs
  int32_t const* group_offsets;
  int32_t const* grouped_row_ids;
  int32_t num_groups;
  double* const* hc_out_f64;
  int64_t* const* hc_out_i64;
  uint8_t* const* hc_out_valid;
};

// Dynamic CUDA driver API entry points via cudaGetDriverEntryPoint (zero -lcuda link dependency!)
struct CudaDriverApi {
  using FnModLoad   = CUresult(CUDAAPI*)(CUmodule*, void const*);
  using FnModUnload = CUresult(CUDAAPI*)(CUmodule);
  using FnGetFunc   = CUresult(CUDAAPI*)(CUfunction*, CUmodule, char const*);
  using FnLaunch    = CUresult(CUDAAPI*)(CUfunction,
                                      unsigned int, unsigned int, unsigned int,
                                      unsigned int, unsigned int, unsigned int,
                                      unsigned int, CUstream, void**, void**);
  FnModLoad mod_load{nullptr};
  FnModUnload mod_unload{nullptr};
  FnGetFunc get_func{nullptr};
  FnLaunch launch{nullptr};

  static CudaDriverApi const& get()
  {
    static CudaDriverApi api = []() {
      CudaDriverApi a{};
      void* p = nullptr;
      CUDF_CUDA_TRY(cudaGetDriverEntryPoint("cuModuleLoadData", &p, cudaEnableDefault));
      a.mod_load = reinterpret_cast<FnModLoad>(p);
      CUDF_CUDA_TRY(cudaGetDriverEntryPoint("cuModuleUnload", &p, cudaEnableDefault));
      a.mod_unload = reinterpret_cast<FnModUnload>(p);
      CUDF_CUDA_TRY(cudaGetDriverEntryPoint("cuModuleGetFunction", &p, cudaEnableDefault));
      a.get_func = reinterpret_cast<FnGetFunc>(p);
      CUDF_CUDA_TRY(cudaGetDriverEntryPoint("cuLaunchKernel", &p, cudaEnableDefault));
      a.launch = reinterpret_cast<FnLaunch>(p);
      return a;
    }();
    return api;
  }
};

__global__ void jr_bool_to_bitmask_kernel(
  uint8_t const* __restrict__ valid_bytes, cudf::bitmask_type* __restrict__ mask, uint32_t n)
{
  uint32_t const widx   = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t const nwords = (n + 31u) >> 5;
  if (widx >= nwords) { return; }
  uint32_t const base  = widx << 5;
  cudf::bitmask_type w = 0u;
  for (uint32_t b = 0; b < 32u && base + b < n; ++b) {
    if (valid_bytes[base + b] != 0) { w |= (1u << b); }
  }
  mask[widx] = w;
}

constexpr char const* kNullGroupKeySentinel = "\x01__LENC_SQL_NULL_KEY__\x02";

static std::unique_ptr<cudf::column> make_string_col_from_host(
  std::vector<std::string> const& strs, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  size_t const m = strs.size();
  std::vector<int32_t> offs(m + 1, 0);
  std::vector<char> chars;
  size_t const mask_words =
    cudf::bitmask_allocation_size_bytes(static_cast<cudf::size_type>(m)) / sizeof(cudf::bitmask_type);
  std::vector<cudf::bitmask_type> h_mask(mask_words, ~cudf::bitmask_type{0});
  cudf::size_type null_cnt = 0;
  for (size_t i = 0; i < m; ++i) {
    if (strs[i] == kNullGroupKeySentinel) {
      h_mask[i >> 5] &= ~(cudf::bitmask_type{1} << (i & 31));
      ++null_cnt;
    } else {
      chars.insert(chars.end(), strs[i].begin(), strs[i].end());
    }
    offs[i + 1] = static_cast<int32_t>(chars.size());
  }
  auto d_offs = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                          static_cast<cudf::size_type>(m + 1),
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_offs->mutable_view().data<int32_t>(),
                                offs.data(),
                                (m + 1) * sizeof(int32_t),
                                cudaMemcpyHostToDevice,
                                stream.get()));
  rmm::device_buffer d_ch(chars.size(), stream, mr);
  if (!chars.empty()) {
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(d_ch.data(), chars.data(), chars.size(), cudaMemcpyHostToDevice, stream.get()));
  }
  auto nbuf = cudf::create_null_mask(
    null_cnt > 0 ? static_cast<cudf::size_type>(m) : 0,
    null_cnt > 0 ? cudf::mask_state::UNINITIALIZED : cudf::mask_state::UNALLOCATED,
    stream,
    mr);
  if (null_cnt > 0) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      nbuf.data(), h_mask.data(), mask_words * sizeof(cudf::bitmask_type), cudaMemcpyHostToDevice, stream.get()));
  }
  return cudf::make_strings_column(static_cast<cudf::size_type>(m),
                                   std::move(d_offs),
                                   std::move(d_ch),
                                   null_cnt,
                                   std::move(nbuf));
}

// Helper to materialize final output columns from Mode 0 (Global) / Mode 1 (Low-Card GroupBy)
static std::unique_ptr<cudf::table> finalize_lowcard_output(
  std::vector<std::vector<std::string>> const& group_key_dicts,
  std::vector<int32_t> const& group_key_strides,
  int32_t max_groups,
  std::vector<uint8_t> const& agg_kinds,
  std::vector<double> const& h_f64,
  std::vector<long long> const& h_i64,
  std::vector<unsigned long long> const& h_cnt,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  size_t const na = agg_kinds.size();
  size_t const ngk = group_key_dicts.size();
  std::vector<int32_t> active_gids;
  if (ngk == 0) {
    // Global reduce ALWAYS outputs 1 row (even if 0 rows matched -> SQL NULL sums / 0 counts!)
    active_gids.push_back(0);
  } else {
    for (int32_t g = 0; g < max_groups; ++g) {
      bool any = false;
      for (size_t a = 0; a < na; ++a) {
        if (h_cnt[static_cast<size_t>(g) * na + a] > 0ULL) {
          any = true;
          break;
        }
      }
      if (any) { active_gids.push_back(g); }
    }
  }

  size_t const out_rows = active_gids.size();
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(ngk + na);

  for (size_t k = 0; k < ngk; ++k) {
    std::vector<std::string> vals(out_rows);
    int32_t const stride = group_key_strides[k];
    int32_t const dsize  = static_cast<int32_t>(group_key_dicts[k].size());
    for (size_t i = 0; i < out_rows; ++i) {
      int32_t const cid = (active_gids[i] / stride) % dsize;
      vals[i]           = group_key_dicts[k][static_cast<size_t>(cid)];
    }
    cols.push_back(make_string_col_from_host(vals, stream, mr));
  }

  for (size_t a = 0; a < na; ++a) {
    uint8_t const kind = agg_kinds[a];
    bool const is_f64  = (kind == 0 || kind == 3 || kind == 4);
    bool const is_cnt  = (kind == 2);
    if (is_f64) {
      std::vector<double> vals(out_rows, 0.0);
      std::vector<uint8_t> valids(out_rows, 0);
      cudf::size_type nnull = 0;
      for (size_t i = 0; i < out_rows; ++i) {
        size_t const off = static_cast<size_t>(active_gids[i]) * na + a;
        if (h_cnt[off] > 0ULL) {
          vals[i]   = h_f64[off];
          valids[i] = 1;
        } else {
          ++nnull;
        }
      }
      auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT64},
                                           static_cast<cudf::size_type>(out_rows),
                                           cudf::mask_state::UNALLOCATED,
                                           stream,
                                           mr);
      if (out_rows > 0) {
        CUDF_CUDA_TRY(cudaMemcpyAsync(col->mutable_view().data<double>(),
                                      vals.data(),
                                      out_rows * sizeof(double),
                                      cudaMemcpyHostToDevice,
                                      stream.get()));
      }
      if (nnull > 0) {
        rmm::device_uvector<uint8_t> d_v(out_rows, stream, mr);
        CUDF_CUDA_TRY(
          cudaMemcpyAsync(d_v.data(), valids.data(), out_rows, cudaMemcpyHostToDevice, stream.get()));
        auto mask             = cudf::create_null_mask(out_rows, cudf::mask_state::UNINITIALIZED, stream, mr);
        uint32_t const nwords = (static_cast<uint32_t>(out_rows) + 31u) >> 5;
        jr_bool_to_bitmask_kernel<<<(nwords + 255u) / 256u, 256, 0, stream.get()>>>(
          d_v.data(), static_cast<cudf::bitmask_type*>(mask.data()), static_cast<uint32_t>(out_rows));
        col->set_null_mask(std::move(mask), nnull);
      }
      cols.push_back(std::move(col));
    } else {
      std::vector<int64_t> vals(out_rows, 0);
      std::vector<uint8_t> valids(out_rows, 1);
      cudf::size_type nnull = 0;
      for (size_t i = 0; i < out_rows; ++i) {
        size_t const off = static_cast<size_t>(active_gids[i]) * na + a;
        if (is_cnt) {
          vals[i] = static_cast<int64_t>(h_cnt[off]);
        } else if (h_cnt[off] > 0ULL) {
          vals[i] = static_cast<int64_t>(h_i64[off]);
        } else {
          valids[i] = 0;
          ++nnull;
        }
      }
      auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                           static_cast<cudf::size_type>(out_rows),
                                           cudf::mask_state::UNALLOCATED,
                                           stream,
                                           mr);
      if (out_rows > 0) {
        CUDF_CUDA_TRY(cudaMemcpyAsync(col->mutable_view().data<int64_t>(),
                                      vals.data(),
                                      out_rows * sizeof(int64_t),
                                      cudaMemcpyHostToDevice,
                                      stream.get()));
      }
      if (!is_cnt && nnull > 0) {
        rmm::device_uvector<uint8_t> d_v(out_rows, stream, mr);
        CUDF_CUDA_TRY(
          cudaMemcpyAsync(d_v.data(), valids.data(), out_rows, cudaMemcpyHostToDevice, stream.get()));
        auto mask             = cudf::create_null_mask(out_rows, cudf::mask_state::UNINITIALIZED, stream, mr);
        uint32_t const nwords = (static_cast<uint32_t>(out_rows) + 31u) >> 5;
        jr_bool_to_bitmask_kernel<<<(nwords + 255u) / 256u, 256, 0, stream.get()>>>(
          d_v.data(), static_cast<cudf::bitmask_type*>(mask.data()), static_cast<uint32_t>(out_rows));
        col->set_null_mask(std::move(mask), nnull);
      }
      cols.push_back(std::move(col));
    }
  }
  stream.sync();
  return std::make_unique<cudf::table>(std::move(cols));
}

template <typename IdxT>
__global__ void dict_remap_kernel(IdxT const* __restrict__ indices,
                                  int32_t const* __restrict__ remap,
                                  cudf::bitmask_type const* __restrict__ null_mask,
                                  int32_t* __restrict__ out,
                                  uint32_t n,
                                  uint32_t dict_size)
{
  uint32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    if (null_mask != nullptr && !cudf::bit_is_set(null_mask, static_cast<cudf::size_type>(idx))) {
      out[idx] = -1;
    } else {
      uint32_t const k = static_cast<uint32_t>(indices[idx]);
      out[idx]         = (k < dict_size) ? remap[k] : -1;
    }
  }
}

}  // namespace

std::unique_ptr<cudf::column> decode_dictionary(cudf::column_view const& dict_col,
                                                cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr)
{
  cudf::dictionary_column_view const dcv(dict_col);
  return cudf::dictionary::decode(dcv, stream, mr);
}

std::unique_ptr<cudf::column> remap_dictionary_to_canonical(
  cudf::column_view const& dict_col,
  std::vector<std::string> const& query_literals,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  cudf::dictionary_column_view const dcv(dict_col);
  uint32_t const num_rows  = static_cast<uint32_t>(dcv.size());
  cudf::column_view keys_v = dcv.keys();
  uint32_t const dict_size = static_cast<uint32_t>(keys_v.size());
  CUDF_EXPECTS(keys_v.type().id() == cudf::type_id::STRING,
               "remap_dictionary_to_canonical: keys must be STRING");

  std::unordered_map<std::string_view, int32_t> lit_map;
  lit_map.reserve(query_literals.size() * 2 + 1);
  for (size_t i = 0; i < query_literals.size(); ++i) {
    lit_map.emplace(std::string_view{query_literals[i]}, static_cast<int32_t>(i));
  }

  std::vector<int32_t> h_remap(dict_size, -1);
  if (dict_size > 0) {
    cudf::strings_column_view const scv(keys_v);
    std::vector<int32_t> h_offs(dict_size + 1, 0);
    CUDF_CUDA_TRY(cudaMemcpyAsync(h_offs.data(),
                                  scv.offsets().data<int32_t>() + keys_v.offset(),
                                  (dict_size + 1) * sizeof(int32_t),
                                  cudaMemcpyDeviceToHost,
                                  stream.get()));
    stream.sync();
    int32_t const base_off = h_offs[0];
    int32_t const total_ch = h_offs[dict_size] - base_off;
    std::vector<char> h_chars(static_cast<size_t>(total_ch));
    if (total_ch > 0) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(h_chars.data(),
                                    scv.chars_begin(stream) + base_off,
                                    static_cast<size_t>(total_ch),
                                    cudaMemcpyDeviceToHost,
                                    stream.get()));
      stream.sync();
    }
    for (uint32_t k = 0; k < dict_size; ++k) {
      int32_t const s = h_offs[k] - base_off;
      int32_t const e = h_offs[k + 1] - base_off;
      auto it         = lit_map.find(std::string_view(h_chars.data() + s, static_cast<size_t>(e - s)));
      if (it != lit_map.end()) { h_remap[k] = it->second; }
    }
  }

  bool const has_nulls = dict_col.nullable() && dict_col.null_count() > 0;
  auto null_buf        = has_nulls
                           ? cudf::copy_bitmask(dict_col, stream, mr)
                           : cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr);
  auto const null_cnt  = has_nulls ? dict_col.null_count() : 0;
  auto out_col         = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                           static_cast<cudf::size_type>(num_rows),
                                           std::move(null_buf),
                                           null_cnt,
                                           stream,
                                           mr);

  if (num_rows > 0) {
    rmm::device_buffer d_remap(std::max<size_t>(dict_size, 1) * sizeof(int32_t), stream, mr);
    if (dict_size > 0) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        d_remap.data(), h_remap.data(), dict_size * sizeof(int32_t), cudaMemcpyHostToDevice, stream.get()));
    }
    cudf::column_view const idx_col = dcv.indices();
    auto const* d_null              = has_nulls ? out_col->view().null_mask() : nullptr;
    auto* d_out                     = out_col->mutable_view().data<int32_t>();
    auto const* d_rm                = static_cast<int32_t const*>(d_remap.data());
    constexpr uint32_t kBlock       = 256;
    uint32_t const grid             = (num_rows + kBlock - 1) / kBlock;
    auto const off                  = dcv.offset();

    switch (idx_col.type().id()) {
      case cudf::type_id::INT32:
        dict_remap_kernel<int32_t><<<grid, kBlock, 0, stream.get()>>>(
          idx_col.data<int32_t>() + off, d_rm, d_null, d_out, num_rows, dict_size);
        break;
      case cudf::type_id::UINT32:
        dict_remap_kernel<uint32_t><<<grid, kBlock, 0, stream.get()>>>(
          idx_col.data<uint32_t>() + off, d_rm, d_null, d_out, num_rows, dict_size);
        break;
      case cudf::type_id::INT16:
        dict_remap_kernel<int16_t><<<grid, kBlock, 0, stream.get()>>>(
          idx_col.data<int16_t>() + off, d_rm, d_null, d_out, num_rows, dict_size);
        break;
      case cudf::type_id::UINT16:
        dict_remap_kernel<uint16_t><<<grid, kBlock, 0, stream.get()>>>(
          idx_col.data<uint16_t>() + off, d_rm, d_null, d_out, num_rows, dict_size);
        break;
      case cudf::type_id::INT8:
        dict_remap_kernel<int8_t><<<grid, kBlock, 0, stream.get()>>>(
          idx_col.data<int8_t>() + off, d_rm, d_null, d_out, num_rows, dict_size);
        break;
      case cudf::type_id::UINT8:
        dict_remap_kernel<uint8_t><<<grid, kBlock, 0, stream.get()>>>(
          idx_col.data<uint8_t>() + off, d_rm, d_null, d_out, num_rows, dict_size);
        break;
      default: CUDF_FAIL("remap_dictionary_to_canonical: unsupported dictionary index type");
    }
    stream.sync();
  }
  return out_col;
}

struct JitReduceProgram {
  CUmodule mod{nullptr};
  CUfunction func{nullptr};
  CUfunction func_blk{nullptr};
};

JitReduceProgram* compile_jit_reduce_program(std::string const& cuda_src,
                                             std::string const& kernel_name)
{
  nvrtcProgram prog{};
  auto rc = nvrtcCreateProgram(&prog, cuda_src.c_str(), "lenc_jit_reduce.cu", 0, nullptr, nullptr);
  CUDF_EXPECTS(rc == NVRTC_SUCCESS, "nvrtcCreateProgram failed");

  int dev = 0, major = 8, minor = 0;
  CUDF_CUDA_TRY(cudaGetDevice(&dev));
  CUDF_CUDA_TRY(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
  CUDF_CUDA_TRY(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));
  std::string arch_opt = "--gpu-architecture=compute_" + std::to_string(major * 10 + minor);
  char const* opts[]   = {arch_opt.c_str(), "--std=c++17", "--fmad=false", "--device-int128"};

  rc = nvrtcCompileProgram(prog, 4, opts);
  if (rc != NVRTC_SUCCESS) {
    size_t log_sz = 0;
    nvrtcGetProgramLogSize(prog, &log_sz);
    std::string log(log_sz, '\0');
    nvrtcGetProgramLog(prog, log.data());
    nvrtcDestroyProgram(&prog);
    CUDF_FAIL("NVRTC compile failed for " + kernel_name + ":\n" + log);
  }
  size_t ptx_sz = 0;
  nvrtcGetPTXSize(prog, &ptx_sz);
  std::string ptx(ptx_sz, '\0');
  nvrtcGetPTX(prog, ptx.data());
  nvrtcDestroyProgram(&prog);

  auto const& drv = CudaDriverApi::get();
  auto* out       = new JitReduceProgram();
  CUresult cr     = drv.mod_load(&out->mod, ptx.data());
  if (cr != CUDA_SUCCESS) {
    delete out;
    CUDF_FAIL("cuModuleLoadData failed: " + std::to_string(static_cast<int>(cr)));
  }
  cr = drv.get_func(&out->func, out->mod, kernel_name.c_str());
  if (cr != CUDA_SUCCESS) {
    drv.mod_unload(out->mod);
    delete out;
    CUDF_FAIL("cuModuleGetFunction failed for " + kernel_name);
  }
  std::string const blk_name = kernel_name + "_blk";
  if (drv.get_func(&out->func_blk, out->mod, blk_name.c_str()) != CUDA_SUCCESS) {
    out->func_blk = nullptr;
  }
  return out;
}

void destroy_jit_reduce_program(JitReduceProgram* prog)
{
  if (prog != nullptr) {
    if (prog->mod != nullptr) { CudaDriverApi::get().mod_unload(prog->mod); }
    delete prog;
  }
}

std::unique_ptr<cudf::table> run_jit_reduce_table(
  JitReduceProgram& prog,
  cudf::table_view const& table,
  std::vector<std::vector<std::string>> const& dict_lits,
  std::vector<int32_t> const& group_key_slots,
  int32_t max_groups,
  std::vector<uint8_t> const& agg_kinds,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  uint32_t const n = static_cast<uint32_t>(table.num_rows());
  size_t const nc  = static_cast<size_t>(table.num_columns());
  size_t const na  = agg_kinds.size();
  CUDF_EXPECTS(nc <= static_cast<size_t>(kMaxJitCols), "jit_reduce_table: max 16 cols");

  JitReduceArgs args{};
  args.num_rows   = n;
  args.max_groups = max_groups;
  args.num_aggs   = static_cast<int32_t>(na);

  std::vector<std::unique_ptr<cudf::column>> owned_cols;
  std::vector<std::vector<std::string>> col_dicts(nc);

  for (size_t s = 0; s < nc; ++s) {
    cudf::column_view cv = table.column(static_cast<cudf::size_type>(s));
    JitColDesc& cd       = args.cols[s];
    cd.mode              = 0;
    cd.has_nulls         = (cv.nullable() && cv.null_count() > 0) ? 1 : 0;
    cd.null_mask         = cd.has_nulls ? reinterpret_cast<uint32_t const*>(cv.null_mask()) : nullptr;

    bool const is_gk =
      std::find(group_key_slots.begin(), group_key_slots.end(), static_cast<int32_t>(s)) !=
      group_key_slots.end();
    bool const has_lits = (s < dict_lits.size() && !dict_lits[s].empty());

    if (cv.type().id() == cudf::type_id::STRING || cv.type().id() == cudf::type_id::DICTIONARY32) {
      cudf::column_view dview = cv;
      if (cv.type().id() == cudf::type_id::STRING) {
        auto enc = cudf::dictionary::encode(cv, cudf::data_type{cudf::type_id::INT32}, stream, mr);
        dview    = enc->view();
        owned_cols.push_back(std::move(enc));
      }
      if (has_lits) {
        auto canon = remap_dictionary_to_canonical(dview, dict_lits[s], stream, mr);
        cd.raw_ptr = canon->view().head();
        owned_cols.push_back(std::move(canon));
      } else if (is_gk) {
        cudf::dictionary_column_view const dcv(dview);
        cudf::strings_column_view const ksv(dcv.keys());
        uint32_t const ds = static_cast<uint32_t>(dcv.keys_size());
        std::vector<int32_t> h_offs(ds + 1u, 0);
        if (ds > 0) {
          CUDF_CUDA_TRY(cudaMemcpyAsync(h_offs.data(),
                                        ksv.offsets().data<int32_t>(),
                                        (ds + 1u) * sizeof(int32_t),
                                        cudaMemcpyDeviceToHost,
                                        stream.get()));
          stream.sync();
        }
        int32_t const kb = h_offs[ds];
        std::vector<char> h_ch(static_cast<size_t>(kb));
        if (kb > 0) {
          CUDF_CUDA_TRY(cudaMemcpyAsync(
            h_ch.data(), ksv.chars_begin(stream), static_cast<size_t>(kb), cudaMemcpyDeviceToHost, stream.get()));
          stream.sync();
        }
        col_dicts[s].resize(ds);
        for (uint32_t k = 0; k < ds; ++k) {
          col_dicts[s][k].assign(h_ch.data() + h_offs[k], static_cast<size_t>(h_offs[k + 1] - h_offs[k]));
        }
        cudf::column_view const idx_col = dcv.indices();
        auto const idx_off              = static_cast<size_t>(dcv.offset());
        switch (idx_col.type().id()) {
          case cudf::type_id::UINT8:
            cd.mode    = 4;
            cd.raw_ptr = idx_col.data<uint8_t>() + idx_off;
            break;
          case cudf::type_id::UINT16:
            cd.mode    = 5;
            cd.raw_ptr = idx_col.data<uint16_t>() + idx_off;
            break;
          case cudf::type_id::INT8:
            cd.mode    = 6;
            cd.raw_ptr = idx_col.data<int8_t>() + idx_off;
            break;
          case cudf::type_id::INT16:
            cd.mode    = 7;
            cd.raw_ptr = idx_col.data<int16_t>() + idx_off;
            break;
          default:
            cd.mode    = 0;
            cd.raw_ptr = static_cast<uint8_t const*>(idx_col.head()) + idx_off * sizeof(int32_t);
            break;
        }
        cd.dict_size = ds;
      }
    } else {
      size_t const esz = cudf::size_of(cv.type());
      cd.raw_ptr       = static_cast<uint8_t const*>(cv.head()) + static_cast<size_t>(cv.offset()) * esz;
    }
  }

  std::vector<std::vector<std::string>> gk_dicts;
  std::vector<int32_t> gk_strides;
  int32_t eff_max_groups = 1;
  if (!group_key_slots.empty()) {
    gk_dicts.resize(group_key_slots.size());
    gk_strides.resize(group_key_slots.size(), 1);
    for (int i = static_cast<int>(group_key_slots.size()) - 1; i >= 0; --i) {
      int32_t const s                    = group_key_slots[static_cast<size_t>(i)];
      gk_dicts[static_cast<size_t>(i)]   = col_dicts[static_cast<size_t>(s)];
      gk_strides[static_cast<size_t>(i)] = eff_max_groups;
      eff_max_groups *= std::max<int32_t>(static_cast<int32_t>(gk_dicts[static_cast<size_t>(i)].size()), 1);
    }
    CUDF_EXPECTS(eff_max_groups <= max_groups, "jit_reduce_table: dict product exceeds max_groups");
    args.max_groups = eff_max_groups;
  }

  size_t const total_slots = static_cast<size_t>(args.max_groups) * na;
  std::vector<double> init_f64(total_slots, 0.0);
  std::vector<long long> init_i64(total_slots, 0LL);
  for (int32_t g = 0; g < args.max_groups; ++g) {
    for (size_t a = 0; a < na; ++a) {
      size_t const off = static_cast<size_t>(g) * na + a;
      if (agg_kinds[a] == 3) init_f64[off] = std::numeric_limits<double>::infinity();
      else if (agg_kinds[a] == 4) init_f64[off] = -std::numeric_limits<double>::infinity();
      else if (agg_kinds[a] == 5) init_i64[off] = std::numeric_limits<long long>::max();
      else if (agg_kinds[a] == 6) init_i64[off] = std::numeric_limits<long long>::min();
    }
  }
  rmm::device_uvector<double> d_f64(total_slots, stream, mr);
  rmm::device_uvector<long long> d_i64(total_slots, stream, mr);
  rmm::device_uvector<unsigned long long> d_cnt(total_slots, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    d_f64.data(), init_f64.data(), total_slots * sizeof(double), cudaMemcpyHostToDevice, stream.get()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    d_i64.data(), init_i64.data(), total_slots * sizeof(long long), cudaMemcpyHostToDevice, stream.get()));
  CUDF_CUDA_TRY(cudaMemsetAsync(d_cnt.data(), 0, total_slots * sizeof(unsigned long long), stream.get()));

  args.out_f64 = d_f64.data();
  args.out_i64 = d_i64.data();
  args.out_cnt = d_cnt.data();

  if (n > 0) {
    rmm::device_buffer d_args(sizeof(JitReduceArgs), stream, mr);
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(d_args.data(), &args, sizeof(JitReduceArgs), cudaMemcpyHostToDevice, stream.get()));
    void* arg_ptr   = d_args.data();
    void* kparams[] = {&arg_ptr};
    uint32_t const blocks = std::min<uint32_t>((n + 255u) / 256u, 2048u);
    CUresult cr = CudaDriverApi::get().launch(
      prog.func, blocks, 1, 1, 256, 1, 1, 0, static_cast<CUstream>(stream.get()), kparams, nullptr);
    CUDF_EXPECTS(cr == CUDA_SUCCESS, "cuLaunchKernel failed in run_jit_reduce_table");
  }

  std::vector<double> h_f64(total_slots);
  std::vector<long long> h_i64(total_slots);
  std::vector<unsigned long long> h_cnt(total_slots);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    h_f64.data(), d_f64.data(), total_slots * sizeof(double), cudaMemcpyDeviceToHost, stream.get()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    h_i64.data(), d_i64.data(), total_slots * sizeof(long long), cudaMemcpyDeviceToHost, stream.get()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    h_cnt.data(), d_cnt.data(), total_slots * sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream.get()));
  stream.sync();

  return finalize_lowcard_output(
    gk_dicts, gk_strides, args.max_groups, agg_kinds, h_f64, h_i64, h_cnt, stream, mr);
}

// ---------------- Generic Per-Task Cross-Batch GPU Accumulator ----------------
// Keeps one persistent d_f64/d_i64/d_cnt GPU buffer across ALL input batches of a Spark task!
// Eliminates per-batch GPU alloc/init/DtoH/sync + per-batch CudfTable creation & downstream shuffle
// fragmentation for EVERY scan format (Parquet, Vortex, ORC, CSV).

namespace {

constexpr uint32_t kSmallDictPackBytes = 4096u;

__global__ void pack_small_dict_keys_kernel(int32_t const* __restrict__ offsets,
                                            char const* __restrict__ chars,
                                            uint32_t ds,
                                            uint8_t* __restrict__ out_buf)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    auto* out_i32   = reinterpret_cast<int32_t*>(out_buf);
    int32_t base    = (ds > 0u) ? offsets[0] : 0;
    int32_t total_c = (ds > 0u) ? (offsets[ds] - base) : 0;
    uint32_t hdr_b  = (ds + 2u) * sizeof(int32_t);
    out_i32[0]      = static_cast<int32_t>(ds);
    if (hdr_b + static_cast<uint32_t>(total_c) > kSmallDictPackBytes) {
      out_i32[0] = -1;  // signal caller to fall back to standard two-step readback
      return;
    }
    for (uint32_t k = 0; k <= ds; ++k) { out_i32[1u + k] = offsets[k] - base; }
    char* dst_c = reinterpret_cast<char*>(out_buf + hdr_b);
    for (int32_t i = 0; i < total_c; ++i) { dst_c[i] = chars[base + i]; }
  }
}

}  // namespace

struct JitReduceTaskAccumulator {
  JitReduceProgram* prog{nullptr};
  std::vector<std::vector<std::string>> dict_lits;
  std::vector<int32_t> group_key_slots;
  int32_t max_groups{1};
  std::vector<uint8_t> agg_kinds;
  std::vector<int32_t> key_caps;
  std::vector<int32_t> key_strides;
  std::vector<std::vector<std::string>> task_gk_dicts;
  std::vector<std::unordered_map<std::string, int32_t>> task_gk_maps;
  int32_t eff_max_groups{1};
  bool any_rows{false};

  rmm::device_uvector<double> d_f64;
  rmm::device_uvector<long long> d_i64;
  rmm::device_uvector<unsigned long long> d_cnt;
  rmm::device_buffer d_args;
  rmm::device_uvector<uint8_t> d_pack_buf;
  std::vector<rmm::device_uvector<int32_t>> d_remaps;
  std::vector<std::unique_ptr<cudf::table>> overflow_tables;

  JitReduceTaskAccumulator(JitReduceProgram& p,
                           std::vector<std::vector<std::string>> const& dl,
                           std::vector<int32_t> const& gks,
                           int32_t mg,
                           std::vector<uint8_t> const& ak,
                           cuda::stream_ref stream,
                           rmm::device_async_resource_ref mr)
    : prog(&p),
      dict_lits(dl),
      group_key_slots(gks),
      max_groups(std::max<int32_t>(mg, 1)),
      agg_kinds(ak),
      d_f64(0, stream, mr),
      d_i64(0, stream, mr),
      d_cnt(0, stream, mr),
      d_args(sizeof(JitReduceArgs), stream, mr),
      d_pack_buf(0, stream, mr)
  {
    size_t const ngk = group_key_slots.size();
    task_gk_dicts.resize(ngk);
    task_gk_maps.resize(ngk);
    key_caps.resize(ngk, 1);
    key_strides.resize(ngk, 1);
    if (ngk == 0) {
      eff_max_groups = 1;
    } else if (ngk == 1) {
      key_caps[0]    = max_groups;
      key_strides[0] = 1;
      eff_max_groups = max_groups;
    } else if (ngk == 2) {
      int32_t const c1 = std::max<int32_t>(1, static_cast<int32_t>(std::sqrt(static_cast<double>(max_groups))));
      int32_t const c0 = std::max<int32_t>(1, max_groups / c1);
      key_caps[0]      = c0;
      key_caps[1]      = c1;
      key_strides[0]   = c1;
      key_strides[1]   = 1;
      eff_max_groups   = c0 * c1;
    } else {
      eff_max_groups = 0;  // fall back to run_jit_reduce_table
    }

    if (eff_max_groups > 0) {
      size_t const na    = agg_kinds.size();
      size_t const slots = static_cast<size_t>(eff_max_groups) * na;
      d_f64.resize(slots, stream);
      d_i64.resize(slots, stream);
      d_cnt.resize(slots, stream);
      if (ngk > 0) {
        d_pack_buf.resize(ngk * kSmallDictPackBytes, stream);
        d_remaps.reserve(ngk);
        for (size_t i = 0; i < ngk; ++i) {
          d_remaps.emplace_back(static_cast<size_t>(std::max<int32_t>(max_groups, 64)), stream, mr);
        }
      }
      reset_device_buffers(stream);
    }
  }

  void reset_device_buffers(cuda::stream_ref stream)
  {
    size_t const na    = agg_kinds.size();
    size_t const slots = static_cast<size_t>(eff_max_groups) * na;
    std::vector<double> init_f64(slots, 0.0);
    std::vector<long long> init_i64(slots, 0LL);
    for (int32_t g = 0; g < eff_max_groups; ++g) {
      for (size_t a = 0; a < na; ++a) {
        size_t const off = static_cast<size_t>(g) * na + a;
        if (agg_kinds[a] == 3) init_f64[off] = std::numeric_limits<double>::infinity();
        else if (agg_kinds[a] == 4) init_f64[off] = -std::numeric_limits<double>::infinity();
        else if (agg_kinds[a] == 5) init_i64[off] = std::numeric_limits<long long>::max();
        else if (agg_kinds[a] == 6) init_i64[off] = std::numeric_limits<long long>::min();
      }
    }
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      d_f64.data(), init_f64.data(), slots * sizeof(double), cudaMemcpyHostToDevice, stream.get()));
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      d_i64.data(), init_i64.data(), slots * sizeof(long long), cudaMemcpyHostToDevice, stream.get()));
    CUDF_CUDA_TRY(cudaMemsetAsync(d_cnt.data(), 0, slots * sizeof(unsigned long long), stream.get()));
  }
};

JitReduceTaskAccumulator* open_jit_reduce_task(
  JitReduceProgram& prog,
  std::vector<std::vector<std::string>> const& dict_lits,
  std::vector<int32_t> const& group_key_slots,
  int32_t max_groups,
  std::vector<uint8_t> const& agg_kinds,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  return new JitReduceTaskAccumulator(prog, dict_lits, group_key_slots, max_groups, agg_kinds, stream, mr);
}

void add_batch_jit_reduce_task(JitReduceTaskAccumulator& acc,
                               cudf::table_view const& table,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
{
  auto const n = static_cast<uint32_t>(table.num_rows());
  if (n == 0) { return; }
  if (acc.eff_max_groups <= 0) {
    acc.overflow_tables.push_back(run_jit_reduce_table(
      *acc.prog, table, acc.dict_lits, acc.group_key_slots, acc.max_groups, acc.agg_kinds, stream, mr));
    return;
  }

  size_t const nc  = static_cast<size_t>(table.num_columns());
  size_t const na  = acc.agg_kinds.size();
  size_t const ngk = acc.group_key_slots.size();

  JitReduceArgs args{};
  args.num_rows   = n;
  args.max_groups = acc.eff_max_groups;
  args.num_aggs   = static_cast<int32_t>(na);
  args.out_f64    = acc.d_f64.data();
  args.out_i64    = acc.d_i64.data();
  args.out_cnt    = acc.d_cnt.data();

  std::vector<std::unique_ptr<cudf::column>> owned_cols;
  struct GkBatchInfo {
    size_t gk_idx;
    size_t slot;
    cudf::column_view dview;
    uint32_t ds;
  };
  std::vector<GkBatchInfo> gk_infos;
  gk_infos.reserve(ngk);

  for (size_t s = 0; s < nc; ++s) {
    cudf::column_view cv = table.column(static_cast<cudf::size_type>(s));
    JitColDesc& cd       = args.cols[s];
    cd.mode              = 0;
    cd.has_nulls         = (cv.nullable() && cv.null_count() > 0) ? 1 : 0;
    cd.null_mask         = cd.has_nulls ? reinterpret_cast<uint32_t const*>(cv.null_mask()) : nullptr;

    auto gk_it = std::find(acc.group_key_slots.begin(), acc.group_key_slots.end(), static_cast<int32_t>(s));
    bool const is_gk    = (gk_it != acc.group_key_slots.end());
    bool const has_lits = (s < acc.dict_lits.size() && !acc.dict_lits[s].empty());

    if (cv.type().id() == cudf::type_id::STRING || cv.type().id() == cudf::type_id::DICTIONARY32) {
      cudf::column_view dview = cv;
      if (cv.type().id() == cudf::type_id::STRING) {
        auto enc = cudf::dictionary::encode(cv, cudf::data_type{cudf::type_id::INT32}, stream, mr);
        dview    = enc->view();
        owned_cols.push_back(std::move(enc));
      }
      if (has_lits) {
        auto canon = remap_dictionary_to_canonical(dview, acc.dict_lits[s], stream, mr);
        cd.raw_ptr = canon->view().head();
        owned_cols.push_back(std::move(canon));
      } else if (is_gk) {
        size_t const gk_idx = static_cast<size_t>(std::distance(acc.group_key_slots.begin(), gk_it));
        cudf::dictionary_column_view const dcv(dview);
        uint32_t const ds = static_cast<uint32_t>(dcv.keys_size());
        gk_infos.push_back(GkBatchInfo{gk_idx, s, dview, ds});

        // Pack small dictionary keys on GPU so ALL group-key cols read back in 1 single sync!
        cudf::strings_column_view const ksv(dcv.keys());
        uint8_t* dst_pack = acc.d_pack_buf.data() + gk_idx * kSmallDictPackBytes;
        pack_small_dict_keys_kernel<<<1, 1, 0, stream.get()>>>(
          ksv.offsets().data<int32_t>() + dcv.keys().offset(),
          ksv.chars_begin(stream),
          ds,
          dst_pack);

        cudf::column_view const idx_col = dcv.indices();
        auto const idx_off              = static_cast<size_t>(dcv.offset());
        switch (idx_col.type().id()) {
          case cudf::type_id::UINT8:
            cd.mode    = 4;
            cd.raw_ptr = idx_col.data<uint8_t>() + idx_off;
            break;
          case cudf::type_id::UINT16:
            cd.mode    = 5;
            cd.raw_ptr = idx_col.data<uint16_t>() + idx_off;
            break;
          case cudf::type_id::INT8:
            cd.mode    = 6;
            cd.raw_ptr = idx_col.data<int8_t>() + idx_off;
            break;
          case cudf::type_id::INT16:
            cd.mode    = 7;
            cd.raw_ptr = idx_col.data<int16_t>() + idx_off;
            break;
          default:
            cd.mode    = 0;
            cd.raw_ptr = static_cast<uint8_t const*>(idx_col.head()) + idx_off * sizeof(int32_t);
            break;
        }
        cd.dict_size = static_cast<uint32_t>(acc.key_caps[gk_idx]);
      }
    } else {
      size_t const esz = cudf::size_of(cv.type());
      cd.raw_ptr       = static_cast<uint8_t const*>(cv.head()) + static_cast<size_t>(cv.offset()) * esz;
    }
  }

  // Single combined DtoH sync for ALL group-key dictionaries (0 syncs for global reduce!)
  bool cap_exceeded = false;
  if (!gk_infos.empty()) {
    std::vector<uint8_t> h_pack(ngk * kSmallDictPackBytes);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      h_pack.data(), acc.d_pack_buf.data(), h_pack.size(), cudaMemcpyDeviceToHost, stream.get()));
    stream.sync();

    for (auto const& gi : gk_infos) {
      uint8_t const* pb  = h_pack.data() + gi.gk_idx * kSmallDictPackBytes;
      auto const* p_i32  = reinterpret_cast<int32_t const*>(pb);
      int32_t const flag = p_i32[0];
      std::vector<std::string> batch_keys(gi.ds);
      if (flag >= 0) {
        char const* ch_ptr = reinterpret_cast<char const*>(pb + (gi.ds + 2u) * sizeof(int32_t));
        for (uint32_t k = 0; k < gi.ds; ++k) {
          batch_keys[k].assign(ch_ptr + p_i32[1u + k], static_cast<size_t>(p_i32[2u + k] - p_i32[1u + k]));
        }
      } else {
        // Fallback for unusually long string keys (> 4 KiB dictionary payload)
        cudf::dictionary_column_view const dcv(gi.dview);
        cudf::strings_column_view const ksv(dcv.keys());
        std::vector<int32_t> h_offs(gi.ds + 1u, 0);
        if (gi.ds > 0) {
          CUDF_CUDA_TRY(cudaMemcpyAsync(h_offs.data(),
                                        ksv.offsets().data<int32_t>() + dcv.keys().offset(),
                                        (gi.ds + 1u) * sizeof(int32_t),
                                        cudaMemcpyDeviceToHost,
                                        stream.get()));
          stream.sync();
        }
        int32_t const base_o = h_offs[0];
        int32_t const kb     = h_offs[gi.ds] - base_o;
        std::vector<char> h_ch(static_cast<size_t>(kb));
        if (kb > 0) {
          CUDF_CUDA_TRY(cudaMemcpyAsync(h_ch.data(),
                                        ksv.chars_begin(stream) + base_o,
                                        static_cast<size_t>(kb),
                                        cudaMemcpyDeviceToHost,
                                        stream.get()));
          stream.sync();
        }
        for (uint32_t k = 0; k < gi.ds; ++k) {
          batch_keys[k].assign(h_ch.data() + (h_offs[k] - base_o),
                               static_cast<size_t>(h_offs[k + 1] - h_offs[k]));
        }
      }

      std::vector<int32_t> h_remap(gi.ds + 1u, -1);
      auto& tdict = acc.task_gk_dicts[gi.gk_idx];
      auto& tmap  = acc.task_gk_maps[gi.gk_idx];
      for (uint32_t k = 0; k < gi.ds; ++k) {
        auto it = tmap.find(batch_keys[k]);
        if (it == tmap.end()) {
          if (static_cast<int32_t>(tdict.size()) >= acc.key_caps[gi.gk_idx]) {
            cap_exceeded = true;
            break;
          }
          int32_t const new_id = static_cast<int32_t>(tdict.size());
          tmap.emplace(batch_keys[k], new_id);
          tdict.push_back(batch_keys[k]);
          h_remap[k] = new_id;
        } else {
          h_remap[k] = it->second;
        }
      }
      // Reserve an explicit NULL group-key code in slot `gi.ds` whenever this batch has NULLs!
      int32_t null_code = -1;
      if (!cap_exceeded && args.cols[gi.slot].has_nulls != 0u) {
        std::string const null_k(kNullGroupKeySentinel);
        auto it = tmap.find(null_k);
        if (it == tmap.end()) {
          if (static_cast<int32_t>(tdict.size()) >= acc.key_caps[gi.gk_idx]) {
            cap_exceeded = true;
          } else {
            null_code = static_cast<int32_t>(tdict.size());
            tmap.emplace(null_k, null_code);
            tdict.push_back(null_k);
          }
        } else {
          null_code = it->second;
        }
      }
      if (cap_exceeded) { break; }
      h_remap[gi.ds] = null_code;
      args.cols[gi.slot].rle_num_runs = gi.ds;  // pass local dict_size so NULL rows read dict_remap[rle_num_runs]
      if (acc.d_remaps[gi.gk_idx].size() < gi.ds + 1u) {
        acc.d_remaps[gi.gk_idx].resize(gi.ds + 1u, stream);
      }
      CUDF_CUDA_TRY(cudaMemcpyAsync(acc.d_remaps[gi.gk_idx].data(),
                                    h_remap.data(),
                                    (gi.ds + 1u) * sizeof(int32_t),
                                    cudaMemcpyHostToDevice,
                                    stream.get()));
      args.cols[gi.slot].dict_remap = acc.d_remaps[gi.gk_idx].data();
    }
  }

  if (cap_exceeded) {
    acc.overflow_tables.push_back(run_jit_reduce_table(
      *acc.prog, table, acc.dict_lits, acc.group_key_slots, acc.max_groups, acc.agg_kinds, stream, mr));
    return;
  }

  // Launch asynchronously directly into persistent task GPU accumulators (ZERO result DtoH/sync!)
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    acc.d_args.data(), &args, sizeof(JitReduceArgs), cudaMemcpyHostToDevice, stream.get()));
  void* arg_ptr   = acc.d_args.data();
  void* kparams[] = {&arg_ptr};
  uint32_t const blocks = std::min<uint32_t>((n + 255u) / 256u, 2048u);
  CUresult cr = CudaDriverApi::get().launch(
    acc.prog->func, blocks, 1, 1, 256, 1, 1, 0, static_cast<CUstream>(stream.get()), kparams, nullptr);
  CUDF_EXPECTS(cr == CUDA_SUCCESS, "cuLaunchKernel failed in add_batch_jit_reduce_task");
  acc.any_rows = true;
}

std::unique_ptr<cudf::table> finish_jit_reduce_task(JitReduceTaskAccumulator& acc,
                                                    cuda::stream_ref stream,
                                                    rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<cudf::table>> parts;
  if (acc.any_rows && acc.eff_max_groups > 0) {
    size_t const na    = acc.agg_kinds.size();
    size_t const slots = static_cast<size_t>(acc.eff_max_groups) * na;
    std::vector<double> h_f64(slots);
    std::vector<long long> h_i64(slots);
    std::vector<unsigned long long> h_cnt(slots);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      h_f64.data(), acc.d_f64.data(), slots * sizeof(double), cudaMemcpyDeviceToHost, stream.get()));
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      h_i64.data(), acc.d_i64.data(), slots * sizeof(long long), cudaMemcpyDeviceToHost, stream.get()));
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      h_cnt.data(), acc.d_cnt.data(), slots * sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream.get()));
    stream.sync();

    // Pad task_gk_dicts to key_caps only via strides so finalize_lowcard_output decodes g correctly
    std::vector<std::vector<std::string>> padded_dicts = acc.task_gk_dicts;
    for (size_t i = 0; i < padded_dicts.size(); ++i) {
      if (padded_dicts[i].size() < static_cast<size_t>(acc.key_caps[i])) {
        padded_dicts[i].resize(static_cast<size_t>(acc.key_caps[i]));
      }
    }
    auto main_tbl = finalize_lowcard_output(
      padded_dicts, acc.key_strides, acc.eff_max_groups, acc.agg_kinds, h_f64, h_i64, h_cnt, stream, mr);
    if (main_tbl->num_rows() > 0) { parts.push_back(std::move(main_tbl)); }
  }
  for (auto& ot : acc.overflow_tables) {
    if (ot && ot->num_rows() > 0) { parts.push_back(std::move(ot)); }
  }
  acc.overflow_tables.clear();

  if (parts.empty()) {
    std::vector<std::vector<std::string>> empty_dicts(acc.group_key_slots.size());
    std::vector<int32_t> empty_strides(acc.group_key_slots.size(), 1);
    std::vector<double> z_f64(acc.agg_kinds.size(), 0.0);
    std::vector<long long> z_i64(acc.agg_kinds.size(), 0LL);
    std::vector<unsigned long long> z_cnt(acc.agg_kinds.size(), 0ULL);
    return finalize_lowcard_output(
      empty_dicts, empty_strides, 0, acc.agg_kinds, z_f64, z_i64, z_cnt, stream, mr);
  }
  if (parts.size() == 1) { return std::move(parts[0]); }
  std::vector<cudf::table_view> views;
  views.reserve(parts.size());
  for (auto const& t : parts) { views.push_back(t->view()); }
  return cudf::concatenate(views, stream, mr);
}

void destroy_jit_reduce_task(JitReduceTaskAccumulator* acc) { delete acc; }

std::unique_ptr<cudf::table> run_jit_reduce_high_card(
  JitReduceProgram& prog,
  cudf::table_view const& group_keys,
  cudf::table_view const& value_cols,
  std::vector<std::vector<std::string>> const& dict_lits,
  std::vector<uint8_t> const& agg_kinds,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto const n    = group_keys.num_rows();
  size_t const nv = static_cast<size_t>(value_cols.num_columns());
  size_t const na = agg_kinds.size();
  CUDF_EXPECTS(nv <= static_cast<size_t>(kMaxJitCols), "jit_reduce_hc: max 16 value cols");
  CUDF_EXPECTS(na <= static_cast<size_t>(kMaxJitAggs), "jit_reduce_hc: max 64 aggs");

  // Step 1: Single cudf::groupby::get_groups() over arbitrary group_keys
  cudf::groupby::groupby gb(group_keys, cudf::null_policy::INCLUDE);
  auto seq_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                           n,
                                           cudf::mask_state::UNALLOCATED,
                                           stream,
                                           mr);
  if (n > 0) {
    thrust::sequence(rmm::exec_policy_nosync(stream),
                     seq_col->mutable_view().data<int32_t>(),
                     seq_col->mutable_view().data<int32_t>() + n);
  }
  auto grouped_seq           = gb.get_groups(cudf::table_view{{seq_col->view()}}, stream, mr);
  auto const grouped_ids_col = grouped_seq.values->view().column(0);
  int32_t const num_groups =
    static_cast<int32_t>(grouped_seq.offsets.empty() ? 0 : (grouped_seq.offsets.size() - 1));

  rmm::device_uvector<int32_t> d_group_offs(grouped_seq.offsets.size(), stream, mr);
  if (!grouped_seq.offsets.empty()) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(d_group_offs.data(),
                                  grouped_seq.offsets.data(),
                                  grouped_seq.offsets.size() * sizeof(int32_t),
                                  cudaMemcpyHostToDevice,
                                  stream.get()));
  }
  cudf::column_view unique_key_idx(
    cudf::data_type{cudf::type_id::INT32}, num_groups, d_group_offs.data(), nullptr, 0);
  auto unique_keys_tbl = cudf::gather(
    grouped_seq.keys->view(), unique_key_idx, cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
  auto unique_keys_cols = unique_keys_tbl->release();

  // Step 2: Populate JitReduceArgs for value_cols (auto-remapping DICTIONARY32/STRING -> canonical INT32)
  JitReduceArgs args{};
  args.num_rows        = static_cast<uint32_t>(n);
  args.num_aggs        = static_cast<int32_t>(na);
  args.num_groups      = num_groups;
  args.group_offsets   = d_group_offs.data();
  args.grouped_row_ids = grouped_ids_col.data<int32_t>();

  std::vector<std::unique_ptr<cudf::column>> owned_cols;
  for (size_t s = 0; s < nv; ++s) {
    cudf::column_view cv = value_cols.column(static_cast<cudf::size_type>(s));
    JitColDesc& cd       = args.cols[s];
    cd.mode              = 0;
    cd.has_nulls         = (cv.nullable() && cv.null_count() > 0) ? 1 : 0;
    cd.null_mask         = cd.has_nulls ? reinterpret_cast<uint32_t const*>(cv.null_mask()) : nullptr;
    if (cv.type().id() == cudf::type_id::STRING || cv.type().id() == cudf::type_id::DICTIONARY32) {
      cudf::column_view dview = cv;
      if (cv.type().id() == cudf::type_id::STRING) {
        auto enc = cudf::dictionary::encode(cv, cudf::data_type{cudf::type_id::INT32}, stream, mr);
        dview    = enc->view();
        owned_cols.push_back(std::move(enc));
      }
      auto canon = remap_dictionary_to_canonical(dview, dict_lits[s], stream, mr);
      cd.raw_ptr = canon->view().head();
      owned_cols.push_back(std::move(canon));
    } else {
      size_t const esz = cudf::size_of(cv.type());
      cd.raw_ptr       = static_cast<uint8_t const*>(cv.head()) + static_cast<size_t>(cv.offset()) * esz;
    }
  }

  // Step 3: Allocate output agg columns & valid-byte arrays
  std::vector<std::unique_ptr<cudf::column>> out_agg_cols;
  std::vector<rmm::device_uvector<uint8_t>> out_valid_bytes;
  std::vector<double*> h_f64_ptrs(na, nullptr);
  std::vector<int64_t*> h_i64_ptrs(na, nullptr);
  std::vector<uint8_t*> h_val_ptrs(na, nullptr);
  out_agg_cols.reserve(na);
  out_valid_bytes.reserve(na);

  for (size_t a = 0; a < na; ++a) {
    bool const is_f64 = (agg_kinds[a] == 0 || agg_kinds[a] == 3 || agg_kinds[a] == 4);
    auto c            = cudf::make_numeric_column(
      cudf::data_type{is_f64 ? cudf::type_id::FLOAT64 : cudf::type_id::INT64},
      num_groups,
      cudf::mask_state::UNALLOCATED,
      stream,
      mr);
    if (is_f64) h_f64_ptrs[a] = c->mutable_view().data<double>();
    else h_i64_ptrs[a] = c->mutable_view().data<int64_t>();
    out_agg_cols.push_back(std::move(c));
    rmm::device_uvector<uint8_t> vb(static_cast<size_t>(num_groups), stream, mr);
    h_val_ptrs[a] = vb.data();
    out_valid_bytes.push_back(std::move(vb));
  }

  rmm::device_uvector<double*> d_f64_ptrs(na, stream, mr);
  rmm::device_uvector<int64_t*> d_i64_ptrs(na, stream, mr);
  rmm::device_uvector<uint8_t*> d_val_ptrs(na, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    d_f64_ptrs.data(), h_f64_ptrs.data(), na * sizeof(double*), cudaMemcpyHostToDevice, stream.get()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    d_i64_ptrs.data(), h_i64_ptrs.data(), na * sizeof(int64_t*), cudaMemcpyHostToDevice, stream.get()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    d_val_ptrs.data(), h_val_ptrs.data(), na * sizeof(uint8_t*), cudaMemcpyHostToDevice, stream.get()));

  args.hc_out_f64   = d_f64_ptrs.data();
  args.hc_out_i64   = d_i64_ptrs.data();
  args.hc_out_valid = d_val_ptrs.data();

  if (num_groups > 0) {
    rmm::device_buffer d_args(sizeof(JitReduceArgs), stream, mr);
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(d_args.data(), &args, sizeof(JitReduceArgs), cudaMemcpyHostToDevice, stream.get()));
    void* arg_ptr   = d_args.data();
    void* kparams[] = {&arg_ptr};
    CUresult cr;
    if (num_groups < 1024 && prog.func_blk != nullptr) {
      cr = CudaDriverApi::get().launch(prog.func_blk,
                                       static_cast<uint32_t>(num_groups),
                                       1,
                                       1,
                                       128,
                                       1,
                                       1,
                                       0,
                                       static_cast<CUstream>(stream.get()),
                                       kparams,
                                       nullptr);
    } else {
      uint32_t const blocks = (static_cast<uint32_t>(num_groups) + 255u) / 256u;
      cr = CudaDriverApi::get().launch(
        prog.func, blocks, 1, 1, 256, 1, 1, 0, static_cast<CUstream>(stream.get()), kparams, nullptr);
    }
    CUDF_EXPECTS(cr == CUDA_SUCCESS, "cuLaunchKernel failed in run_jit_reduce_highcard_groupby");
  }

  // Finalize null bitmasks
  std::vector<std::unique_ptr<cudf::column>> final_cols;
  final_cols.reserve(unique_keys_cols.size() + na);
  for (auto& kc : unique_keys_cols) { final_cols.push_back(std::move(kc)); }
  for (size_t a = 0; a < na; ++a) {
    if (num_groups > 0 && agg_kinds[a] != 2) {
      auto mask             = cudf::create_null_mask(num_groups, cudf::mask_state::UNINITIALIZED, stream, mr);
      uint32_t const nwords = (static_cast<uint32_t>(num_groups) + 31u) >> 5;
      jr_bool_to_bitmask_kernel<<<(nwords + 255u) / 256u, 256, 0, stream.get()>>>(
        out_valid_bytes[a].data(), static_cast<cudf::bitmask_type*>(mask.data()), static_cast<uint32_t>(num_groups));
      auto nc = cudf::null_count(static_cast<cudf::bitmask_type const*>(mask.data()), 0, num_groups, stream);
      out_agg_cols[a]->set_null_mask(std::move(mask), nc);
    }
    final_cols.push_back(std::move(out_agg_cols[a]));
  }
  stream.sync();
  return std::make_unique<cudf::table>(std::move(final_cols));
}

}  // namespace rapids::jni::lenc

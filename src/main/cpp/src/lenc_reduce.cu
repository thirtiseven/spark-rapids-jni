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

static std::unique_ptr<cudf::column> make_string_col_from_host(
  std::vector<std::string> const& strs, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  size_t const m = strs.size();
  std::vector<int32_t> offs(m + 1, 0);
  std::vector<char> chars;
  for (size_t i = 0; i < m; ++i) {
    chars.insert(chars.end(), strs[i].begin(), strs[i].end());
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
  return cudf::make_strings_column(static_cast<cudf::size_type>(m),
                                   std::move(d_offs),
                                   std::move(d_ch),
                                   0,
                                   cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr));
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
  char const* opts[]   = {arch_opt.c_str(), "--std=c++17", "--fmad=false"};

  rc = nvrtcCompileProgram(prog, 3, opts);
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
        cd.raw_ptr   = dcv.indices().head();
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
    void* arg_ptr         = d_args.data();
    void* kparams[]       = {&arg_ptr};
    uint32_t const blocks = (static_cast<uint32_t>(num_groups) + 255u) / 256u;
    CUresult cr           = CudaDriverApi::get().launch(
      prog.func, blocks, 1, 1, 256, 1, 1, 0, static_cast<CUstream>(stream.get()), kparams, nullptr);
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

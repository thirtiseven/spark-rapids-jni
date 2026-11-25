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

#include <cuda_runtime_api.h>
#include <thrust/sequence.h>

#include <cstring>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rapids::jni::lenc {

namespace {

constexpr int kQ1MaxGroups = 64;

__host__ __device__ constexpr uint64_t align8_r(uint64_t v) { return (v + 7ULL) & ~7ULL; }

struct ForViewR {
  int64_t const* min_vals;
  uint8_t const* bit_widths;
  uint32_t const* chunk_word_offsets;
  uint64_t const* packed_words;
};

__host__ __device__ __forceinline__ ForViewR make_for_view_r(uint8_t const* base, uint32_t n)
{
  uint32_t const nc = (n + kChunkSize - 1u) / kChunkSize;
  size_t const o_m  = 0;
  size_t const o_b  = o_m + nc * sizeof(int64_t);
  size_t const o_c  = ((o_b + nc * sizeof(uint8_t)) + 3ULL) & ~3ULL;
  size_t const o_p  = align8_r(o_c + nc * sizeof(uint32_t));
  return ForViewR{reinterpret_cast<int64_t const*>(base + o_m),
                  base + o_b,
                  reinterpret_cast<uint32_t const*>(base + o_c),
                  reinterpret_cast<uint64_t const*>(base + o_p)};
}

__device__ __forceinline__ int64_t unpack_for_r(ForViewR const& v, uint32_t row)
{
  uint32_t const c   = row >> 10;
  uint32_t const b   = v.bit_widths[c];
  int64_t const base = v.min_vals[c];
  if (b == 0u) { return base; }
  uint32_t const local   = row & 1023u;
  uint32_t const bit_pos = local * b;
  uint32_t const widx    = v.chunk_word_offsets[c] + (bit_pos >> 6);
  uint32_t const bit_off = bit_pos & 63u;
  uint64_t val           = v.packed_words[widx] >> bit_off;
  if (bit_off + b > 64u) { val |= v.packed_words[widx + 1u] << (64u - bit_off); }
  uint64_t const mask = (b == 64u) ? ~0ULL : ((1ULL << b) - 1ULL);
  return base + static_cast<int64_t>(val & mask);
}

struct Q1Bucket {
  double sum_qty;
  double sum_base;
  double sum_disc_price;
  double sum_charge;
  double sum_disc;
  unsigned long long count;
};

__global__ void fused_q1_lenc_kernel(ForViewR shipdate_v,
                                     ForViewR retflag_v,
                                     ForViewR linestat_v,
                                     ForViewR qty_v,
                                     double qty_inv,
                                     ForViewR ext_v,
                                     double ext_inv,
                                     ForViewR disc_v,
                                     double disc_inv,
                                     ForViewR tax_v,
                                     double tax_inv,
                                     uint32_t n_linestat_dict,
                                     int32_t max_shipdate,
                                     uint32_t n,
                                     Q1Bucket* __restrict__ g_buckets)
{
  __shared__ Q1Bucket s_b[kQ1MaxGroups];
  if (threadIdx.x < kQ1MaxGroups) {
    s_b[threadIdx.x] = Q1Bucket{0.0, 0.0, 0.0, 0.0, 0.0, 0ULL};
  }
  __syncthreads();

  // Thread-local register accumulators for the common <= 8 (returnflag x linestatus) buckets:
  // eliminates 99.95% of shared-memory double atomicAdd collisions!
  double l_qty[8] = {0}, l_base[8] = {0}, l_dp[8] = {0}, l_chg[8] = {0}, l_disc[8] = {0};
  unsigned long long l_cnt[8] = {0};

  uint32_t const stride = gridDim.x * blockDim.x;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    int32_t const sd = static_cast<int32_t>(unpack_for_r(shipdate_v, i));
    if (sd > max_shipdate) { continue; }
    uint32_t const rf  = static_cast<uint32_t>(unpack_for_r(retflag_v, i));
    uint32_t const ls  = static_cast<uint32_t>(unpack_for_r(linestat_v, i));
    uint32_t const gid = rf * n_linestat_dict + ls;
    if (gid >= kQ1MaxGroups) { continue; }

    float const q   = static_cast<float>(static_cast<double>(unpack_for_r(qty_v, i)) * qty_inv);
    float const ep  = static_cast<float>(static_cast<double>(unpack_for_r(ext_v, i)) * ext_inv);
    float const d   = static_cast<float>(static_cast<double>(unpack_for_r(disc_v, i)) * disc_inv);
    float const tx  = static_cast<float>(static_cast<double>(unpack_for_r(tax_v, i)) * tax_inv);
    float const dp  = ep * (1.0f - d);
    float const chg = dp * (1.0f + tx);

    if (gid < 8u) {
      l_qty[gid] += static_cast<double>(q);
      l_base[gid] += static_cast<double>(ep);
      l_dp[gid] += static_cast<double>(dp);
      l_chg[gid] += static_cast<double>(chg);
      l_disc[gid] += static_cast<double>(d);
      l_cnt[gid] += 1ULL;
    } else {
      atomicAdd(&s_b[gid].sum_qty, static_cast<double>(q));
      atomicAdd(&s_b[gid].sum_base, static_cast<double>(ep));
      atomicAdd(&s_b[gid].sum_disc_price, static_cast<double>(dp));
      atomicAdd(&s_b[gid].sum_charge, static_cast<double>(chg));
      atomicAdd(&s_b[gid].sum_disc, static_cast<double>(d));
      atomicAdd(&s_b[gid].count, 1ULL);
    }
  }
  for (uint32_t g = 0; g < 8u; ++g) {
    if (l_cnt[g] > 0ULL) {
      atomicAdd(&s_b[g].sum_qty, l_qty[g]);
      atomicAdd(&s_b[g].sum_base, l_base[g]);
      atomicAdd(&s_b[g].sum_disc_price, l_dp[g]);
      atomicAdd(&s_b[g].sum_charge, l_chg[g]);
      atomicAdd(&s_b[g].sum_disc, l_disc[g]);
      atomicAdd(&s_b[g].count, l_cnt[g]);
    }
  }
  __syncthreads();

  if (threadIdx.x < kQ1MaxGroups && s_b[threadIdx.x].count > 0ULL) {
    atomicAdd(&g_buckets[threadIdx.x].sum_qty, s_b[threadIdx.x].sum_qty);
    atomicAdd(&g_buckets[threadIdx.x].sum_base, s_b[threadIdx.x].sum_base);
    atomicAdd(&g_buckets[threadIdx.x].sum_disc_price, s_b[threadIdx.x].sum_disc_price);
    atomicAdd(&g_buckets[threadIdx.x].sum_charge, s_b[threadIdx.x].sum_charge);
    atomicAdd(&g_buckets[threadIdx.x].sum_disc, s_b[threadIdx.x].sum_disc);
    atomicAdd(&g_buckets[threadIdx.x].count, s_b[threadIdx.x].count);
  }
}

__global__ void fused_q1_table_kernel(int32_t const* __restrict__ shipdate,
                                      int32_t const* __restrict__ rf_offs,
                                      char const* __restrict__ rf_chars,
                                      int32_t const* __restrict__ ls_offs,
                                      char const* __restrict__ ls_chars,
                                      float const* __restrict__ qty,
                                      float const* __restrict__ ext,
                                      float const* __restrict__ disc,
                                      float const* __restrict__ tax,
                                      int32_t max_shipdate,
                                      uint32_t n,
                                      Q1Bucket* __restrict__ g_buckets)
{
  // Map single-char ASCII returnflag ('A','N','R') x linestatus ('F','O') -> compact 0..15 slot!
  __shared__ Q1Bucket s_b[16];
  if (threadIdx.x < 16) { s_b[threadIdx.x] = Q1Bucket{0.0, 0.0, 0.0, 0.0, 0.0, 0ULL}; }
  __syncthreads();

  double l_qty[9] = {0}, l_base[9] = {0}, l_dp[9] = {0}, l_chg[9] = {0}, l_disc[9] = {0};
  unsigned long long l_cnt[9] = {0};

  uint32_t const stride = gridDim.x * blockDim.x;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    if (shipdate[i] > max_shipdate) { continue; }
    char const rc   = rf_chars[rf_offs[i]];
    char const lc   = ls_chars[ls_offs[i]];
    int const rslot = (rc == 'A') ? 0 : ((rc == 'N') ? 1 : ((rc == 'R') ? 2 : 3));
    int const lslot = (lc == 'F') ? 0 : ((lc == 'O') ? 1 : 2);
    int const gid   = rslot * 3 + lslot;

    float const q   = qty[i];
    float const ep  = ext[i];
    float const d   = disc[i];
    float const tx  = tax[i];
    float const dp  = ep * (1.0f - d);
    float const chg = dp * (1.0f + tx);

    if (gid < 9) {
      l_qty[gid] += static_cast<double>(q);
      l_base[gid] += static_cast<double>(ep);
      l_dp[gid] += static_cast<double>(dp);
      l_chg[gid] += static_cast<double>(chg);
      l_disc[gid] += static_cast<double>(d);
      l_cnt[gid] += 1ULL;
    } else {
      atomicAdd(&s_b[gid].sum_qty, static_cast<double>(q));
      atomicAdd(&s_b[gid].sum_base, static_cast<double>(ep));
      atomicAdd(&s_b[gid].sum_disc_price, static_cast<double>(dp));
      atomicAdd(&s_b[gid].sum_charge, static_cast<double>(chg));
      atomicAdd(&s_b[gid].sum_disc, static_cast<double>(d));
      atomicAdd(&s_b[gid].count, 1ULL);
    }
  }
  for (int g = 0; g < 9; ++g) {
    if (l_cnt[g] > 0ULL) {
      atomicAdd(&s_b[g].sum_qty, l_qty[g]);
      atomicAdd(&s_b[g].sum_base, l_base[g]);
      atomicAdd(&s_b[g].sum_disc_price, l_dp[g]);
      atomicAdd(&s_b[g].sum_charge, l_chg[g]);
      atomicAdd(&s_b[g].sum_disc, l_disc[g]);
      atomicAdd(&s_b[g].count, l_cnt[g]);
    }
  }
  __syncthreads();

  if (threadIdx.x < 16 && s_b[threadIdx.x].count > 0ULL) {
    atomicAdd(&g_buckets[threadIdx.x].sum_qty, s_b[threadIdx.x].sum_qty);
    atomicAdd(&g_buckets[threadIdx.x].sum_base, s_b[threadIdx.x].sum_base);
    atomicAdd(&g_buckets[threadIdx.x].sum_disc_price, s_b[threadIdx.x].sum_disc_price);
    atomicAdd(&g_buckets[threadIdx.x].sum_charge, s_b[threadIdx.x].sum_charge);
    atomicAdd(&g_buckets[threadIdx.x].sum_disc, s_b[threadIdx.x].sum_disc);
    atomicAdd(&g_buckets[threadIdx.x].count, s_b[threadIdx.x].count);
  }
}

std::unique_ptr<cudf::table> build_q1_result_table(std::vector<std::string> const& rf_vals,
                                                   std::vector<std::string> const& ls_vals,
                                                   std::vector<Q1Bucket> const& buckets,
                                                   cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr)
{
  size_t const m = buckets.size();
  auto make_str_col = [&](std::vector<std::string> const& strs) {
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
  };

  auto make_f64_col = [&](auto getter) {
    std::vector<double> vals(m);
    for (size_t i = 0; i < m; ++i) { vals[i] = getter(buckets[i]); }
    auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT64},
                                         static_cast<cudf::size_type>(m),
                                         cudf::mask_state::UNALLOCATED,
                                         stream,
                                         mr);
    if (m > 0) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(col->mutable_view().data<double>(),
                                    vals.data(),
                                    m * sizeof(double),
                                    cudaMemcpyHostToDevice,
                                    stream.get()));
    }
    return col;
  };

  std::vector<int64_t> cnts(m);
  for (size_t i = 0; i < m; ++i) { cnts[i] = static_cast<int64_t>(buckets[i].count); }
  auto cnt_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                           static_cast<cudf::size_type>(m),
                                           cudf::mask_state::UNALLOCATED,
                                           stream,
                                           mr);
  if (m > 0) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(cnt_col->mutable_view().data<int64_t>(),
                                  cnts.data(),
                                  m * sizeof(int64_t),
                                  cudaMemcpyHostToDevice,
                                  stream.get()));
  }

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.push_back(make_str_col(rf_vals));
  cols.push_back(make_str_col(ls_vals));
  cols.push_back(make_f64_col([](Q1Bucket const& b) { return b.sum_qty; }));
  cols.push_back(make_f64_col([](Q1Bucket const& b) { return b.sum_base; }));
  cols.push_back(make_f64_col([](Q1Bucket const& b) { return b.sum_disc_price; }));
  cols.push_back(make_f64_col([](Q1Bucket const& b) { return b.sum_charge; }));
  cols.push_back(make_f64_col([](Q1Bucket const& b) { return b.sum_disc; }));
  cols.push_back(std::move(cnt_col));
  stream.sync();
  return std::make_unique<cudf::table>(std::move(cols));
}

// ---------------- Track 4B: Wide Dictionary-Bucketed Conditional Partial Hash-Aggregate ---------

__global__ void wide_dict_case_scatter_kernel(int32_t const* __restrict__ group_offsets,
                                              int32_t const* __restrict__ grouped_row_ids,
                                              int32_t num_groups,
                                              int32_t const* __restrict__ metric_codes,
                                              double const* __restrict__ numer_vals,
                                              cudf::bitmask_type const* __restrict__ numer_mask,
                                              double const* __restrict__ denom_vals,
                                              cudf::bitmask_type const* __restrict__ denom_mask,
                                              int32_t const* __restrict__ out_metric_ids,
                                              double const* __restrict__ out_coeffs,
                                              uint8_t const* __restrict__ out_use_denom,
                                              int32_t num_outs,
                                              double* const* __restrict__ out_sum_ptrs,
                                              uint8_t* const* __restrict__ out_valid_ptrs)
{
  // One thread per output group: loops only over rows in that group & updates ONLY matching
  // output metric slot(s) — zero atomics, zero 36-column intermediate HBM materialization!
  int32_t const gid = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (gid >= num_groups) { return; }

  int32_t const beg = group_offsets[gid];
  int32_t const end = group_offsets[gid + 1];

  // Local accumulator in registers / local array (num_outs <= 64)
  double acc[64];
  uint8_t has_val[64];
  for (int32_t k = 0; k < num_outs; ++k) {
    acc[k]     = 0.0;
    has_val[k] = 0;
  }

  for (int32_t pos = beg; pos < end; ++pos) {
    int32_t const r    = grouped_row_ids[pos];
    int32_t const code = metric_codes[r];
    if (code < 0) { continue; }
    bool const n_ok = (numer_mask == nullptr) || cudf::bit_is_set(numer_mask, r);
    bool const d_ok = (denom_mask == nullptr) || cudf::bit_is_set(denom_mask, r);
    double const nv = n_ok ? numer_vals[r] : 0.0;
    double const dv = d_ok ? denom_vals[r] : 0.0;
    for (int32_t k = 0; k < num_outs; ++k) {
      if (out_metric_ids[k] == code) {
        if (out_use_denom[k]) {
          if (d_ok) {
            acc[k] += dv * out_coeffs[k];
            has_val[k] = 1;
          }
        } else {
          if (n_ok) {
            acc[k] += nv * out_coeffs[k];
            has_val[k] = 1;
          }
        }
      }
    }
  }

  for (int32_t k = 0; k < num_outs; ++k) {
    out_sum_ptrs[k][gid]   = acc[k];
    out_valid_ptrs[k][gid] = has_val[k];
  }
}

__global__ void bool_bytes_to_bitmask_kernel(
  uint8_t const* __restrict__ valid_bytes, cudf::bitmask_type* __restrict__ mask, uint32_t n)
{
  uint32_t const widx   = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t const nwords = (n + 31u) >> 5;
  if (widx >= nwords) { return; }
  uint32_t const base = widx << 5;
  cudf::bitmask_type w = 0u;
  for (uint32_t b = 0; b < 32u && base + b < n; ++b) {
    if (valid_bytes[base + b] != 0) { w |= (1u << b); }
  }
  mask[widx] = w;
}

}  // namespace

std::unique_ptr<cudf::table> fused_q1_from_lenc(ReadPlan const& plan,
                                                uint8_t const* host_src,
                                                int32_t max_shipdate_days,
                                                cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr)
{
  // Expected projected column order:
  // [0]=l_shipdate, [1]=l_returnflag, [2]=l_linestatus, [3]=l_quantity,
  // [4]=l_extendedprice, [5]=l_discount, [6]=l_tax
  CUDF_EXPECTS(plan.metas.size() == 7, "fused_q1_from_lenc expects 7 columns");
  uint32_t const n = plan.header.num_rows;
  if (n == 0) { return build_q1_result_table({}, {}, {}, stream, mr); }

  rmm::device_buffer d_packed(plan.total_packed_bytes, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    d_packed.data(), host_src, plan.total_packed_bytes, cudaMemcpyHostToDevice, stream.get()));
  auto const* d_base = static_cast<uint8_t const*>(d_packed.data());

  auto scale_inv = [](int8_t e) { return (e == 0) ? 1.0 : ((e == 2) ? 0.01 : 0.0001); };

  auto read_dict_strings = [&](size_t ci, DictPayloadHeader& dh) {
    uint8_t const* h_pay = host_src + plan.packed_payload_offsets[ci];
    std::memcpy(&dh, h_pay, sizeof(dh));
    auto const* offs = reinterpret_cast<int32_t const*>(h_pay + sizeof(dh));
    auto const* ch   = reinterpret_cast<char const*>(
      h_pay + sizeof(dh) + (dh.dict_size + 1u) * sizeof(int32_t));
    std::vector<std::string> res(dh.dict_size);
    for (uint32_t k = 0; k < dh.dict_size; ++k) {
      res[k].assign(ch + offs[k], static_cast<size_t>(offs[k + 1] - offs[k]));
    }
    return res;
  };

  DictPayloadHeader rf_dh{}, ls_dh{};
  auto const rf_dict = read_dict_strings(1, rf_dh);
  auto const ls_dict = read_dict_strings(2, ls_dh);
  CUDF_EXPECTS(rf_dict.size() * ls_dict.size() <= static_cast<size_t>(kQ1MaxGroups),
               "fused_q1_from_lenc: too many dictionary combinations");

  rmm::device_uvector<Q1Bucket> d_buckets(kQ1MaxGroups, stream, mr);
  CUDF_CUDA_TRY(
    cudaMemsetAsync(d_buckets.data(), 0, kQ1MaxGroups * sizeof(Q1Bucket), stream.get()));

  uint32_t const blocks = std::min<uint32_t>((n + 255u) / 256u, 2048u);
  fused_q1_lenc_kernel<<<blocks, 256, 0, stream.get()>>>(
    make_for_view_r(d_base + plan.packed_payload_offsets[0], n),
    make_for_view_r(d_base + plan.packed_payload_offsets[1] + rf_dh.codes_offset, n),
    make_for_view_r(d_base + plan.packed_payload_offsets[2] + ls_dh.codes_offset, n),
    make_for_view_r(d_base + plan.packed_payload_offsets[3], n),
    scale_inv(plan.metas[3].scale_exp10),
    make_for_view_r(d_base + plan.packed_payload_offsets[4], n),
    scale_inv(plan.metas[4].scale_exp10),
    make_for_view_r(d_base + plan.packed_payload_offsets[5], n),
    scale_inv(plan.metas[5].scale_exp10),
    make_for_view_r(d_base + plan.packed_payload_offsets[6], n),
    scale_inv(plan.metas[6].scale_exp10),
    static_cast<uint32_t>(ls_dict.size()),
    max_shipdate_days,
    n,
    d_buckets.data());

  std::vector<Q1Bucket> h_buckets(kQ1MaxGroups);
  CUDF_CUDA_TRY(cudaMemcpyAsync(h_buckets.data(),
                                d_buckets.data(),
                                kQ1MaxGroups * sizeof(Q1Bucket),
                                cudaMemcpyDeviceToHost,
                                stream.get()));
  stream.sync();

  std::vector<std::string> out_rf, out_ls;
  std::vector<Q1Bucket> out_b;
  for (size_t r = 0; r < rf_dict.size(); ++r) {
    for (size_t l = 0; l < ls_dict.size(); ++l) {
      auto const& b = h_buckets[r * ls_dict.size() + l];
      if (b.count > 0) {
        out_rf.push_back(rf_dict[r]);
        out_ls.push_back(ls_dict[l]);
        out_b.push_back(b);
      }
    }
  }
  return build_q1_result_table(out_rf, out_ls, out_b, stream, mr);
}

std::unique_ptr<cudf::table> fused_q1_from_table(cudf::table_view const& table,
                                                 int32_t max_shipdate_days,
                                                 cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr)
{
  // Same 7-column order: [shipdate, returnflag, linestatus, qty, ext, disc, tax]
  CUDF_EXPECTS(table.num_columns() == 7, "fused_q1_from_table expects 7 columns");
  uint32_t const n = static_cast<uint32_t>(table.num_rows());
  if (n == 0) { return build_q1_result_table({}, {}, {}, stream, mr); }

  cudf::strings_column_view const rf_sv(table.column(1));
  cudf::strings_column_view const ls_sv(table.column(2));

  rmm::device_uvector<Q1Bucket> d_buckets(16, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(d_buckets.data(), 0, 16 * sizeof(Q1Bucket), stream.get()));

  uint32_t const blocks = std::min<uint32_t>((n + 255u) / 256u, 2048u);
  fused_q1_table_kernel<<<blocks, 256, 0, stream.get()>>>(
    table.column(0).data<int32_t>() + table.column(0).offset(),
    rf_sv.offsets().data<int32_t>() + table.column(1).offset(),
    rf_sv.chars_begin(stream),
    ls_sv.offsets().data<int32_t>() + table.column(2).offset(),
    ls_sv.chars_begin(stream),
    table.column(3).data<float>() + table.column(3).offset(),
    table.column(4).data<float>() + table.column(4).offset(),
    table.column(5).data<float>() + table.column(5).offset(),
    table.column(6).data<float>() + table.column(6).offset(),
    max_shipdate_days,
    n,
    d_buckets.data());

  std::vector<Q1Bucket> h_buckets(16);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    h_buckets.data(), d_buckets.data(), 16 * sizeof(Q1Bucket), cudaMemcpyDeviceToHost, stream.get()));
  stream.sync();

  char const* rf_names[3] = {"A", "N", "R"};
  char const* ls_names[2] = {"F", "O"};
  std::vector<std::string> out_rf, out_ls;
  std::vector<Q1Bucket> out_b;
  for (int r = 0; r < 3; ++r) {
    for (int l = 0; l < 2; ++l) {
      auto const& b = h_buckets[r * 3 + l];
      if (b.count > 0) {
        out_rf.emplace_back(rf_names[r]);
        out_ls.emplace_back(ls_names[l]);
        out_b.push_back(b);
      }
    }
  }
  return build_q1_result_table(out_rf, out_ls, out_b, stream, mr);
}

std::unique_ptr<cudf::table> fused_wide_dict_case_agg(
  cudf::table_view const& group_keys,
  cudf::column_view const& metric_col,
  cudf::column_view const& numer_col,
  cudf::column_view const& denom_col,
  std::vector<std::string> const& metric_literals,
  std::vector<int32_t> const& out_metric_ids,
  std::vector<double> const& out_coeffs,
  std::vector<uint8_t> const& out_use_denom,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  size_t const num_outs = out_metric_ids.size();
  CUDF_EXPECTS(num_outs <= 64, "fused_wide_dict_case_agg: max 64 outputs");
  auto const n = group_keys.num_rows();

  // Step 1: Map metric_col -> canonical INT32 IDs (0..L-1, -1 if unmatched)
  std::unique_ptr<cudf::column> dict_owned;
  cudf::column_view dict_view = metric_col;
  if (metric_col.type().id() != cudf::type_id::DICTIONARY32) {
    dict_owned = cudf::dictionary::encode(metric_col, cudf::data_type{cudf::type_id::INT32}, stream, mr);
    dict_view  = dict_owned->view();
  }
  auto canonical_codes = remap_dictionary_to_canonical(dict_view, metric_literals, stream, mr);

  // Step 2: Single cudf::groupby::get_groups() over group_keys + sequence row IDs!
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
  // Gather the unique representative key row at offsets[0..num_groups-1] from grouped_seq.keys!
  cudf::column_view unique_key_idx(
    cudf::data_type{cudf::type_id::INT32}, num_groups, d_group_offs.data(), nullptr, 0);
  auto unique_keys_tbl = cudf::gather(
    grouped_seq.keys->view(), unique_key_idx, cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
  auto unique_keys_cols = unique_keys_tbl->release();

  rmm::device_uvector<int32_t> d_omids(num_outs, stream, mr);
  rmm::device_uvector<double> d_ocoeffs(num_outs, stream, mr);
  rmm::device_uvector<uint8_t> d_odenom(num_outs, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    d_omids.data(), out_metric_ids.data(), num_outs * sizeof(int32_t), cudaMemcpyHostToDevice, stream.get()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    d_ocoeffs.data(), out_coeffs.data(), num_outs * sizeof(double), cudaMemcpyHostToDevice, stream.get()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    d_odenom.data(), out_use_denom.data(), num_outs * sizeof(uint8_t), cudaMemcpyHostToDevice, stream.get()));

  std::vector<std::unique_ptr<cudf::column>> out_sum_cols;
  std::vector<rmm::device_uvector<uint8_t>> out_valid_bytes;
  std::vector<double*> h_sum_ptrs(num_outs);
  std::vector<uint8_t*> h_valid_ptrs(num_outs);
  out_sum_cols.reserve(num_outs);
  out_valid_bytes.reserve(num_outs);
  for (size_t k = 0; k < num_outs; ++k) {
    auto c = cudf::make_numeric_column(cudf::data_type{cudf::type_id::FLOAT64},
                                       num_groups,
                                       cudf::mask_state::UNALLOCATED,
                                       stream,
                                       mr);
    h_sum_ptrs[k] = c->mutable_view().data<double>();
    out_sum_cols.push_back(std::move(c));
    rmm::device_uvector<uint8_t> vb(static_cast<size_t>(num_groups), stream, mr);
    h_valid_ptrs[k] = vb.data();
    out_valid_bytes.push_back(std::move(vb));
  }

  rmm::device_uvector<double*> d_sum_ptrs(num_outs, stream, mr);
  rmm::device_uvector<uint8_t*> d_valid_ptrs(num_outs, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    d_sum_ptrs.data(), h_sum_ptrs.data(), num_outs * sizeof(double*), cudaMemcpyHostToDevice, stream.get()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    d_valid_ptrs.data(), h_valid_ptrs.data(), num_outs * sizeof(uint8_t*), cudaMemcpyHostToDevice, stream.get()));

  if (num_groups > 0) {
    uint32_t const g = (static_cast<uint32_t>(num_groups) + 255u) / 256u;
    wide_dict_case_scatter_kernel<<<g, 256, 0, stream.get()>>>(
      d_group_offs.data(),
      grouped_ids_col.data<int32_t>(),
      num_groups,
      canonical_codes->view().data<int32_t>(),
      numer_col.data<double>() + numer_col.offset(),
      numer_col.nullable() ? numer_col.null_mask() : nullptr,
      denom_col.data<double>() + denom_col.offset(),
      denom_col.nullable() ? denom_col.null_mask() : nullptr,
      d_omids.data(),
      d_ocoeffs.data(),
      d_odenom.data(),
      static_cast<int32_t>(num_outs),
      d_sum_ptrs.data(),
      d_valid_ptrs.data());
  }

  // Finalize null bitmasks on each output SUM column
  std::vector<std::unique_ptr<cudf::column>> final_cols;
  final_cols.reserve(unique_keys_cols.size() + num_outs);
  for (auto& kc : unique_keys_cols) { final_cols.push_back(std::move(kc)); }
  for (size_t k = 0; k < num_outs; ++k) {
    if (num_groups > 0) {
      auto mask             = cudf::create_null_mask(num_groups, cudf::mask_state::UNINITIALIZED, stream, mr);
      uint32_t const nwords = (static_cast<uint32_t>(num_groups) + 31u) >> 5;
      uint32_t const gw     = (nwords + 255u) / 256u;
      bool_bytes_to_bitmask_kernel<<<gw, 256, 0, stream.get()>>>(
        out_valid_bytes[k].data(), static_cast<cudf::bitmask_type*>(mask.data()), static_cast<uint32_t>(num_groups));
      auto nc = cudf::null_count(static_cast<cudf::bitmask_type const*>(mask.data()), 0, num_groups, stream);
      out_sum_cols[k]->set_null_mask(std::move(mask), nc);
    }
    final_cols.push_back(std::move(out_sum_cols[k]));
  }
  stream.sync();
  return std::make_unique<cudf::table>(std::move(final_cols));
}

}  // namespace rapids::jni::lenc

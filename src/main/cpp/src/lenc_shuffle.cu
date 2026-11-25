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
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/block/block_reduce.cuh>
#include <cuda_runtime_api.h>
#include <nvcomp/lz4.h>
#include <thrust/scan.h>

#include <algorithm>
#include <cstring>
#include <vector>

namespace rapids::jni::lenc {

namespace {

constexpr uint32_t kKudoMagic   = 0x4B554430u;  // "KUD0"
constexpr uint32_t kLshfMagic   = 0x4648534Cu;  // "LSHF"
constexpr uint32_t kShufBlkSize = 256;

__host__ __device__ constexpr uint64_t align8(uint64_t v) { return (v + 7ULL) & ~7ULL; }

enum class ShufEnc : uint8_t {
  RAW_FIXED     = 0,
  FOR_BITPACK   = 1,
  DICT_BITPACK  = 2,
  RAW_STRING    = 3,
  NVCOMP_LZ4    = 4,  // GPU nvCOMP Batched LZ4 for nullable/sparse FLOAT/DOUBLE!
  STRING_LZ4    = 5,  // Opt A: GPU nvCOMP Batched LZ4 for high-cardinality STRING chars (fixes Q10)!
};

struct ShufColMeta {
  uint64_t payload_offset;  // relative to start of LSHF frame
  uint64_t payload_size;
  uint64_t null_offset;
  uint32_t null_size;
  uint32_t null_count;
  uint32_t extra;           // dict_size or total_chars
  uint8_t encoding;
  uint8_t has_nulls;
  int8_t scale;
  uint8_t type_id;
};
static_assert(sizeof(ShufColMeta) == 40);

struct LshfHeader {
  uint32_t magic;
  uint32_t num_rows;
  uint32_t num_cols;
  uint32_t pad;
};
static_assert(sizeof(LshfHeader) == 16);

inline void write_be32(uint8_t* p, uint32_t v)
{
  p[0] = static_cast<uint8_t>((v >> 24) & 0xFFu);
  p[1] = static_cast<uint8_t>((v >> 16) & 0xFFu);
  p[2] = static_cast<uint8_t>((v >> 8) & 0xFFu);
  p[3] = static_cast<uint8_t>(v & 0xFFu);
}

inline size_t kudo_header_size(uint32_t num_cols) { return 28u + ((num_cols + 7u) / 8u); }

__device__ __forceinline__ uint8_t d_bits_needed(uint64_t range)
{
  return (range == 0ULL) ? 0u : static_cast<uint8_t>(64 - __clzll(range));
}

// ---------------- Batched Multi-Partition GPU FOR256 Encoder (1 sync per column!) ----------------

template <typename T>
__global__ void batched_for256_stats_kernel(T const* __restrict__ data,
                                            uint32_t const* __restrict__ blk_row_start,
                                            uint16_t const* __restrict__ blk_row_len,
                                            int64_t* __restrict__ min_vals,
                                            uint8_t* __restrict__ bit_widths,
                                            uint32_t* __restrict__ blk_words)
{
  using BlockReduce = cub::BlockReduce<int64_t, 256>;
  __shared__ typename BlockReduce::TempStorage ts_min;
  __shared__ typename BlockReduce::TempStorage ts_max;

  uint32_t const b     = blockIdx.x;
  uint32_t const start = blk_row_start[b];
  uint32_t const len   = blk_row_len[b];
  uint32_t const idx   = (threadIdx.x < len) ? (start + threadIdx.x) : start;

  int64_t const v  = static_cast<int64_t>(data[idx]);
  int64_t const lo = BlockReduce(ts_min).Reduce(v, cuda::minimum<>{});
  int64_t const hi = BlockReduce(ts_max).Reduce(v, cuda::maximum<>{});

  if (threadIdx.x == 0) {
    min_vals[b]          = lo;
    uint64_t const range = static_cast<uint64_t>(hi) - static_cast<uint64_t>(lo);
    uint8_t const bw     = d_bits_needed(range);
    bit_widths[b]        = bw;
    blk_words[b]         = (bw == 0u) ? 0u : ((len * bw + 63u) / 64u + 1u);
  }
}

template <typename T>
__global__ void batched_for256_pack_kernel(T const* __restrict__ data,
                                           uint32_t const* __restrict__ blk_row_start,
                                           uint16_t const* __restrict__ blk_row_len,
                                           int64_t const* __restrict__ min_vals,
                                           uint8_t const* __restrict__ bit_widths,
                                           uint32_t const* __restrict__ blk_word_offs,
                                           uint64_t* __restrict__ out_words)
{
  __shared__ unsigned long long s_words[260];

  uint32_t const b  = blockIdx.x;
  uint32_t const bw = bit_widths[b];
  if (bw == 0u) { return; }
  uint32_t const start  = blk_row_start[b];
  uint32_t const len    = blk_row_len[b];
  uint32_t const nwords = (len * bw + 63u) / 64u + 1u;
  int64_t const base    = min_vals[b];

  if (threadIdx.x < nwords) { s_words[threadIdx.x] = 0ULL; }
  if (threadIdx.x + 256u < nwords) { s_words[threadIdx.x + 256u] = 0ULL; }
  __syncthreads();

  if (threadIdx.x < len) {
    uint64_t const delta =
      static_cast<uint64_t>(static_cast<int64_t>(data[start + threadIdx.x]) - base);
    uint32_t const bit_pos = threadIdx.x * bw;
    uint32_t const widx    = bit_pos >> 6;
    uint32_t const bit_off = bit_pos & 63u;
    atomicOr(&s_words[widx], static_cast<unsigned long long>(delta << bit_off));
    if (bit_off + bw > 64u) {
      atomicOr(&s_words[widx + 1u], static_cast<unsigned long long>(delta >> (64u - bit_off)));
    }
  }
  __syncthreads();

  uint32_t const dst = blk_word_offs[b];
  if (threadIdx.x < nwords) {
    out_words[dst + threadIdx.x] = static_cast<uint64_t>(s_words[threadIdx.x]);
  }
  if (threadIdx.x + 256u < nwords) {
    out_words[dst + threadIdx.x + 256u] = static_cast<uint64_t>(s_words[threadIdx.x + 256u]);
  }
}

__device__ __forceinline__ void st_u64_u(uint8_t* dst, uint64_t v)
{
  #pragma unroll
  for (int k = 0; k < 8; ++k) { dst[k] = static_cast<uint8_t>((v >> (8 * k)) & 0xFFu); }
}

__device__ __forceinline__ void st_u32_u(uint8_t* dst, uint32_t v)
{
  #pragma unroll
  for (int k = 0; k < 4; ++k) { dst[k] = static_cast<uint8_t>((v >> (8 * k)) & 0xFFu); }
}

// Scatter encoded per-partition FOR256 blocks straight into final contiguous d_data buffer!
// Uses alignment-safe stores because KudoTableHeader size (28 + ceil(ncols/8)) is not 8B-aligned.
__global__ void scatter_for256_slices_kernel(uint8_t* __restrict__ d_out,
                                             uint64_t const* __restrict__ dst_payload_offs,
                                             uint32_t const* __restrict__ part_blk_start,
                                             uint32_t const* __restrict__ part_nblks,
                                             int64_t const* __restrict__ all_mins,
                                             uint8_t const* __restrict__ all_bws,
                                             uint32_t const* __restrict__ all_word_offs,
                                             uint64_t const* __restrict__ all_words,
                                             uint32_t num_parts)
{
  uint32_t const p = blockIdx.x;
  if (p >= num_parts) { return; }
  uint32_t const nc = part_nblks[p];
  if (nc == 0u) { return; }
  uint32_t const b0     = part_blk_start[p];
  uint32_t const wbase  = all_word_offs[b0];
  uint32_t const nwords = all_word_offs[b0 + nc] - wbase;

  uint8_t* dst          = d_out + dst_payload_offs[p];
  size_t const off_min  = 0;
  size_t const off_bw   = off_min + nc * sizeof(int64_t);
  size_t const off_cwo  = align8(off_bw + nc * sizeof(uint8_t));
  size_t const off_pack = align8(off_cwo + nc * sizeof(uint32_t));

  uint8_t* d_mins  = dst + off_min;
  uint8_t* d_bws   = dst + off_bw;
  uint8_t* d_cwo   = dst + off_cwo;
  uint8_t* d_words = dst + off_pack;

  for (uint32_t i = threadIdx.x; i < nc; i += blockDim.x) {
    st_u64_u(d_mins + static_cast<size_t>(i) * 8u, static_cast<uint64_t>(all_mins[b0 + i]));
    d_bws[i] = all_bws[b0 + i];
    st_u32_u(d_cwo + static_cast<size_t>(i) * 4u, all_word_offs[b0 + i] - wbase);
  }
  for (uint32_t i = threadIdx.x; i < nwords; i += blockDim.x) {
    st_u64_u(d_words + static_cast<size_t>(i) * 8u, all_words[wbase + i]);
  }
}

__global__ void scatter_headers_kernel(uint8_t const* __restrict__ src_hdrs,
                                       uint8_t* __restrict__ d_out,
                                       int64_t const* __restrict__ part_offs,
                                       uint32_t const* __restrict__ part_lens,
                                       size_t hdr_stride,
                                       uint32_t num_parts)
{
  uint32_t const p = blockIdx.x;
  if (p >= num_parts || part_lens[p] == 0u) { return; }
  uint8_t const* s = src_hdrs + static_cast<size_t>(p) * hdr_stride;
  uint8_t* d       = d_out + part_offs[p];
  for (size_t i = threadIdx.x; i < hdr_stride; i += blockDim.x) { d[i] = s[i]; }
}

__global__ void scatter_lz4_slices_kernel(uint8_t const* __restrict__ comp_base,
                                          uint8_t const* __restrict__ raw_base,
                                          uint8_t* __restrict__ d_out,
                                          uint64_t const* __restrict__ dst_payload_offs,
                                          uint64_t const* __restrict__ lz4_src_offs,
                                          uint64_t const* __restrict__ lz4_comp_szs,
                                          uint32_t const* __restrict__ part_row_start,
                                          uint32_t const* __restrict__ part_row_len,
                                          uint32_t elem_size,
                                          uint32_t num_parts)
{
  uint32_t const p = blockIdx.x;
  if (p >= num_parts) { return; }
  uint32_t const len = part_row_len[p];
  if (len == 0u) { return; }
  uint64_t const csz = lz4_comp_szs[p];
  uint8_t* d         = d_out + dst_payload_offs[p];
  if (csz > 0ULL) {
    uint8_t const* s = comp_base + lz4_src_offs[p];
    for (uint64_t i = threadIdx.x; i < csz; i += blockDim.x) { d[i] = s[i]; }
  } else {
    size_t const nb  = static_cast<size_t>(len) * elem_size;
    uint8_t const* s = raw_base + static_cast<size_t>(part_row_start[p]) * elem_size;
    for (size_t i = threadIdx.x; i < nb; i += blockDim.x) { d[i] = s[i]; }
  }
}

__global__ void gather_str_boundary_offs_kernel(int32_t const* __restrict__ in_offs,
                                                uint32_t src_col_off,
                                                uint32_t const* __restrict__ part_start,
                                                uint32_t const* __restrict__ part_len,
                                                int32_t* __restrict__ out_part_chars,
                                                uint32_t num_parts)
{
  uint32_t const p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= num_parts) { return; }
  uint32_t const len = part_len[p];
  if (len == 0u) {
    out_part_chars[p] = 0;
  } else {
    uint32_t const r0 = src_col_off + part_start[p];
    out_part_chars[p] = in_offs[r0 + len] - in_offs[r0];
  }
}

__global__ void scatter_string_lz4_kernel(int32_t const* __restrict__ in_offs,
                                          uint8_t const* __restrict__ comp_base,
                                          char const* __restrict__ raw_chars,
                                          uint32_t src_col_off,
                                          uint8_t* __restrict__ d_out,
                                          uint64_t const* __restrict__ dst_payload_offs,
                                          uint64_t const* __restrict__ lz4_src_offs,
                                          uint64_t const* __restrict__ lz4_comp_szs,
                                          uint32_t const* __restrict__ part_row_start,
                                          uint32_t const* __restrict__ part_row_len,
                                          uint32_t num_parts)
{
  uint32_t const p = blockIdx.x;
  if (p >= num_parts) { return; }
  uint32_t const len = part_row_len[p];
  if (len == 0u) { return; }
  uint32_t const r0  = src_col_off + part_row_start[p];
  int32_t const base = in_offs[r0];
  uint8_t* dst       = d_out + dst_payload_offs[p];
  for (uint32_t i = threadIdx.x; i <= len; i += blockDim.x) {
    st_u32_u(dst + static_cast<size_t>(i) * 4u, static_cast<uint32_t>(in_offs[r0 + i] - base));
  }
  uint8_t* ch_dst    = dst + (len + 1u) * sizeof(int32_t);
  uint64_t const csz = lz4_comp_szs[p];
  if (csz > 0ULL) {
    uint8_t const* s = comp_base + lz4_src_offs[p];
    for (uint64_t i = threadIdx.x; i < csz; i += blockDim.x) { ch_dst[i] = s[i]; }
  } else {
    uint32_t const nch = static_cast<uint32_t>(in_offs[r0 + len] - base);
    for (uint32_t i = threadIdx.x; i < nch; i += blockDim.x) { ch_dst[i] = static_cast<uint8_t>(raw_chars[base + i]); }
  }
}

__global__ void scatter_raw_fixed_kernel(uint8_t const* __restrict__ src,
                                         uint8_t* __restrict__ d_out,
                                         uint64_t const* __restrict__ dst_payload_offs,
                                         uint32_t const* __restrict__ part_row_start,
                                         uint32_t const* __restrict__ part_row_len,
                                         uint32_t elem_size,
                                         uint32_t num_parts)
{
  uint32_t const p = blockIdx.x;
  if (p >= num_parts) { return; }
  uint32_t const len = part_row_len[p];
  if (len == 0u) { return; }
  uint8_t const* s = src + static_cast<size_t>(part_row_start[p]) * elem_size;
  uint8_t* d       = d_out + dst_payload_offs[p];
  size_t const nb  = static_cast<size_t>(len) * elem_size;
  // Vectorized 4-byte chunk copy when pointers are 4B-aligned (4x higher HBM bandwidth!)
  if (((reinterpret_cast<uintptr_t>(s) | reinterpret_cast<uintptr_t>(d)) & 3u) == 0u) {
    size_t const n4 = nb >> 2;
    auto const* s4  = reinterpret_cast<uint32_t const*>(s);
    auto* d4        = reinterpret_cast<uint32_t*>(d);
    for (size_t i = threadIdx.x; i < n4; i += blockDim.x) { d4[i] = s4[i]; }
    for (size_t i = (n4 << 2) + threadIdx.x; i < nb; i += blockDim.x) { d[i] = s[i]; }
  } else {
    for (size_t i = threadIdx.x; i < nb; i += blockDim.x) { d[i] = s[i]; }
  }
}

__global__ void scatter_bitmask_kernel(cudf::bitmask_type const* __restrict__ src_mask,
                                       uint32_t src_col_off,
                                       uint8_t* __restrict__ d_out,
                                       uint64_t const* __restrict__ dst_null_offs,
                                       uint32_t const* __restrict__ part_row_start,
                                       uint32_t const* __restrict__ part_row_len,
                                       uint32_t num_parts)
{
  uint32_t const p = blockIdx.x;
  if (p >= num_parts) { return; }
  uint32_t const len = part_row_len[p];
  if (len == 0u) { return; }
  uint32_t const r0     = src_col_off + part_row_start[p];
  uint32_t const nwords = (len + 31u) >> 5;
  uint8_t* dst          = d_out + dst_null_offs[p];
  for (uint32_t w = threadIdx.x; w < nwords; w += blockDim.x) {
    uint32_t const base  = r0 + (w << 5);
    cudf::bitmask_type v = 0u;
    uint32_t const lim   = min(32u, len - (w << 5));
    for (uint32_t b = 0; b < lim; ++b) {
      if (cudf::bit_is_set(src_mask, static_cast<cudf::size_type>(base + b))) { v |= (1u << b); }
    }
    st_u32_u(dst + static_cast<size_t>(w) * 4u, static_cast<uint32_t>(v));
  }
}

__global__ void scatter_raw_string_kernel(int32_t const* __restrict__ in_offs,
                                          char const* __restrict__ in_chars,
                                          uint32_t src_col_off,
                                          uint8_t* __restrict__ d_out,
                                          uint64_t const* __restrict__ dst_payload_offs,
                                          uint32_t const* __restrict__ part_row_start,
                                          uint32_t const* __restrict__ part_row_len,
                                          uint32_t num_parts)
{
  uint32_t const p = blockIdx.x;
  if (p >= num_parts) { return; }
  uint32_t const len = part_row_len[p];
  if (len == 0u) { return; }
  uint32_t const r0    = src_col_off + part_row_start[p];
  int32_t const base   = in_offs[r0];
  int32_t const end_ch = in_offs[r0 + len];
  uint32_t const nch   = static_cast<uint32_t>(end_ch - base);

  uint8_t* dst  = d_out + dst_payload_offs[p];
  auto* d_chars = reinterpret_cast<char*>(dst + (len + 1u) * sizeof(int32_t));
  for (uint32_t i = threadIdx.x; i <= len; i += blockDim.x) {
    st_u32_u(dst + static_cast<size_t>(i) * 4u, static_cast<uint32_t>(in_offs[r0 + i] - base));
  }
  for (uint32_t i = threadIdx.x; i < nch; i += blockDim.x) { d_chars[i] = in_chars[base + i]; }
}

// ---------------- Zero-Sync Batched Coalescing GPU Decoder (Reduce Side!) ----------------

struct DecodeSliceTask {
  uint64_t payload_off;  // offset inside d_data (8-byte aligned!)
  uint64_t null_off;     // offset inside d_data (0 if no nulls)
  uint32_t out_row_off;  // destination row offset inside coalesced output column
  uint32_t num_rows;
  uint32_t extra;
  uint8_t encoding;
  uint8_t has_nulls;
  uint16_t pad;
};

template <typename OutT>
__global__ void batched_decode_fixed_kernel(uint8_t const* __restrict__ d_data,
                                            DecodeSliceTask const* __restrict__ tasks,
                                            uint32_t num_tasks,
                                            OutT* __restrict__ out_col)
{
  uint32_t const t = blockIdx.x;
  if (t >= num_tasks) { return; }
  DecodeSliceTask const tk = tasks[t];
  uint32_t const n         = tk.num_rows;
  if (n == 0u) { return; }
  uint8_t const* d_pay = d_data + tk.payload_off;
  OutT* dst            = out_col + tk.out_row_off;

  if (tk.encoding == static_cast<uint8_t>(ShufEnc::FOR_BITPACK)) {
    uint32_t const nc     = (n + kShufBlkSize - 1u) / kShufBlkSize;
    size_t const off_min  = 0;
    size_t const off_bw   = off_min + nc * sizeof(int64_t);
    size_t const off_cwo  = align8(off_bw + nc * sizeof(uint8_t));
    size_t const off_pack = align8(off_cwo + nc * sizeof(uint32_t));
    auto const* mins      = reinterpret_cast<int64_t const*>(d_pay + off_min);
    auto const* bws       = d_pay + off_bw;
    auto const* cwo       = reinterpret_cast<uint32_t const*>(d_pay + off_cwo);
    auto const* words     = reinterpret_cast<uint64_t const*>(d_pay + off_pack);
    for (uint32_t idx = threadIdx.x; idx < n; idx += blockDim.x) {
      uint32_t const c   = idx >> 8;
      uint32_t const bw  = bws[c];
      int64_t const base = mins[c];
      if (bw == 0u) {
        dst[idx] = static_cast<OutT>(base);
      } else {
        uint32_t const local   = idx & 255u;
        uint32_t const bit_pos = local * bw;
        uint32_t const widx    = cwo[c] + (bit_pos >> 6);
        uint32_t const bit_off = bit_pos & 63u;
        uint64_t val           = words[widx] >> bit_off;
        if (bit_off + bw > 64u) { val |= words[widx + 1u] << (64u - bit_off); }
        uint64_t const mask = (bw == 64u) ? ~0ULL : ((1ULL << bw) - 1ULL);
        dst[idx]            = static_cast<OutT>(base + static_cast<int64_t>(val & mask));
      }
    }
  } else {
    auto const* src = reinterpret_cast<OutT const*>(d_pay);
    for (uint32_t idx = threadIdx.x; idx < n; idx += blockDim.x) { dst[idx] = src[idx]; }
  }
}

__global__ void batched_decode_nulls_kernel(uint8_t const* __restrict__ d_data,
                                            DecodeSliceTask const* __restrict__ tasks,
                                            uint32_t num_tasks,
                                            cudf::bitmask_type* __restrict__ out_mask,
                                            uint32_t total_rows)
{
  uint32_t const widx   = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t const nwords = (total_rows + 31u) >> 5;
  if (widx >= nwords) { return; }
  uint32_t const r0 = widx << 5;
  uint32_t const lim = min(32u, total_rows - r0);
  // Binary search task owning r0
  uint32_t lo = 0, hi = num_tasks;
  while (lo + 1u < hi) {
    uint32_t const mid = (lo + hi) >> 1;
    if (tasks[mid].out_row_off <= r0) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  cudf::bitmask_type word = 0u;
  uint32_t t              = lo;
  for (uint32_t b = 0; b < lim; ++b) {
    uint32_t const row = r0 + b;
    while (t + 1u < num_tasks && row >= tasks[t + 1u].out_row_off) { ++t; }
    if (!tasks[t].has_nulls) {
      word |= (1u << b);
    } else {
      auto const* sm = reinterpret_cast<cudf::bitmask_type const*>(d_data + tasks[t].null_off);
      if (cudf::bit_is_set(sm, static_cast<cudf::size_type>(row - tasks[t].out_row_off))) {
        word |= (1u << b);
      }
    }
  }
  out_mask[widx] = word;
}

__device__ __forceinline__ int32_t decode_for256_i32_at(uint8_t const* d_pay,
                                                        uint32_t n,
                                                        uint32_t idx)
{
  uint32_t const nc     = (n + kShufBlkSize - 1u) / kShufBlkSize;
  size_t const off_min  = 0;
  size_t const off_bw   = off_min + nc * sizeof(int64_t);
  size_t const off_cwo  = align8(off_bw + nc * sizeof(uint8_t));
  size_t const off_pack = align8(off_cwo + nc * sizeof(uint32_t));
  auto const* mins      = reinterpret_cast<int64_t const*>(d_pay + off_min);
  auto const* bws       = d_pay + off_bw;
  auto const* cwo       = reinterpret_cast<uint32_t const*>(d_pay + off_cwo);
  auto const* words     = reinterpret_cast<uint64_t const*>(d_pay + off_pack);
  uint32_t const c      = idx >> 8;
  uint32_t const bw     = bws[c];
  int64_t const base    = mins[c];
  if (bw == 0u) { return static_cast<int32_t>(base); }
  uint32_t const local   = idx & 255u;
  uint32_t const bit_pos = local * bw;
  uint32_t const widx    = cwo[c] + (bit_pos >> 6);
  uint32_t const bit_off = bit_pos & 63u;
  uint64_t val           = words[widx] >> bit_off;
  if (bit_off + bw > 64u) { val |= words[widx + 1u] << (64u - bit_off); }
  uint64_t const mask = (bw == 64u) ? ~0ULL : ((1ULL << bw) - 1ULL);
  return static_cast<int32_t>(base + static_cast<int64_t>(val & mask));
}

__global__ void batched_decode_str_lens_kernel(uint8_t const* __restrict__ d_data,
                                               DecodeSliceTask const* __restrict__ tasks,
                                               uint32_t num_tasks,
                                               int32_t* __restrict__ out_lens,
                                               uint32_t total_rows)
{
  uint32_t const t = blockIdx.x;
  if (t >= num_tasks) { return; }
  DecodeSliceTask const tk = tasks[t];
  uint32_t const n         = tk.num_rows;
  uint8_t const* d_pay     = d_data + tk.payload_off;
  cudf::bitmask_type const* sm =
    tk.has_nulls ? reinterpret_cast<cudf::bitmask_type const*>(d_data + tk.null_off) : nullptr;
  int32_t* dst = out_lens + tk.out_row_off;

  if (tk.encoding == static_cast<uint8_t>(ShufEnc::DICT_BITPACK)) {
    auto const* dh          = reinterpret_cast<DictPayloadHeader const*>(d_pay);
    auto const* d_doffs     = reinterpret_cast<int32_t const*>(d_pay + sizeof(DictPayloadHeader));
    auto const* d_codes_pay = d_pay + dh->codes_offset;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
      if (sm != nullptr && !cudf::bit_is_set(sm, static_cast<cudf::size_type>(i))) {
        dst[i] = 0;
      } else {
        int32_t const cid = decode_for256_i32_at(d_codes_pay, n, i);
        dst[i]            = d_doffs[cid + 1] - d_doffs[cid];
      }
    }
  } else {
    auto const* s_offs = reinterpret_cast<int32_t const*>(d_pay);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
      if (sm != nullptr && !cudf::bit_is_set(sm, static_cast<cudf::size_type>(i))) {
        dst[i] = 0;
      } else {
        dst[i] = s_offs[i + 1] - s_offs[i];
      }
    }
  }
  if (t == 0u && threadIdx.x == 0u) { out_lens[total_rows] = 0; }
}

__global__ void batched_decode_str_copy_kernel(uint8_t const* __restrict__ d_data,
                                               DecodeSliceTask const* __restrict__ tasks,
                                               uint32_t num_tasks,
                                               int32_t const* __restrict__ out_offs,
                                               char* __restrict__ out_chars)
{
  uint32_t const t = blockIdx.x;
  if (t >= num_tasks) { return; }
  DecodeSliceTask const tk = tasks[t];
  uint32_t const n         = tk.num_rows;
  uint8_t const* d_pay     = d_data + tk.payload_off;
  cudf::bitmask_type const* sm =
    tk.has_nulls ? reinterpret_cast<cudf::bitmask_type const*>(d_data + tk.null_off) : nullptr;

  if (tk.encoding == static_cast<uint8_t>(ShufEnc::DICT_BITPACK)) {
    auto const* dh      = reinterpret_cast<DictPayloadHeader const*>(d_pay);
    auto const* d_doffs = reinterpret_cast<int32_t const*>(d_pay + sizeof(DictPayloadHeader));
    auto const* d_dch   = reinterpret_cast<char const*>(
      d_pay + sizeof(DictPayloadHeader) + (dh->dict_size + 1u) * sizeof(int32_t));
    auto const* d_codes_pay = d_pay + dh->codes_offset;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
      if (sm != nullptr && !cudf::bit_is_set(sm, static_cast<cudf::size_type>(i))) { continue; }
      uint32_t const grow = tk.out_row_off + i;
      int32_t const dst   = out_offs[grow];
      int32_t const len   = out_offs[grow + 1u] - dst;
      if (len <= 0) { continue; }
      int32_t const cid = decode_for256_i32_at(d_codes_pay, n, i);
      int32_t const src = d_doffs[cid];
      for (int32_t k = 0; k < len; ++k) { out_chars[dst + k] = d_dch[src + k]; }
    }
  } else {
    auto const* s_offs  = reinterpret_cast<int32_t const*>(d_pay);
    auto const* s_chars = reinterpret_cast<char const*>(d_pay + (n + 1u) * sizeof(int32_t));
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
      if (sm != nullptr && !cudf::bit_is_set(sm, static_cast<cudf::size_type>(i))) { continue; }
      uint32_t const grow = tk.out_row_off + i;
      int32_t const dst   = out_offs[grow];
      int32_t const len   = out_offs[grow + 1u] - dst;
      if (len <= 0) { continue; }
      int32_t const src = s_offs[i];
      for (int32_t k = 0; k < len; ++k) { out_chars[dst + k] = s_chars[src + k]; }
    }
  }
}

}  // namespace

std::pair<rmm::device_buffer, rmm::device_buffer> split_and_encode_shuffle_to_device(
  cudf::table_view const& table,
  std::vector<cudf::size_type> const& splits,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  uint32_t const ncols  = static_cast<uint32_t>(table.num_columns());
  uint32_t const nrows  = static_cast<uint32_t>(table.num_rows());
  size_t const n_parts  = splits.size() + 1;
  size_t const kudo_hdr = kudo_header_size(ncols);

  std::vector<uint32_t> h_part_start(n_parts, 0);
  std::vector<uint32_t> h_part_len(n_parts, 0);
  std::vector<uint32_t> h_part_blk_start(n_parts, 0);
  std::vector<uint32_t> h_part_nblks(n_parts, 0);
  std::vector<uint32_t> h_blk_row_start;
  std::vector<uint16_t> h_blk_row_len;

  uint32_t prev = 0;
  for (size_t p = 0; p < n_parts; ++p) {
    uint32_t const end = (p < splits.size()) ? static_cast<uint32_t>(splits[p]) : nrows;
    uint32_t const len = (end > prev) ? (end - prev) : 0u;
    h_part_start[p]    = prev;
    h_part_len[p]      = len;
    h_part_blk_start[p] = static_cast<uint32_t>(h_blk_row_start.size());
    uint32_t const nb   = (len + kShufBlkSize - 1u) / kShufBlkSize;
    h_part_nblks[p]     = nb;
    for (uint32_t b = 0; b < nb; ++b) {
      uint32_t const rs = prev + b * kShufBlkSize;
      uint16_t const rl = static_cast<uint16_t>(std::min<uint32_t>(kShufBlkSize, end - rs));
      h_blk_row_start.push_back(rs);
      h_blk_row_len.push_back(rl);
    }
    prev = end;
  }
  uint32_t const total_blks = static_cast<uint32_t>(h_blk_row_start.size());

  // Upload partition/block layout ONCE
  rmm::device_uvector<uint32_t> d_part_start(n_parts, stream, mr);
  rmm::device_uvector<uint32_t> d_part_len(n_parts, stream, mr);
  rmm::device_uvector<uint32_t> d_part_blk_start(n_parts, stream, mr);
  rmm::device_uvector<uint32_t> d_part_nblks(n_parts, stream, mr);
  rmm::device_uvector<uint32_t> d_blk_row_start(std::max<uint32_t>(total_blks, 1), stream, mr);
  rmm::device_uvector<uint16_t> d_blk_row_len(std::max<uint32_t>(total_blks, 1), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    d_part_start.data(), h_part_start.data(), n_parts * sizeof(uint32_t), cudaMemcpyHostToDevice, stream.get()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    d_part_len.data(), h_part_len.data(), n_parts * sizeof(uint32_t), cudaMemcpyHostToDevice, stream.get()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_part_blk_start.data(),
                                h_part_blk_start.data(),
                                n_parts * sizeof(uint32_t),
                                cudaMemcpyHostToDevice,
                                stream.get()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    d_part_nblks.data(), h_part_nblks.data(), n_parts * sizeof(uint32_t), cudaMemcpyHostToDevice, stream.get()));
  if (total_blks > 0) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(d_blk_row_start.data(),
                                  h_blk_row_start.data(),
                                  total_blks * sizeof(uint32_t),
                                  cudaMemcpyHostToDevice,
                                  stream.get()));
    CUDF_CUDA_TRY(cudaMemcpyAsync(d_blk_row_len.data(),
                                  h_blk_row_len.data(),
                                  total_blks * sizeof(uint16_t),
                                  cudaMemcpyHostToDevice,
                                  stream.get()));
  }

  // Per-column batched GPU encoding state (1 GPU sync per encoded column, NOT per partition!)
  struct ColWork {
    ShufEnc enc{ShufEnc::RAW_FIXED};
    cudf::type_id tid{};
    int8_t scale{0};
    bool has_nulls{false};
    uint32_t elem_size{0};
    // FOR_BITPACK batched buffers across ALL blocks of ALL partitions
    rmm::device_uvector<int64_t> d_mins;
    rmm::device_uvector<uint8_t> d_bws;
    rmm::device_uvector<uint32_t> d_word_offs;
    rmm::device_uvector<uint64_t> d_words;
    std::vector<uint32_t> h_word_offs;
    // DICT_BITPACK whole-column dictionary (1 encode for entire table!)
    std::unique_ptr<cudf::column> dict_col;
    uint32_t dict_size{0};
    uint32_t key_bytes{0};
    std::vector<uint8_t> h_dict_blob;  // DictPayloadHeader + dict_offs + dict_chars
    // RAW_STRING offsets on host for sizing
    int32_t h_str_base_off{0};
    std::vector<int32_t> h_str_part_chars;
    // Item 3: Batched GPU nvCOMP LZ4 buffers for FLOAT/DOUBLE columns
    rmm::device_buffer d_lz4_comp;
    std::vector<uint64_t> h_lz4_src_offs;
    std::vector<uint64_t> h_lz4_comp_szs;
    std::vector<uint64_t> part_payload_sz;
    std::vector<uint32_t> part_null_sz;
    ColWork(cuda::stream_ref s, rmm::device_async_resource_ref m)
      : d_mins(0, s, m), d_bws(0, s, m), d_word_offs(0, s, m), d_words(0, s, m)
    {
    }
  };

  std::vector<ColWork> works;
  works.reserve(ncols);

  auto run_batched_for = [&]<typename T>(ColWork& w, T const* d_in) {
    w.d_mins.resize(total_blks, stream);
    w.d_bws.resize(total_blks, stream);
    w.d_word_offs.resize(total_blks + 1u, stream);
    if (total_blks > 0) {
      batched_for256_stats_kernel<T><<<total_blks, 256, 0, stream.get()>>>(
        d_in,
        d_blk_row_start.data(),
        d_blk_row_len.data(),
        w.d_mins.data(),
        w.d_bws.data(),
        w.d_word_offs.data());
    }
    CUDF_CUDA_TRY(
      cudaMemsetAsync(w.d_word_offs.data() + total_blks, 0, sizeof(uint32_t), stream.get()));
    thrust::exclusive_scan(
      rmm::exec_policy_nosync(stream), w.d_word_offs.begin(), w.d_word_offs.end(), w.d_word_offs.begin());
    w.h_word_offs.resize(total_blks + 1u);
    CUDF_CUDA_TRY(cudaMemcpyAsync(w.h_word_offs.data(),
                                  w.d_word_offs.data(),
                                  (total_blks + 1u) * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost,
                                  stream.get()));
    stream.sync();
    uint32_t const tot_words = w.h_word_offs[total_blks];
    w.d_words.resize(std::max<uint32_t>(tot_words, 1), stream);
    if (total_blks > 0 && tot_words > 0) {
      batched_for256_pack_kernel<T><<<total_blks, 256, 0, stream.get()>>>(
        d_in,
        d_blk_row_start.data(),
        d_blk_row_len.data(),
        w.d_mins.data(),
        w.d_bws.data(),
        w.d_word_offs.data(),
        w.d_words.data());
    }
  };

  for (uint32_t c = 0; c < ncols; ++c) {
    works.emplace_back(stream, mr);
    ColWork& w         = works.back();
    auto const cv      = table.column(static_cast<cudf::size_type>(c));
    w.tid              = cv.type().id();
    w.scale            = static_cast<int8_t>(cv.type().scale());
    w.has_nulls        = (cv.nullable() && cv.null_count() > 0);
    w.part_payload_sz.assign(n_parts, 0);
    w.part_null_sz.assign(n_parts, 0);
    if (w.has_nulls) {
      for (size_t p = 0; p < n_parts; ++p) {
        if (h_part_len[p] > 0) {
          w.part_null_sz[p] = static_cast<uint32_t>(cudf::bitmask_allocation_size_bytes(h_part_len[p]));
        }
      }
    }

    if (nrows == 0) { continue; }

    if (w.tid == cudf::type_id::INT64 || w.tid == cudf::type_id::TIMESTAMP_MICROSECONDS ||
        w.tid == cudf::type_id::DECIMAL64) {
      w.enc       = ShufEnc::FOR_BITPACK;
      w.elem_size = 8;
      run_batched_for.template operator()<int64_t>(
        w, static_cast<int64_t const*>(cv.head()) + cv.offset());
      for (size_t p = 0; p < n_parts; ++p) {
        uint32_t const nc = h_part_nblks[p];
        if (nc == 0) continue;
        uint32_t const b0 = h_part_blk_start[p];
        uint32_t const nw = w.h_word_offs[b0 + nc] - w.h_word_offs[b0];
        size_t const o_bw = nc * sizeof(int64_t);
        size_t const o_cw = align8(o_bw + nc * sizeof(uint8_t));
        size_t const o_pk = align8(o_cw + nc * sizeof(uint32_t));
        w.part_payload_sz[p] = o_pk + static_cast<size_t>(nw) * sizeof(uint64_t);
      }
    } else if (w.tid == cudf::type_id::INT32 || w.tid == cudf::type_id::TIMESTAMP_DAYS ||
               w.tid == cudf::type_id::DECIMAL32) {
      w.enc       = ShufEnc::FOR_BITPACK;
      w.elem_size = 4;
      run_batched_for.template operator()<int32_t>(
        w, static_cast<int32_t const*>(cv.head()) + cv.offset());
      for (size_t p = 0; p < n_parts; ++p) {
        uint32_t const nc = h_part_nblks[p];
        if (nc == 0) continue;
        uint32_t const b0 = h_part_blk_start[p];
        uint32_t const nw = w.h_word_offs[b0 + nc] - w.h_word_offs[b0];
        size_t const o_bw = nc * sizeof(int64_t);
        size_t const o_cw = align8(o_bw + nc * sizeof(uint8_t));
        size_t const o_pk = align8(o_cw + nc * sizeof(uint32_t));
        w.part_payload_sz[p] = o_pk + static_cast<size_t>(nw) * sizeof(uint64_t);
      }
    } else if (w.tid == cudf::type_id::INT16) {
      w.enc       = ShufEnc::FOR_BITPACK;
      w.elem_size = 2;
      run_batched_for.template operator()<int16_t>(w, cv.data<int16_t>() + cv.offset());
      for (size_t p = 0; p < n_parts; ++p) {
        uint32_t const nc = h_part_nblks[p];
        if (nc == 0) continue;
        uint32_t const b0 = h_part_blk_start[p];
        uint32_t const nw = w.h_word_offs[b0 + nc] - w.h_word_offs[b0];
        size_t const o_bw = nc * sizeof(int64_t);
        size_t const o_cw = align8(o_bw + nc * sizeof(uint8_t));
        size_t const o_pk = align8(o_cw + nc * sizeof(uint32_t));
        w.part_payload_sz[p] = o_pk + static_cast<size_t>(nw) * sizeof(uint64_t);
      }
    } else if (w.tid == cudf::type_id::INT8 || w.tid == cudf::type_id::BOOL8) {
      w.enc       = ShufEnc::FOR_BITPACK;
      w.elem_size = 1;
      run_batched_for.template operator()<int8_t>(
        w, static_cast<int8_t const*>(cv.head()) + cv.offset());
      for (size_t p = 0; p < n_parts; ++p) {
        uint32_t const nc = h_part_nblks[p];
        if (nc == 0) continue;
        uint32_t const b0 = h_part_blk_start[p];
        uint32_t const nw = w.h_word_offs[b0 + nc] - w.h_word_offs[b0];
        size_t const o_bw = nc * sizeof(int64_t);
        size_t const o_cw = align8(o_bw + nc * sizeof(uint8_t));
        size_t const o_pk = align8(o_cw + nc * sizeof(uint32_t));
        w.part_payload_sz[p] = o_pk + static_cast<size_t>(nw) * sizeof(uint64_t);
      }
    } else if (w.tid == cudf::type_id::FLOAT32 || w.tid == cudf::type_id::FLOAT64) {
      w.elem_size = static_cast<uint32_t>(cudf::size_of(cv.type()));
      // Temporarily mark as RAW_FIXED; all FLOAT/DOUBLE columns are batch-LZ4-compressed TOGETHER
      // in ONE single nvcompBatchedLZ4CompressAsync call below (1 GPU sync instead of N_cols syncs!).
      w.enc = ShufEnc::RAW_FIXED;
      for (size_t p = 0; p < n_parts; ++p) {
        w.part_payload_sz[p] = static_cast<size_t>(h_part_len[p]) * w.elem_size;
      }
    } else if (w.tid == cudf::type_id::STRING) {
      bool used_dict = false;
      if (nrows >= 256) {
        auto dc = cudf::dictionary::encode(cv, cudf::data_type{cudf::type_id::INT32}, stream, mr);
        cudf::dictionary_column_view const dcv(dc->view());
        uint32_t const ds = static_cast<uint32_t>(dcv.keys_size());
        if (ds > 0 && ds <= 1024) {
          cudf::strings_column_view const ksv(dcv.keys());
          int32_t kb = 0;
          CUDF_CUDA_TRY(cudaMemcpyAsync(
            &kb, ksv.offsets().data<int32_t>() + ds, sizeof(int32_t), cudaMemcpyDeviceToHost, stream.get()));
          stream.sync();
          if (kb <= 16384) {
            used_dict   = true;
            w.enc       = ShufEnc::DICT_BITPACK;
            w.dict_size = ds;
            w.key_bytes = static_cast<uint32_t>(kb);
            size_t const o_do = sizeof(DictPayloadHeader);
            size_t const o_dc = o_do + (ds + 1u) * sizeof(int32_t);
            size_t const o_cd = align8(o_dc + kb);
            w.h_dict_blob.assign(o_cd, 0);
            DictPayloadHeader dh{ds, static_cast<uint32_t>(kb), o_cd};
            std::memcpy(w.h_dict_blob.data(), &dh, sizeof(dh));
            CUDF_CUDA_TRY(cudaMemcpyAsync(w.h_dict_blob.data() + o_do,
                                          ksv.offsets().data<int32_t>(),
                                          (ds + 1u) * sizeof(int32_t),
                                          cudaMemcpyDeviceToHost,
                                          stream.get()));
            if (kb > 0) {
              CUDF_CUDA_TRY(cudaMemcpyAsync(w.h_dict_blob.data() + o_dc,
                                            ksv.chars_begin(stream),
                                            static_cast<size_t>(kb),
                                            cudaMemcpyDeviceToHost,
                                            stream.get()));
            }
            w.dict_col = std::move(dc);
            run_batched_for.template operator()<int32_t>(w, dcv.indices().data<int32_t>());
            for (size_t p = 0; p < n_parts; ++p) {
              uint32_t const nc = h_part_nblks[p];
              if (nc == 0) continue;
              uint32_t const b0 = h_part_blk_start[p];
              uint32_t const nw = w.h_word_offs[b0 + nc] - w.h_word_offs[b0];
              size_t const o_bw = nc * sizeof(int64_t);
              size_t const o_cw = align8(o_bw + nc * sizeof(uint8_t));
              size_t const o_pk = align8(o_cw + nc * sizeof(uint32_t));
              w.part_payload_sz[p] = o_cd + o_pk + static_cast<size_t>(nw) * sizeof(uint64_t);
            }
          }
        }
      }
      if (!used_dict) {
        w.enc = ShufEnc::RAW_STRING;
        cudf::strings_column_view const scv(cv);
        // Copy ONLY split boundary offsets (n_parts + 1 ints!) -> 1 tiny sync!
        // Gather ONLY n_parts boundary char counts on GPU (800 bytes D2H instead of 80 MiB full offsets!)
        w.h_str_base_off = 0;
        w.h_str_part_chars.assign(n_parts, 0);
        rmm::device_uvector<int32_t> d_part_chars(n_parts, stream, mr);
        uint32_t const gpb = (static_cast<uint32_t>(n_parts) + 255u) / 256u;
        gather_str_boundary_offs_kernel<<<gpb, 256, 0, stream.get()>>>(
          scv.offsets().data<int32_t>(),
          static_cast<uint32_t>(cv.offset()),
          d_part_start.data(),
          d_part_len.data(),
          d_part_chars.data(),
          static_cast<uint32_t>(n_parts));
        CUDF_CUDA_TRY(cudaMemcpyAsync(w.h_str_part_chars.data(),
                                      d_part_chars.data(),
                                      n_parts * sizeof(int32_t),
                                      cudaMemcpyDeviceToHost,
                                      stream.get()));
        stream.sync();
        for (size_t p = 0; p < n_parts; ++p) {
          if (h_part_len[p] == 0) continue;
          int32_t const nch      = w.h_str_part_chars[p];
          w.part_payload_sz[p]   = (h_part_len[p] + 1u) * sizeof(int32_t) + static_cast<size_t>(nch);
        }
      }
    } else {
      CUDF_FAIL("lenc_shuffle: unsupported column type");
    }
  }

  // Item 3: ONE SINGLE batched nvCOMP LZ4 call across ALL FLOAT32/FLOAT64 columns & partitions
  // combined (1 GPU sync per batch instead of 36 syncs on wide tables like NF_J0048!)
  rmm::device_buffer d_all_lz4_comp;
  {
    struct Lz4Job {
      uint32_t c;
      size_t p;
    };
    std::vector<Lz4Job> jobs;
    std::vector<void const*> h_in_ptrs;
    std::vector<size_t> h_in_szs;
    size_t max_chunk_b = 0;
    size_t total_raw_b = 0;
    for (uint32_t c = 0; c < ncols; ++c) {
      auto& w = works[c];
      // Run ONE single batched GPU nvCOMP LZ4 across ALL nullable/sparse FLOAT/DOUBLE columns
      // (skip short random RAW_STRING columns like TPC-H c_address/c_comment where LZ4 overhead > gain).
      if ((w.tid != cudf::type_id::FLOAT32 && w.tid != cudf::type_id::FLOAT64) || !w.has_nulls) {
        continue;
      }
      auto const cv        = table.column(static_cast<cudf::size_type>(c));
      auto const* col_base =
        static_cast<uint8_t const*>(cv.head()) + static_cast<size_t>(cv.offset()) * w.elem_size;
      for (size_t p = 0; p < n_parts; ++p) {
        size_t const rb = static_cast<size_t>(h_part_len[p]) * w.elem_size;
        if (rb >= 256) {
          jobs.push_back(Lz4Job{c, p});
          h_in_ptrs.push_back(col_base + static_cast<size_t>(h_part_start[p]) * w.elem_size);
          h_in_szs.push_back(rb);
          max_chunk_b = std::max(max_chunk_b, rb);
          total_raw_b += rb;
        }
      }
    }
    if (!jobs.empty() && total_raw_b >= 65536 && max_chunk_b <= (16u << 20)) {
      size_t const bn                     = jobs.size();
      nvcompBatchedLZ4CompressOpts_t opts = nvcompBatchedLZ4CompressDefaultOpts;
      size_t temp_b = 0, max_out_b = 0;
      if (nvcompBatchedLZ4CompressGetTempSize(bn, max_chunk_b, opts, &temp_b, total_raw_b, stream.get()) ==
            nvcompSuccess &&
          nvcompBatchedLZ4CompressGetMaxOutputChunkSize(max_chunk_b, opts, &max_out_b) ==
            nvcompSuccess) {
        d_all_lz4_comp  = rmm::device_buffer(bn * max_out_b, stream, mr);
        auto* comp_base = static_cast<uint8_t*>(d_all_lz4_comp.data());
        std::vector<void*> h_out_ptrs(bn);
        for (size_t i = 0; i < bn; ++i) { h_out_ptrs[i] = comp_base + i * max_out_b; }
        rmm::device_uvector<void const*> d_ip(bn, stream, mr);
        rmm::device_uvector<size_t> d_is(bn, stream, mr);
        rmm::device_uvector<void*> d_op(bn, stream, mr);
        rmm::device_uvector<size_t> d_os(bn, stream, mr);
        rmm::device_uvector<nvcompStatus_t> d_st(bn, stream, mr);
        rmm::device_buffer d_tmp(std::max<size_t>(temp_b, 64), stream, mr);
        CUDF_CUDA_TRY(cudaMemcpyAsync(
          d_ip.data(), h_in_ptrs.data(), bn * sizeof(void*), cudaMemcpyHostToDevice, stream.get()));
        CUDF_CUDA_TRY(cudaMemcpyAsync(
          d_is.data(), h_in_szs.data(), bn * sizeof(size_t), cudaMemcpyHostToDevice, stream.get()));
        CUDF_CUDA_TRY(cudaMemcpyAsync(
          d_op.data(), h_out_ptrs.data(), bn * sizeof(void*), cudaMemcpyHostToDevice, stream.get()));
        if (nvcompBatchedLZ4CompressAsync(d_ip.data(),
                                          d_is.data(),
                                          max_chunk_b,
                                          bn,
                                          d_tmp.data(),
                                          temp_b,
                                          d_op.data(),
                                          d_os.data(),
                                          opts,
                                          d_st.data(),
                                          stream.get()) == nvcompSuccess) {
          std::vector<size_t> h_os(bn);
          CUDF_CUDA_TRY(cudaMemcpyAsync(
            h_os.data(), d_os.data(), bn * sizeof(size_t), cudaMemcpyDeviceToHost, stream.get()));
          stream.sync();
          // Decide per-column whether LZ4 won (>=15% smaller)
          std::vector<size_t> col_raw(ncols, 0), col_comp(ncols, 0);
          for (size_t i = 0; i < bn; ++i) {
            col_raw[jobs[i].c] += h_in_szs[i];
            col_comp[jobs[i].c] += h_os[i];
          }
          for (uint32_t c = 0; c < ncols; ++c) {
            if (col_raw[c] > 0 && col_comp[c] * 100 < col_raw[c] * 85) {
              works[c].enc = (works[c].enc == ShufEnc::RAW_STRING) ? ShufEnc::STRING_LZ4
                                                                   : ShufEnc::NVCOMP_LZ4;
              works[c].h_lz4_src_offs.assign(n_parts, 0);
              works[c].h_lz4_comp_szs.assign(n_parts, 0);
            }
          }
          for (size_t i = 0; i < bn; ++i) {
            auto& w = works[jobs[i].c];
            if (w.enc == ShufEnc::NVCOMP_LZ4) {
              w.h_lz4_src_offs[jobs[i].p]  = i * max_out_b;
              w.h_lz4_comp_szs[jobs[i].p]  = h_os[i];
              w.part_payload_sz[jobs[i].p] = h_os[i];
            } else if (w.enc == ShufEnc::STRING_LZ4) {
              w.h_lz4_src_offs[jobs[i].p] = i * max_out_b;
              w.h_lz4_comp_szs[jobs[i].p] = h_os[i];
              w.part_payload_sz[jobs[i].p] =
                (h_part_len[jobs[i].p] + 1u) * sizeof(int32_t) + h_os[i];
            }
          }
        }
      }
    }
  }

  // Compute final contiguous d_data layout for ALL partitions on CPU (zero syncs!)
  std::vector<int64_t> h_part_offsets(n_parts + 1, 0);
  std::vector<std::vector<uint64_t>> col_dst_pay(ncols, std::vector<uint64_t>(n_parts, 0));
  std::vector<std::vector<uint64_t>> col_dst_null(ncols, std::vector<uint64_t>(n_parts, 0));

  size_t total_out = 0;
  for (size_t p = 0; p < n_parts; ++p) {
    h_part_offsets[p] = static_cast<int64_t>(total_out);
    if (h_part_len[p] == 0) { continue; }
    size_t const lshf_base = total_out + kudo_hdr;
    size_t cur             = align8(sizeof(LshfHeader) + ncols * sizeof(ShufColMeta));
    for (uint32_t c = 0; c < ncols; ++c) {
      col_dst_pay[c][p] = lshf_base + cur;
      cur               = align8(cur + works[c].part_payload_sz[p]);
      if (works[c].has_nulls) {
        col_dst_null[c][p] = lshf_base + cur;
        cur                = align8(cur + works[c].part_null_sz[p]);
      }
    }
    total_out += kudo_hdr + cur;
  }
  h_part_offsets[n_parts] = static_cast<int64_t>(total_out);

  rmm::device_buffer d_data(std::max<size_t>(total_out, 8), stream, mr);
  auto* d_base = static_cast<uint8_t*>(d_data.data());

  // Pack ALL Kudo + LSHF + ShufColMeta headers (+ small dict blobs) into ONE host buffer and H2D once!
  // Scatter headers per partition via async H2D (tiny ~200B per partition, pipelined without sync!)
  size_t const hdr_stride = kudo_hdr + sizeof(LshfHeader) + ncols * sizeof(ShufColMeta);
  std::vector<uint8_t> h_hdrs(n_parts * hdr_stride, 0);
  for (size_t p = 0; p < n_parts; ++p) {
    if (h_part_len[p] == 0) continue;
    uint8_t* hp = h_hdrs.data() + p * hdr_stride;
    size_t const lshf_bytes =
      static_cast<size_t>(h_part_offsets[p + 1] - h_part_offsets[p]) - kudo_hdr;
    write_be32(hp + 0, kKudoMagic);
    write_be32(hp + 4, 0u);
    write_be32(hp + 8, h_part_len[p]);
    write_be32(hp + 12, 0u);
    write_be32(hp + 16, 0u);
    write_be32(hp + 20, static_cast<uint32_t>(lshf_bytes));
    write_be32(hp + 24, ncols);
    LshfHeader lh{kLshfMagic, h_part_len[p], ncols, 0u};
    std::memcpy(hp + kudo_hdr, &lh, sizeof(lh));
    size_t const lshf_base = static_cast<size_t>(h_part_offsets[p]) + kudo_hdr;
    for (uint32_t c = 0; c < ncols; ++c) {
      ShufColMeta m{};
      m.payload_offset = col_dst_pay[c][p] - lshf_base;
      m.payload_size   = works[c].part_payload_sz[p];
      m.null_offset    = works[c].has_nulls ? (col_dst_null[c][p] - lshf_base) : 0ULL;
      m.null_size      = works[c].part_null_sz[p];
      m.null_count     = 0;  // computed at reduce if needed
      m.extra = (works[c].enc == ShufEnc::DICT_BITPACK)
                  ? works[c].dict_size
                  : ((works[c].enc == ShufEnc::RAW_STRING)
                       ? static_cast<uint32_t>(works[c].h_str_part_chars[p])
                       : ((works[c].enc == ShufEnc::NVCOMP_LZ4 || works[c].enc == ShufEnc::STRING_LZ4)
                            ? static_cast<uint32_t>(works[c].h_lz4_comp_szs[p])
                            : 0u));
      m.encoding =
        (works[c].enc == ShufEnc::NVCOMP_LZ4 && works[c].h_lz4_comp_szs[p] == 0)
          ? static_cast<uint8_t>(ShufEnc::RAW_FIXED)
          : ((works[c].enc == ShufEnc::STRING_LZ4 && works[c].h_lz4_comp_szs[p] == 0)
               ? static_cast<uint8_t>(ShufEnc::RAW_STRING)
               : static_cast<uint8_t>(works[c].enc));
      m.has_nulls      = works[c].has_nulls ? 1 : 0;
      m.scale          = works[c].scale;
      m.type_id        = static_cast<uint8_t>(works[c].tid);
      std::memcpy(hp + kudo_hdr + sizeof(lh) + c * sizeof(ShufColMeta), &m, sizeof(m));
    }
  }
  rmm::device_buffer d_offs((n_parts + 1) * sizeof(int64_t), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(d_offs.data(),
                                h_part_offsets.data(),
                                (n_parts + 1) * sizeof(int64_t),
                                cudaMemcpyHostToDevice,
                                stream.get()));
  rmm::device_buffer d_hdrs(h_hdrs.size(), stream, mr);
  if (!h_hdrs.empty()) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      d_hdrs.data(), h_hdrs.data(), h_hdrs.size(), cudaMemcpyHostToDevice, stream.get()));
    scatter_headers_kernel<<<static_cast<uint32_t>(n_parts), 64, 0, stream.get()>>>(
      static_cast<uint8_t const*>(d_hdrs.data()),
      d_base,
      static_cast<int64_t const*>(d_offs.data()),
      d_part_len.data(),
      hdr_stride,
      static_cast<uint32_t>(n_parts));
  }

  // Launch ONE GPU scatter kernel per column across ALL 200 partitions in parallel!
  rmm::device_uvector<uint64_t> d_dst_offs(n_parts, stream, mr);
  for (uint32_t c = 0; c < ncols; ++c) {
    ColWork& w    = works[c];
    auto const cv = table.column(static_cast<cudf::size_type>(c));
    if (nrows == 0) continue;

    if (w.enc == ShufEnc::FOR_BITPACK) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(d_dst_offs.data(),
                                    col_dst_pay[c].data(),
                                    n_parts * sizeof(uint64_t),
                                    cudaMemcpyHostToDevice,
                                    stream.get()));
      scatter_for256_slices_kernel<<<static_cast<uint32_t>(n_parts), 256, 0, stream.get()>>>(
        d_base,
        d_dst_offs.data(),
        d_part_blk_start.data(),
        d_part_nblks.data(),
        w.d_mins.data(),
        w.d_bws.data(),
        w.d_word_offs.data(),
        w.d_words.data(),
        static_cast<uint32_t>(n_parts));
    } else if (w.enc == ShufEnc::DICT_BITPACK) {
      // Write tiny dict blob at start of each non-empty partition's payload + scatter FOR codes!
      size_t const dblob_sz = w.h_dict_blob.size();
      std::vector<uint64_t> code_dst_offs(n_parts, 0);
      for (size_t p = 0; p < n_parts; ++p) {
        if (h_part_len[p] == 0) continue;
        CUDF_CUDA_TRY(cudaMemcpyAsync(d_base + col_dst_pay[c][p],
                                      w.h_dict_blob.data(),
                                      dblob_sz,
                                      cudaMemcpyHostToDevice,
                                      stream.get()));
        code_dst_offs[p] = col_dst_pay[c][p] + dblob_sz;
      }
      CUDF_CUDA_TRY(cudaMemcpyAsync(d_dst_offs.data(),
                                    code_dst_offs.data(),
                                    n_parts * sizeof(uint64_t),
                                    cudaMemcpyHostToDevice,
                                    stream.get()));
      scatter_for256_slices_kernel<<<static_cast<uint32_t>(n_parts), 256, 0, stream.get()>>>(
        d_base,
        d_dst_offs.data(),
        d_part_blk_start.data(),
        d_part_nblks.data(),
        w.d_mins.data(),
        w.d_bws.data(),
        w.d_word_offs.data(),
        w.d_words.data(),
        static_cast<uint32_t>(n_parts));
    } else if (w.enc == ShufEnc::NVCOMP_LZ4) {
      rmm::device_uvector<uint64_t> d_src_offs(n_parts, stream, mr);
      rmm::device_uvector<uint64_t> d_comp_szs(n_parts, stream, mr);
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        d_dst_offs.data(), col_dst_pay[c].data(), n_parts * sizeof(uint64_t), cudaMemcpyHostToDevice, stream.get()));
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        d_src_offs.data(), w.h_lz4_src_offs.data(), n_parts * sizeof(uint64_t), cudaMemcpyHostToDevice, stream.get()));
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        d_comp_szs.data(), w.h_lz4_comp_szs.data(), n_parts * sizeof(uint64_t), cudaMemcpyHostToDevice, stream.get()));
      scatter_lz4_slices_kernel<<<static_cast<uint32_t>(n_parts), 256, 0, stream.get()>>>(
        static_cast<uint8_t const*>(d_all_lz4_comp.data()),
        static_cast<uint8_t const*>(cv.head()) + static_cast<size_t>(cv.offset()) * w.elem_size,
        d_base,
        d_dst_offs.data(),
        d_src_offs.data(),
        d_comp_szs.data(),
        d_part_start.data(),
        d_part_len.data(),
        w.elem_size,
        static_cast<uint32_t>(n_parts));
    } else if (w.enc == ShufEnc::RAW_FIXED) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(d_dst_offs.data(),
                                    col_dst_pay[c].data(),
                                    n_parts * sizeof(uint64_t),
                                    cudaMemcpyHostToDevice,
                                    stream.get()));
      scatter_raw_fixed_kernel<<<static_cast<uint32_t>(n_parts), 256, 0, stream.get()>>>(
        static_cast<uint8_t const*>(cv.head()) + static_cast<size_t>(cv.offset()) * w.elem_size,
        d_base,
        d_dst_offs.data(),
        d_part_start.data(),
        d_part_len.data(),
        w.elem_size,
        static_cast<uint32_t>(n_parts));
    } else if (w.enc == ShufEnc::STRING_LZ4) {
      cudf::strings_column_view const scv(cv);
      rmm::device_uvector<uint64_t> d_src_offs(n_parts, stream, mr);
      rmm::device_uvector<uint64_t> d_comp_szs(n_parts, stream, mr);
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        d_dst_offs.data(), col_dst_pay[c].data(), n_parts * sizeof(uint64_t), cudaMemcpyHostToDevice, stream.get()));
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        d_src_offs.data(), w.h_lz4_src_offs.data(), n_parts * sizeof(uint64_t), cudaMemcpyHostToDevice, stream.get()));
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        d_comp_szs.data(), w.h_lz4_comp_szs.data(), n_parts * sizeof(uint64_t), cudaMemcpyHostToDevice, stream.get()));
      scatter_string_lz4_kernel<<<static_cast<uint32_t>(n_parts), 256, 0, stream.get()>>>(
        scv.offsets().data<int32_t>(),
        static_cast<uint8_t const*>(d_all_lz4_comp.data()),
        scv.chars_begin(stream),
        static_cast<uint32_t>(cv.offset()),
        d_base,
        d_dst_offs.data(),
        d_src_offs.data(),
        d_comp_szs.data(),
        d_part_start.data(),
        d_part_len.data(),
        static_cast<uint32_t>(n_parts));
    } else if (w.enc == ShufEnc::RAW_STRING) {
      cudf::strings_column_view const scv(cv);
      CUDF_CUDA_TRY(cudaMemcpyAsync(d_dst_offs.data(),
                                    col_dst_pay[c].data(),
                                    n_parts * sizeof(uint64_t),
                                    cudaMemcpyHostToDevice,
                                    stream.get()));
      scatter_raw_string_kernel<<<static_cast<uint32_t>(n_parts), 256, 0, stream.get()>>>(
        scv.offsets().data<int32_t>(),
        scv.chars_begin(stream),
        static_cast<uint32_t>(cv.offset()),
        d_base,
        d_dst_offs.data(),
        d_part_start.data(),
        d_part_len.data(),
        static_cast<uint32_t>(n_parts));
    }

    if (w.has_nulls) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(d_dst_offs.data(),
                                    col_dst_null[c].data(),
                                    n_parts * sizeof(uint64_t),
                                    cudaMemcpyHostToDevice,
                                    stream.get()));
      scatter_bitmask_kernel<<<static_cast<uint32_t>(n_parts), 256, 0, stream.get()>>>(
        cv.null_mask(),
        static_cast<uint32_t>(cv.offset()),
        d_base,
        d_dst_offs.data(),
        d_part_start.data(),
        d_part_len.data(),
        static_cast<uint32_t>(n_parts));
    }
  }

  stream.sync();
  return {std::move(d_data), std::move(d_offs)};
}

std::unique_ptr<cudf::table> assemble_and_decode_shuffle_from_device(
  uint8_t const* h_data,
  uint8_t const* d_data,
  size_t /*data_bytes*/,
  int64_t const* h_offsets,
  size_t num_tables,
  std::vector<cudf::data_type> const& col_types,
  std::vector<uint8_t> const& /*dict_as_dictionary32*/,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  uint32_t const ncols = static_cast<uint32_t>(col_types.size());
  // Parse ALL headers directly from h_data on CPU -> ZERO D2H syncs!!
  std::vector<std::vector<DecodeSliceTask>> col_tasks(ncols);
  std::vector<uint8_t> col_any_nulls(ncols, 0);
  uint32_t total_rows = 0;

  for (size_t t = 0; t < num_tables; ++t) {
    size_t const plen = static_cast<size_t>(h_offsets[t + 1] - h_offsets[t]);
    if (plen < sizeof(LshfHeader)) { continue; }
    uint8_t const* h_lshf = h_data + h_offsets[t];
    LshfHeader lh{};
    std::memcpy(&lh, h_lshf, sizeof(lh));
    CUDF_EXPECTS(lh.magic == kLshfMagic, "assemble_and_decode_shuffle: bad LSHF magic");
    if (lh.num_rows == 0) { continue; }
    auto const* metas = reinterpret_cast<ShufColMeta const*>(h_lshf + sizeof(lh));
    for (uint32_t c = 0; c < ncols; ++c) {
      ShufColMeta const& m = metas[c];
      DecodeSliceTask tk{};
      tk.payload_off = static_cast<uint64_t>(h_offsets[t]) + m.payload_offset;
      tk.null_off    = m.has_nulls ? (static_cast<uint64_t>(h_offsets[t]) + m.null_offset) : 0ULL;
      tk.out_row_off = total_rows;
      tk.num_rows    = lh.num_rows;
      tk.extra       = m.extra;
      tk.encoding    = m.encoding;
      tk.has_nulls   = m.has_nulls;
      if (m.has_nulls) { col_any_nulls[c] = 1; }
      col_tasks[c].push_back(tk);
    }
    total_rows += lh.num_rows;
  }

  if (total_rows == 0) {
    std::vector<std::unique_ptr<cudf::column>> empty_cols;
    for (auto const& dt : col_types) { empty_cols.push_back(cudf::make_empty_column(dt)); }
    return std::make_unique<cudf::table>(std::move(empty_cols));
  }

  // Launch ONE batched coalescing GPU decode kernel per column across ALL incoming slices!
  std::vector<std::unique_ptr<cudf::column>> out_cols;
  out_cols.reserve(ncols);
  for (uint32_t c = 0; c < ncols; ++c) {
    auto const dtype   = col_types[c];
    auto const tid     = dtype.id();
    uint32_t const ntk = static_cast<uint32_t>(col_tasks[c].size());
    rmm::device_uvector<DecodeSliceTask> d_tasks(ntk, stream, mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(d_tasks.data(),
                                  col_tasks[c].data(),
                                  ntk * sizeof(DecodeSliceTask),
                                  cudaMemcpyHostToDevice,
                                  stream.get()));

    auto make_coalesced_nulls = [&](cudf::size_type& out_nc) {
      if (!col_any_nulls[c]) {
        out_nc = 0;
        return cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr);
      }
      auto mask             = cudf::create_null_mask(total_rows, cudf::mask_state::UNINITIALIZED, stream, mr);
      uint32_t const nwords = (total_rows + 31u) >> 5;
      uint32_t const gw     = (nwords + 255u) / 256u;
      batched_decode_nulls_kernel<<<gw, 256, 0, stream.get()>>>(
        d_data, d_tasks.data(), ntk, static_cast<cudf::bitmask_type*>(mask.data()), total_rows);
      out_nc = cudf::null_count(
        static_cast<cudf::bitmask_type const*>(mask.data()), 0, static_cast<cudf::size_type>(total_rows), stream);
      return mask;
    };

    if (tid == cudf::type_id::STRING) {
      auto out_offs = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                static_cast<cudf::size_type>(total_rows + 1u),
                                                cudf::mask_state::UNALLOCATED,
                                                stream,
                                                mr);
      auto* d_oo    = out_offs->mutable_view().data<int32_t>();
      batched_decode_str_lens_kernel<<<ntk, 256, 0, stream.get()>>>(
        d_data, d_tasks.data(), ntk, d_oo, total_rows);
      thrust::exclusive_scan(rmm::exec_policy_nosync(stream), d_oo, d_oo + total_rows + 1u, d_oo);
      int32_t total_ch = 0;
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        &total_ch, d_oo + total_rows, sizeof(int32_t), cudaMemcpyDeviceToHost, stream.get()));
      stream.sync();
      rmm::device_buffer out_ch(static_cast<size_t>(total_ch), stream, mr);
      if (total_ch > 0) {
        // Opt A: Decompress STRING_LZ4 slices directly into out_ch via ONE batched nvCOMP LZ4 call!
        std::vector<DecodeSliceTask> non_lz4_str;
        std::vector<void const*> h_ip;
        std::vector<size_t> h_is;
        std::vector<void*> h_op;
        std::vector<size_t> h_os;
        size_t max_ub = 0, tot_ub = 0;
        bool any_str_lz4 = false;
        for (auto const& tk : col_tasks[c]) {
          if (tk.encoding == static_cast<uint8_t>(ShufEnc::STRING_LZ4)) { any_str_lz4 = true; break; }
        }
        if (any_str_lz4) {
          // Copy slice starting char offsets from d_oo (1 tiny gather or full host copy of boundary offs)
          std::vector<int32_t> h_slice_dst_ch(ntk, 0);
          for (size_t i = 0; i < ntk; ++i) {
            auto const& tk = col_tasks[c][i];
            if (tk.encoding == static_cast<uint8_t>(ShufEnc::STRING_LZ4)) {
              CUDF_CUDA_TRY(cudaMemcpyAsync(&h_slice_dst_ch[i],
                                            d_oo + tk.out_row_off,
                                            sizeof(int32_t),
                                            cudaMemcpyDeviceToHost,
                                            stream.get()));
            }
          }
          stream.sync();
          auto* out_ch_u8 = static_cast<uint8_t*>(out_ch.data());
          for (size_t i = 0; i < ntk; ++i) {
            auto const& tk = col_tasks[c][i];
            if (tk.encoding == static_cast<uint8_t>(ShufEnc::STRING_LZ4)) {
              int32_t uncomp_ch = 0;
              std::memcpy(&uncomp_ch,
                          h_data + tk.payload_off + static_cast<size_t>(tk.num_rows) * sizeof(int32_t),
                          sizeof(int32_t));
              if (uncomp_ch > 0) {
                size_t const ub = static_cast<size_t>(uncomp_ch);
                h_ip.push_back(d_data + tk.payload_off + (tk.num_rows + 1u) * sizeof(int32_t));
                h_is.push_back(tk.extra);
                h_op.push_back(out_ch_u8 + h_slice_dst_ch[i]);
                h_os.push_back(ub);
                max_ub = std::max(max_ub, ub);
                tot_ub += ub;
              }
            } else {
              non_lz4_str.push_back(tk);
            }
          }
          if (!h_ip.empty()) {
            size_t const bn = h_ip.size();
            size_t temp_b   = 0;
            nvcompBatchedLZ4DecompressOpts_t dopts = nvcompBatchedLZ4DecompressDefaultOpts;
            CUDF_EXPECTS(nvcompBatchedLZ4DecompressGetTempSize(
                           bn, max_ub, dopts, &temp_b, tot_ub, stream.get()) == nvcompSuccess,
                         "lenc_shuffle: STRING_LZ4 GetTempSize failed");
            rmm::device_uvector<void const*> d_ip(bn, stream, mr);
            rmm::device_uvector<size_t> d_is(bn, stream, mr);
            rmm::device_uvector<void*> d_op(bn, stream, mr);
            rmm::device_uvector<size_t> d_os(bn, stream, mr);
            rmm::device_uvector<size_t> d_act(bn, stream, mr);
            rmm::device_uvector<nvcompStatus_t> d_st(bn, stream, mr);
            rmm::device_buffer d_tmp(std::max<size_t>(temp_b, 64), stream, mr);
            CUDF_CUDA_TRY(cudaMemcpyAsync(
              d_ip.data(), h_ip.data(), bn * sizeof(void*), cudaMemcpyHostToDevice, stream.get()));
            CUDF_CUDA_TRY(cudaMemcpyAsync(
              d_is.data(), h_is.data(), bn * sizeof(size_t), cudaMemcpyHostToDevice, stream.get()));
            CUDF_CUDA_TRY(cudaMemcpyAsync(
              d_op.data(), h_op.data(), bn * sizeof(void*), cudaMemcpyHostToDevice, stream.get()));
            CUDF_CUDA_TRY(cudaMemcpyAsync(
              d_os.data(), h_os.data(), bn * sizeof(size_t), cudaMemcpyHostToDevice, stream.get()));
            CUDF_EXPECTS(nvcompBatchedLZ4DecompressAsync(d_ip.data(),
                                                         d_is.data(),
                                                         d_os.data(),
                                                         d_act.data(),
                                                         bn,
                                                         d_tmp.data(),
                                                         temp_b,
                                                         d_op.data(),
                                                         dopts,
                                                         d_st.data(),
                                                         stream.get()) == nvcompSuccess,
                         "lenc_shuffle: STRING_LZ4 decompress failed");
          }
          if (!non_lz4_str.empty()) {
            uint32_t const n2 = static_cast<uint32_t>(non_lz4_str.size());
            rmm::device_uvector<DecodeSliceTask> d_t2(n2, stream, mr);
            CUDF_CUDA_TRY(cudaMemcpyAsync(
              d_t2.data(), non_lz4_str.data(), n2 * sizeof(DecodeSliceTask), cudaMemcpyHostToDevice, stream.get()));
            batched_decode_str_copy_kernel<<<n2, 256, 0, stream.get()>>>(
              d_data, d_t2.data(), n2, d_oo, static_cast<char*>(out_ch.data()));
          }
        } else {
          batched_decode_str_copy_kernel<<<ntk, 256, 0, stream.get()>>>(
            d_data, d_tasks.data(), ntk, d_oo, static_cast<char*>(out_ch.data()));
        }
      }
      cudf::size_type nc = 0;
      auto nbuf          = make_coalesced_nulls(nc);
      out_cols.push_back(cudf::make_strings_column(static_cast<cudf::size_type>(total_rows),
                                                   std::move(out_offs),
                                                   std::move(out_ch),
                                                   nc,
                                                   std::move(nbuf)));
    } else {
      cudf::size_type nc = 0;
      auto nbuf          = make_coalesced_nulls(nc);
      auto col           = cudf::make_fixed_width_column(
        dtype, static_cast<cudf::size_type>(total_rows), std::move(nbuf), nc, stream, mr);
      size_t const esz = cudf::size_of(dtype);
      // Separate NVCOMP_LZ4 tasks (batch decompress in ONE nvCOMP call) from FOR/RAW tasks
      std::vector<DecodeSliceTask> non_lz4;
      std::vector<void const*> h_in_ptrs;
      std::vector<size_t> h_in_szs;
      std::vector<void*> h_out_ptrs;
      std::vector<size_t> h_out_szs;
      size_t max_uncomp_b = 0;
      size_t tot_uncomp_b = 0;
      auto* col_out_base  = static_cast<uint8_t*>(col->mutable_view().head());
      for (size_t i = 0; i < col_tasks[c].size(); ++i) {
        auto const& tk = col_tasks[c][i];
        if (tk.encoding == static_cast<uint8_t>(ShufEnc::NVCOMP_LZ4)) {
          size_t const ub = static_cast<size_t>(tk.num_rows) * esz;
          h_in_ptrs.push_back(d_data + tk.payload_off);
          h_in_szs.push_back(tk.extra);  // payload_size stored in extra for LZ4 decode
          h_out_ptrs.push_back(col_out_base + static_cast<size_t>(tk.out_row_off) * esz);
          h_out_szs.push_back(ub);
          max_uncomp_b = std::max(max_uncomp_b, ub);
          tot_uncomp_b += ub;
        } else {
          non_lz4.push_back(tk);
        }
      }
      if (!h_in_ptrs.empty()) {
        size_t const bn = h_in_ptrs.size();
        size_t temp_b   = 0;
        nvcompBatchedLZ4DecompressOpts_t dopts = nvcompBatchedLZ4DecompressDefaultOpts;
        CUDF_EXPECTS(nvcompBatchedLZ4DecompressGetTempSize(
                       bn, max_uncomp_b, dopts, &temp_b, tot_uncomp_b, stream.get()) ==
                       nvcompSuccess,
                     "lenc_shuffle: nvCOMP LZ4 decompress GetTempSize failed");
        rmm::device_uvector<void const*> d_ip(bn, stream, mr);
        rmm::device_uvector<size_t> d_is(bn, stream, mr);
        rmm::device_uvector<void*> d_op(bn, stream, mr);
        rmm::device_uvector<size_t> d_os(bn, stream, mr);
        rmm::device_uvector<size_t> d_act(bn, stream, mr);
        rmm::device_uvector<nvcompStatus_t> d_st(bn, stream, mr);
        rmm::device_buffer d_tmp(std::max<size_t>(temp_b, 64), stream, mr);
        CUDF_CUDA_TRY(cudaMemcpyAsync(
          d_ip.data(), h_in_ptrs.data(), bn * sizeof(void*), cudaMemcpyHostToDevice, stream.get()));
        CUDF_CUDA_TRY(cudaMemcpyAsync(
          d_is.data(), h_in_szs.data(), bn * sizeof(size_t), cudaMemcpyHostToDevice, stream.get()));
        CUDF_CUDA_TRY(cudaMemcpyAsync(
          d_op.data(), h_out_ptrs.data(), bn * sizeof(void*), cudaMemcpyHostToDevice, stream.get()));
        CUDF_CUDA_TRY(cudaMemcpyAsync(
          d_os.data(), h_out_szs.data(), bn * sizeof(size_t), cudaMemcpyHostToDevice, stream.get()));
        CUDF_EXPECTS(nvcompBatchedLZ4DecompressAsync(d_ip.data(),
                                                     d_is.data(),
                                                     d_os.data(),
                                                     d_act.data(),
                                                     bn,
                                                     d_tmp.data(),
                                                     temp_b,
                                                     d_op.data(),
                                                     dopts,
                                                     d_st.data(),
                                                     stream.get()) == nvcompSuccess,
                     "lenc_shuffle: nvCOMP LZ4 decompress failed");
      }
      if (!non_lz4.empty()) {
        uint32_t const n2 = static_cast<uint32_t>(non_lz4.size());
        rmm::device_uvector<DecodeSliceTask> d_t2(n2, stream, mr);
        CUDF_CUDA_TRY(cudaMemcpyAsync(
          d_t2.data(), non_lz4.data(), n2 * sizeof(DecodeSliceTask), cudaMemcpyHostToDevice, stream.get()));
        if (esz == 8) {
          batched_decode_fixed_kernel<int64_t><<<n2, 256, 0, stream.get()>>>(
            d_data, d_t2.data(), n2, static_cast<int64_t*>(col->mutable_view().head()));
        } else if (esz == 4) {
          batched_decode_fixed_kernel<int32_t><<<n2, 256, 0, stream.get()>>>(
            d_data, d_t2.data(), n2, static_cast<int32_t*>(col->mutable_view().head()));
        } else if (esz == 2) {
          batched_decode_fixed_kernel<int16_t><<<n2, 256, 0, stream.get()>>>(
            d_data, d_t2.data(), n2, static_cast<int16_t*>(col->mutable_view().head()));
        } else {
          batched_decode_fixed_kernel<int8_t><<<n2, 256, 0, stream.get()>>>(
            d_data, d_t2.data(), n2, static_cast<int8_t*>(col->mutable_view().head()));
        }
      }
      out_cols.push_back(std::move(col));
    }
  }

  stream.sync();
  return std::make_unique<cudf::table>(std::move(out_cols));
}

}  // namespace rapids::jni::lenc

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

#include <unordered_map>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
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

struct DictPayloadHeader {
  uint32_t dict_size;
  uint32_t key_bytes;
  uint64_t codes_offset;
};

__host__ __device__ constexpr uint64_t align8(uint64_t v) { return (v + 7ULL) & ~7ULL; }

enum class ShufEnc : uint8_t {
  RAW_FIXED     = 0,
  FOR_BITPACK   = 1,
  DICT_BITPACK  = 2,
  RAW_STRING    = 3,
  NVCOMP_LZ4    = 4,  // GPU nvCOMP Batched LZ4 for nullable/sparse FLOAT/DOUBLE!
  STRING_LZ4    = 5,  // Optional GPU nvCOMP LZ4 for strings
  STRING_BIGRAM = 6,  // Opt V3: Zero-sync dynamic 127-bigram GPU mini-FSST string codec (1.5x-2.0x smaller, 250+ GB/s)!
  FLOAT_ALP     = 7,  // Item 4: Exact bit-for-bit lossless decimal-scaled FLOAT32/FLOAT64 -> FOR256 (1.5x-8x smaller!)
  STRING_FSST   = 8,  // P1-B: 255-symbol multi-length (1..4B) GPU FSST string codec (2.2x-3.0x smaller, 250+ GB/s!)
};

constexpr size_t kFsstHdrBytes = 1280;  // 255*4B sym_word + 255B sym_len + 5B pad (8B aligned!)

struct alignas(8) FsstEncodeTable {
  uint32_t sym_word[255];
  uint8_t sym_len[255];
  uint8_t pad0[5];
  uint8_t single_code[256];   // 0..254 or 0xFF (escape)
  uint8_t bucket_head[1024];  // 2-byte prefix hash -> first multi-byte sym code (longest-first!), 0xFF=empty
  uint8_t sym_next[255];      // linked list next symbol code in bucket, 0xFF=end
  uint8_t pad1;
};

__host__ __device__ __forceinline__ uint32_t fsst_hash2(uint8_t b0, uint8_t b1)
{
  return ((static_cast<uint32_t>(b0) * 2654435761u) ^ (static_cast<uint32_t>(b1) * 40503u)) & 1023u;
}

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
                                            cudf::bitmask_type const* __restrict__ null_mask,
                                            cudf::size_type col_off,
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
  bool const in_bounds = (threadIdx.x < len);
  uint32_t const r     = start + threadIdx.x;
  bool const valid =
    in_bounds && (null_mask == nullptr || cudf::bit_is_set(null_mask, col_off + static_cast<cudf::size_type>(r)));

  int64_t const v_lo = valid ? static_cast<int64_t>(data[r]) : INT64_MAX;
  int64_t const v_hi = valid ? static_cast<int64_t>(data[r]) : INT64_MIN;
  int64_t const lo   = BlockReduce(ts_min).Reduce(v_lo, cuda::minimum<>{});
  int64_t const hi   = BlockReduce(ts_max).Reduce(v_hi, cuda::maximum<>{});

  if (threadIdx.x == 0) {
    if (lo > hi) {
      // All-null 256-row block: 0 bits needed, zero payload!
      min_vals[b]   = 0;
      bit_widths[b] = 0u;
      blk_words[b]  = 0u;
    } else {
      min_vals[b]          = lo;
      uint64_t const range = static_cast<uint64_t>(hi) - static_cast<uint64_t>(lo);
      uint8_t const bw     = d_bits_needed(range);
      bit_widths[b]        = bw;
      blk_words[b]         = (bw == 0u) ? 0u : ((len * bw + 63u) / 64u + 1u);
    }
  }
}

template <typename T>
__global__ void batched_for256_pack_kernel(T const* __restrict__ data,
                                           cudf::bitmask_type const* __restrict__ null_mask,
                                           cudf::size_type col_off,
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
    uint32_t const r = start + threadIdx.x;
    bool const valid =
      (null_mask == nullptr || cudf::bit_is_set(null_mask, col_off + static_cast<cudf::size_type>(r)));
    uint64_t const delta =
      valid ? static_cast<uint64_t>(static_cast<int64_t>(data[r]) - base) : 0ULL;
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

// Item 4: Exact bit-for-bit lossless GPU ALP-Lite decimal scaling for FLOAT32/FLOAT64 -> FOR256!
__device__ __forceinline__ uint64_t float_raw_bits(float v)
{
  return static_cast<uint64_t>(__float_as_uint(v));
}
__device__ __forceinline__ uint64_t float_raw_bits(double v)
{
  return static_cast<uint64_t>(__double_as_longlong(v));
}

__device__ __forceinline__ double alp_factor_for_exp(uint32_t exp)
{
  switch (exp) {
    case 0u: return 1.0;
    case 1u: return 10.0;
    case 2u: return 100.0;
    case 3u: return 1000.0;
    default: return 10000.0;
  }
}

template <typename T>
__device__ __forceinline__ bool alp_exact_quantize(T x, double f, int64_t& out_q)
{
  double const d = static_cast<double>(x);
  if (!isfinite(d)) { return false; }
  // Preserve exact -0.0 sign bit by rejecting negative zero (falls back cleanly to RAW_FIXED)
  if (d == 0.0 && signbit(d)) { return false; }
  double const scaled = d * f;
  if (fabs(scaled) > 4.5e15) { return false; }
  long long const q = llround(scaled);
  T const recon     = static_cast<T>(static_cast<double>(q) / f);
  if (float_raw_bits(recon) != float_raw_bits(x)) { return false; }
  out_q = static_cast<int64_t>(q);
  return true;
}

// Fast 1-block 2,048-row probe across exponents 0..4 (1, 10, 100, 1000, 10000)
template <typename T>
__global__ void float_alp_probe_kernel(T const* __restrict__ in_data,
                                       cudf::bitmask_type const* __restrict__ null_mask,
                                       uint32_t col_off,
                                       uint32_t nrows,
                                       uint32_t* __restrict__ d_ok)
{
  uint32_t const n_samp = min(nrows, 2048u);
  uint32_t const step   = max(1u, nrows / n_samp);
  for (uint32_t i = threadIdx.x; i < n_samp; i += blockDim.x) {
    uint32_t const r = i * step;
    if (null_mask != nullptr && !cudf::bit_is_set(null_mask, static_cast<cudf::size_type>(col_off + r))) {
      continue;
    }
    T const x = in_data[r];
    for (uint32_t e = 0u; e < 5u; ++e) {
      if (d_ok[e] != 0u) {
        int64_t q = 0;
        if (!alp_exact_quantize(x, alp_factor_for_exp(e), q)) { d_ok[e] = 0u; }
      }
    }
  }
}

// Full-column exact bit-for-bit quantization into int64_t + mismatch counter
template <typename T>
__global__ void float_alp_encode_kernel(T const* __restrict__ in_data,
                                        cudf::bitmask_type const* __restrict__ null_mask,
                                        uint32_t col_off,
                                        uint32_t nrows,
                                        double f,
                                        int64_t* __restrict__ out_q,
                                        uint32_t* __restrict__ d_mismatch)
{
  uint32_t const tid    = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t const stride = gridDim.x * blockDim.x;
  for (uint32_t r = tid; r < nrows; r += stride) {
    if (null_mask != nullptr && !cudf::bit_is_set(null_mask, static_cast<cudf::size_type>(col_off + r))) {
      out_q[r] = 0;
      continue;
    }
    int64_t q = 0;
    if (alp_exact_quantize(in_data[r], f, q)) {
      out_q[r] = q;
    } else {
      out_q[r] = 0;
      atomicAdd(d_mismatch, 1u);
    }
  }
}

// Item 5: Single GPU kernel scatters one small dictionary blob into all non-empty partitions
// (replaces 200 separate per-partition cudaMemcpyAsync H2D calls per DICT_BITPACK column!)
__global__ void scatter_dict_blob_kernel(uint8_t const* __restrict__ d_blob,
                                         size_t blob_sz,
                                         uint8_t* __restrict__ d_out,
                                         uint64_t const* __restrict__ dst_payload_offs,
                                         uint32_t const* __restrict__ part_row_len,
                                         uint32_t num_parts)
{
  uint32_t const p = blockIdx.x;
  if (p >= num_parts || part_row_len[p] == 0u) { return; }
  uint8_t* dst          = d_out + dst_payload_offs[p];
  uint32_t const tid    = blockIdx.y * blockDim.x + threadIdx.x;
  uint32_t const stride = gridDim.y * blockDim.x;
  for (size_t i = tid; i < blob_sz; i += stride) { dst[i] = d_blob[i]; }
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

  uint8_t* d_mins       = dst + off_min;
  uint8_t* d_bws        = dst + off_bw;
  uint8_t* d_cwo        = dst + off_cwo;
  uint8_t* d_words      = dst + off_pack;
  uint32_t const tid    = blockIdx.y * blockDim.x + threadIdx.x;
  uint32_t const stride = gridDim.y * blockDim.x;

  for (uint32_t i = tid; i < nc; i += stride) {
    st_u64_u(d_mins + static_cast<size_t>(i) * 8u, static_cast<uint64_t>(all_mins[b0 + i]));
    d_bws[i] = all_bws[b0 + i];
    st_u32_u(d_cwo + static_cast<size_t>(i) * 4u, all_word_offs[b0 + i] - wbase);
  }
  for (uint32_t i = tid; i < nwords; i += stride) {
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
  uint64_t const csz    = lz4_comp_szs[p];
  uint8_t* d            = d_out + dst_payload_offs[p];
  uint32_t const tid    = blockIdx.y * blockDim.x + threadIdx.x;
  uint32_t const stride = gridDim.y * blockDim.x;
  if (csz > 0ULL) {
    uint8_t const* s = comp_base + lz4_src_offs[p];
    for (uint64_t i = tid; i < csz; i += stride) { d[i] = s[i]; }
  } else {
    size_t const nb  = static_cast<size_t>(len) * elem_size;
    uint8_t const* s = raw_base + static_cast<size_t>(part_row_start[p]) * elem_size;
    for (size_t i = tid; i < nb; i += stride) { d[i] = s[i]; }
  }
}

// Opt V3: Zero-sync GPU Dynamic Bigram (mini-FSST) String Codec
// Encoding rules (100% lossless for arbitrary UTF-8 / binary):
//   - Byte < 0x80: literal ASCII byte (or paired with next ASCII byte -> symbol 0x80..0xFE if in 127-entry LUT)
//   - Byte >= 0x80: escaped as [0xFF, byte]
// Payload layout per partition:
//   [uint16_t sym_table[128] (256B)] [int32_t comp_offs[n+1]] [uint8_t comp_chars[...]]

__global__ void bigram_measure_rows_kernel(int32_t const* __restrict__ in_offs,
                                           char const* __restrict__ in_chars,
                                           uint32_t src_col_off,
                                           uint32_t nrows,
                                           uint8_t const* __restrict__ bigram_lut,
                                           int32_t* __restrict__ out_comp_lens)
{
  uint32_t const r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= nrows) { return; }
  int32_t const s = in_offs[src_col_off + r];
  int32_t const e = in_offs[src_col_off + r + 1u];
  int32_t clen    = 0;
  int32_t i       = s;
  while (i < e) {
    uint8_t const b0 = static_cast<uint8_t>(in_chars[i]);
    if (b0 >= 0x80u) {
      clen += 2;
      i += 1;
    } else if (i + 1 < e) {
      uint8_t const b1 = static_cast<uint8_t>(in_chars[i + 1]);
      if (b1 < 0x80u && bigram_lut[(static_cast<uint32_t>(b0) << 7) | b1] != 0u) {
        clen += 1;
        i += 2;
      } else {
        clen += 1;
        i += 1;
      }
    } else {
      clen += 1;
      i += 1;
    }
  }
  out_comp_lens[r] = clen;
  if (r == 0u) { out_comp_lens[nrows] = 0; }
}

__global__ void scatter_bigram_string_kernel(int32_t const* __restrict__ in_offs,
                                             char const* __restrict__ in_chars,
                                             int32_t const* __restrict__ comp_offs,
                                             uint32_t src_col_off,
                                             uint8_t const* __restrict__ bigram_lut,
                                             uint16_t const* __restrict__ sym_table,
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
  uint32_t const p_start = part_row_start[p];
  uint8_t* dst           = d_out + dst_payload_offs[p];
  uint32_t const tid     = blockIdx.y * blockDim.x + threadIdx.x;
  uint32_t const stride  = gridDim.y * blockDim.x;

  // Write 256-byte symbol table at start of payload (only blockIdx.y == 0)
  if (blockIdx.y == 0u) {
    for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) {
      uint16_t const v = sym_table[i];
      dst[i * 2u]      = static_cast<uint8_t>(v & 0xFFu);
      dst[i * 2u + 1u] = static_cast<uint8_t>((v >> 8) & 0xFFu);
    }
  }
  uint8_t* offs_dst = dst + 256u;
  int32_t const cb0 = comp_offs[p_start];
  for (uint32_t i = tid; i <= len; i += stride) {
    st_u32_u(offs_dst + static_cast<size_t>(i) * 4u, static_cast<uint32_t>(comp_offs[p_start + i] - cb0));
  }
  uint8_t* ch_dst = offs_dst + static_cast<size_t>(len + 1u) * sizeof(int32_t);

  // Compress every string row in parallel across all SMs directly into ch_dst!
  for (uint32_t r = tid; r < len; r += stride) {
    uint32_t const gr = p_start + r;
    int32_t const s   = in_offs[src_col_off + gr];
    int32_t const e   = in_offs[src_col_off + gr + 1u];
    uint8_t* wp       = ch_dst + (comp_offs[gr] - cb0);
    int32_t i         = s;
    while (i < e) {
      uint8_t const b0 = static_cast<uint8_t>(in_chars[i]);
      if (b0 >= 0x80u) {
        *wp++ = 0xFFu;
        *wp++ = b0;
        i += 1;
      } else if (i + 1 < e) {
        uint8_t const b1   = static_cast<uint8_t>(in_chars[i + 1]);
        uint8_t const code = (b1 < 0x80u) ? bigram_lut[(static_cast<uint32_t>(b0) << 7) | b1] : 0u;
        if (code != 0u) {
          *wp++ = code;
          i += 2;
        } else {
          *wp++ = b0;
          i += 1;
        }
      } else {
        *wp++ = b0;
        i += 1;
      }
    }
  }
}

// ---------------- P1-B: 255-Symbol Multi-Length (1..4B) GPU FSST String Codec ----------------

__device__ __forceinline__ uint8_t fsst_find_match(
  uint8_t const* __restrict__ p,
  int32_t rem,
  uint32_t const* __restrict__ s_word,
  uint8_t const* __restrict__ s_len,
  uint8_t const* __restrict__ s_single,
  uint8_t const* __restrict__ s_head,
  uint8_t const* __restrict__ s_next,
  int32_t& out_adv)
{
  if (rem >= 2) {
    uint32_t const h = fsst_hash2(p[0], p[1]);
    uint8_t c        = s_head[h];
    // Bucket chains are sorted longest-first (4 -> 3 -> 2), so first match is greedy longest!
    while (c != 0xFFu) {
      uint8_t const sl = s_len[c];
      if (rem >= static_cast<int32_t>(sl)) {
        uint32_t const w = s_word[c];
        bool ok          = (p[0] == static_cast<uint8_t>(w & 0xFFu)) &&
                  (p[1] == static_cast<uint8_t>((w >> 8) & 0xFFu));
        if (ok && sl >= 3u) { ok = (p[2] == static_cast<uint8_t>((w >> 16) & 0xFFu)); }
        if (ok && sl >= 4u) { ok = (p[3] == static_cast<uint8_t>((w >> 24) & 0xFFu)); }
        if (ok) {
          out_adv = static_cast<int32_t>(sl);
          return c;
        }
      }
      c = s_next[c];
    }
  }
  out_adv = 1;
  return s_single[p[0]];
}

__global__ void fsst_measure_rows_kernel(int32_t const* __restrict__ in_offs,
                                         char const* __restrict__ in_chars,
                                         uint32_t src_col_off,
                                         uint32_t nrows,
                                         FsstEncodeTable const* __restrict__ tbl,
                                         int32_t* __restrict__ out_lens)
{
  __shared__ FsstEncodeTable st;
  auto const* src_u32 = reinterpret_cast<uint32_t const*>(tbl);
  auto* dst_u32       = reinterpret_cast<uint32_t*>(&st);
  constexpr uint32_t n_u32 = sizeof(FsstEncodeTable) / sizeof(uint32_t);
  for (uint32_t i = threadIdx.x; i < n_u32; i += blockDim.x) { dst_u32[i] = src_u32[i]; }
  __syncthreads();

  uint32_t const r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= nrows) {
    if (r == nrows) { out_lens[nrows] = 0; }
    return;
  }
  int32_t const s = in_offs[src_col_off + r];
  int32_t const e = in_offs[src_col_off + r + 1u];
  int32_t clen    = 0;
  int32_t i       = s;
  while (i < e) {
    int32_t adv = 1;
    uint8_t const c = fsst_find_match(
      reinterpret_cast<uint8_t const*>(in_chars + i),
      e - i,
      st.sym_word,
      st.sym_len,
      st.single_code,
      st.bucket_head,
      st.sym_next,
      adv);
    clen += (c == 0xFFu) ? 2 : 1;
    i += adv;
  }
  out_lens[r] = clen;
}

__global__ void scatter_fsst_string_kernel(int32_t const* __restrict__ in_offs,
                                           char const* __restrict__ in_chars,
                                           int32_t const* __restrict__ comp_offs,
                                           uint32_t src_col_off,
                                           FsstEncodeTable const* __restrict__ tbl,
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

  __shared__ FsstEncodeTable st;
  auto const* src_u32 = reinterpret_cast<uint32_t const*>(tbl);
  auto* dst_u32       = reinterpret_cast<uint32_t*>(&st);
  constexpr uint32_t n_u32 = sizeof(FsstEncodeTable) / sizeof(uint32_t);
  for (uint32_t i = threadIdx.x; i < n_u32; i += blockDim.x) { dst_u32[i] = src_u32[i]; }
  __syncthreads();

  uint8_t* dst = d_out + dst_payload_offs[p];
  // First tile writes the 1,280-byte FSST header (sym_word[255] + sym_len[255])
  if (blockIdx.y == 0u) {
    auto const* hdr_src = reinterpret_cast<uint8_t const*>(&st);
    for (uint32_t i = threadIdx.x; i < kFsstHdrBytes; i += blockDim.x) { dst[i] = hdr_src[i]; }
  }
  uint32_t const ps      = part_row_start[p];
  int32_t const cb0      = comp_offs[ps];
  uint8_t* off_dst       = dst + kFsstHdrBytes;
  uint8_t* ch_dst        = off_dst + static_cast<size_t>(len + 1u) * sizeof(int32_t);
  uint32_t const tid     = blockIdx.y * blockDim.x + threadIdx.x;
  uint32_t const stride  = gridDim.y * blockDim.x;
  for (uint32_t i = tid; i <= len; i += stride) {
    st_u32_u(off_dst + static_cast<size_t>(i) * 4u, static_cast<uint32_t>(comp_offs[ps + i] - cb0));
  }
  for (uint32_t r = tid; r < len; r += stride) {
    uint32_t const gr = ps + r;
    int32_t const s   = in_offs[src_col_off + gr];
    int32_t const e   = in_offs[src_col_off + gr + 1u];
    uint8_t* wp       = ch_dst + (comp_offs[gr] - cb0);
    int32_t i         = s;
    while (i < e) {
      int32_t adv = 1;
      uint8_t const c = fsst_find_match(
        reinterpret_cast<uint8_t const*>(in_chars + i),
        e - i,
        st.sym_word,
        st.sym_len,
        st.single_code,
        st.bucket_head,
        st.sym_next,
        adv);
      if (c == 0xFFu) {
        *wp++ = 0xFFu;
        *wp++ = static_cast<uint8_t>(in_chars[i]);
      } else {
        *wp++ = c;
      }
      i += adv;
    }
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
    size_t const n4       = nb >> 2;
    auto const* s4        = reinterpret_cast<uint32_t const*>(s);
    auto* d4              = reinterpret_cast<uint32_t*>(d);
    uint32_t const tid    = blockIdx.y * blockDim.x + threadIdx.x;
    uint32_t const stride = gridDim.y * blockDim.x;
    for (size_t i = tid; i < n4; i += stride) { d4[i] = s4[i]; }
    for (size_t i = (n4 << 2) + tid; i < nb; i += stride) { d[i] = s[i]; }
  } else {
    uint32_t const tid    = blockIdx.y * blockDim.x + threadIdx.x;
    uint32_t const stride = gridDim.y * blockDim.x;
    for (size_t i = tid; i < nb; i += stride) { d[i] = s[i]; }
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

  uint8_t* dst          = d_out + dst_payload_offs[p];
  auto* d_chars         = reinterpret_cast<char*>(dst + (len + 1u) * sizeof(int32_t));
  uint32_t const tid    = blockIdx.y * blockDim.x + threadIdx.x;
  uint32_t const stride = gridDim.y * blockDim.x;
  for (uint32_t i = tid; i <= len; i += stride) {
    st_u32_u(dst + static_cast<size_t>(i) * 4u, static_cast<uint32_t>(in_offs[r0 + i] - base));
  }
  for (uint32_t i = tid; i < nch; i += stride) { d_chars[i] = in_chars[base + i]; }
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
  uint8_t const* d_pay  = d_data + tk.payload_off;
  OutT* dst             = out_col + tk.out_row_off;
  uint32_t const tid    = blockIdx.y * blockDim.x + threadIdx.x;
  uint32_t const stride = gridDim.y * blockDim.x;

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
    for (uint32_t idx = tid; idx < n; idx += stride) {
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
  } else if (tk.encoding == static_cast<uint8_t>(ShufEnc::FLOAT_ALP)) {
    if constexpr (std::is_floating_point_v<OutT>) {
      uint32_t const nc     = (n + kShufBlkSize - 1u) / kShufBlkSize;
      size_t const off_min  = 0;
      size_t const off_bw   = off_min + nc * sizeof(int64_t);
      size_t const off_cwo  = align8(off_bw + nc * sizeof(uint8_t));
      size_t const off_pack = align8(off_cwo + nc * sizeof(uint32_t));
      auto const* mins      = reinterpret_cast<int64_t const*>(d_pay + off_min);
      auto const* bws       = d_pay + off_bw;
      auto const* cwo       = reinterpret_cast<uint32_t const*>(d_pay + off_cwo);
      auto const* words     = reinterpret_cast<uint64_t const*>(d_pay + off_pack);
      double const f        = alp_factor_for_exp(tk.extra);
      for (uint32_t idx = tid; idx < n; idx += stride) {
        uint32_t const c   = idx >> 8;
        uint32_t const bw  = bws[c];
        int64_t const base = mins[c];
        int64_t q          = base;
        if (bw != 0u) {
          uint32_t const local   = idx & 255u;
          uint32_t const bit_pos = local * bw;
          uint32_t const widx    = cwo[c] + (bit_pos >> 6);
          uint32_t const bit_off = bit_pos & 63u;
          uint64_t val           = words[widx] >> bit_off;
          if (bit_off + bw > 64u) { val |= words[widx + 1u] << (64u - bit_off); }
          uint64_t const mask = (bw == 64u) ? ~0ULL : ((1ULL << bw) - 1ULL);
          q                   = base + static_cast<int64_t>(val & mask);
        }
        dst[idx] = static_cast<OutT>(static_cast<double>(q) / f);
      }
    }
  } else {
    auto const* src = reinterpret_cast<OutT const*>(d_pay);
    for (uint32_t idx = tid; idx < n; idx += stride) { dst[idx] = src[idx]; }
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
  int32_t* dst          = out_lens + tk.out_row_off;
  uint32_t const tid    = blockIdx.y * blockDim.x + threadIdx.x;
  uint32_t const stride = gridDim.y * blockDim.x;

  if (tk.encoding == static_cast<uint8_t>(ShufEnc::DICT_BITPACK)) {
    auto const* dh          = reinterpret_cast<DictPayloadHeader const*>(d_pay);
    auto const* d_doffs     = reinterpret_cast<int32_t const*>(d_pay + sizeof(DictPayloadHeader));
    auto const* d_codes_pay = d_pay + dh->codes_offset;
    for (uint32_t i = tid; i < n; i += stride) {
      if (sm != nullptr && !cudf::bit_is_set(sm, static_cast<cudf::size_type>(i))) {
        dst[i] = 0;
      } else {
        int32_t const cid = decode_for256_i32_at(d_codes_pay, n, i);
        dst[i]            = d_doffs[cid + 1] - d_doffs[cid];
      }
    }
  } else if (tk.encoding == static_cast<uint8_t>(ShufEnc::STRING_BIGRAM)) {
    auto const* comp_offs = reinterpret_cast<int32_t const*>(d_pay + 256u);
    uint8_t const* ch_ptr = d_pay + 256u + static_cast<size_t>(n + 1u) * sizeof(int32_t);
    for (uint32_t i = tid; i < n; i += stride) {
      if (sm != nullptr && !cudf::bit_is_set(sm, static_cast<cudf::size_type>(i))) {
        dst[i] = 0;
      } else {
        int32_t const cs = comp_offs[i];
        int32_t const ce = comp_offs[i + 1u];
        int32_t ulen     = 0;
        int32_t k        = cs;
        while (k < ce) {
          uint8_t const b = ch_ptr[k];
          if (b < 0x80u) {
            ulen += 1;
            k += 1;
          } else if (b == 0xFFu) {
            ulen += 1;
            k += 2;
          } else {
            ulen += 2;
            k += 1;
          }
        }
        dst[i] = ulen;
      }
    }
  } else if (tk.encoding == static_cast<uint8_t>(ShufEnc::STRING_FSST)) {
    __shared__ uint8_t s_len[255];
    for (uint32_t i = threadIdx.x; i < 255u; i += blockDim.x) {
      s_len[i] = d_pay[255u * 4u + i];
    }
    __syncthreads();
    auto const* comp_offs = reinterpret_cast<int32_t const*>(d_pay + kFsstHdrBytes);
    uint8_t const* ch_ptr = d_pay + kFsstHdrBytes + static_cast<size_t>(n + 1u) * sizeof(int32_t);
    for (uint32_t i = tid; i < n; i += stride) {
      if (sm != nullptr && !cudf::bit_is_set(sm, static_cast<cudf::size_type>(i))) {
        dst[i] = 0;
      } else {
        int32_t const cs = comp_offs[i];
        int32_t const ce = comp_offs[i + 1u];
        int32_t ulen     = 0;
        int32_t k        = cs;
        while (k < ce) {
          uint8_t const c = ch_ptr[k];
          if (c == 0xFFu) {
            ulen += 1;
            k += 2;
          } else {
            ulen += static_cast<int32_t>(s_len[c]);
            k += 1;
          }
        }
        dst[i] = ulen;
      }
    }
  } else {
    auto const* s_offs = reinterpret_cast<int32_t const*>(d_pay);
    for (uint32_t i = tid; i < n; i += stride) {
      if (sm != nullptr && !cudf::bit_is_set(sm, static_cast<cudf::size_type>(i))) {
        dst[i] = 0;
      } else {
        dst[i] = s_offs[i + 1] - s_offs[i];
      }
    }
  }
  if (t == 0u && tid == 0u) { out_lens[total_rows] = 0; }
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
  uint32_t const tid    = blockIdx.y * blockDim.x + threadIdx.x;
  uint32_t const stride = gridDim.y * blockDim.x;

  if (tk.encoding == static_cast<uint8_t>(ShufEnc::DICT_BITPACK)) {
    auto const* dh      = reinterpret_cast<DictPayloadHeader const*>(d_pay);
    auto const* d_doffs = reinterpret_cast<int32_t const*>(d_pay + sizeof(DictPayloadHeader));
    auto const* d_dch   = reinterpret_cast<char const*>(
      d_pay + sizeof(DictPayloadHeader) + (dh->dict_size + 1u) * sizeof(int32_t));
    auto const* d_codes_pay = d_pay + dh->codes_offset;
    for (uint32_t i = tid; i < n; i += stride) {
      if (sm != nullptr && !cudf::bit_is_set(sm, static_cast<cudf::size_type>(i))) { continue; }
      uint32_t const grow = tk.out_row_off + i;
      int32_t const dst   = out_offs[grow];
      int32_t const len   = out_offs[grow + 1u] - dst;
      if (len <= 0) { continue; }
      int32_t const cid = decode_for256_i32_at(d_codes_pay, n, i);
      int32_t const src = d_doffs[cid];
      for (int32_t k = 0; k < len; ++k) { out_chars[dst + k] = d_dch[src + k]; }
    }
  } else if (tk.encoding == static_cast<uint8_t>(ShufEnc::STRING_BIGRAM)) {
    __shared__ uint16_t s_sym[128];
    for (uint32_t i = threadIdx.x; i < 128u; i += blockDim.x) {
      s_sym[i] = static_cast<uint16_t>(d_pay[i * 2u]) |
                 (static_cast<uint16_t>(d_pay[i * 2u + 1u]) << 8);
    }
    __syncthreads();
    auto const* comp_offs = reinterpret_cast<int32_t const*>(d_pay + 256u);
    uint8_t const* ch_ptr = d_pay + 256u + static_cast<size_t>(n + 1u) * sizeof(int32_t);
    for (uint32_t i = tid; i < n; i += stride) {
      if (sm != nullptr && !cudf::bit_is_set(sm, static_cast<cudf::size_type>(i))) { continue; }
      uint32_t const grow = tk.out_row_off + i;
      int32_t const cs    = comp_offs[i];
      int32_t const ce    = comp_offs[i + 1u];
      char* wp            = out_chars + out_offs[grow];
      int32_t k           = cs;
      while (k < ce) {
        uint8_t const b = ch_ptr[k];
        if (b < 0x80u) {
          *wp++ = static_cast<char>(b);
          k += 1;
        } else if (b == 0xFFu) {
          *wp++ = static_cast<char>(ch_ptr[k + 1]);
          k += 2;
        } else {
          uint16_t const pair = s_sym[b - 0x80u];
          *wp++               = static_cast<char>(pair & 0xFFu);
          *wp++               = static_cast<char>((pair >> 8) & 0xFFu);
          k += 1;
        }
      }
    }
  } else if (tk.encoding == static_cast<uint8_t>(ShufEnc::STRING_FSST)) {
    __shared__ uint32_t s_word[255];
    __shared__ uint8_t s_len[255];
    for (uint32_t i = threadIdx.x; i < 255u; i += blockDim.x) {
      size_t const o = static_cast<size_t>(i) * 4u;
      s_word[i] = static_cast<uint32_t>(d_pay[o]) |
                  (static_cast<uint32_t>(d_pay[o + 1u]) << 8) |
                  (static_cast<uint32_t>(d_pay[o + 2u]) << 16) |
                  (static_cast<uint32_t>(d_pay[o + 3u]) << 24);
      s_len[i]  = d_pay[255u * 4u + i];
    }
    __syncthreads();
    auto const* comp_offs = reinterpret_cast<int32_t const*>(d_pay + kFsstHdrBytes);
    uint8_t const* ch_ptr = d_pay + kFsstHdrBytes + static_cast<size_t>(n + 1u) * sizeof(int32_t);
    for (uint32_t i = tid; i < n; i += stride) {
      if (sm != nullptr && !cudf::bit_is_set(sm, static_cast<cudf::size_type>(i))) { continue; }
      uint32_t const grow = tk.out_row_off + i;
      int32_t const cs    = comp_offs[i];
      int32_t const ce    = comp_offs[i + 1u];
      char* wp            = out_chars + out_offs[grow];
      int32_t k           = cs;
      while (k < ce) {
        uint8_t const c = ch_ptr[k];
        if (c == 0xFFu) {
          *wp++ = static_cast<char>(ch_ptr[k + 1]);
          k += 2;
        } else {
          uint32_t const w = s_word[c];
          uint8_t const sl = s_len[c];
          *wp++            = static_cast<char>(w & 0xFFu);
          if (sl >= 2u) { *wp++ = static_cast<char>((w >> 8) & 0xFFu); }
          if (sl >= 3u) { *wp++ = static_cast<char>((w >> 16) & 0xFFu); }
          if (sl >= 4u) { *wp++ = static_cast<char>((w >> 24) & 0xFFu); }
          k += 1;
        }
      }
    }
  } else {
    auto const* s_offs  = reinterpret_cast<int32_t const*>(d_pay);
    auto const* s_chars = reinterpret_cast<char const*>(d_pay + (n + 1u) * sizeof(int32_t));
    for (uint32_t i = tid; i < n; i += stride) {
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
    // DICT_BITPACK whole-column dictionary (1 encode for entire table, or zero-decode pass-through!)
    std::unique_ptr<cudf::column> dict_col;
    std::unique_ptr<cudf::column> decoded_str_col;
    uint32_t dict_size{0};
    uint32_t key_bytes{0};
    uint32_t alp_exp{0};
    std::vector<uint8_t> h_dict_blob;  // DictPayloadHeader + dict_offs + dict_chars
    // RAW_STRING offsets on host for sizing
    int32_t h_str_base_off{0};
    std::vector<int32_t> h_str_part_chars;
    // Opt V3: Zero-sync GPU Dynamic Bigram + P1-B 255-Symbol Multi-Length GPU FSST buffers
    rmm::device_buffer d_bigram_lut;
    rmm::device_buffer d_bigram_sym;
    rmm::device_buffer d_fsst_tbl;
    rmm::device_uvector<int32_t> d_bigram_offs{0, cudf::get_default_stream()};
    std::vector<int32_t> h_bigram_part_chars;
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

  // Milestone 2: Thread-local Pinned Host Staging Pool so batched FOR256 D2H copies are
  // 100% truly async (eliminating CUDA's implicit sync on pageable std::vector D2H!).
  struct PinnedU32Pool {
    uint32_t* ptr{nullptr};
    size_t cap{0};
    ~PinnedU32Pool() {
      if (ptr != nullptr) { cudaFreeHost(ptr); }
    }
    uint32_t* ensure(size_t n) {
      if (n > cap) {
        if (ptr != nullptr) { cudaFreeHost(ptr); ptr = nullptr; cap = 0; }
        size_t alloc_n = std::max<size_t>(n, 65536u);
        if (cudaHostAlloc(reinterpret_cast<void**>(&ptr), alloc_n * sizeof(uint32_t), cudaHostAllocDefault) == cudaSuccess) {
          cap = alloc_n;
        } else {
          ptr = nullptr;
          cap = 0;
        }
      }
      return ptr;
    }
  };
  thread_local PinnedU32Pool tl_pinned_word_offs;
  uint32_t const stride_wo = total_blks + 1u;
  uint32_t* pinned_base    = tl_pinned_word_offs.ensure(static_cast<size_t>(ncols) * stride_wo);

  auto launch_for_stats = [&]<typename T>(ColWork& w, T const* d_in, cudf::column_view const& cv, uint32_t col_slot) {
    cudf::bitmask_type const* nmask = w.has_nulls ? cv.null_mask() : nullptr;
    cudf::size_type const coff      = cv.offset();
    w.d_mins.resize(total_blks, stream);
    w.d_bws.resize(total_blks, stream);
    w.d_word_offs.resize(stride_wo, stream);
    if (total_blks > 0) {
      batched_for256_stats_kernel<T><<<total_blks, 256, 0, stream.get()>>>(
        d_in,
        nmask,
        coff,
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
    uint32_t* h_dst = nullptr;
    if (pinned_base != nullptr) {
      h_dst = pinned_base + static_cast<size_t>(col_slot) * stride_wo;
    } else {
      w.h_word_offs.resize(stride_wo);
      h_dst = w.h_word_offs.data();
    }
    CUDF_CUDA_TRY(cudaMemcpyAsync(h_dst,
                                  w.d_word_offs.data(),
                                  stride_wo * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost,
                                  stream.get()));
  };

  auto sync_copy_pinned_wo = [&](ColWork& w, uint32_t col_slot) {
    if (pinned_base != nullptr) {
      uint32_t const* src = pinned_base + static_cast<size_t>(col_slot) * stride_wo;
      w.h_word_offs.assign(src, src + stride_wo);
    }
  };

  auto launch_for_pack = [&]<typename T>(ColWork& w, T const* d_in, cudf::column_view const& cv) {
    cudf::bitmask_type const* nmask = w.has_nulls ? cv.null_mask() : nullptr;
    cudf::size_type const coff      = cv.offset();
    uint32_t const tot_words        = w.h_word_offs[total_blks];
    w.d_words.resize(std::max<uint32_t>(tot_words, 1), stream);
    if (total_blks > 0 && tot_words > 0) {
      batched_for256_pack_kernel<T><<<total_blks, 256, 0, stream.get()>>>(
        d_in,
        nmask,
        coff,
        d_blk_row_start.data(),
        d_blk_row_len.data(),
        w.d_mins.data(),
        w.d_bws.data(),
        w.d_word_offs.data(),
        w.d_words.data());
    }
  };

  auto run_batched_for = [&]<typename T>(ColWork& w, T const* d_in, cudf::column_view const& cv) {
    launch_for_stats.template operator()<T>(w, d_in, cv, 0u);
    stream.sync();
    sync_copy_pinned_wo(w, 0u);
    launch_for_pack.template operator()<T>(w, d_in, cv);
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
  }

  std::vector<uint32_t> batched_for_cols;
  if (nrows > 0) {
    for (uint32_t c = 0; c < ncols; ++c) {
      ColWork& w    = works[c];
      auto const cv = table.column(static_cast<cudf::size_type>(c));
      if (w.tid == cudf::type_id::INT64 || w.tid == cudf::type_id::TIMESTAMP_MICROSECONDS ||
          w.tid == cudf::type_id::DECIMAL64) {
        w.enc       = ShufEnc::FOR_BITPACK;
        w.elem_size = 8;
        launch_for_stats.template operator()<int64_t>(
          w, static_cast<int64_t const*>(cv.head()) + cv.offset(), cv, c);
        batched_for_cols.push_back(c);
      } else if (w.tid == cudf::type_id::INT32 || w.tid == cudf::type_id::TIMESTAMP_DAYS ||
                 w.tid == cudf::type_id::DECIMAL32) {
        w.enc       = ShufEnc::FOR_BITPACK;
        w.elem_size = 4;
        launch_for_stats.template operator()<int32_t>(
          w, static_cast<int32_t const*>(cv.head()) + cv.offset(), cv, c);
        batched_for_cols.push_back(c);
      } else if (w.tid == cudf::type_id::INT16) {
        w.enc       = ShufEnc::FOR_BITPACK;
        w.elem_size = 2;
        launch_for_stats.template operator()<int16_t>(w, cv.data<int16_t>() + cv.offset(), cv, c);
        batched_for_cols.push_back(c);
      } else if (w.tid == cudf::type_id::INT8 || w.tid == cudf::type_id::BOOL8) {
        w.enc       = ShufEnc::FOR_BITPACK;
        w.elem_size = 1;
        launch_for_stats.template operator()<int8_t>(
          w, static_cast<int8_t const*>(cv.head()) + cv.offset(), cv, c);
        batched_for_cols.push_back(c);
      }
    }
    if (!batched_for_cols.empty()) {
      stream.sync();
      for (uint32_t c : batched_for_cols) {
        ColWork& w    = works[c];
        sync_copy_pinned_wo(w, c);
        auto const cv = table.column(static_cast<cudf::size_type>(c));
        if (w.elem_size == 8) {
          launch_for_pack.template operator()<int64_t>(
            w, static_cast<int64_t const*>(cv.head()) + cv.offset(), cv);
        } else if (w.elem_size == 4) {
          launch_for_pack.template operator()<int32_t>(
            w, static_cast<int32_t const*>(cv.head()) + cv.offset(), cv);
        } else if (w.elem_size == 2) {
          launch_for_pack.template operator()<int16_t>(w, cv.data<int16_t>() + cv.offset(), cv);
        } else {
          launch_for_pack.template operator()<int8_t>(
            w, static_cast<int8_t const*>(cv.head()) + cv.offset(), cv);
        }
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
      }
    }
  }

  for (uint32_t c = 0; c < ncols; ++c) {
    ColWork& w    = works[c];
    auto const cv = table.column(static_cast<cudf::size_type>(c));
    if (nrows == 0 || w.enc == ShufEnc::FOR_BITPACK) { continue; }

    if (w.tid == cudf::type_id::FLOAT32 || w.tid == cudf::type_id::FLOAT64) {
      w.elem_size = static_cast<uint32_t>(cudf::size_of(cv.type()));
      w.enc       = ShufEnc::RAW_FIXED;
      size_t const tot_raw_b = static_cast<size_t>(nrows) * w.elem_size;
      for (size_t p = 0; p < n_parts; ++p) {
        w.part_payload_sz[p] = static_cast<size_t>(h_part_len[p]) * w.elem_size;
      }
      // Item 4: Exact bit-for-bit lossless GPU ALP-Lite decimal scaling -> FOR256 for dense floats!
      // Compresses TPC-H price/discount/tax/quantity columns 1.4x-8x with ZERO rounding error;
      // nullable/sparse float columns (e.g. NF_J0048) still go to batched nvCOMP LZ4 below!
      if (nrows >= 1024u && !w.has_nulls) {
        auto try_float_alp = [&]<typename FT>(FT const* d_in) {
          uint32_t h_ok[5] = {1u, 1u, 1u, 1u, 1u};
          rmm::device_uvector<uint32_t> d_ok(5, stream, mr);
          CUDF_CUDA_TRY(cudaMemcpyAsync(
            d_ok.data(), h_ok, sizeof(h_ok), cudaMemcpyHostToDevice, stream.get()));
          cudf::bitmask_type const* nmask = w.has_nulls ? cv.null_mask() : nullptr;
          float_alp_probe_kernel<FT><<<1, 256, 0, stream.get()>>>(
            d_in, nmask, static_cast<uint32_t>(cv.offset()), nrows, d_ok.data());
          CUDF_CUDA_TRY(cudaMemcpyAsync(
            h_ok, d_ok.data(), sizeof(h_ok), cudaMemcpyDeviceToHost, stream.get()));
          stream.sync();
          int best_exp = -1;
          for (int e = 0; e < 5; ++e) {
            if (h_ok[e] != 0u) {
              best_exp = e;
              break;
            }
          }
          if (best_exp < 0) { return; }
          static constexpr double kHostFactors[5] = {1.0, 10.0, 100.0, 1000.0, 10000.0};
          rmm::device_uvector<int64_t> d_alp_q(nrows, stream, mr);
          uint32_t h_mis = 0u;
          rmm::device_uvector<uint32_t> d_mis(1, stream, mr);
          CUDF_CUDA_TRY(cudaMemsetAsync(d_mis.data(), 0, sizeof(uint32_t), stream.get()));
          uint32_t const g_alp = std::min<uint32_t>((nrows + 255u) / 256u, 4096u);
          float_alp_encode_kernel<FT><<<g_alp, 256, 0, stream.get()>>>(
            d_in, nmask, static_cast<uint32_t>(cv.offset()), nrows, kHostFactors[best_exp], d_alp_q.data(), d_mis.data());
          CUDF_CUDA_TRY(cudaMemcpyAsync(
            &h_mis, d_mis.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.get()));
          stream.sync();
          if (h_mis != 0u) { return; }

          run_batched_for.template operator()<int64_t>(w, d_alp_q.data(), cv);
          size_t tot_alp_b = 0;
          std::vector<uint64_t> cand_sz(n_parts, 0);
          for (size_t p = 0; p < n_parts; ++p) {
            uint32_t const nc = h_part_nblks[p];
            if (nc == 0) continue;
            uint32_t const b0 = h_part_blk_start[p];
            uint32_t const nw = w.h_word_offs[b0 + nc] - w.h_word_offs[b0];
            size_t const o_bw = nc * sizeof(int64_t);
            size_t const o_cw = align8(o_bw + nc * sizeof(uint8_t));
            size_t const o_pk = align8(o_cw + nc * sizeof(uint32_t));
            cand_sz[p]        = o_pk + static_cast<size_t>(nw) * sizeof(uint64_t);
            tot_alp_b += cand_sz[p];
          }
          if (tot_alp_b * 100u < tot_raw_b * 85u) {
            w.enc             = ShufEnc::FLOAT_ALP;
            w.alp_exp         = static_cast<uint32_t>(best_exp);
            w.part_payload_sz = std::move(cand_sz);
          }
        };
        if (w.tid == cudf::type_id::FLOAT32) {
          try_float_alp(static_cast<float const*>(cv.head()) + cv.offset());
        } else {
          try_float_alp(static_cast<double const*>(cv.head()) + cv.offset());
        }
      }
    } else if (w.tid == cudf::type_id::STRING || w.tid == cudf::type_id::DICTIONARY32) {
      bool const was_dict32 = (w.tid == cudf::type_id::DICTIONARY32);
      // Reduce side always expects logical STRING column
      w.tid = cudf::type_id::STRING;
      bool used_dict = false;
      auto pack_dict_view = [&](cudf::dictionary_column_view const& dcv,
                                std::unique_ptr<cudf::column> owned_dc) {
        uint32_t const ds = static_cast<uint32_t>(dcv.keys_size());
        if (ds > 0 && ds <= 1024 && dcv.keys().type().id() == cudf::type_id::STRING &&
            dcv.indices().type().id() == cudf::type_id::INT32) {
          cudf::strings_column_view const ksv(dcv.keys());
          int32_t kb = 0;
          CUDF_CUDA_TRY(cudaMemcpyAsync(
            &kb, ksv.offsets().data<int32_t>() + ds, sizeof(int32_t), cudaMemcpyDeviceToHost, stream.get()));
          stream.sync();
          if (kb <= 20480) {
            used_dict         = true;
            w.enc             = ShufEnc::DICT_BITPACK;
            w.dict_size       = ds;
            w.key_bytes       = static_cast<uint32_t>(kb);
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
            w.dict_col = std::move(owned_dc);
            run_batched_for.template operator()<int32_t>(
              w, dcv.indices().data<int32_t>() + dcv.offset(), cv);
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
      };

      if (was_dict32) {
        // Item 2: Zero-decode pass-through directly from Parquet/Vortex DICTIONARY32 -> DICT_BITPACK!
        pack_dict_view(cudf::dictionary_column_view(cv), nullptr);
      } else if (nrows >= 256) {
        // Fast 2048-row cardinality probe: skips wasted full-column GPU sort/hash on
        // high-cardinality strings (c_name/c_address/c_phone/c_comment/l_comment)!
        bool try_full_dict = true;
        if (nrows > 4096u) {
          auto samp = cudf::slice(cv, {0, 2048}, stream)[0];
          auto sdc  = cudf::dictionary::encode(samp, cudf::data_type{cudf::type_id::INT32}, stream, mr);
          uint32_t const sds =
            static_cast<uint32_t>(cudf::dictionary_column_view(sdc->view()).keys_size());
          if (sds > 512u) { try_full_dict = false; }
        }
        if (try_full_dict) {
          auto dc = cudf::dictionary::encode(cv, cudf::data_type{cudf::type_id::INT32}, stream, mr);
          cudf::dictionary_column_view const dcv(dc->view());
          pack_dict_view(dcv, std::move(dc));
        }
      }
      if (!used_dict) {
        cudf::column_view str_cv = cv;
        if (was_dict32) {
          w.decoded_str_col = cudf::dictionary::decode(cudf::dictionary_column_view(cv), stream, mr);
          str_cv            = w.decoded_str_col->view();
        }
        w.enc = ShufEnc::RAW_STRING;
        cudf::strings_column_view const scv(str_cv);
        // Gather ONLY n_parts boundary char counts on GPU (800 bytes D2H instead of 80 MiB full offsets!)
        w.h_str_base_off = 0;
        w.h_str_part_chars.assign(n_parts, 0);
        rmm::device_uvector<int32_t> d_part_chars(n_parts, stream, mr);
        uint32_t const gpb = (static_cast<uint32_t>(n_parts) + 255u) / 256u;
        gather_str_boundary_offs_kernel<<<gpb, 256, 0, stream.get()>>>(
          scv.offsets().data<int32_t>(),
          static_cast<uint32_t>(str_cv.offset()),
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
        size_t tot_raw_ch = 0;
        for (size_t p = 0; p < n_parts; ++p) {
          if (h_part_len[p] == 0) continue;
          int32_t const nch    = w.h_str_part_chars[p];
          w.part_payload_sz[p] = (h_part_len[p] + 1u) * sizeof(int32_t) + static_cast<size_t>(nch);
          tot_raw_ch += static_cast<size_t>(nch);
        }

        // Opt V3: Zero-sync GPU Dynamic 127-Bigram (mini-FSST) compression for high-card ASCII strings!
        // P1-B: Try 255-Symbol Multi-Length (1..4B) GPU FSST first for large high-card string cols (>=512 KiB)!
        // Achieves 2.2x-3.0x compression on arbitrary UTF-8 (vs Bigram 1.5x ASCII-only), with 1.25 KiB table.
        if (tot_raw_ch >= 262144u && nrows >= 1024u) {
          int32_t base_ch = 0;
          CUDF_CUDA_TRY(cudaMemcpyAsync(
            &base_ch, scv.offsets().data<int32_t>() + str_cv.offset(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream.get()));
          stream.sync();
          size_t const samp_len = std::min<size_t>(tot_raw_ch, 65536u);
          std::vector<uint8_t> h_samp(samp_len);
          CUDF_CUDA_TRY(cudaMemcpyAsync(
            h_samp.data(), scv.chars_begin(stream) + base_ch, samp_len, cudaMemcpyDeviceToHost, stream.get()));
          stream.sync();

          bool used_fsst = false;
          static bool const fsst_opt_in = []() {
            char const* e = std::getenv("LENC_SHUF_FSST");
            return e != nullptr && e[0] == '1';
          }();
          if (fsst_opt_in && tot_raw_ch >= 524288u) {
            // Fast 64 KiB CPU FSST symbol trainer (<0.25 ms): score 4B, 3B, 2B n-grams by freq*(len-1)!
            std::unordered_map<uint64_t, uint32_t> ng_cnt;
            ng_cnt.reserve(8192);
            uint32_t b1_cnt[256] = {};
            for (size_t i = 0; i < samp_len; ++i) {
              uint8_t const b0 = h_samp[i];
              b1_cnt[b0]++;
              if (i + 2u <= samp_len) {
                uint64_t const k2 = (2ULL << 32) | static_cast<uint64_t>(b0) |
                                    (static_cast<uint64_t>(h_samp[i + 1u]) << 8);
                ng_cnt[k2]++;
              }
              if (i + 3u <= samp_len) {
                uint64_t const k3 = (3ULL << 32) | static_cast<uint64_t>(b0) |
                                    (static_cast<uint64_t>(h_samp[i + 1u]) << 8) |
                                    (static_cast<uint64_t>(h_samp[i + 2u]) << 16);
                ng_cnt[k3]++;
              }
              if (i + 4u <= samp_len) {
                uint64_t const k4 = (4ULL << 32) | static_cast<uint64_t>(b0) |
                                    (static_cast<uint64_t>(h_samp[i + 1u]) << 8) |
                                    (static_cast<uint64_t>(h_samp[i + 2u]) << 16) |
                                    (static_cast<uint64_t>(h_samp[i + 3u]) << 24);
                ng_cnt[k4]++;
              }
            }
            struct Cand { uint32_t score; uint32_t word; uint8_t len; };
            std::vector<Cand> cands;
            cands.reserve(ng_cnt.size());
            for (auto const& kv : ng_cnt) {
              uint8_t const l  = static_cast<uint8_t>(kv.first >> 32);
              uint32_t const c = kv.second;
              if (c >= 6u) {
                cands.push_back(Cand{c * static_cast<uint32_t>(l - 1u), static_cast<uint32_t>(kv.first), l});
              }
            }
            size_t const max_multi = std::min<size_t>(cands.size(), 150u);
            if (max_multi >= 24u) {
              std::partial_sort(cands.begin(), cands.begin() + max_multi, cands.end(),
                                [](Cand const& a, Cand const& b) { return a.score > b.score; });
              // Sort selected multi-byte symbols longest-first (4 -> 3 -> 2) for greedy longest match!
              std::vector<Cand> chosen(cands.begin(), cands.begin() + max_multi);
              std::sort(chosen.begin(), chosen.end(), [](Cand const& a, Cand const& b) {
                return (a.len != b.len) ? (a.len > b.len) : (a.score > b.score);
              });

              FsstEncodeTable h_ft{};
              std::memset(&h_ft, 0, sizeof(h_ft));
              std::memset(h_ft.single_code, 0xFF, sizeof(h_ft.single_code));
              std::memset(h_ft.bucket_head, 0xFF, sizeof(h_ft.bucket_head));
              std::memset(h_ft.sym_next, 0xFF, sizeof(h_ft.sym_next));
              uint8_t bucket_tail[1024];
              std::memset(bucket_tail, 0xFF, sizeof(bucket_tail));

              uint32_t next_code = 0;
              for (auto const& sc : chosen) {
                uint8_t const code = static_cast<uint8_t>(next_code++);
                h_ft.sym_word[code] = sc.word;
                h_ft.sym_len[code]  = sc.len;
                uint32_t const h    = fsst_hash2(static_cast<uint8_t>(sc.word & 0xFFu),
                                                 static_cast<uint8_t>((sc.word >> 8) & 0xFFu));
                if (h_ft.bucket_head[h] == 0xFFu) {
                  h_ft.bucket_head[h] = code;
                  bucket_tail[h]      = code;
                } else {
                  h_ft.sym_next[bucket_tail[h]] = code;
                  bucket_tail[h]                = code;
                }
              }
              // Fill remaining codes (..254) with top frequent 1-byte literals (avoiding 0xFF escapes!)
              std::vector<std::pair<uint32_t, uint8_t>> b1s;
              b1s.reserve(256);
              for (uint32_t b = 0; b < 256u; ++b) {
                if (b1_cnt[b] > 0u) { b1s.emplace_back(b1_cnt[b], static_cast<uint8_t>(b)); }
              }
              std::sort(b1s.begin(), b1s.end(), [](auto const& a, auto const& b) { return a.first > b.first; });
              for (auto const& p1 : b1s) {
                if (next_code >= 255u) { break; }
                uint8_t const code    = static_cast<uint8_t>(next_code++);
                h_ft.sym_word[code]   = static_cast<uint32_t>(p1.second);
                h_ft.sym_len[code]    = 1u;
                h_ft.single_code[p1.second] = code;
              }

              w.d_fsst_tbl    = rmm::device_buffer(&h_ft, sizeof(FsstEncodeTable), stream, mr);
              w.d_bigram_offs = rmm::device_uvector<int32_t>(nrows + 1u, stream, mr);
              uint32_t const gr = (nrows + 255u) / 256u;
              fsst_measure_rows_kernel<<<gr, 256, 0, stream.get()>>>(
                scv.offsets().data<int32_t>(),
                scv.chars_begin(stream),
                static_cast<uint32_t>(str_cv.offset()),
                nrows,
                static_cast<FsstEncodeTable const*>(w.d_fsst_tbl.data()),
                w.d_bigram_offs.data());
              thrust::exclusive_scan(rmm::exec_policy_nosync(stream),
                                     w.d_bigram_offs.data(),
                                     w.d_bigram_offs.data() + nrows + 1u,
                                     w.d_bigram_offs.data());
              w.h_bigram_part_chars.assign(n_parts, 0);
              gather_str_boundary_offs_kernel<<<gpb, 256, 0, stream.get()>>>(
                w.d_bigram_offs.data(),
                0u,
                d_part_start.data(),
                d_part_len.data(),
                d_part_chars.data(),
                static_cast<uint32_t>(n_parts));
              CUDF_CUDA_TRY(cudaMemcpyAsync(w.h_bigram_part_chars.data(),
                                            d_part_chars.data(),
                                            n_parts * sizeof(int32_t),
                                            cudaMemcpyDeviceToHost,
                                            stream.get()));
              stream.sync();
              size_t tot_comp_ch = 0;
              size_t nonempty_p  = 0;
              for (size_t p = 0; p < n_parts; ++p) {
                if (h_part_len[p] > 0) {
                  tot_comp_ch += static_cast<size_t>(w.h_bigram_part_chars[p]);
                  ++nonempty_p;
                }
              }
              if ((tot_comp_ch + nonempty_p * kFsstHdrBytes) * 100u < tot_raw_ch * 78u) {
                used_fsst = true;
                w.enc     = ShufEnc::STRING_FSST;
                for (size_t p = 0; p < n_parts; ++p) {
                  if (h_part_len[p] == 0) continue;
                  w.part_payload_sz[p] = kFsstHdrBytes + (h_part_len[p] + 1u) * sizeof(int32_t) +
                                         static_cast<size_t>(w.h_bigram_part_chars[p]);
                }
              }
            }
          }

          if (!used_fsst) {
          std::vector<uint32_t> counts(128u * 128u, 0u);
          size_t ascii_cnt = 0;
          for (size_t i = 0; i + 1u < samp_len; ++i) {
            uint8_t const b0 = h_samp[i];
            uint8_t const b1 = h_samp[i + 1u];
            if (b0 < 0x80u) {
              ++ascii_cnt;
              if (b1 < 0x80u) { ++counts[(static_cast<size_t>(b0) << 7) | b1]; }
            }
          }
          if (ascii_cnt * 10u >= samp_len * 9u) {
            std::vector<std::pair<uint32_t, uint16_t>> ranked;
            ranked.reserve(2048);
            for (uint32_t k = 0; k < 128u * 128u; ++k) {
              if (counts[k] >= 4u) { ranked.emplace_back(counts[k], static_cast<uint16_t>(k)); }
            }
            size_t const topk = std::min<size_t>(ranked.size(), 127u);
            if (topk >= 16u) {
              std::partial_sort(ranked.begin(),
                                ranked.begin() + topk,
                                ranked.end(),
                                [](auto const& a, auto const& b) { return a.first > b.first; });
              std::vector<uint8_t> h_lut(128u * 128u, 0u);
              std::vector<uint16_t> h_sym(128u, 0u);
              for (size_t idx = 0; idx < topk; ++idx) {
                uint16_t const key = ranked[idx].second;
                uint8_t const b0   = static_cast<uint8_t>(key >> 7);
                uint8_t const b1   = static_cast<uint8_t>(key & 0x7Fu);
                h_lut[key]         = static_cast<uint8_t>(0x80u + idx);
                h_sym[idx]         = static_cast<uint16_t>(b0) | (static_cast<uint16_t>(b1) << 8);
              }

              w.d_bigram_lut = rmm::device_buffer(h_lut.data(), h_lut.size(), stream, mr);
              w.d_bigram_sym = rmm::device_buffer(h_sym.data(), h_sym.size() * sizeof(uint16_t), stream, mr);
              w.d_bigram_offs = rmm::device_uvector<int32_t>(nrows + 1u, stream, mr);
              uint32_t const gr = (nrows + 255u) / 256u;
              bigram_measure_rows_kernel<<<gr, 256, 0, stream.get()>>>(
                scv.offsets().data<int32_t>(),
                scv.chars_begin(stream),
                static_cast<uint32_t>(str_cv.offset()),
                nrows,
                static_cast<uint8_t const*>(w.d_bigram_lut.data()),
                w.d_bigram_offs.data());
              thrust::exclusive_scan(rmm::exec_policy_nosync(stream),
                                     w.d_bigram_offs.data(),
                                     w.d_bigram_offs.data() + nrows + 1u,
                                     w.d_bigram_offs.data());
              w.h_bigram_part_chars.assign(n_parts, 0);
              gather_str_boundary_offs_kernel<<<gpb, 256, 0, stream.get()>>>(
                w.d_bigram_offs.data(),
                0u,
                d_part_start.data(),
                d_part_len.data(),
                d_part_chars.data(),
                static_cast<uint32_t>(n_parts));
              CUDF_CUDA_TRY(cudaMemcpyAsync(w.h_bigram_part_chars.data(),
                                            d_part_chars.data(),
                                            n_parts * sizeof(int32_t),
                                            cudaMemcpyDeviceToHost,
                                            stream.get()));
              stream.sync();
              size_t tot_comp_ch = 0;
              size_t nonempty_p  = 0;
              for (size_t p = 0; p < n_parts; ++p) {
                if (h_part_len[p] > 0) {
                  tot_comp_ch += static_cast<size_t>(w.h_bigram_part_chars[p]);
                  ++nonempty_p;
                }
              }
              // Adopt STRING_BIGRAM if total bytes (including 256B table per partition) save >= 15%!
              if ((tot_comp_ch + nonempty_p * 256u) * 100u < tot_raw_ch * 85u) {
                w.enc = ShufEnc::STRING_BIGRAM;
                for (size_t p = 0; p < n_parts; ++p) {
                  if (h_part_len[p] == 0) continue;
                  w.part_payload_sz[p] = 256u + (h_part_len[p] + 1u) * sizeof(int32_t) +
                                         static_cast<size_t>(w.h_bigram_part_chars[p]);
                }
              }
            }
          }
          }  // !used_fsst
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
      // Skip columns already compressed by FLOAT_ALP!
      if ((w.tid != cudf::type_id::FLOAT32 && w.tid != cudf::type_id::FLOAT64) ||
          w.enc == ShufEnc::FLOAT_ALP || !w.has_nulls) {
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
                  : ((works[c].enc == ShufEnc::FLOAT_ALP)
                       ? works[c].alp_exp
                       : ((works[c].enc == ShufEnc::RAW_STRING)
                            ? static_cast<uint32_t>(works[c].h_str_part_chars[p])
                            : ((works[c].enc == ShufEnc::NVCOMP_LZ4 || works[c].enc == ShufEnc::STRING_LZ4)
                                 ? static_cast<uint32_t>(works[c].h_lz4_comp_szs[p])
                                 : 0u)));
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

  // Item 5: Upload ALL columns' destination offsets in ONE batched H2D copy + scatter dict blobs on GPU!
  uint32_t const y_sc = (n_parts < 64u) ? 256u : 1u;
  dim3 const sc_grid(static_cast<uint32_t>(n_parts), y_sc);
  std::vector<uint64_t> h_all_dst_offs(static_cast<size_t>(ncols) * n_parts, 0);
  for (uint32_t c = 0; c < ncols; ++c) {
    std::memcpy(h_all_dst_offs.data() + static_cast<size_t>(c) * n_parts,
                col_dst_pay[c].data(),
                n_parts * sizeof(uint64_t));
  }
  rmm::device_uvector<uint64_t> d_all_dst_offs(h_all_dst_offs.size(), stream, mr);
  if (!h_all_dst_offs.empty()) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(d_all_dst_offs.data(),
                                  h_all_dst_offs.data(),
                                  h_all_dst_offs.size() * sizeof(uint64_t),
                                  cudaMemcpyHostToDevice,
                                  stream.get()));
  }
  rmm::device_uvector<uint64_t> d_code_dst_offs(n_parts, stream, mr);
  for (uint32_t c = 0; c < ncols; ++c) {
    ColWork& w    = works[c];
    auto const cv = w.decoded_str_col ? w.decoded_str_col->view()
                                      : table.column(static_cast<cudf::size_type>(c));
    if (nrows == 0) continue;
    uint64_t const* d_col_dst = d_all_dst_offs.data() + static_cast<size_t>(c) * n_parts;

    if (w.enc == ShufEnc::FOR_BITPACK || w.enc == ShufEnc::FLOAT_ALP) {
      scatter_for256_slices_kernel<<<sc_grid, 256, 0, stream.get()>>>(
        d_base,
        d_col_dst,
        d_part_blk_start.data(),
        d_part_nblks.data(),
        w.d_mins.data(),
        w.d_bws.data(),
        w.d_word_offs.data(),
        w.d_words.data(),
        static_cast<uint32_t>(n_parts));
    } else if (w.enc == ShufEnc::DICT_BITPACK) {
      // ONE H2D copy of tiny dict blob + ONE GPU kernel scatters it to all 200 partitions!
      size_t const dblob_sz = w.h_dict_blob.size();
      rmm::device_buffer d_blob(w.h_dict_blob.data(), dblob_sz, stream, mr);
      scatter_dict_blob_kernel<<<sc_grid, 256, 0, stream.get()>>>(
        static_cast<uint8_t const*>(d_blob.data()),
        dblob_sz,
        d_base,
        d_col_dst,
        d_part_len.data(),
        static_cast<uint32_t>(n_parts));
      std::vector<uint64_t> code_dst_offs(n_parts, 0);
      for (size_t p = 0; p < n_parts; ++p) {
        if (h_part_len[p] > 0) { code_dst_offs[p] = col_dst_pay[c][p] + dblob_sz; }
      }
      CUDF_CUDA_TRY(cudaMemcpyAsync(d_code_dst_offs.data(),
                                    code_dst_offs.data(),
                                    n_parts * sizeof(uint64_t),
                                    cudaMemcpyHostToDevice,
                                    stream.get()));
      scatter_for256_slices_kernel<<<sc_grid, 256, 0, stream.get()>>>(
        d_base,
        d_code_dst_offs.data(),
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
        d_src_offs.data(), w.h_lz4_src_offs.data(), n_parts * sizeof(uint64_t), cudaMemcpyHostToDevice, stream.get()));
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        d_comp_szs.data(), w.h_lz4_comp_szs.data(), n_parts * sizeof(uint64_t), cudaMemcpyHostToDevice, stream.get()));
      scatter_lz4_slices_kernel<<<sc_grid, 256, 0, stream.get()>>>(
        static_cast<uint8_t const*>(d_all_lz4_comp.data()),
        static_cast<uint8_t const*>(cv.head()) + static_cast<size_t>(cv.offset()) * w.elem_size,
        d_base,
        d_col_dst,
        d_src_offs.data(),
        d_comp_szs.data(),
        d_part_start.data(),
        d_part_len.data(),
        w.elem_size,
        static_cast<uint32_t>(n_parts));
    } else if (w.enc == ShufEnc::RAW_FIXED) {
      scatter_raw_fixed_kernel<<<sc_grid, 256, 0, stream.get()>>>(
        static_cast<uint8_t const*>(cv.head()) + static_cast<size_t>(cv.offset()) * w.elem_size,
        d_base,
        d_col_dst,
        d_part_start.data(),
        d_part_len.data(),
        w.elem_size,
        static_cast<uint32_t>(n_parts));
    } else if (w.enc == ShufEnc::STRING_BIGRAM) {
      cudf::strings_column_view const scv(cv);
      scatter_bigram_string_kernel<<<sc_grid, 256, 0, stream.get()>>>(
        scv.offsets().data<int32_t>(),
        scv.chars_begin(stream),
        w.d_bigram_offs.data(),
        static_cast<uint32_t>(cv.offset()),
        static_cast<uint8_t const*>(w.d_bigram_lut.data()),
        static_cast<uint16_t const*>(w.d_bigram_sym.data()),
        d_base,
        d_col_dst,
        d_part_start.data(),
        d_part_len.data(),
        static_cast<uint32_t>(n_parts));
    } else if (w.enc == ShufEnc::STRING_FSST) {
      cudf::strings_column_view const scv(cv);
      scatter_fsst_string_kernel<<<sc_grid, 256, 0, stream.get()>>>(
        scv.offsets().data<int32_t>(),
        scv.chars_begin(stream),
        w.d_bigram_offs.data(),
        static_cast<uint32_t>(cv.offset()),
        static_cast<FsstEncodeTable const*>(w.d_fsst_tbl.data()),
        d_base,
        d_col_dst,
        d_part_start.data(),
        d_part_len.data(),
        static_cast<uint32_t>(n_parts));
    } else if (w.enc == ShufEnc::STRING_LZ4) {
      cudf::strings_column_view const scv(cv);
      rmm::device_uvector<uint64_t> d_src_offs(n_parts, stream, mr);
      rmm::device_uvector<uint64_t> d_comp_szs(n_parts, stream, mr);
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
        d_col_dst,
        d_src_offs.data(),
        d_comp_szs.data(),
        d_part_start.data(),
        d_part_len.data(),
        static_cast<uint32_t>(n_parts));
    } else if (w.enc == ShufEnc::RAW_STRING) {
      cudf::strings_column_view const scv(cv);
      scatter_raw_string_kernel<<<sc_grid, 256, 0, stream.get()>>>(
        scv.offsets().data<int32_t>(),
        scv.chars_begin(stream),
        static_cast<uint32_t>(cv.offset()),
        d_base,
        d_col_dst,
        d_part_start.data(),
        d_part_len.data(),
        static_cast<uint32_t>(n_parts));
    }

    if (w.has_nulls) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(d_code_dst_offs.data(),
                                    col_dst_null[c].data(),
                                    n_parts * sizeof(uint64_t),
                                    cudaMemcpyHostToDevice,
                                    stream.get()));
      scatter_bitmask_kernel<<<static_cast<uint32_t>(n_parts), 256, 0, stream.get()>>>(
        cv.null_mask(),
        static_cast<uint32_t>(cv.offset()),
        d_base,
        d_code_dst_offs.data(),
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
      uint32_t const y_tiles = (ntk < 64u) ? 256u : 1u;
      batched_decode_str_lens_kernel<<<dim3(ntk, y_tiles), 256, 0, stream.get()>>>(
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
          batched_decode_str_copy_kernel<<<dim3(ntk, y_tiles), 256, 0, stream.get()>>>(
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
        uint32_t const y2 = (n2 < 64u) ? 256u : 1u;
        if (tid == cudf::type_id::FLOAT64) {
          batched_decode_fixed_kernel<double><<<dim3(n2, y2), 256, 0, stream.get()>>>(
            d_data, d_t2.data(), n2, static_cast<double*>(col->mutable_view().head()));
        } else if (tid == cudf::type_id::FLOAT32) {
          batched_decode_fixed_kernel<float><<<dim3(n2, y2), 256, 0, stream.get()>>>(
            d_data, d_t2.data(), n2, static_cast<float*>(col->mutable_view().head()));
        } else if (esz == 8) {
          batched_decode_fixed_kernel<int64_t><<<dim3(n2, y2), 256, 0, stream.get()>>>(
            d_data, d_t2.data(), n2, static_cast<int64_t*>(col->mutable_view().head()));
        } else if (esz == 4) {
          batched_decode_fixed_kernel<int32_t><<<dim3(n2, y2), 256, 0, stream.get()>>>(
            d_data, d_t2.data(), n2, static_cast<int32_t*>(col->mutable_view().head()));
        } else if (esz == 2) {
          batched_decode_fixed_kernel<int16_t><<<dim3(n2, y2), 256, 0, stream.get()>>>(
            d_data, d_t2.data(), n2, static_cast<int16_t*>(col->mutable_view().head()));
        } else {
          batched_decode_fixed_kernel<int8_t><<<dim3(n2, y2), 256, 0, stream.get()>>>(
            d_data, d_t2.data(), n2, static_cast<int8_t*>(col->mutable_view().head()));
        }
      }
      out_cols.push_back(std::move(col));
    }
  }

  stream.sync();
  return std::make_unique<cudf::table>(std::move(out_cols));
}

// Column-pruned GPU decode for LightweightCachedBatchSerializer:
// parses the LSHF header on CPU, repacks ONLY the selected columns' compressed bytes into a
// tiny compact frame, H2D-copies ONLY those bytes, and decodes ONLY the selected columns!
std::unique_ptr<cudf::table> decode_cached_lshf_projected(
  uint8_t const* h_lshf,
  size_t lshf_bytes,
  std::vector<int32_t> const& selected_cols,
  std::vector<cudf::data_type> const& selected_dtypes,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(lshf_bytes >= sizeof(LshfHeader), "decode_cached_lshf_projected: truncated header");
  LshfHeader lh{};
  std::memcpy(&lh, h_lshf, sizeof(lh));
  CUDF_EXPECTS(lh.magic == kLshfMagic, "decode_cached_lshf_projected: bad LSHF magic");

  auto const* src_metas = reinterpret_cast<ShufColMeta const*>(h_lshf + sizeof(lh));
  uint32_t const nsel   = static_cast<uint32_t>(selected_cols.size());
  size_t const hdr_sz   = align8(sizeof(LshfHeader) + static_cast<size_t>(nsel) * sizeof(ShufColMeta));

  // Compute exact compact size of ONLY the selected columns
  size_t cur = hdr_sz;
  std::vector<ShufColMeta> dst_metas(nsel);
  for (uint32_t i = 0; i < nsel; ++i) {
    int32_t const sc = selected_cols[i];
    CUDF_EXPECTS(sc >= 0 && static_cast<uint32_t>(sc) < lh.num_cols,
                 "decode_cached_lshf_projected: column index out of range");
    ShufColMeta const& sm = src_metas[sc];
    ShufColMeta dm        = sm;
    dm.payload_offset     = cur;
    cur                   = align8(cur + sm.payload_size);
    if (sm.has_nulls) {
      dm.null_offset = cur;
      cur            = align8(cur + sm.null_size);
    } else {
      dm.null_offset = 0;
    }
    dst_metas[i] = dm;
  }

  // Pack ONLY selected columns' compressed slices into compact host buffer & single H2D copy!
  std::vector<uint8_t> compact(cur, 0);
  LshfHeader new_lh = lh;
  new_lh.num_cols   = nsel;
  std::memcpy(compact.data(), &new_lh, sizeof(new_lh));
  std::memcpy(compact.data() + sizeof(new_lh), dst_metas.data(), nsel * sizeof(ShufColMeta));
  for (uint32_t i = 0; i < nsel; ++i) {
    ShufColMeta const& sm = src_metas[selected_cols[i]];
    ShufColMeta const& dm = dst_metas[i];
    if (sm.payload_size > 0) {
      std::memcpy(compact.data() + dm.payload_offset, h_lshf + sm.payload_offset, sm.payload_size);
    }
    if (sm.has_nulls && sm.null_size > 0) {
      std::memcpy(compact.data() + dm.null_offset, h_lshf + sm.null_offset, sm.null_size);
    }
  }

  rmm::device_buffer d_compact(compact.data(), compact.size(), stream, mr);
  int64_t const h_offs[2] = {0, static_cast<int64_t>(compact.size())};
  auto tbl                = assemble_and_decode_shuffle_from_device(
    compact.data(), static_cast<uint8_t const*>(d_compact.data()), compact.size(), h_offs, 1u,
    selected_dtypes, {}, stream, mr);
  stream.sync();
  return tbl;
}

}  // namespace rapids::jni::lenc

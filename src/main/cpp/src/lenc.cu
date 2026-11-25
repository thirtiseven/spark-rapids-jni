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
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/udf.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <nvcomp/lz4.h>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/block/block_scan.cuh>
#include <cuda_runtime_api.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string_view>
#include <unistd.h>
#include <unordered_map>

namespace rapids::jni::lenc {

namespace {

__host__ __device__ constexpr uint64_t align_up(uint64_t v, uint64_t align)
{
  return (v + align - 1) & ~(align - 1);
}

constexpr uint8_t bits_needed(uint64_t range)
{
  if (range == 0) { return 0; }
  uint8_t b = 0;
  while (range > 0) {
    range >>= 1;
    ++b;
  }
  return b;
}

void pread_exact(int fd, void* dst, size_t nbytes, off_t offset)
{
  auto* out       = static_cast<uint8_t*>(dst);
  size_t done     = 0;
  while (done < nbytes) {
    ssize_t const r = ::pread(fd, out + done, nbytes - done, offset + static_cast<off_t>(done));
    CUDF_EXPECTS(r > 0, "lenc: pread failed or hit unexpected EOF");
    done += static_cast<size_t>(r);
  }
}

void write_exact(int fd, void const* src, size_t nbytes)
{
  auto const* in = static_cast<uint8_t const*>(src);
  size_t done    = 0;
  while (done < nbytes) {
    ssize_t const w = ::write(fd, in + done, nbytes - done);
    CUDF_EXPECTS(w > 0, "lenc: write failed");
    done += static_cast<size_t>(w);
  }
}

// ---------------- Host Encoders ----------------

std::vector<uint8_t> encode_for_bitpack_i64(int64_t const* vals, uint32_t num_rows)
{
  uint32_t const num_chunks = (num_rows + kChunkSize - 1) / kChunkSize;
  std::vector<int64_t> min_vals(num_chunks, 0);
  std::vector<uint8_t> bit_widths(num_chunks, 0);
  std::vector<uint32_t> chunk_word_offsets(num_chunks, 0);
  std::vector<uint64_t> packed_words;
  packed_words.reserve(static_cast<size_t>(num_rows) / 2 + 16);

  for (uint32_t c = 0; c < num_chunks; ++c) {
    uint32_t const start = c * kChunkSize;
    uint32_t const end   = std::min(start + kChunkSize, num_rows);
    int64_t mn           = vals[start];
    int64_t mx           = vals[start];
    for (uint32_t i = start + 1; i < end; ++i) {
      mn = std::min(mn, vals[i]);
      mx = std::max(mx, vals[i]);
    }
    uint64_t const udelta = static_cast<uint64_t>(mx) - static_cast<uint64_t>(mn);
    uint8_t const b       = bits_needed(udelta);
    min_vals[c]           = mn;
    bit_widths[c]         = b;
    chunk_word_offsets[c] = static_cast<uint32_t>(packed_words.size());
    if (b == 0) { continue; }

    uint32_t const count      = end - start;
    uint32_t const total_bits = count * static_cast<uint32_t>(b);
    uint32_t const num_words  = (total_bits + 63u) / 64u;
    size_t const base_word    = packed_words.size();
    packed_words.resize(base_word + num_words, 0ULL);

    for (uint32_t l = 0; l < count; ++l) {
      uint64_t const d       = static_cast<uint64_t>(vals[start + l]) - static_cast<uint64_t>(mn);
      uint32_t const bit_pos = l * static_cast<uint32_t>(b);
      size_t const w_idx     = base_word + (bit_pos >> 6);
      uint32_t const bit_off = bit_pos & 63u;
      packed_words[w_idx] |= (d << bit_off);
      if (bit_off + b > 64u) { packed_words[w_idx + 1] |= (d >> (64u - bit_off)); }
    }
  }
  // One trailing sentinel word so branchless 2-word reads on GPU never read out of bounds.
  packed_words.push_back(0ULL);

  size_t const off_min  = 0;
  size_t const off_bw   = off_min + num_chunks * sizeof(int64_t);
  size_t const off_cwo  = align_up(off_bw + num_chunks * sizeof(uint8_t), 4);
  size_t const off_pack = align_up(off_cwo + num_chunks * sizeof(uint32_t), 8);
  size_t const total    = off_pack + packed_words.size() * sizeof(uint64_t);

  std::vector<uint8_t> buf(total, 0);
  if (num_chunks > 0) {
    std::memcpy(buf.data() + off_min, min_vals.data(), num_chunks * sizeof(int64_t));
    std::memcpy(buf.data() + off_bw, bit_widths.data(), num_chunks * sizeof(uint8_t));
    std::memcpy(buf.data() + off_cwo, chunk_word_offsets.data(), num_chunks * sizeof(uint32_t));
  }
  std::memcpy(buf.data() + off_pack, packed_words.data(), packed_words.size() * sizeof(uint64_t));
  return buf;
}

std::vector<uint8_t> encode_rle_i64(int64_t const* vals, uint32_t num_rows, uint32_t& out_num_runs)
{
  uint32_t const num_chunks = (num_rows + kChunkSize - 1) / kChunkSize;
  std::vector<uint32_t> run_ends;
  std::vector<int64_t> run_vals;
  if (num_rows > 0) {
    int64_t cur = vals[0];
    for (uint32_t i = 1; i < num_rows; ++i) {
      if (vals[i] != cur) {
        run_ends.push_back(i);
        run_vals.push_back(cur);
        cur = vals[i];
      }
    }
    run_ends.push_back(num_rows);
    run_vals.push_back(cur);
  }
  out_num_runs = static_cast<uint32_t>(run_vals.size());

  std::vector<uint32_t> chunk_start_run(num_chunks, 0);
  uint32_t ridx = 0;
  for (uint32_t c = 0; c < num_chunks; ++c) {
    uint32_t const row_start = c * kChunkSize;
    while (ridx < out_num_runs && run_ends[ridx] <= row_start) { ++ridx; }
    chunk_start_run[c] = ridx;
  }

  size_t const off_hdr  = 0;
  size_t const off_csr  = 8;
  size_t const off_ends = off_csr + num_chunks * sizeof(uint32_t);
  size_t const off_vals = align_up(off_ends + out_num_runs * sizeof(uint32_t), 8);
  size_t const total    = off_vals + out_num_runs * sizeof(int64_t);
  std::vector<uint8_t> buf(total, 0);

  uint32_t hdr[2] = {out_num_runs, num_chunks};
  std::memcpy(buf.data() + off_hdr, hdr, sizeof(hdr));
  if (num_chunks > 0) {
    std::memcpy(buf.data() + off_csr, chunk_start_run.data(), num_chunks * sizeof(uint32_t));
  }
  if (out_num_runs > 0) {
    std::memcpy(buf.data() + off_ends, run_ends.data(), out_num_runs * sizeof(uint32_t));
    std::memcpy(buf.data() + off_vals, run_vals.data(), out_num_runs * sizeof(int64_t));
  }
  return buf;
}

template <typename FloatT>
bool try_scale_float(FloatT const* vals,
                     uint32_t num_rows,
                     int8_t& out_exp10,
                     std::vector<int64_t>& out_scaled)
{
  static constexpr int8_t kExps[]   = {0, 2, 4};
  static constexpr double kPow10[]  = {1.0, 100.0, 10000.0};
  static constexpr double kInv10[]  = {1.0, 0.01, 0.0001};

  out_scaled.resize(num_rows);
  for (int e = 0; e < 3; ++e) {
    double const mul = kPow10[e];
    double const inv = kInv10[e];
    bool ok          = true;
    for (uint32_t i = 0; i < num_rows; ++i) {
      FloatT const v = vals[i];
      if (!std::isfinite(v)) {
        ok = false;
        break;
      }
      double const scaled = std::rint(static_cast<double>(v) * mul);
      if (std::fabs(scaled) > 1e15) {
        ok = false;
        break;
      }
      int64_t const iv    = static_cast<int64_t>(scaled);
      FloatT const recon  = static_cast<FloatT>(static_cast<double>(iv) * inv);
      // Bit-exact check (including -0.0 vs +0.0)
      if (std::memcmp(&recon, &v, sizeof(FloatT)) != 0) {
        ok = false;
        break;
      }
      out_scaled[i] = iv;
    }
    if (ok) {
      out_exp10 = kExps[e];
      return true;
    }
  }
  return false;
}

// ---------------- Device Bit-Unpacking Primitives ----------------

struct ForBitpackView {
  int64_t const* min_vals;
  uint8_t const* bit_widths;
  uint32_t const* chunk_word_offsets;
  uint64_t const* packed_words;
};

__host__ __device__ __forceinline__ ForBitpackView make_for_view(uint8_t const* base,
                                                                 uint32_t num_rows)
{
  uint32_t const num_chunks = (num_rows + kChunkSize - 1) / kChunkSize;
  size_t const off_min      = 0;
  size_t const off_bw       = off_min + num_chunks * sizeof(int64_t);
  size_t const off_cwo      = align_up(off_bw + num_chunks * sizeof(uint8_t), 4);
  size_t const off_pack     = align_up(off_cwo + num_chunks * sizeof(uint32_t), 8);
  return ForBitpackView{
    reinterpret_cast<int64_t const*>(base + off_min),
    reinterpret_cast<uint8_t const*>(base + off_bw),
    reinterpret_cast<uint32_t const*>(base + off_cwo),
    reinterpret_cast<uint64_t const*>(base + off_pack),
  };
}

__device__ __forceinline__ int64_t decode_for_row(ForBitpackView const& v, uint32_t row)
{
  uint32_t const c    = row >> 10;
  uint32_t const b    = v.bit_widths[c];
  int64_t const base  = v.min_vals[c];
  if (b == 0u) { return base; }
  uint32_t const local    = row & 1023u;
  uint32_t const bit_pos  = local * b;
  uint32_t const word_idx = v.chunk_word_offsets[c] + (bit_pos >> 6);
  uint32_t const bit_off  = bit_pos & 63u;
  uint64_t val            = v.packed_words[word_idx] >> bit_off;
  if (bit_off + b > 64u) { val |= v.packed_words[word_idx + 1] << (64u - bit_off); }
  uint64_t const mask = (b == 64u) ? ~0ULL : ((1ULL << b) - 1ULL);
  return base + static_cast<int64_t>(val & mask);
}

struct RleView {
  uint32_t num_runs;
  uint32_t const* chunk_start_run;
  uint32_t const* run_ends;
  int64_t const* run_vals;
};

__host__ __device__ __forceinline__ RleView make_rle_view(uint8_t const* base,
                                                          uint32_t num_rows,
                                                          uint32_t num_runs)
{
  uint32_t const num_chunks = (num_rows + kChunkSize - 1) / kChunkSize;
  size_t const off_csr      = 8;
  size_t const off_ends     = off_csr + num_chunks * sizeof(uint32_t);
  size_t const off_vals     = align_up(off_ends + num_runs * sizeof(uint32_t), 8);
  return RleView{
    num_runs,
    reinterpret_cast<uint32_t const*>(base + off_csr),
    reinterpret_cast<uint32_t const*>(base + off_ends),
    reinterpret_cast<int64_t const*>(base + off_vals),
  };
}

__device__ __forceinline__ int64_t decode_rle_row(RleView const& v, uint32_t row)
{
  uint32_t idx = v.chunk_start_run[row >> 10];
  while (idx + 1u < v.num_runs && v.run_ends[idx] <= row) { ++idx; }
  return v.run_vals[idx];
}

// ---------------- Decode Kernels ----------------

template <typename OutT>
__global__ void for_bitpack_decode_kernel(ForBitpackView view, OutT* __restrict__ out, uint32_t n)
{
  uint32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) { out[idx] = static_cast<OutT>(decode_for_row(view, idx)); }
}

template <typename FloatT>
__global__ void scaled_float_for_decode_kernel(ForBitpackView view,
                                               double inv_scale,
                                               FloatT* __restrict__ out,
                                               uint32_t n)
{
  uint32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int64_t const iv = decode_for_row(view, idx);
    out[idx]         = static_cast<FloatT>(static_cast<double>(iv) * inv_scale);
  }
}

template <typename OutT>
__global__ void rle_decode_kernel(RleView view, OutT* __restrict__ out, uint32_t n)
{
  uint32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) { out[idx] = static_cast<OutT>(decode_rle_row(view, idx)); }
}

// ---------------- DELTA_FOR_BITPACK Codec (1024-row mini-blocks) ----------------

struct DeltaForView {
  int64_t const* first_vals;
  int64_t const* min_deltas;
  uint8_t const* bit_widths;
  uint32_t const* chunk_word_offsets;
  uint64_t const* packed_words;
};

__host__ __device__ __forceinline__ DeltaForView make_delta_for_view(uint8_t const* base,
                                                                     uint32_t num_rows)
{
  uint32_t const num_chunks = (num_rows + kChunkSize - 1) / kChunkSize;
  size_t const off_first    = 0;
  size_t const off_mind     = off_first + num_chunks * sizeof(int64_t);
  size_t const off_bw       = off_mind + num_chunks * sizeof(int64_t);
  size_t const off_cwo      = align_up(off_bw + num_chunks * sizeof(uint8_t), 4);
  size_t const off_pack     = align_up(off_cwo + num_chunks * sizeof(uint32_t), 8);
  return DeltaForView{
    reinterpret_cast<int64_t const*>(base + off_first),
    reinterpret_cast<int64_t const*>(base + off_mind),
    reinterpret_cast<uint8_t const*>(base + off_bw),
    reinterpret_cast<uint32_t const*>(base + off_cwo),
    reinterpret_cast<uint64_t const*>(base + off_pack),
  };
}

bool encode_delta_for_bitpack_i64(int64_t const* data, uint32_t n, std::vector<uint8_t>& out_buf)
{
  if (n == 0) { return false; }
  uint32_t const num_chunks = (n + kChunkSize - 1) / kChunkSize;
  std::vector<int64_t> first_vals(num_chunks, 0);
  std::vector<int64_t> min_deltas(num_chunks, 0);
  std::vector<uint8_t> bit_widths(num_chunks, 0);
  std::vector<uint32_t> chunk_word_offsets(num_chunks, 0);
  std::vector<uint64_t> packed_words;

  for (uint32_t c = 0; c < num_chunks; ++c) {
    uint32_t const start = c * kChunkSize;
    uint32_t const len   = std::min<uint32_t>(kChunkSize, n - start);
    first_vals[c]        = data[start];
    if (len <= 1) {
      min_deltas[c]         = 0;
      bit_widths[c]         = 0;
      chunk_word_offsets[c] = static_cast<uint32_t>(packed_words.size());
      continue;
    }
    __int128 dmin = static_cast<__int128>(data[start + 1]) - static_cast<__int128>(data[start]);
    __int128 dmax = dmin;
    for (uint32_t l = 2; l < len; ++l) {
      __int128 const d =
        static_cast<__int128>(data[start + l]) - static_cast<__int128>(data[start + l - 1]);
      if (d < dmin) { dmin = d; }
      if (d > dmax) { dmax = d; }
    }
    if (dmin < static_cast<__int128>(std::numeric_limits<int64_t>::min()) ||
        dmin > static_cast<__int128>(std::numeric_limits<int64_t>::max())) {
      return false;
    }
    __int128 const range128 = dmax - dmin;
    if (range128 > static_cast<__int128>(std::numeric_limits<uint64_t>::max())) { return false; }
    int64_t const md      = static_cast<int64_t>(dmin);
    uint8_t const bw      = bits_needed(static_cast<uint64_t>(range128));
    min_deltas[c]         = md;
    bit_widths[c]         = bw;
    chunk_word_offsets[c] = static_cast<uint32_t>(packed_words.size());
    if (bw == 0) { continue; }
    uint32_t const total_bits = (len - 1) * bw;
    uint32_t const nwords     = (total_bits + 63) / 64 + 1;
    size_t const base_word    = packed_words.size();
    packed_words.resize(base_word + nwords, 0ULL);
    for (uint32_t l = 1; l < len; ++l) {
      uint64_t const u = static_cast<uint64_t>(
        static_cast<__int128>(data[start + l]) - static_cast<__int128>(data[start + l - 1]) - dmin);
      uint32_t const bit_pos  = (l - 1) * bw;
      uint32_t const word_idx = base_word + (bit_pos >> 6);
      uint32_t const bit_off  = bit_pos & 63u;
      packed_words[word_idx] |= (u << bit_off);
      if (bit_off + bw > 64u) { packed_words[word_idx + 1] |= (u >> (64u - bit_off)); }
    }
  }

  size_t const off_first = 0;
  size_t const off_mind  = off_first + num_chunks * sizeof(int64_t);
  size_t const off_bw    = off_mind + num_chunks * sizeof(int64_t);
  size_t const off_cwo   = align_up(off_bw + num_chunks * sizeof(uint8_t), 4);
  size_t const off_pack  = align_up(off_cwo + num_chunks * sizeof(uint32_t), 8);
  size_t const total_sz  = off_pack + packed_words.size() * sizeof(uint64_t);
  out_buf.assign(total_sz, 0);
  std::memcpy(out_buf.data() + off_first, first_vals.data(), num_chunks * sizeof(int64_t));
  std::memcpy(out_buf.data() + off_mind, min_deltas.data(), num_chunks * sizeof(int64_t));
  std::memcpy(out_buf.data() + off_bw, bit_widths.data(), num_chunks * sizeof(uint8_t));
  std::memcpy(out_buf.data() + off_cwo, chunk_word_offsets.data(), num_chunks * sizeof(uint32_t));
  if (!packed_words.empty()) {
    std::memcpy(
      out_buf.data() + off_pack, packed_words.data(), packed_words.size() * sizeof(uint64_t));
  }
  return true;
}

template <typename OutT>
__global__ void delta_for_decode_kernel(DeltaForView view, OutT* __restrict__ out, uint32_t n)
{
  using BlockScan = cub::BlockScan<int64_t, 256>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  uint32_t const c     = blockIdx.x;
  uint32_t const start = c * kChunkSize;
  if (start >= n) { return; }
  uint32_t const len = (n - start < kChunkSize) ? (n - start) : kChunkSize;
  uint32_t const bw  = view.bit_widths[c];
  int64_t const fv   = view.first_vals[c];
  int64_t const md   = view.min_deltas[c];
  uint32_t const cwo = view.chunk_word_offsets[c];
  uint64_t const msk = (bw == 64u) ? ~0ULL : ((1ULL << bw) - 1ULL);

  int64_t items[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    uint32_t const local = threadIdx.x * 4u + static_cast<uint32_t>(k);
    if (local == 0u) {
      items[k] = fv;
    } else if (local < len) {
      int64_t d = md;
      if (bw > 0u) {
        uint32_t const bit_pos  = (local - 1u) * bw;
        uint32_t const word_idx = cwo + (bit_pos >> 6);
        uint32_t const bit_off  = bit_pos & 63u;
        uint64_t val            = view.packed_words[word_idx] >> bit_off;
        if (bit_off + bw > 64u) { val |= view.packed_words[word_idx + 1] << (64u - bit_off); }
        d += static_cast<int64_t>(val & msk);
      }
      items[k] = d;
    } else {
      items[k] = 0;
    }
  }
  BlockScan(temp_storage).InclusiveSum(items, items);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    uint32_t const local = threadIdx.x * 4u + static_cast<uint32_t>(k);
    if (local < len) { out[start + local] = static_cast<OutT>(items[k]); }
  }
}

// ---------------- GPU nvCOMP Batched LZ4 Fallback (High-Card Strings / High-Entropy Fixed) --------

constexpr uint32_t kLz4ChunkSize = 65536;

bool nvcomp_lz4_compress_host(uint8_t const* raw_data,
                              size_t raw_size,
                              uint64_t total_chars,
                              std::vector<uint8_t>& out_buf,
                              cuda::stream_ref stream)
{
  if (raw_size < 4096) { return false; }
  uint32_t const num_chunks = static_cast<uint32_t>((raw_size + kLz4ChunkSize - 1) / kLz4ChunkSize);
  size_t max_out_chunk      = 0;
  auto st = nvcompBatchedLZ4CompressGetMaxOutputChunkSize(
    kLz4ChunkSize, nvcompBatchedLZ4CompressDefaultOpts, &max_out_chunk);
  if (st != nvcompSuccess) { return false; }

  size_t temp_bytes = 0;
  st                = nvcompBatchedLZ4CompressGetTempSize(
    num_chunks, kLz4ChunkSize, nvcompBatchedLZ4CompressDefaultOpts, &temp_bytes, raw_size, stream.get());
  if (st != nvcompSuccess) { return false; }

  auto mr = cudf::get_current_device_resource_ref();
  rmm::device_buffer d_in(raw_size, stream, mr);
  CUDF_CUDA_TRY(
    cudaMemcpyAsync(d_in.data(), raw_data, raw_size, cudaMemcpyHostToDevice, stream.get()));

  std::vector<void const*> h_in_ptrs(num_chunks);
  std::vector<size_t> h_in_sizes(num_chunks);
  std::vector<void*> h_out_ptrs(num_chunks);
  rmm::device_buffer d_out(static_cast<size_t>(num_chunks) * max_out_chunk, stream, mr);
  for (uint32_t i = 0; i < num_chunks; ++i) {
    size_t const off = static_cast<size_t>(i) * kLz4ChunkSize;
    h_in_ptrs[i]     = static_cast<uint8_t const*>(d_in.data()) + off;
    h_in_sizes[i]    = std::min<size_t>(kLz4ChunkSize, raw_size - off);
    h_out_ptrs[i]    = static_cast<uint8_t*>(d_out.data()) + static_cast<size_t>(i) * max_out_chunk;
  }

  rmm::device_uvector<void const*> d_in_ptrs(num_chunks, stream, mr);
  rmm::device_uvector<size_t> d_in_sizes(num_chunks, stream, mr);
  rmm::device_uvector<void*> d_out_ptrs(num_chunks, stream, mr);
  rmm::device_uvector<size_t> d_out_sizes(num_chunks, stream, mr);
  rmm::device_uvector<nvcompStatus_t> d_statuses(num_chunks, stream, mr);
  rmm::device_buffer d_temp(temp_bytes, stream, mr);

  CUDF_CUDA_TRY(cudaMemcpyAsync(
    d_in_ptrs.data(), h_in_ptrs.data(), num_chunks * sizeof(void const*), cudaMemcpyHostToDevice, stream.get()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    d_in_sizes.data(), h_in_sizes.data(), num_chunks * sizeof(size_t), cudaMemcpyHostToDevice, stream.get()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    d_out_ptrs.data(), h_out_ptrs.data(), num_chunks * sizeof(void*), cudaMemcpyHostToDevice, stream.get()));

  st = nvcompBatchedLZ4CompressAsync(d_in_ptrs.data(),
                                     d_in_sizes.data(),
                                     kLz4ChunkSize,
                                     num_chunks,
                                     d_temp.data(),
                                     temp_bytes,
                                     d_out_ptrs.data(),
                                     d_out_sizes.data(),
                                     nvcompBatchedLZ4CompressDefaultOpts,
                                     d_statuses.data(),
                                     stream.get());
  if (st != nvcompSuccess) { return false; }

  std::vector<size_t> h_out_sizes(num_chunks);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    h_out_sizes.data(), d_out_sizes.data(), num_chunks * sizeof(size_t), cudaMemcpyDeviceToHost, stream.get()));
  stream.sync();

  uint64_t total_comp = 0;
  std::vector<uint64_t> comp_offs(num_chunks + 1, 0);
  for (uint32_t i = 0; i < num_chunks; ++i) {
    total_comp += h_out_sizes[i];
    comp_offs[i + 1] = total_comp;
  }
  size_t const hdr_bytes = sizeof(Lz4PayloadHeader) + (num_chunks + 1) * sizeof(uint64_t);
  if (hdr_bytes + total_comp >= (raw_size * 95) / 100) { return false; }

  out_buf.assign(hdr_bytes + total_comp, 0);
  Lz4PayloadHeader lh{num_chunks, kLz4ChunkSize, raw_size, total_chars};
  std::memcpy(out_buf.data(), &lh, sizeof(lh));
  std::memcpy(out_buf.data() + sizeof(lh), comp_offs.data(), (num_chunks + 1) * sizeof(uint64_t));

  // Copy compressed chunks from device to contiguous host buffer
  std::vector<uint8_t> h_staged(static_cast<size_t>(num_chunks) * max_out_chunk);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    h_staged.data(), d_out.data(), h_staged.size(), cudaMemcpyDeviceToHost, stream.get()));
  stream.sync();
  for (uint32_t i = 0; i < num_chunks; ++i) {
    std::memcpy(out_buf.data() + hdr_bytes + comp_offs[i],
                h_staged.data() + static_cast<size_t>(i) * max_out_chunk,
                h_out_sizes[i]);
  }
  return true;
}

rmm::device_buffer nvcomp_lz4_decompress_device(uint8_t const* h_pay,
                                                uint8_t const* d_pay,
                                                Lz4PayloadHeader& out_lh,
                                                cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr)
{
  std::memcpy(&out_lh, h_pay, sizeof(out_lh));
  uint32_t const nc = out_lh.num_chunks;
  size_t const cs   = out_lh.chunk_size;
  auto const* h_offs = reinterpret_cast<uint64_t const*>(h_pay + sizeof(out_lh));
  size_t const data_off = sizeof(out_lh) + (nc + 1) * sizeof(uint64_t);

  rmm::device_buffer d_uncomp(out_lh.uncompressed_bytes, stream, mr);
  if (nc == 0 || out_lh.uncompressed_bytes == 0) { return d_uncomp; }

  std::vector<void const*> h_in_ptrs(nc);
  std::vector<size_t> h_in_sizes(nc);
  std::vector<void*> h_out_ptrs(nc);
  std::vector<size_t> h_out_buf_sizes(nc);
  for (uint32_t i = 0; i < nc; ++i) {
    h_in_ptrs[i]       = d_pay + data_off + h_offs[i];
    h_in_sizes[i]      = static_cast<size_t>(h_offs[i + 1] - h_offs[i]);
    size_t const uoff  = static_cast<size_t>(i) * cs;
    h_out_ptrs[i]      = static_cast<uint8_t*>(d_uncomp.data()) + uoff;
    h_out_buf_sizes[i] = std::min<size_t>(cs, out_lh.uncompressed_bytes - uoff);
  }

  size_t temp_bytes = 0;
  auto st           = nvcompBatchedLZ4DecompressGetTempSize(
    nc, cs, nvcompBatchedLZ4DecompressDefaultOpts, &temp_bytes, out_lh.uncompressed_bytes, stream.get());
  CUDF_EXPECTS(st == nvcompSuccess, "nvcompBatchedLZ4DecompressGetTempSize failed");

  rmm::device_uvector<void const*> d_in_ptrs(nc, stream, mr);
  rmm::device_uvector<size_t> d_in_sizes(nc, stream, mr);
  rmm::device_uvector<void*> d_out_ptrs(nc, stream, mr);
  rmm::device_uvector<size_t> d_out_buf_sizes(nc, stream, mr);
  rmm::device_uvector<size_t> d_actual_sizes(nc, stream, mr);
  rmm::device_uvector<nvcompStatus_t> d_statuses(nc, stream, mr);
  rmm::device_buffer d_temp(temp_bytes, stream, mr);

  CUDF_CUDA_TRY(cudaMemcpyAsync(
    d_in_ptrs.data(), h_in_ptrs.data(), nc * sizeof(void const*), cudaMemcpyHostToDevice, stream.get()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    d_in_sizes.data(), h_in_sizes.data(), nc * sizeof(size_t), cudaMemcpyHostToDevice, stream.get()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    d_out_ptrs.data(), h_out_ptrs.data(), nc * sizeof(void*), cudaMemcpyHostToDevice, stream.get()));
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    d_out_buf_sizes.data(), h_out_buf_sizes.data(), nc * sizeof(size_t), cudaMemcpyHostToDevice, stream.get()));

  st = nvcompBatchedLZ4DecompressAsync(d_in_ptrs.data(),
                                       d_in_sizes.data(),
                                       d_out_buf_sizes.data(),
                                       d_actual_sizes.data(),
                                       nc,
                                       d_temp.data(),
                                       temp_bytes,
                                       d_out_ptrs.data(),
                                       nvcompBatchedLZ4DecompressDefaultOpts,
                                       d_statuses.data(),
                                       stream.get());
  CUDF_EXPECTS(st == nvcompSuccess, "nvcompBatchedLZ4DecompressAsync failed");
  return d_uncomp;
}

// ---------------- Late-Materialized (Selected Rows) Decode Kernels ----------------

template <typename OutT>
__global__ void for_bitpack_decode_selected_kernel(
  ForBitpackView view, int32_t const* __restrict__ sel, OutT* __restrict__ out, uint32_t m)
{
  uint32_t const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m) { out[i] = static_cast<OutT>(decode_for_row(view, static_cast<uint32_t>(sel[i]))); }
}

template <typename FloatT>
__global__ void scaled_float_for_decode_selected_kernel(ForBitpackView view,
                                                        double inv_scale,
                                                        int32_t const* __restrict__ sel,
                                                        FloatT* __restrict__ out,
                                                        uint32_t m)
{
  uint32_t const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m) {
    int64_t const iv = decode_for_row(view, static_cast<uint32_t>(sel[i]));
    out[i]           = static_cast<FloatT>(static_cast<double>(iv) * inv_scale);
  }
}

template <typename OutT>
__global__ void rle_decode_selected_kernel(
  RleView view, int32_t const* __restrict__ sel, OutT* __restrict__ out, uint32_t m)
{
  uint32_t const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m) { out[i] = static_cast<OutT>(decode_rle_row(view, static_cast<uint32_t>(sel[i]))); }
}

template <typename T>
__global__ void raw_gather_kernel(
  T const* __restrict__ src, int32_t const* __restrict__ sel, T* __restrict__ out, uint32_t m)
{
  uint32_t const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m) { out[i] = src[sel[i]]; }
}

__global__ void dict_lengths_selected_kernel(ForBitpackView codes_view,
                                             int32_t const* __restrict__ dict_offsets,
                                             cudf::bitmask_type const* __restrict__ null_mask,
                                             int32_t const* __restrict__ sel,
                                             int32_t* __restrict__ out_lengths,
                                             uint32_t m)
{
  uint32_t const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m) {
    uint32_t const r = static_cast<uint32_t>(sel[i]);
    if (null_mask != nullptr && !cudf::bit_is_set(null_mask, static_cast<cudf::size_type>(r))) {
      out_lengths[i] = 0;
    } else {
      uint32_t const cid = static_cast<uint32_t>(decode_for_row(codes_view, r));
      out_lengths[i]     = dict_offsets[cid + 1] - dict_offsets[cid];
    }
  } else if (i == m) {
    out_lengths[m] = 0;
  }
}

__global__ void dict_copy_chars_selected_kernel(ForBitpackView codes_view,
                                                int32_t const* __restrict__ dict_offsets,
                                                char const* __restrict__ dict_chars,
                                                cudf::bitmask_type const* __restrict__ null_mask,
                                                int32_t const* __restrict__ sel,
                                                int32_t const* __restrict__ out_offsets,
                                                char* __restrict__ out_chars,
                                                uint32_t m)
{
  uint32_t const i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < m) {
    uint32_t const r = static_cast<uint32_t>(sel[i]);
    if (null_mask != nullptr && !cudf::bit_is_set(null_mask, static_cast<cudf::size_type>(r))) {
      return;
    }
    int32_t const dst_off = out_offsets[i];
    int32_t const len     = out_offsets[i + 1] - dst_off;
    if (len <= 0) { return; }
    uint32_t const cid    = static_cast<uint32_t>(decode_for_row(codes_view, r));
    int32_t const src_off = dict_offsets[cid];
    char const* src       = dict_chars + src_off;
    char* dst             = out_chars + dst_off;
    for (int32_t k = 0; k < len; ++k) { dst[k] = src[k]; }
  }
}

__global__ void gather_null_mask_kernel(cudf::bitmask_type const* __restrict__ src_mask,
                                        int32_t const* __restrict__ sel,
                                        cudf::bitmask_type* __restrict__ dst_mask,
                                        uint32_t m)
{
  uint32_t const word_idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t const nwords   = (m + 31u) >> 5;
  if (word_idx >= nwords) { return; }
  uint32_t const base = word_idx << 5;
  cudf::bitmask_type w = 0u;
  for (uint32_t b = 0; b < 32u && base + b < m; ++b) {
    uint32_t const r = static_cast<uint32_t>(sel[base + b]);
    if (cudf::bit_is_set(src_mask, static_cast<cudf::size_type>(r))) { w |= (1u << b); }
  }
  dst_mask[word_idx] = w;
}

struct MaskIsTruePred {
  bool const* data;
  cudf::bitmask_type const* mask;
  __device__ bool operator()(int32_t idx) const
  {
    if (mask != nullptr && !cudf::bit_is_set(mask, idx)) { return false; }
    return data[idx];
  }
};

__global__ void dict_lengths_kernel(ForBitpackView codes_view,
                                    int32_t const* __restrict__ dict_offsets,
                                    cudf::bitmask_type const* __restrict__ null_mask,
                                    int32_t* __restrict__ out_lengths,
                                    uint32_t n)
{
  uint32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    if (null_mask != nullptr && !cudf::bit_is_set(null_mask, static_cast<cudf::size_type>(idx))) {
      out_lengths[idx] = 0;
    } else {
      uint32_t const cid = static_cast<uint32_t>(decode_for_row(codes_view, idx));
      out_lengths[idx]   = dict_offsets[cid + 1] - dict_offsets[cid];
    }
  } else if (idx == n) {
    out_lengths[n] = 0;
  }
}

__global__ void dict_copy_chars_kernel(ForBitpackView codes_view,
                                       int32_t const* __restrict__ dict_offsets,
                                       char const* __restrict__ dict_chars,
                                       cudf::bitmask_type const* __restrict__ null_mask,
                                       int32_t const* __restrict__ out_offsets,
                                       char* __restrict__ out_chars,
                                       uint32_t n)
{
  uint32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    if (null_mask != nullptr && !cudf::bit_is_set(null_mask, static_cast<cudf::size_type>(idx))) {
      return;
    }
    int32_t const dst_off = out_offsets[idx];
    int32_t const len     = out_offsets[idx + 1] - dst_off;
    if (len <= 0) { return; }
    uint32_t const cid    = static_cast<uint32_t>(decode_for_row(codes_view, idx));
    int32_t const src_off = dict_offsets[cid];
    char const* src       = dict_chars + src_off;
    char* dst             = out_chars + dst_off;
    for (int32_t i = 0; i < len; ++i) { dst[i] = src[i]; }
  }
}

// ---------------- Phase 3B Fused Decode-in-Register Kernels ----------------

__global__ void fused_w1a_1_kernel(ForBitpackView c0v,
                                   ForBitpackView c1v,
                                   ForBitpackView c2v,
                                   ForBitpackView c3v,
                                   int64_t* __restrict__ out,
                                   uint32_t n)
{
  uint32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int64_t const c0 = decode_for_row(c0v, idx);
    int64_t const c1 = decode_for_row(c1v, idx);
    int64_t const c2 = decode_for_row(c2v, idx);
    int64_t const c3 = decode_for_row(c3v, idx);
    int64_t const t  = ((c0 + c1) - c2) - c3;
    int64_t const a  = t < 0 ? -t : t;
    out[idx]         = (-a * 2LL) + 17LL + c1 - c2;
  }
}

struct Ptr8 {
  int64_t* p[8];
};

__global__ void fused_w1a_8_kernel(ForBitpackView c0v,
                                   ForBitpackView c1v,
                                   ForBitpackView c2v,
                                   ForBitpackView c3v,
                                   ForBitpackView c4v,
                                   Ptr8 out,
                                   uint32_t n)
{
  uint32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int64_t const c0 = decode_for_row(c0v, idx);
    int64_t const c1 = decode_for_row(c1v, idx);
    int64_t const c2 = decode_for_row(c2v, idx);
    int64_t const c3 = decode_for_row(c3v, idx);
    int64_t const c4 = decode_for_row(c4v, idx);
    int64_t const t  = ((c0 + c1) - c2) - c3;
    int64_t const na = -(t < 0 ? -t : t);
    int64_t const cm = c1 - c2;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      out.p[k][idx] = na * (2LL + (k % 3)) + (17LL + k * 13LL) + cm + (c4 * (k + 1LL));
    }
  }
}

struct DictBranchLut24 {
  int32_t target_code[24];
};

template <int Branches>
__global__ void fused_w1b_case_kernel(ForBitpackView c0v,
                                      ForBitpackView c1v,
                                      ForBitpackView c2v,
                                      ForBitpackView c3v,
                                      ForBitpackView c4v,
                                      ForBitpackView s1_codes_v,
                                      DictBranchLut24 lut,
                                      int64_t* __restrict__ out,
                                      uint32_t n)
{
  uint32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int32_t const s1_code = static_cast<int32_t>(decode_for_row(s1_codes_v, idx));
    int32_t const c4      = static_cast<int32_t>(decode_for_row(c4v, idx));
    int64_t result        = 0;
    bool matched          = false;
#pragma unroll
    for (int i = 0; i < Branches; ++i) {
      if (!matched && s1_code == lut.target_code[i] && c4 > i * 100) {
        int64_t const c0 = decode_for_row(c0v, idx);
        int64_t const c1 = decode_for_row(c1v, idx);
        int64_t const c2 = decode_for_row(c2v, idx);
        result           = c0 * static_cast<int64_t>(i + 1) + c1 - c2;
        matched          = true;
      }
    }
    if (!matched) { result = decode_for_row(c3v, idx) + 7LL; }
    out[idx] = result;
  }
}

// Ultimate ceiling micro-kernel: decode-in-register + eval + block-sum reduction (writes 8 bytes!)
__global__ void fused_w1a_1_sum_kernel(ForBitpackView c0v,
                                       ForBitpackView c1v,
                                       ForBitpackView c2v,
                                       ForBitpackView c3v,
                                       unsigned long long* __restrict__ out_sum,
                                       uint32_t n)
{
  __shared__ int64_t smem[256];
  uint32_t const tid = threadIdx.x;
  int64_t acc        = 0;
  for (uint32_t idx = blockIdx.x * blockDim.x + tid; idx < n; idx += gridDim.x * blockDim.x) {
    int64_t const c0 = decode_for_row(c0v, idx);
    int64_t const c1 = decode_for_row(c1v, idx);
    int64_t const c2 = decode_for_row(c2v, idx);
    int64_t const c3 = decode_for_row(c3v, idx);
    int64_t const t  = ((c0 + c1) - c2) - c3;
    int64_t const a  = t < 0 ? -t : t;
    acc += (-a * 2LL) + 17LL + c1 - c2;
  }
  smem[tid] = acc;
  __syncthreads();
  for (uint32_t s = 128; s > 0; s >>= 1) {
    if (tid < s) { smem[tid] += smem[tid + s]; }
    __syncthreads();
  }
  if (tid == 0) { atomicAdd(out_sum, static_cast<unsigned long long>(smem[0])); }
}

// Helper to resolve dictionary string -> code on host directly from the packed host buffer
int32_t find_dict_code_host(uint8_t const* payload_host, std::string_view target)
{
  DictPayloadHeader dh{};
  std::memcpy(&dh, payload_host, sizeof(dh));
  auto const* dict_offsets = reinterpret_cast<int32_t const*>(payload_host + sizeof(dh));
  auto const* dict_chars   = reinterpret_cast<char const*>(
    payload_host + sizeof(dh) + (dh.dict_size + 1) * sizeof(int32_t));
  for (uint32_t id = 0; id < dh.dict_size; ++id) {
    int32_t const s = dict_offsets[id];
    int32_t const e = dict_offsets[id + 1];
    if (static_cast<size_t>(e - s) == target.size() &&
        std::memcmp(dict_chars + s, target.data(), target.size()) == 0) {
      return static_cast<int32_t>(id);
    }
  }
  return -1;
}

}  // namespace

// ---------------- Public API ----------------

void write_table(cudf::table_view const& table,
                 std::string const& path,
                 cuda::stream_ref stream)
{
  uint32_t const num_rows = static_cast<uint32_t>(table.num_rows());
  uint32_t const num_cols = static_cast<uint32_t>(table.num_columns());

  std::vector<ColumnMeta> metas(num_cols);
  std::vector<std::vector<uint8_t>> payloads(num_cols);
  std::vector<std::vector<uint8_t>> null_masks(num_cols);

  for (uint32_t c = 0; c < num_cols; ++c) {
    cudf::column_view const col = table.column(static_cast<cudf::size_type>(c));
    ColumnMeta& meta            = metas[c];
    std::memset(&meta, 0, sizeof(meta));
    meta.null_count = static_cast<uint32_t>(col.null_count());
    meta.has_nulls  = (col.nullable() && col.null_count() > 0) ? 1 : 0;

    if (meta.has_nulls) {
      size_t const mask_bytes = cudf::bitmask_allocation_size_bytes(col.size());
      null_masks[c].resize(mask_bytes, 0);
      if (col.offset() == 0) {
        CUDF_CUDA_TRY(cudaMemcpyAsync(
          null_masks[c].data(), col.null_mask(), mask_bytes, cudaMemcpyDeviceToHost, stream.get()));
        stream.sync();
      } else {
        auto sliced_mask = cudf::copy_bitmask(col, stream);
        CUDF_CUDA_TRY(cudaMemcpyAsync(
          null_masks[c].data(), sliced_mask.data(), mask_bytes, cudaMemcpyDeviceToHost, stream.get()));
        stream.sync();
      }
      meta.null_mask_size = static_cast<uint32_t>(mask_bytes);
    }

    auto const tid = col.type().id();
    if (tid == cudf::type_id::INT32 || tid == cudf::type_id::INT64 ||
        tid == cudf::type_id::TIMESTAMP_DAYS || tid == cudf::type_id::BOOL8 ||
        tid == cudf::type_id::INT8 || tid == cudf::type_id::INT16 ||
        tid == cudf::type_id::TIMESTAMP_MICROSECONDS || tid == cudf::type_id::DECIMAL32 ||
        tid == cudf::type_id::DECIMAL64) {
      if (tid == cudf::type_id::INT32) {
        meta.type = static_cast<uint8_t>(ColType::INT32);
      } else if (tid == cudf::type_id::INT64) {
        meta.type = static_cast<uint8_t>(ColType::INT64);
      } else if (tid == cudf::type_id::TIMESTAMP_DAYS) {
        meta.type = static_cast<uint8_t>(ColType::DATE32);
      } else if (tid == cudf::type_id::BOOL8) {
        meta.type = static_cast<uint8_t>(ColType::BOOL8);
      } else if (tid == cudf::type_id::INT8) {
        meta.type = static_cast<uint8_t>(ColType::INT8);
      } else if (tid == cudf::type_id::INT16) {
        meta.type = static_cast<uint8_t>(ColType::INT16);
      } else if (tid == cudf::type_id::TIMESTAMP_MICROSECONDS) {
        meta.type = static_cast<uint8_t>(ColType::TIMESTAMP_US);
      } else if (tid == cudf::type_id::DECIMAL32) {
        meta.type        = static_cast<uint8_t>(ColType::DECIMAL32);
        meta.scale_exp10 = static_cast<int8_t>(col.type().scale());
      } else {
        meta.type        = static_cast<uint8_t>(ColType::DECIMAL64);
        meta.scale_exp10 = static_cast<int8_t>(col.type().scale());
      }
      std::vector<int64_t> h_vals(num_rows, 0);
      if (num_rows > 0) {
        if (tid == cudf::type_id::INT64 || tid == cudf::type_id::TIMESTAMP_MICROSECONDS ||
            tid == cudf::type_id::DECIMAL64) {
          CUDF_CUDA_TRY(cudaMemcpyAsync(h_vals.data(),
                                        static_cast<int64_t const*>(col.head()) + col.offset(),
                                        num_rows * sizeof(int64_t),
                                        cudaMemcpyDeviceToHost,
                                        stream.get()));
          stream.sync();
        } else if (tid == cudf::type_id::INT32 || tid == cudf::type_id::TIMESTAMP_DAYS ||
                   tid == cudf::type_id::DECIMAL32) {
          std::vector<int32_t> h32(num_rows);
          CUDF_CUDA_TRY(cudaMemcpyAsync(h32.data(),
                                        static_cast<int32_t const*>(col.head()) + col.offset(),
                                        num_rows * sizeof(int32_t),
                                        cudaMemcpyDeviceToHost,
                                        stream.get()));
          stream.sync();
          for (uint32_t i = 0; i < num_rows; ++i) { h_vals[i] = static_cast<int64_t>(h32[i]); }
        } else if (tid == cudf::type_id::INT16) {
          std::vector<int16_t> h16(num_rows);
          CUDF_CUDA_TRY(cudaMemcpyAsync(h16.data(),
                                        col.data<int16_t>() + col.offset(),
                                        num_rows * sizeof(int16_t),
                                        cudaMemcpyDeviceToHost,
                                        stream.get()));
          stream.sync();
          for (uint32_t i = 0; i < num_rows; ++i) { h_vals[i] = static_cast<int64_t>(h16[i]); }
        } else {
          std::vector<int8_t> h8(num_rows);
          CUDF_CUDA_TRY(cudaMemcpyAsync(h8.data(),
                                        static_cast<int8_t const*>(col.head()) + col.offset(),
                                        num_rows * sizeof(int8_t),
                                        cudaMemcpyDeviceToHost,
                                        stream.get()));
          stream.sync();
          for (uint32_t i = 0; i < num_rows; ++i) {
            h_vals[i] = (tid == cudf::type_id::BOOL8) ? (h8[i] != 0 ? 1 : 0)
                                                      : static_cast<int64_t>(h8[i]);
          }
        }
      }
      auto for_buf      = encode_for_bitpack_i64(h_vals.data(), num_rows);
      uint32_t num_runs = 0;
      auto rle_buf      = encode_rle_i64(h_vals.data(), num_rows, num_runs);
      std::vector<uint8_t> delta_buf;
      bool const has_delta =
        (num_rows > 0 && tid != cudf::type_id::BOOL8 &&
         encode_delta_for_bitpack_i64(h_vals.data(), num_rows, delta_buf));
      if (num_rows > 0 && rle_buf.size() * 2 < for_buf.size() &&
          (!has_delta || rle_buf.size() < delta_buf.size())) {
        meta.encoding = static_cast<uint8_t>(Encoding::RLE_INT);
        meta.extra    = num_runs;
        payloads[c]   = std::move(rle_buf);
      } else if (has_delta && delta_buf.size() * 100 < for_buf.size() * 85) {
        meta.encoding = static_cast<uint8_t>(Encoding::DELTA_FOR_BITPACK);
        payloads[c]   = std::move(delta_buf);
      } else {
        meta.encoding = static_cast<uint8_t>(Encoding::FOR_BITPACK);
        payloads[c]   = std::move(for_buf);
      }
    } else if (tid == cudf::type_id::FLOAT32 || tid == cudf::type_id::FLOAT64) {
      bool const is_f32 = (tid == cudf::type_id::FLOAT32);
      meta.type         = static_cast<uint8_t>(is_f32 ? ColType::FLOAT32 : ColType::FLOAT64);
      size_t const esz  = is_f32 ? sizeof(float) : sizeof(double);
      std::vector<uint8_t> raw(num_rows * esz);
      if (num_rows > 0) {
        CUDF_CUDA_TRY(cudaMemcpyAsync(raw.data(),
                                      static_cast<uint8_t const*>(col.head()) + col.offset() * esz,
                                      raw.size(),
                                      cudaMemcpyDeviceToHost,
                                      stream.get()));
        stream.sync();
      }
      std::vector<int64_t> scaled;
      int8_t exp10 = 0;
      bool can_scale =
        (num_rows > 0 && !meta.has_nulls) &&
        (is_f32 ? try_scale_float(reinterpret_cast<float const*>(raw.data()), num_rows, exp10, scaled)
                : try_scale_float(reinterpret_cast<double const*>(raw.data()), num_rows, exp10, scaled));
      if (can_scale) {
        auto for_buf = encode_for_bitpack_i64(scaled.data(), num_rows);
        if (for_buf.size() < raw.size()) {
          meta.encoding    = static_cast<uint8_t>(Encoding::SCALED_FLOAT_FOR);
          meta.scale_exp10 = exp10;
          payloads[c]      = std::move(for_buf);
        } else {
          std::vector<uint8_t> lz4_buf;
          if (nvcomp_lz4_compress_host(raw.data(), raw.size(), 0, lz4_buf, stream)) {
            meta.encoding = static_cast<uint8_t>(Encoding::NVCOMP_LZ4_FIXED);
            payloads[c]   = std::move(lz4_buf);
          } else {
            meta.encoding = static_cast<uint8_t>(Encoding::RAW_FIXED);
            payloads[c]   = std::move(raw);
          }
        }
      } else {
        std::vector<uint8_t> lz4_buf;
        if (nvcomp_lz4_compress_host(raw.data(), raw.size(), 0, lz4_buf, stream)) {
          meta.encoding = static_cast<uint8_t>(Encoding::NVCOMP_LZ4_FIXED);
          payloads[c]   = std::move(lz4_buf);
        } else {
          meta.encoding = static_cast<uint8_t>(Encoding::RAW_FIXED);
          payloads[c]   = std::move(raw);
        }
      }
    } else if (tid == cudf::type_id::STRING) {
      meta.type = static_cast<uint8_t>(ColType::STRING);
      cudf::strings_column_view const scv(col);
      std::vector<int32_t> h_offs(num_rows + 1, 0);
      if (num_rows > 0) {
        CUDF_CUDA_TRY(cudaMemcpyAsync(h_offs.data(),
                                      scv.offsets().data<int32_t>() + col.offset(),
                                      (num_rows + 1) * sizeof(int32_t),
                                      cudaMemcpyDeviceToHost,
                                      stream.get()));
        stream.sync();
      }
      int32_t const base_off   = h_offs[0];
      int32_t const total_ch   = h_offs[num_rows] - base_off;
      std::vector<char> h_chars(static_cast<size_t>(total_ch));
      if (total_ch > 0) {
        CUDF_CUDA_TRY(cudaMemcpyAsync(h_chars.data(),
                                      scv.chars_begin(stream) + base_off,
                                      static_cast<size_t>(total_ch),
                                      cudaMemcpyDeviceToHost,
                                      stream.get()));
        stream.sync();
      }
      // Build dictionary (up to 65,536 distinct strings). For low-cardinality (<= 4096) always
      // prefer DICT_BITPACK (enables DICTIONARY32 & LTO dict_code); for higher cardinality,
      // compare against GPU nvCOMP Batched LZ4 and pick whichever is smaller!
      std::unordered_map<std::string_view, uint32_t> dict_map;
      std::vector<int32_t> dict_offs;
      std::vector<char> dict_chars;
      dict_offs.push_back(0);
      std::vector<int64_t> codes(num_rows, 0);
      bool dict_ok = true;
      for (uint32_t i = 0; i < num_rows; ++i) {
        int32_t const s = h_offs[i] - base_off;
        int32_t const e = h_offs[i + 1] - base_off;
        std::string_view sv(h_chars.data() + s, static_cast<size_t>(e - s));
        auto it = dict_map.find(sv);
        if (it == dict_map.end()) {
          if (dict_map.size() >= 65536) {
            dict_ok = false;
            break;
          }
          uint32_t const new_id = static_cast<uint32_t>(dict_map.size());
          dict_map.emplace(sv, new_id);
          dict_chars.insert(dict_chars.end(), sv.begin(), sv.end());
          dict_offs.push_back(static_cast<int32_t>(dict_chars.size()));
          codes[i] = static_cast<int64_t>(new_id);
        } else {
          codes[i] = static_cast<int64_t>(it->second);
        }
      }
      size_t const off_bytes  = (num_rows + 1) * sizeof(int32_t);
      size_t const raw_str_sz = off_bytes + static_cast<size_t>(total_ch);
      std::vector<uint8_t> dict_buf;
      uint32_t dsize = 0;
      if (dict_ok) {
        auto codes_buf         = encode_for_bitpack_i64(codes.data(), num_rows);
        dsize                  = static_cast<uint32_t>(dict_offs.size() - 1);
        size_t const off_doff  = sizeof(DictPayloadHeader);
        size_t const off_dchr  = off_doff + dict_offs.size() * sizeof(int32_t);
        size_t const off_codes = align_up(off_dchr + dict_chars.size(), 8);
        dict_buf.assign(off_codes + codes_buf.size(), 0);
        DictPayloadHeader dh{dsize, static_cast<uint32_t>(dict_chars.size()), off_codes};
        std::memcpy(dict_buf.data(), &dh, sizeof(dh));
        std::memcpy(dict_buf.data() + off_doff, dict_offs.data(), dict_offs.size() * sizeof(int32_t));
        if (!dict_chars.empty()) {
          std::memcpy(dict_buf.data() + off_dchr, dict_chars.data(), dict_chars.size());
        }
        std::memcpy(dict_buf.data() + off_codes, codes_buf.data(), codes_buf.size());
      }
      if (dict_ok && dsize <= 4096) {
        meta.encoding = static_cast<uint8_t>(Encoding::DICT_BITPACK);
        meta.extra    = dsize;
        payloads[c]   = std::move(dict_buf);
      } else {
        // Medium/high-cardinality string: try GPU nvCOMP Batched LZ4 and pick smallest
        for (uint32_t i = 0; i <= num_rows; ++i) { h_offs[i] -= base_off; }
        std::vector<uint8_t> raw_buf(raw_str_sz, 0);
        std::memcpy(raw_buf.data(), h_offs.data(), off_bytes);
        if (total_ch > 0) {
          std::memcpy(raw_buf.data() + off_bytes, h_chars.data(), static_cast<size_t>(total_ch));
        }
        std::vector<uint8_t> lz4_buf;
        bool const has_lz4 = nvcomp_lz4_compress_host(
          raw_buf.data(), raw_buf.size(), static_cast<uint64_t>(total_ch), lz4_buf, stream);
        if (dict_ok && (!has_lz4 || dict_buf.size() <= lz4_buf.size()) &&
            dict_buf.size() < raw_buf.size()) {
          meta.encoding = static_cast<uint8_t>(Encoding::DICT_BITPACK);
          meta.extra    = dsize;
          payloads[c]   = std::move(dict_buf);
        } else if (has_lz4) {
          meta.encoding = static_cast<uint8_t>(Encoding::NVCOMP_LZ4_STR);
          meta.extra    = static_cast<uint32_t>(total_ch);
          payloads[c]   = std::move(lz4_buf);
        } else {
          meta.encoding = static_cast<uint8_t>(Encoding::RAW_STRING);
          meta.extra    = static_cast<uint32_t>(total_ch);
          payloads[c]   = std::move(raw_buf);
        }
      }
    } else {
      CUDF_FAIL("lenc::write_table: unsupported cuDF column type");
    }
  }

  // Write 64-byte aligned file
  int const fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  CUDF_EXPECTS(fd >= 0, "lenc::write_table: failed to open output file");

  FileHeader hdr{};
  std::memcpy(hdr.magic, kMagic, 8);
  hdr.num_rows   = num_rows;
  hdr.num_cols   = num_cols;
  hdr.chunk_size = kChunkSize;

  uint64_t cur_off = sizeof(FileHeader);
  std::vector<uint8_t> zeros(64, 0);
  write_exact(fd, &hdr, sizeof(hdr));

  for (uint32_t c = 0; c < num_cols; ++c) {
    uint64_t const aligned = align_up(cur_off, 64);
    if (aligned > cur_off) {
      write_exact(fd, zeros.data(), aligned - cur_off);
      cur_off = aligned;
    }
    metas[c].payload_offset = cur_off;
    metas[c].payload_size   = payloads[c].size();
    if (!payloads[c].empty()) {
      write_exact(fd, payloads[c].data(), payloads[c].size());
      cur_off += payloads[c].size();
    }
    if (metas[c].has_nulls) {
      uint64_t const n_aligned = align_up(cur_off, 64);
      if (n_aligned > cur_off) {
        write_exact(fd, zeros.data(), n_aligned - cur_off);
        cur_off = n_aligned;
      }
      metas[c].null_mask_offset = cur_off;
      write_exact(fd, null_masks[c].data(), null_masks[c].size());
      cur_off += null_masks[c].size();
    }
  }

  uint64_t const footer_off = align_up(cur_off, 64);
  if (footer_off > cur_off) {
    write_exact(fd, zeros.data(), footer_off - cur_off);
    cur_off = footer_off;
  }
  uint32_t const footer_size = static_cast<uint32_t>(metas.size() * sizeof(ColumnMeta));
  if (footer_size > 0) { write_exact(fd, metas.data(), footer_size); }

  hdr.footer_offset = footer_off;
  hdr.footer_size   = footer_size;
  CUDF_EXPECTS(::pwrite(fd, &hdr, sizeof(hdr), 0) == static_cast<ssize_t>(sizeof(hdr)),
               "lenc::write_table: failed to write header");
  ::close(fd);
}

ReadPlan plan_read(std::string const& path, std::vector<int32_t> const& col_indices)
{
  int const fd = ::open(path.c_str(), O_RDONLY);
  CUDF_EXPECTS(fd >= 0, "lenc::plan_read: failed to open file");

  ReadPlan plan{};
  pread_exact(fd, &plan.header, sizeof(FileHeader), 0);
  CUDF_EXPECTS(std::memcmp(plan.header.magic, kMagic, 8) == 0, "lenc::plan_read: bad magic");

  std::vector<ColumnMeta> all_metas(plan.header.num_cols);
  if (plan.header.num_cols > 0) {
    pread_exact(fd,
                all_metas.data(),
                plan.header.num_cols * sizeof(ColumnMeta),
                static_cast<off_t>(plan.header.footer_offset));
  }
  plan.col_indices = col_indices;
  plan.metas.reserve(col_indices.size());
  plan.packed_payload_offsets.reserve(col_indices.size());
  plan.packed_null_offsets.reserve(col_indices.size());
  plan.col_mins.assign(col_indices.size(), 0);
  plan.col_maxs.assign(col_indices.size(), 0);
  plan.col_has_minmax.assign(col_indices.size(), 0);

  uint32_t const num_chunks = (plan.header.num_rows + kChunkSize - 1) / kChunkSize;
  uint64_t packed_off       = 0;
  for (size_t idx = 0; idx < col_indices.size(); ++idx) {
    int32_t const ci = col_indices[idx];
    CUDF_EXPECTS(ci >= 0 && static_cast<uint32_t>(ci) < plan.header.num_cols,
                 "lenc::plan_read: column index out of range");
    ColumnMeta const& m = all_metas[static_cast<size_t>(ci)];
    plan.metas.push_back(m);
    packed_off = align_up(packed_off, 64);
    plan.packed_payload_offsets.push_back(packed_off);
    packed_off += m.payload_size;
    if (m.has_nulls) {
      packed_off = align_up(packed_off, 64);
      plan.packed_null_offsets.push_back(packed_off);
      packed_off += m.null_mask_size;
    } else {
      plan.packed_null_offsets.push_back(0);
    }

    // Track 3: Derive exact file-level [min, max] from FOR/SCALED_FLOAT/RLE mini-block headers
    // (zero data re-conversion required on existing .lenc files!)
    auto const enc = static_cast<Encoding>(m.encoding);
    if (plan.header.num_rows > 0 && !m.has_nulls &&
        (enc == Encoding::FOR_BITPACK || enc == Encoding::SCALED_FLOAT_FOR) && num_chunks > 0) {
      size_t const hdr_sz = num_chunks * (sizeof(int64_t) + sizeof(uint8_t));
      if (hdr_sz <= m.payload_size) {
        std::vector<uint8_t> tmp(hdr_sz);
        pread_exact(fd, tmp.data(), hdr_sz, static_cast<off_t>(m.payload_offset));
        auto const* mins = reinterpret_cast<int64_t const*>(tmp.data());
        auto const* bws  = tmp.data() + num_chunks * sizeof(int64_t);
        int64_t gmin     = std::numeric_limits<int64_t>::max();
        int64_t gmax     = std::numeric_limits<int64_t>::min();
        bool ok          = true;
        for (uint32_t c = 0; c < num_chunks; ++c) {
          if (bws[c] >= 63) {
            ok = false;
            break;
          }
          int64_t const lo  = mins[c];
          int64_t const rng = (bws[c] == 0) ? 0LL : static_cast<int64_t>((1ULL << bws[c]) - 1ULL);
          __int128 const hi = static_cast<__int128>(lo) + static_cast<__int128>(rng);
          if (hi > static_cast<__int128>(std::numeric_limits<int64_t>::max())) {
            ok = false;
            break;
          }
          if (lo < gmin) { gmin = lo; }
          if (static_cast<int64_t>(hi) > gmax) { gmax = static_cast<int64_t>(hi); }
        }
        if (ok && gmin <= gmax) {
          plan.col_mins[idx]       = gmin;
          plan.col_maxs[idx]       = gmax;
          plan.col_has_minmax[idx] = 1;
        }
      }
    }
  }
  ::close(fd);
  plan.total_packed_bytes = align_up(packed_off, 64);
  return plan;
}

void read_to_host(std::string const& path, ReadPlan const& plan, uint8_t* host_dst)
{
  if (plan.total_packed_bytes == 0 || plan.metas.empty()) { return; }
  int const fd = ::open(path.c_str(), O_RDONLY);
  CUDF_EXPECTS(fd >= 0, "lenc::read_to_host: failed to open file");
  for (size_t i = 0; i < plan.metas.size(); ++i) {
    ColumnMeta const& m = plan.metas[i];
    if (m.payload_size > 0) {
      pread_exact(fd,
                  host_dst + plan.packed_payload_offsets[i],
                  m.payload_size,
                  static_cast<off_t>(m.payload_offset));
    }
    if (m.has_nulls && m.null_mask_size > 0) {
      pread_exact(fd,
                  host_dst + plan.packed_null_offsets[i],
                  m.null_mask_size,
                  static_cast<off_t>(m.null_mask_offset));
    }
  }
  ::close(fd);
}

std::unique_ptr<cudf::table> decode_from_host(ReadPlan const& plan,
                                              uint8_t const* host_src,
                                              std::vector<uint8_t> const& dict_as_dictionary32,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  uint32_t const num_rows = plan.header.num_rows;
  size_t const ncols      = plan.metas.size();
  std::vector<std::unique_ptr<cudf::column>> out_cols;
  out_cols.reserve(ncols);

  rmm::device_buffer d_packed(plan.total_packed_bytes, stream, mr);
  if (plan.total_packed_bytes > 0) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      d_packed.data(), host_src, plan.total_packed_bytes, cudaMemcpyHostToDevice, stream.get()));
  }
  auto const* d_base = static_cast<uint8_t const*>(d_packed.data());

  constexpr uint32_t kBlock = 256;
  uint32_t const grid       = (num_rows + kBlock - 1) / kBlock;

  for (size_t i = 0; i < ncols; ++i) {
    ColumnMeta const& m  = plan.metas[i];
    auto const* d_pay    = d_base + plan.packed_payload_offsets[i];
    auto const enc       = static_cast<Encoding>(m.encoding);
    auto const ctype     = static_cast<ColType>(m.type);
    auto const null_cnt  = static_cast<cudf::size_type>(m.null_count);

    auto make_null_buf = [&]() {
      if (m.has_nulls && m.null_mask_size > 0) {
        auto mask = cudf::create_null_mask(
          static_cast<cudf::size_type>(num_rows), cudf::mask_state::UNINITIALIZED, stream, mr);
        CUDF_CUDA_TRY(cudaMemcpyAsync(mask.data(),
                                      d_base + plan.packed_null_offsets[i],
                                      m.null_mask_size,
                                      cudaMemcpyDeviceToDevice,
                                      stream.get()));
        return mask;
      }
      return cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr);
    };

    if (ctype == ColType::INT32 || ctype == ColType::INT64 || ctype == ColType::DATE32 ||
        ctype == ColType::BOOL8 || ctype == ColType::INT8 || ctype == ColType::INT16 ||
        ctype == ColType::TIMESTAMP_US || ctype == ColType::DECIMAL32 ||
        ctype == ColType::DECIMAL64) {
      cudf::data_type const dtype = [&]() {
        switch (ctype) {
          case ColType::INT32: return cudf::data_type{cudf::type_id::INT32};
          case ColType::INT64: return cudf::data_type{cudf::type_id::INT64};
          case ColType::DATE32: return cudf::data_type{cudf::type_id::TIMESTAMP_DAYS};
          case ColType::BOOL8: return cudf::data_type{cudf::type_id::BOOL8};
          case ColType::INT8: return cudf::data_type{cudf::type_id::INT8};
          case ColType::INT16: return cudf::data_type{cudf::type_id::INT16};
          case ColType::TIMESTAMP_US: return cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS};
          case ColType::DECIMAL32: return cudf::data_type{cudf::type_id::DECIMAL32, m.scale_exp10};
          case ColType::DECIMAL64: return cudf::data_type{cudf::type_id::DECIMAL64, m.scale_exp10};
          default: CUDF_FAIL("unreachable integral ColType");
        }
      }();
      auto col = cudf::make_fixed_width_column(
        dtype, static_cast<cudf::size_type>(num_rows), make_null_buf(), null_cnt, stream, mr);
      if (num_rows > 0) {
        size_t const esz      = cudf::size_of(dtype);
        uint32_t const nchunk = (num_rows + kChunkSize - 1) / kChunkSize;
        if (enc == Encoding::FOR_BITPACK) {
          auto fv = make_for_view(d_pay, num_rows);
          if (esz == 8) {
            for_bitpack_decode_kernel<int64_t><<<grid, kBlock, 0, stream.get()>>>(
              fv, static_cast<int64_t*>(col->mutable_view().head()), num_rows);
          } else if (esz == 4) {
            for_bitpack_decode_kernel<int32_t><<<grid, kBlock, 0, stream.get()>>>(
              fv, static_cast<int32_t*>(col->mutable_view().head()), num_rows);
          } else if (esz == 2) {
            for_bitpack_decode_kernel<int16_t><<<grid, kBlock, 0, stream.get()>>>(
              fv, static_cast<int16_t*>(col->mutable_view().head()), num_rows);
          } else {
            for_bitpack_decode_kernel<int8_t><<<grid, kBlock, 0, stream.get()>>>(
              fv, static_cast<int8_t*>(col->mutable_view().head()), num_rows);
          }
        } else if (enc == Encoding::DELTA_FOR_BITPACK) {
          auto dv = make_delta_for_view(d_pay, num_rows);
          if (esz == 8) {
            delta_for_decode_kernel<int64_t><<<nchunk, 256, 0, stream.get()>>>(
              dv, static_cast<int64_t*>(col->mutable_view().head()), num_rows);
          } else if (esz == 4) {
            delta_for_decode_kernel<int32_t><<<nchunk, 256, 0, stream.get()>>>(
              dv, static_cast<int32_t*>(col->mutable_view().head()), num_rows);
          } else if (esz == 2) {
            delta_for_decode_kernel<int16_t><<<nchunk, 256, 0, stream.get()>>>(
              dv, static_cast<int16_t*>(col->mutable_view().head()), num_rows);
          } else {
            delta_for_decode_kernel<int8_t><<<nchunk, 256, 0, stream.get()>>>(
              dv, static_cast<int8_t*>(col->mutable_view().head()), num_rows);
          }
        } else if (enc == Encoding::RLE_INT) {
          auto rv = make_rle_view(d_pay, num_rows, m.extra);
          if (esz == 8) {
            rle_decode_kernel<int64_t><<<grid, kBlock, 0, stream.get()>>>(
              rv, static_cast<int64_t*>(col->mutable_view().head()), num_rows);
          } else if (esz == 4) {
            rle_decode_kernel<int32_t><<<grid, kBlock, 0, stream.get()>>>(
              rv, static_cast<int32_t*>(col->mutable_view().head()), num_rows);
          } else if (esz == 2) {
            rle_decode_kernel<int16_t><<<grid, kBlock, 0, stream.get()>>>(
              rv, static_cast<int16_t*>(col->mutable_view().head()), num_rows);
          } else {
            rle_decode_kernel<int8_t><<<grid, kBlock, 0, stream.get()>>>(
              rv, static_cast<int8_t*>(col->mutable_view().head()), num_rows);
          }
        } else {
          CUDF_CUDA_TRY(cudaMemcpyAsync(col->mutable_view().head(),
                                        d_pay,
                                        num_rows * esz,
                                        cudaMemcpyDeviceToDevice,
                                        stream.get()));
        }
      }
      out_cols.push_back(std::move(col));
    } else if (ctype == ColType::FLOAT32 || ctype == ColType::FLOAT64) {
      bool const is_f32           = (ctype == ColType::FLOAT32);
      cudf::data_type const dtype = cudf::data_type{is_f32 ? cudf::type_id::FLOAT32 : cudf::type_id::FLOAT64};
      auto col                    = cudf::make_fixed_width_column(
        dtype, static_cast<cudf::size_type>(num_rows), make_null_buf(), null_cnt, stream, mr);
      if (num_rows > 0) {
        if (enc == Encoding::SCALED_FLOAT_FOR) {
          auto fv = make_for_view(d_pay, num_rows);
          double const inv =
            (m.scale_exp10 == 0) ? 1.0 : ((m.scale_exp10 == 2) ? 0.01 : 0.0001);
          if (is_f32) {
            scaled_float_for_decode_kernel<float><<<grid, kBlock, 0, stream.get()>>>(
              fv, inv, col->mutable_view().data<float>(), num_rows);
          } else {
            scaled_float_for_decode_kernel<double><<<grid, kBlock, 0, stream.get()>>>(
              fv, inv, col->mutable_view().data<double>(), num_rows);
          }
        } else if (enc == Encoding::NVCOMP_LZ4_FIXED) {
          Lz4PayloadHeader lh{};
          auto d_uncomp = nvcomp_lz4_decompress_device(
            host_src + plan.packed_payload_offsets[i], d_pay, lh, stream, mr);
          CUDF_CUDA_TRY(cudaMemcpyAsync(col->mutable_view().head(),
                                        d_uncomp.data(),
                                        num_rows * cudf::size_of(dtype),
                                        cudaMemcpyDeviceToDevice,
                                        stream.get()));
        } else {
          CUDF_CUDA_TRY(cudaMemcpyAsync(col->mutable_view().head(),
                                        d_pay,
                                        num_rows * cudf::size_of(dtype),
                                        cudaMemcpyDeviceToDevice,
                                        stream.get()));
        }
      }
      out_cols.push_back(std::move(col));
    } else if (ctype == ColType::STRING) {
      bool const want_dict32 =
        (i < dict_as_dictionary32.size()) && (dict_as_dictionary32[i] != 0) &&
        (enc == Encoding::DICT_BITPACK);
      if (enc == Encoding::DICT_BITPACK) {
        // Read tiny 16-byte DictPayloadHeader directly from host_src (already on host!)
        DictPayloadHeader dh{};
        std::memcpy(&dh, host_src + plan.packed_payload_offsets[i], sizeof(dh));
        auto const* d_dict_offs = reinterpret_cast<int32_t const*>(d_pay + sizeof(dh));
        auto const* d_dict_ch   = reinterpret_cast<char const*>(
          d_pay + sizeof(dh) + (dh.dict_size + 1) * sizeof(int32_t));
        auto codes_view = make_for_view(d_pay + dh.codes_offset, num_rows);

        if (want_dict32) {
          // Direct DICTIONARY32 output: zero string expansion!
          auto keys_offs = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                     static_cast<cudf::size_type>(dh.dict_size + 1),
                                                     cudf::mask_state::UNALLOCATED,
                                                     stream,
                                                     mr);
          CUDF_CUDA_TRY(cudaMemcpyAsync(keys_offs->mutable_view().data<int32_t>(),
                                        d_dict_offs,
                                        (dh.dict_size + 1) * sizeof(int32_t),
                                        cudaMemcpyDeviceToDevice,
                                        stream.get()));
          rmm::device_buffer keys_ch(dh.total_dict_bytes, stream, mr);
          if (dh.total_dict_bytes > 0) {
            CUDF_CUDA_TRY(cudaMemcpyAsync(keys_ch.data(),
                                          d_dict_ch,
                                          dh.total_dict_bytes,
                                          cudaMemcpyDeviceToDevice,
                                          stream.get()));
          }
          auto keys_col = cudf::make_strings_column(
            static_cast<cudf::size_type>(dh.dict_size),
            std::move(keys_offs),
            std::move(keys_ch),
            0,
            cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr));
          auto indices_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                       static_cast<cudf::size_type>(num_rows),
                                                       cudf::mask_state::UNALLOCATED,
                                                       stream,
                                                       mr);
          if (num_rows > 0) {
            for_bitpack_decode_kernel<int32_t><<<grid, kBlock, 0, stream.get()>>>(
              codes_view, indices_col->mutable_view().data<int32_t>(), num_rows);
          }
          out_cols.push_back(cudf::make_dictionary_column(
            std::move(keys_col), std::move(indices_col), make_null_buf(), null_cnt));
        } else {
          // Standard Arrow STRING output via 2-pass parallel GPU decode
          auto null_buf = make_null_buf();
          auto const* d_null =
            (m.has_nulls && m.null_mask_size > 0)
              ? reinterpret_cast<cudf::bitmask_type const*>(null_buf.data())
              : nullptr;
          auto out_offs = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                    static_cast<cudf::size_type>(num_rows + 1),
                                                    cudf::mask_state::UNALLOCATED,
                                                    stream,
                                                    mr);
          auto* d_offs  = out_offs->mutable_view().data<int32_t>();
          uint32_t const grid_p1 = (num_rows + 1 + kBlock - 1) / kBlock;
          dict_lengths_kernel<<<grid_p1, kBlock, 0, stream.get()>>>(
            codes_view, d_dict_offs, d_null, d_offs, num_rows);
          thrust::exclusive_scan(
            rmm::exec_policy_nosync(stream), d_offs, d_offs + num_rows + 1, d_offs);

          int32_t total_chars = 0;
          CUDF_CUDA_TRY(cudaMemcpyAsync(
            &total_chars, d_offs + num_rows, sizeof(int32_t), cudaMemcpyDeviceToHost, stream.get()));
          stream.sync();

          rmm::device_buffer out_chars(static_cast<size_t>(total_chars), stream, mr);
          if (num_rows > 0 && total_chars > 0) {
            dict_copy_chars_kernel<<<grid, kBlock, 0, stream.get()>>>(codes_view,
                                                                        d_dict_offs,
                                                                        d_dict_ch,
                                                                        d_null,
                                                                        d_offs,
                                                                        static_cast<char*>(out_chars.data()),
                                                                        num_rows);
          }
          out_cols.push_back(cudf::make_strings_column(static_cast<cudf::size_type>(num_rows),
                                                       std::move(out_offs),
                                                       std::move(out_chars),
                                                       null_cnt,
                                                       std::move(null_buf)));
        }
      } else if (enc == Encoding::NVCOMP_LZ4_STR) {
        auto null_buf          = make_null_buf();
        size_t const off_bytes = (num_rows + 1) * sizeof(int32_t);
        Lz4PayloadHeader lh{};
        auto d_uncomp = nvcomp_lz4_decompress_device(
          host_src + plan.packed_payload_offsets[i], d_pay, lh, stream, mr);
        auto out_offs = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                  static_cast<cudf::size_type>(num_rows + 1),
                                                  cudf::mask_state::UNALLOCATED,
                                                  stream,
                                                  mr);
        CUDF_CUDA_TRY(cudaMemcpyAsync(out_offs->mutable_view().data<int32_t>(),
                                      d_uncomp.data(),
                                      off_bytes,
                                      cudaMemcpyDeviceToDevice,
                                      stream.get()));
        rmm::device_buffer out_chars(lh.total_chars, stream, mr);
        if (lh.total_chars > 0) {
          CUDF_CUDA_TRY(cudaMemcpyAsync(out_chars.data(),
                                        static_cast<uint8_t const*>(d_uncomp.data()) + off_bytes,
                                        lh.total_chars,
                                        cudaMemcpyDeviceToDevice,
                                        stream.get()));
        }
        out_cols.push_back(cudf::make_strings_column(static_cast<cudf::size_type>(num_rows),
                                                     std::move(out_offs),
                                                     std::move(out_chars),
                                                     null_cnt,
                                                     std::move(null_buf)));
      } else {
        // RAW_STRING fallback
        auto null_buf          = make_null_buf();
        size_t const off_bytes = (num_rows + 1) * sizeof(int32_t);
        auto out_offs          = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                  static_cast<cudf::size_type>(num_rows + 1),
                                                  cudf::mask_state::UNALLOCATED,
                                                  stream,
                                                  mr);
        CUDF_CUDA_TRY(cudaMemcpyAsync(out_offs->mutable_view().data<int32_t>(),
                                      d_pay,
                                      off_bytes,
                                      cudaMemcpyDeviceToDevice,
                                      stream.get()));
        rmm::device_buffer out_chars(m.extra, stream, mr);
        if (m.extra > 0) {
          CUDF_CUDA_TRY(cudaMemcpyAsync(
            out_chars.data(), d_pay + off_bytes, m.extra, cudaMemcpyDeviceToDevice, stream.get()));
        }
        out_cols.push_back(cudf::make_strings_column(static_cast<cudf::size_type>(num_rows),
                                                     std::move(out_offs),
                                                     std::move(out_chars),
                                                     null_cnt,
                                                     std::move(null_buf)));
      }
    }
  }

  stream.sync();
  return std::make_unique<cudf::table>(std::move(out_cols));
}

std::unique_ptr<cudf::table> decode_and_eval_fused(ReadPlan const& plan,
                                                   uint8_t const* host_src,
                                                   int32_t expr_kind,
                                                   int64_t /*param0*/,
                                                   cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr)
{
  uint32_t const num_rows = plan.header.num_rows;
  rmm::device_buffer d_packed(plan.total_packed_bytes, stream, mr);
  if (plan.total_packed_bytes > 0) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      d_packed.data(), host_src, plan.total_packed_bytes, cudaMemcpyHostToDevice, stream.get()));
  }
  auto const* d_base        = static_cast<uint8_t const*>(d_packed.data());
  constexpr uint32_t kBlock = 256;
  uint32_t const grid       = (num_rows + kBlock - 1) / kBlock;

  std::vector<std::unique_ptr<cudf::column>> out_cols;
  if (expr_kind == 0) {
    // W1a_1: projected cols [c0, c1, c2, c3] -> 1 INT64 output
    CUDF_EXPECTS(plan.metas.size() == 4, "fused W1a_1 requires 4 projected cols");
    auto c0v = make_for_view(d_base + plan.packed_payload_offsets[0], num_rows);
    auto c1v = make_for_view(d_base + plan.packed_payload_offsets[1], num_rows);
    auto c2v = make_for_view(d_base + plan.packed_payload_offsets[2], num_rows);
    auto c3v = make_for_view(d_base + plan.packed_payload_offsets[3], num_rows);
    auto out = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                         static_cast<cudf::size_type>(num_rows),
                                         cudf::mask_state::UNALLOCATED,
                                         stream,
                                         mr);
    if (num_rows > 0) {
      fused_w1a_1_kernel<<<grid, kBlock, 0, stream.get()>>>(
        c0v, c1v, c2v, c3v, out->mutable_view().data<int64_t>(), num_rows);
    }
    out_cols.push_back(std::move(out));
  } else if (expr_kind == 1) {
    // W1a_8: projected cols [c0, c1, c2, c3, c4] -> 8 INT64 outputs
    CUDF_EXPECTS(plan.metas.size() == 5, "fused W1a_8 requires 5 projected cols");
    auto c0v = make_for_view(d_base + plan.packed_payload_offsets[0], num_rows);
    auto c1v = make_for_view(d_base + plan.packed_payload_offsets[1], num_rows);
    auto c2v = make_for_view(d_base + plan.packed_payload_offsets[2], num_rows);
    auto c3v = make_for_view(d_base + plan.packed_payload_offsets[3], num_rows);
    auto c4v = make_for_view(d_base + plan.packed_payload_offsets[4], num_rows);
    Ptr8 ptrs{};
    for (int k = 0; k < 8; ++k) {
      auto col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                           static_cast<cudf::size_type>(num_rows),
                                           cudf::mask_state::UNALLOCATED,
                                           stream,
                                           mr);
      ptrs.p[k] = col->mutable_view().data<int64_t>();
      out_cols.push_back(std::move(col));
    }
    if (num_rows > 0) {
      fused_w1a_8_kernel<<<grid, kBlock, 0, stream.get()>>>(
        c0v, c1v, c2v, c3v, c4v, ptrs, num_rows);
    }
  } else if (expr_kind == 2 || expr_kind == 3) {
    // W1b_8 (kind=2) or W1b_24 (kind=3): projected cols [c0, c1, c2, c3, c4, s1] -> 1 INT64 output
    CUDF_EXPECTS(plan.metas.size() == 6, "fused W1b requires 6 projected cols [c0..c4, s1]");
    auto c0v = make_for_view(d_base + plan.packed_payload_offsets[0], num_rows);
    auto c1v = make_for_view(d_base + plan.packed_payload_offsets[1], num_rows);
    auto c2v = make_for_view(d_base + plan.packed_payload_offsets[2], num_rows);
    auto c3v = make_for_view(d_base + plan.packed_payload_offsets[3], num_rows);
    auto c4v = make_for_view(d_base + plan.packed_payload_offsets[4], num_rows);

    DictPayloadHeader dh{};
    uint8_t const* h_s1 = host_src + plan.packed_payload_offsets[5];
    std::memcpy(&dh, h_s1, sizeof(dh));
    auto s1_codes_v = make_for_view(d_base + plan.packed_payload_offsets[5] + dh.codes_offset, num_rows);

    DictBranchLut24 lut{};
    int const branches = (expr_kind == 2) ? 8 : 24;
    for (int i = 0; i < 24; ++i) {
      if (i < branches) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "m%02d", i);
        lut.target_code[i] = find_dict_code_host(h_s1, std::string_view(buf, 3));
      } else {
        lut.target_code[i] = -2;
      }
    }
    auto out = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                         static_cast<cudf::size_type>(num_rows),
                                         cudf::mask_state::UNALLOCATED,
                                         stream,
                                         mr);
    if (num_rows > 0) {
      if (expr_kind == 2) {
        fused_w1b_case_kernel<8><<<grid, kBlock, 0, stream.get()>>>(
          c0v, c1v, c2v, c3v, c4v, s1_codes_v, lut, out->mutable_view().data<int64_t>(), num_rows);
      } else {
        fused_w1b_case_kernel<24><<<grid, kBlock, 0, stream.get()>>>(
          c0v, c1v, c2v, c3v, c4v, s1_codes_v, lut, out->mutable_view().data<int64_t>(), num_rows);
      }
    }
    out_cols.push_back(std::move(out));
  } else if (expr_kind == 4) {
    // Ultimate ceiling: W1a_1 fused decode + eval + sum reduction -> 1-row INT64 output!
    CUDF_EXPECTS(plan.metas.size() == 4, "fused W1a_1_sum requires 4 projected cols");
    auto c0v = make_for_view(d_base + plan.packed_payload_offsets[0], num_rows);
    auto c1v = make_for_view(d_base + plan.packed_payload_offsets[1], num_rows);
    auto c2v = make_for_view(d_base + plan.packed_payload_offsets[2], num_rows);
    auto c3v = make_for_view(d_base + plan.packed_payload_offsets[3], num_rows);
    auto out = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT64},
                                         1,
                                         cudf::mask_state::UNALLOCATED,
                                         stream,
                                         mr);
    CUDF_CUDA_TRY(cudaMemsetAsync(out->mutable_view().data<int64_t>(), 0, sizeof(int64_t), stream.get()));
    if (num_rows > 0) {
      uint32_t const blocks = std::min(grid, 1024u);
      fused_w1a_1_sum_kernel<<<blocks, kBlock, 0, stream.get()>>>(
        c0v, c1v, c2v, c3v, reinterpret_cast<unsigned long long*>(out->mutable_view().data<int64_t>()), num_rows);
    }
    out_cols.push_back(std::move(out));
  } else {
    CUDF_FAIL("lenc::decode_and_eval_fused: unknown expr_kind");
  }

  stream.sync();
  return std::make_unique<cudf::table>(std::move(out_cols));
}

namespace {

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

// ---------------- Direction B: Generic Dynamic LTO Decode-in-Register ----------------

namespace {

constexpr int kMaxLtoCols = 16;

struct LencLtoColDesc {
  // FOR+BitPack (also used for SCALED_FLOAT_FOR and DICT_BITPACK codes)
  int64_t const* min_vals;
  uint8_t const* bit_widths;
  uint32_t const* chunk_word_offsets;
  uint64_t const* packed_words;
  // RLE
  uint32_t const* rle_chunk_start;
  uint32_t const* rle_run_ends;
  int64_t const* rle_run_vals;
  uint32_t rle_num_runs;
  // DICT canonical remap & RAW pointer & optional null mask
  int32_t const* dict_remap;
  uint32_t dict_size;
  uint8_t const* raw_ptr;
  cudf::bitmask_type const* null_mask;
  double inv_scale;
};

struct LencLtoBatchDesc {
  int32_t const* selected_rows;  // nullptr -> full scan; non-null -> late-materialized gather!
  LencLtoColDesc cols[kMaxLtoCols];
};

// Device CUDA source compiled ONCE into an LTO-IR module via cudf::compile_cuda_to_lto_ir()
// and linked into ANY dynamically generated Spark AST-JIT kernel via nvJitLink!
char const* const kLencLtoDeviceSrc = R"CUDA(
#include <cudf/errc.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/timestamps.hpp>
#include <cstdint>

struct LencLtoColDesc {
  int64_t const* min_vals;
  uint8_t const* bit_widths;
  uint32_t const* chunk_word_offsets;
  uint64_t const* packed_words;
  uint32_t const* rle_chunk_start;
  uint32_t const* rle_run_ends;
  int64_t const* rle_run_vals;
  uint32_t rle_num_runs;
  int32_t const* dict_remap;
  uint32_t dict_size;
  uint8_t const* raw_ptr;
  cudf::bitmask_type const* null_mask;
  double inv_scale;
};

struct LencLtoBatchDesc {
  int32_t const* selected_rows;
  LencLtoColDesc cols[16];
};

__device__ __forceinline__ uint32_t lenc_resolve_row(LencLtoBatchDesc const* desc, cudf::size_type row)
{
  return (desc->selected_rows != nullptr) ? static_cast<uint32_t>(desc->selected_rows[row])
                                          : static_cast<uint32_t>(row);
}

__device__ __forceinline__ bool lenc_is_null(LencLtoColDesc const& col, uint32_t row)
{
  if (col.null_mask == nullptr) { return false; }
  uint32_t const word = col.null_mask[row >> 5];
  return ((word >> (row & 31u)) & 1u) == 0u;
}

__device__ __forceinline__ int64_t lenc_unpack_for(LencLtoColDesc const& col, uint32_t row)
{
  uint32_t const c   = row >> 10;
  uint32_t const b   = col.bit_widths[c];
  int64_t const base = col.min_vals[c];
  if (b == 0u) { return base; }
  uint32_t const local    = row & 1023u;
  uint32_t const bit_pos  = local * b;
  uint32_t const word_idx = col.chunk_word_offsets[c] + (bit_pos >> 6);
  uint32_t const bit_off  = bit_pos & 63u;
  uint64_t val            = col.packed_words[word_idx] >> bit_off;
  if (bit_off + b > 64u) { val |= col.packed_words[word_idx + 1] << (64u - bit_off); }
  uint64_t const mask = (b == 64u) ? ~0ULL : ((1ULL << b) - 1ULL);
  return base + static_cast<int64_t>(val & mask);
}

__device__ __forceinline__ int64_t lenc_unpack_rle(LencLtoColDesc const& col, uint32_t row)
{
  uint32_t idx = col.rle_chunk_start[row >> 10];
  while (idx + 1u < col.rle_num_runs && col.rle_run_ends[idx] <= row) { ++idx; }
  return col.rle_run_vals[idx];
}

extern "C" __device__ cudf::errc lenc_decode_for_i64(
  int64_t* out, void* ud, cudf::size_type row, int32_t slot)
{
  auto const* d = static_cast<LencLtoBatchDesc const*>(ud);
  uint32_t const r = lenc_resolve_row(d, row);
  auto const& col = d->cols[slot];
  if (lenc_is_null(col, r)) { return cudf::errc::USER_RESERVED; }
  *out = lenc_unpack_for(col, r);
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc lenc_decode_for_i32(
  int32_t* out, void* ud, cudf::size_type row, int32_t slot)
{
  auto const* d = static_cast<LencLtoBatchDesc const*>(ud);
  uint32_t const r = lenc_resolve_row(d, row);
  auto const& col = d->cols[slot];
  if (lenc_is_null(col, r)) { return cudf::errc::USER_RESERVED; }
  *out = static_cast<int32_t>(lenc_unpack_for(col, r));
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc lenc_decode_for_date32(
  cudf::timestamp_D* out, void* ud, cudf::size_type row, int32_t slot)
{
  auto const* d = static_cast<LencLtoBatchDesc const*>(ud);
  uint32_t const r = lenc_resolve_row(d, row);
  auto const& col = d->cols[slot];
  if (lenc_is_null(col, r)) { return cudf::errc::USER_RESERVED; }
  *out = cudf::timestamp_D{cudf::duration_D{static_cast<int32_t>(lenc_unpack_for(col, r))}};
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc lenc_decode_scaled_f32(
  float* out, void* ud, cudf::size_type row, int32_t slot)
{
  auto const* d = static_cast<LencLtoBatchDesc const*>(ud);
  uint32_t const r = lenc_resolve_row(d, row);
  auto const& col = d->cols[slot];
  if (lenc_is_null(col, r)) { return cudf::errc::USER_RESERVED; }
  int64_t const iv = lenc_unpack_for(col, r);
  *out             = static_cast<float>(static_cast<double>(iv) * col.inv_scale);
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc lenc_decode_scaled_f64(
  double* out, void* ud, cudf::size_type row, int32_t slot)
{
  auto const* d = static_cast<LencLtoBatchDesc const*>(ud);
  uint32_t const r = lenc_resolve_row(d, row);
  auto const& col = d->cols[slot];
  if (lenc_is_null(col, r)) { return cudf::errc::USER_RESERVED; }
  int64_t const iv = lenc_unpack_for(col, r);
  *out             = static_cast<double>(iv) * col.inv_scale;
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc lenc_decode_rle_i64(
  int64_t* out, void* ud, cudf::size_type row, int32_t slot)
{
  auto const* d = static_cast<LencLtoBatchDesc const*>(ud);
  uint32_t const r = lenc_resolve_row(d, row);
  auto const& col = d->cols[slot];
  if (lenc_is_null(col, r)) { return cudf::errc::USER_RESERVED; }
  *out = lenc_unpack_rle(col, r);
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc lenc_decode_rle_i32(
  int32_t* out, void* ud, cudf::size_type row, int32_t slot)
{
  auto const* d = static_cast<LencLtoBatchDesc const*>(ud);
  uint32_t const r = lenc_resolve_row(d, row);
  auto const& col = d->cols[slot];
  if (lenc_is_null(col, r)) { return cudf::errc::USER_RESERVED; }
  *out = static_cast<int32_t>(lenc_unpack_rle(col, r));
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc lenc_decode_dict_code(
  int32_t* out, void* ud, cudf::size_type row, int32_t slot)
{
  auto const* d = static_cast<LencLtoBatchDesc const*>(ud);
  uint32_t const r = lenc_resolve_row(d, row);
  auto const& col = d->cols[slot];
  if (lenc_is_null(col, r)) { return cudf::errc::USER_RESERVED; }
  uint32_t const code = static_cast<uint32_t>(lenc_unpack_for(col, r));
  *out                = (code < col.dict_size) ? col.dict_remap[code] : -1;
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc lenc_decode_raw_i64(
  int64_t* out, void* ud, cudf::size_type row, int32_t slot)
{
  auto const* d = static_cast<LencLtoBatchDesc const*>(ud);
  uint32_t const r = lenc_resolve_row(d, row);
  auto const& col = d->cols[slot];
  if (lenc_is_null(col, r)) { return cudf::errc::USER_RESERVED; }
  *out = reinterpret_cast<int64_t const*>(col.raw_ptr)[r];
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc lenc_decode_raw_i32(
  int32_t* out, void* ud, cudf::size_type row, int32_t slot)
{
  auto const* d = static_cast<LencLtoBatchDesc const*>(ud);
  uint32_t const r = lenc_resolve_row(d, row);
  auto const& col = d->cols[slot];
  if (lenc_is_null(col, r)) { return cudf::errc::USER_RESERVED; }
  *out = reinterpret_cast<int32_t const*>(col.raw_ptr)[r];
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc lenc_decode_raw_date32(
  cudf::timestamp_D* out, void* ud, cudf::size_type row, int32_t slot)
{
  auto const* d = static_cast<LencLtoBatchDesc const*>(ud);
  uint32_t const r = lenc_resolve_row(d, row);
  auto const& col = d->cols[slot];
  if (lenc_is_null(col, r)) { return cudf::errc::USER_RESERVED; }
  *out = cudf::timestamp_D{cudf::duration_D{reinterpret_cast<int32_t const*>(col.raw_ptr)[r]}};
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc lenc_decode_raw_f32(
  float* out, void* ud, cudf::size_type row, int32_t slot)
{
  auto const* d = static_cast<LencLtoBatchDesc const*>(ud);
  uint32_t const r = lenc_resolve_row(d, row);
  auto const& col = d->cols[slot];
  if (lenc_is_null(col, r)) { return cudf::errc::USER_RESERVED; }
  *out = reinterpret_cast<float const*>(col.raw_ptr)[r];
  return cudf::errc::SUCCESS;
}

extern "C" __device__ cudf::errc lenc_decode_raw_f64(
  double* out, void* ud, cudf::size_type row, int32_t slot)
{
  auto const* d = static_cast<LencLtoBatchDesc const*>(ud);
  uint32_t const r = lenc_resolve_row(d, row);
  auto const& col = d->cols[slot];
  if (lenc_is_null(col, r)) { return cudf::errc::USER_RESERVED; }
  *out = reinterpret_cast<double const*>(col.raw_ptr)[r];
  return cudf::errc::SUCCESS;
}

#define LENC_SLOT_WRAPPERS(S) \
  extern "C" __device__ cudf::errc lenc_decode_for_i64_s##S(int64_t* o, void* u, cudf::size_type r) { return lenc_decode_for_i64(o, u, r, S); } \
  extern "C" __device__ cudf::errc lenc_decode_for_i32_s##S(int32_t* o, void* u, cudf::size_type r) { return lenc_decode_for_i32(o, u, r, S); } \
  extern "C" __device__ cudf::errc lenc_decode_for_date32_s##S(cudf::timestamp_D* o, void* u, cudf::size_type r) { return lenc_decode_for_date32(o, u, r, S); } \
  extern "C" __device__ cudf::errc lenc_decode_scaled_f32_s##S(float* o, void* u, cudf::size_type r) { return lenc_decode_scaled_f32(o, u, r, S); } \
  extern "C" __device__ cudf::errc lenc_decode_scaled_f64_s##S(double* o, void* u, cudf::size_type r) { return lenc_decode_scaled_f64(o, u, r, S); } \
  extern "C" __device__ cudf::errc lenc_decode_rle_i64_s##S(int64_t* o, void* u, cudf::size_type r) { return lenc_decode_rle_i64(o, u, r, S); } \
  extern "C" __device__ cudf::errc lenc_decode_rle_i32_s##S(int32_t* o, void* u, cudf::size_type r) { return lenc_decode_rle_i32(o, u, r, S); } \
  extern "C" __device__ cudf::errc lenc_decode_dict_code_s##S(int32_t* o, void* u, cudf::size_type r) { return lenc_decode_dict_code(o, u, r, S); } \
  extern "C" __device__ cudf::errc lenc_decode_raw_i64_s##S(int64_t* o, void* u, cudf::size_type r) { return lenc_decode_raw_i64(o, u, r, S); } \
  extern "C" __device__ cudf::errc lenc_decode_raw_i32_s##S(int32_t* o, void* u, cudf::size_type r) { return lenc_decode_raw_i32(o, u, r, S); } \
  extern "C" __device__ cudf::errc lenc_decode_raw_date32_s##S(cudf::timestamp_D* o, void* u, cudf::size_type r) { return lenc_decode_raw_date32(o, u, r, S); } \
  extern "C" __device__ cudf::errc lenc_decode_raw_f32_s##S(float* o, void* u, cudf::size_type r) { return lenc_decode_raw_f32(o, u, r, S); } \
  extern "C" __device__ cudf::errc lenc_decode_raw_f64_s##S(double* o, void* u, cudf::size_type r) { return lenc_decode_raw_f64(o, u, r, S); }

LENC_SLOT_WRAPPERS(0)  LENC_SLOT_WRAPPERS(1)  LENC_SLOT_WRAPPERS(2)  LENC_SLOT_WRAPPERS(3)
LENC_SLOT_WRAPPERS(4)  LENC_SLOT_WRAPPERS(5)  LENC_SLOT_WRAPPERS(6)  LENC_SLOT_WRAPPERS(7)
LENC_SLOT_WRAPPERS(8)  LENC_SLOT_WRAPPERS(9)  LENC_SLOT_WRAPPERS(10) LENC_SLOT_WRAPPERS(11)
LENC_SLOT_WRAPPERS(12) LENC_SLOT_WRAPPERS(13) LENC_SLOT_WRAPPERS(14) LENC_SLOT_WRAPPERS(15)
)CUDA";

cudf::compiled_fragment const& get_lenc_lto_fragment()
{
  static cudf::compiled_fragment const frag =
    cudf::compile_cuda_to_lto_ir(kLencLtoDeviceSrc, {}, {}, {}, false);
  return frag;
}

}  // namespace

std::unique_ptr<cudf::transform_program> compile_lto_program(
  std::vector<std::reference_wrapper<cudf::ast::expression const>> const& expressions,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto const& frag          = get_lenc_lto_fragment();
  cudf::lto_fragment view[] = {frag.view()};
  cudf::table_view empty_table{};
  return std::make_unique<cudf::transform_program>(
    empty_table, expressions, cudf::lto_fragments{view}, stream, mr);
}

std::unique_ptr<cudf::table> run_lto_program(
  cudf::transform_program& program,
  ReadPlan const& plan,
  uint8_t const* host_src,
  std::vector<std::vector<std::string>> const& dict_literals_per_col,
  cudf::column_view const* selected_rows,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  uint32_t const num_rows = plan.header.num_rows;
  uint32_t const out_rows =
    (selected_rows != nullptr) ? static_cast<uint32_t>(selected_rows->size()) : num_rows;
  size_t const ncols = plan.metas.size();
  CUDF_EXPECTS(ncols <= static_cast<size_t>(kMaxLtoCols), "lenc LTO: max 16 projected cols");

  rmm::device_buffer d_packed(plan.total_packed_bytes, stream, mr);
  if (plan.total_packed_bytes > 0) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      d_packed.data(), host_src, plan.total_packed_bytes, cudaMemcpyHostToDevice, stream.get()));
  }
  auto const* d_base = static_cast<uint8_t const*>(d_packed.data());

  LencLtoBatchDesc h_desc{};
  h_desc.selected_rows =
    (selected_rows != nullptr && selected_rows->size() > 0)
      ? (selected_rows->data<int32_t>() + selected_rows->offset())
      : nullptr;
  std::vector<rmm::device_buffer> d_temps;

  for (size_t i = 0; i < ncols; ++i) {
    ColumnMeta const& m = plan.metas[i];
    auto const* d_pay   = d_base + plan.packed_payload_offsets[i];
    auto const enc      = static_cast<Encoding>(m.encoding);
    auto const ctype    = static_cast<ColType>(m.type);
    LencLtoColDesc& cd  = h_desc.cols[i];
    if (m.has_nulls && m.null_mask_size > 0) {
      cd.null_mask = reinterpret_cast<cudf::bitmask_type const*>(d_base + plan.packed_null_offsets[i]);
    }
    if (enc == Encoding::FOR_BITPACK || enc == Encoding::SCALED_FLOAT_FOR) {
      auto fv               = make_for_view(d_pay, num_rows);
      cd.min_vals           = fv.min_vals;
      cd.bit_widths         = fv.bit_widths;
      cd.chunk_word_offsets = fv.chunk_word_offsets;
      cd.packed_words       = fv.packed_words;
      cd.inv_scale =
        (m.scale_exp10 == 0) ? 1.0 : ((m.scale_exp10 == 2) ? 0.01 : 0.0001);
    } else if (enc == Encoding::RLE_INT) {
      auto rv            = make_rle_view(d_pay, num_rows, m.extra);
      cd.rle_chunk_start = rv.chunk_start_run;
      cd.rle_run_ends    = rv.run_ends;
      cd.rle_run_vals    = rv.run_vals;
      cd.rle_num_runs    = rv.num_runs;
    } else if (enc == Encoding::DELTA_FOR_BITPACK) {
      uint32_t const nchunk = (num_rows + kChunkSize - 1) / kChunkSize;
      auto dv               = make_delta_for_view(d_pay, num_rows);
      bool const is64       = (ctype == ColType::INT64 || ctype == ColType::TIMESTAMP_US ||
                         ctype == ColType::DECIMAL64);
      rmm::device_buffer d_dec(static_cast<size_t>(num_rows) * (is64 ? 8 : 4), stream, mr);
      if (num_rows > 0) {
        if (is64) {
          delta_for_decode_kernel<int64_t><<<nchunk, 256, 0, stream.get()>>>(
            dv, static_cast<int64_t*>(d_dec.data()), num_rows);
        } else {
          delta_for_decode_kernel<int32_t><<<nchunk, 256, 0, stream.get()>>>(
            dv, static_cast<int32_t*>(d_dec.data()), num_rows);
        }
      }
      cd.raw_ptr = static_cast<uint8_t const*>(d_dec.data());
      d_temps.push_back(std::move(d_dec));
    } else if (enc == Encoding::NVCOMP_LZ4_FIXED) {
      Lz4PayloadHeader lh{};
      auto d_dec = nvcomp_lz4_decompress_device(
        host_src + plan.packed_payload_offsets[i], d_pay, lh, stream, mr);
      cd.raw_ptr = static_cast<uint8_t const*>(d_dec.data());
      d_temps.push_back(std::move(d_dec));
    } else if (enc == Encoding::DICT_BITPACK) {
      DictPayloadHeader dh{};
      uint8_t const* h_pay = host_src + plan.packed_payload_offsets[i];
      std::memcpy(&dh, h_pay, sizeof(dh));
      auto fv               = make_for_view(d_pay + dh.codes_offset, num_rows);
      cd.min_vals           = fv.min_vals;
      cd.bit_widths         = fv.bit_widths;
      cd.chunk_word_offsets = fv.chunk_word_offsets;
      cd.packed_words       = fv.packed_words;
      cd.dict_size          = dh.dict_size;

      // Resolve dictionary keys -> canonical query literal IDs on host (~2 us!)
      std::vector<int32_t> h_remap(dh.dict_size, -1);
      if (i < dict_literals_per_col.size() && !dict_literals_per_col[i].empty()) {
        std::unordered_map<std::string_view, int32_t> lit_map;
        auto const& lits = dict_literals_per_col[i];
        lit_map.reserve(lits.size() * 2 + 1);
        for (size_t l = 0; l < lits.size(); ++l) {
          lit_map.emplace(std::string_view{lits[l]}, static_cast<int32_t>(l));
        }
        auto const* dict_offs = reinterpret_cast<int32_t const*>(h_pay + sizeof(dh));
        auto const* dict_ch   = reinterpret_cast<char const*>(
          h_pay + sizeof(dh) + (dh.dict_size + 1) * sizeof(int32_t));
        for (uint32_t k = 0; k < dh.dict_size; ++k) {
          int32_t const s = dict_offs[k];
          int32_t const e = dict_offs[k + 1];
          auto it         = lit_map.find(std::string_view(dict_ch + s, static_cast<size_t>(e - s)));
          if (it != lit_map.end()) { h_remap[k] = it->second; }
        }
      }
      rmm::device_buffer d_rm(std::max<size_t>(dh.dict_size, 1) * sizeof(int32_t), stream, mr);
      if (dh.dict_size > 0) {
        CUDF_CUDA_TRY(cudaMemcpyAsync(
          d_rm.data(), h_remap.data(), dh.dict_size * sizeof(int32_t), cudaMemcpyHostToDevice, stream.get()));
      }
      cd.dict_remap = static_cast<int32_t const*>(d_rm.data());
      d_temps.push_back(std::move(d_rm));
    } else if (enc == Encoding::RAW_FIXED) {
      cd.raw_ptr = d_pay;
    }
  }

  rmm::device_buffer d_desc(sizeof(LencLtoBatchDesc), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    d_desc.data(), &h_desc, sizeof(LencLtoBatchDesc), cudaMemcpyHostToDevice, stream.get()));

  auto result = program.run_with_user_data(
    d_desc.data(), static_cast<cudf::size_type>(out_rows), stream, mr);
  stream.sync();
  return result;
}

std::unique_ptr<cudf::column> compact_mask_indices(cudf::column_view const& bool_mask,
                                                   cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(bool_mask.type().id() == cudf::type_id::BOOL8, "compact_mask_indices expects BOOL8");
  auto const n = bool_mask.size();
  rmm::device_uvector<int32_t> tmp(static_cast<size_t>(n), stream, mr);
  MaskIsTruePred pred{bool_mask.data<bool>() + bool_mask.offset(),
                      bool_mask.nullable() ? bool_mask.null_mask() : nullptr};
  auto end_it = thrust::copy_if(rmm::exec_policy_nosync(stream),
                                thrust::make_counting_iterator<int32_t>(0),
                                thrust::make_counting_iterator<int32_t>(n),
                                tmp.begin(),
                                pred);
  auto const m = static_cast<cudf::size_type>(end_it - tmp.begin());
  tmp.resize(static_cast<size_t>(m), stream);
  return std::make_unique<cudf::column>(
    cudf::data_type{cudf::type_id::INT32},
    m,
    tmp.release(),
    cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr),
    0);
}

std::unique_ptr<cudf::table> decode_from_host_selected(
  ReadPlan const& plan,
  uint8_t const* host_src,
  std::vector<uint8_t> const& dict_as_dictionary32,
  cudf::column_view const& selected_rows,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  uint32_t const num_rows = plan.header.num_rows;
  uint32_t const m_rows   = static_cast<uint32_t>(selected_rows.size());
  size_t const ncols      = plan.metas.size();

  // If > 60% of rows survive or any column is non-random-access, check per column
  rmm::device_buffer d_packed(plan.total_packed_bytes, stream, mr);
  if (plan.total_packed_bytes > 0) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      d_packed.data(), host_src, plan.total_packed_bytes, cudaMemcpyHostToDevice, stream.get()));
  }
  auto const* d_base = static_cast<uint8_t const*>(d_packed.data());
  auto const* d_sel  = (m_rows > 0) ? (selected_rows.data<int32_t>() + selected_rows.offset()) : nullptr;

  constexpr uint32_t kBlock = 256;
  uint32_t const grid       = (m_rows + kBlock - 1) / kBlock;
  std::vector<std::unique_ptr<cudf::column>> out_cols;
  out_cols.reserve(ncols);

  for (size_t i = 0; i < ncols; ++i) {
    ColumnMeta const& m  = plan.metas[i];
    auto const* d_pay    = d_base + plan.packed_payload_offsets[i];
    auto const enc       = static_cast<Encoding>(m.encoding);
    auto const ctype     = static_cast<ColType>(m.type);
    auto const* d_src_nm =
      (m.has_nulls && m.null_mask_size > 0)
        ? reinterpret_cast<cudf::bitmask_type const*>(d_base + plan.packed_null_offsets[i])
        : nullptr;

    auto gather_null_buf = [&](cudf::size_type& out_null_cnt) {
      if (d_src_nm != nullptr && m_rows > 0) {
        auto mask = cudf::create_null_mask(
          static_cast<cudf::size_type>(m_rows), cudf::mask_state::UNINITIALIZED, stream, mr);
        uint32_t const nwords = (m_rows + 31u) >> 5;
        uint32_t const gwords = (nwords + kBlock - 1) / kBlock;
        gather_null_mask_kernel<<<gwords, kBlock, 0, stream.get()>>>(
          d_src_nm, d_sel, static_cast<cudf::bitmask_type*>(mask.data()), m_rows);
        out_null_cnt = cudf::null_count(
          static_cast<cudf::bitmask_type const*>(mask.data()), 0, static_cast<cudf::size_type>(m_rows), stream);
        return mask;
      }
      out_null_cnt = 0;
      return cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr);
    };

    // Fast late-materialized O(M) decode for random-access encodings
    if ((ctype == ColType::INT32 || ctype == ColType::INT64 || ctype == ColType::DATE32 ||
         ctype == ColType::BOOL8 || ctype == ColType::INT8 || ctype == ColType::INT16 ||
         ctype == ColType::TIMESTAMP_US || ctype == ColType::DECIMAL32 ||
         ctype == ColType::DECIMAL64) &&
        (enc == Encoding::FOR_BITPACK || enc == Encoding::RLE_INT || enc == Encoding::RAW_FIXED)) {
      cudf::data_type const dtype = [&]() {
        switch (ctype) {
          case ColType::INT32: return cudf::data_type{cudf::type_id::INT32};
          case ColType::INT64: return cudf::data_type{cudf::type_id::INT64};
          case ColType::DATE32: return cudf::data_type{cudf::type_id::TIMESTAMP_DAYS};
          case ColType::BOOL8: return cudf::data_type{cudf::type_id::BOOL8};
          case ColType::INT8: return cudf::data_type{cudf::type_id::INT8};
          case ColType::INT16: return cudf::data_type{cudf::type_id::INT16};
          case ColType::TIMESTAMP_US: return cudf::data_type{cudf::type_id::TIMESTAMP_MICROSECONDS};
          case ColType::DECIMAL32: return cudf::data_type{cudf::type_id::DECIMAL32, m.scale_exp10};
          case ColType::DECIMAL64: return cudf::data_type{cudf::type_id::DECIMAL64, m.scale_exp10};
          default: CUDF_FAIL("unreachable");
        }
      }();
      cudf::size_type nc = 0;
      auto nbuf          = gather_null_buf(nc);
      auto col           = cudf::make_fixed_width_column(
        dtype, static_cast<cudf::size_type>(m_rows), std::move(nbuf), nc, stream, mr);
      if (m_rows > 0) {
        size_t const esz = cudf::size_of(dtype);
        if (enc == Encoding::FOR_BITPACK) {
          auto fv = make_for_view(d_pay, num_rows);
          if (esz == 8) {
            for_bitpack_decode_selected_kernel<int64_t><<<grid, kBlock, 0, stream.get()>>>(
              fv, d_sel, static_cast<int64_t*>(col->mutable_view().head()), m_rows);
          } else if (esz == 4) {
            for_bitpack_decode_selected_kernel<int32_t><<<grid, kBlock, 0, stream.get()>>>(
              fv, d_sel, static_cast<int32_t*>(col->mutable_view().head()), m_rows);
          } else if (esz == 2) {
            for_bitpack_decode_selected_kernel<int16_t><<<grid, kBlock, 0, stream.get()>>>(
              fv, d_sel, static_cast<int16_t*>(col->mutable_view().head()), m_rows);
          } else {
            for_bitpack_decode_selected_kernel<int8_t><<<grid, kBlock, 0, stream.get()>>>(
              fv, d_sel, static_cast<int8_t*>(col->mutable_view().head()), m_rows);
          }
        } else if (enc == Encoding::RLE_INT) {
          auto rv = make_rle_view(d_pay, num_rows, m.extra);
          if (esz == 8) {
            rle_decode_selected_kernel<int64_t><<<grid, kBlock, 0, stream.get()>>>(
              rv, d_sel, static_cast<int64_t*>(col->mutable_view().head()), m_rows);
          } else if (esz == 4) {
            rle_decode_selected_kernel<int32_t><<<grid, kBlock, 0, stream.get()>>>(
              rv, d_sel, static_cast<int32_t*>(col->mutable_view().head()), m_rows);
          } else if (esz == 2) {
            rle_decode_selected_kernel<int16_t><<<grid, kBlock, 0, stream.get()>>>(
              rv, d_sel, static_cast<int16_t*>(col->mutable_view().head()), m_rows);
          } else {
            rle_decode_selected_kernel<int8_t><<<grid, kBlock, 0, stream.get()>>>(
              rv, d_sel, static_cast<int8_t*>(col->mutable_view().head()), m_rows);
          }
        } else {
          if (esz == 8) {
            raw_gather_kernel<int64_t><<<grid, kBlock, 0, stream.get()>>>(
              reinterpret_cast<int64_t const*>(d_pay), d_sel, static_cast<int64_t*>(col->mutable_view().head()), m_rows);
          } else if (esz == 4) {
            raw_gather_kernel<int32_t><<<grid, kBlock, 0, stream.get()>>>(
              reinterpret_cast<int32_t const*>(d_pay), d_sel, static_cast<int32_t*>(col->mutable_view().head()), m_rows);
          } else if (esz == 2) {
            raw_gather_kernel<int16_t><<<grid, kBlock, 0, stream.get()>>>(
              reinterpret_cast<int16_t const*>(d_pay), d_sel, static_cast<int16_t*>(col->mutable_view().head()), m_rows);
          } else {
            raw_gather_kernel<int8_t><<<grid, kBlock, 0, stream.get()>>>(
              reinterpret_cast<int8_t const*>(d_pay), d_sel, static_cast<int8_t*>(col->mutable_view().head()), m_rows);
          }
        }
      }
      out_cols.push_back(std::move(col));
    } else if ((ctype == ColType::FLOAT32 || ctype == ColType::FLOAT64) &&
               (enc == Encoding::SCALED_FLOAT_FOR || enc == Encoding::RAW_FIXED)) {
      bool const is_f32           = (ctype == ColType::FLOAT32);
      cudf::data_type const dtype = cudf::data_type{is_f32 ? cudf::type_id::FLOAT32 : cudf::type_id::FLOAT64};
      cudf::size_type nc          = 0;
      auto nbuf                   = gather_null_buf(nc);
      auto col                    = cudf::make_fixed_width_column(
        dtype, static_cast<cudf::size_type>(m_rows), std::move(nbuf), nc, stream, mr);
      if (m_rows > 0) {
        if (enc == Encoding::SCALED_FLOAT_FOR) {
          auto fv = make_for_view(d_pay, num_rows);
          double const inv =
            (m.scale_exp10 == 0) ? 1.0 : ((m.scale_exp10 == 2) ? 0.01 : 0.0001);
          if (is_f32) {
            scaled_float_for_decode_selected_kernel<float><<<grid, kBlock, 0, stream.get()>>>(
              fv, inv, d_sel, col->mutable_view().data<float>(), m_rows);
          } else {
            scaled_float_for_decode_selected_kernel<double><<<grid, kBlock, 0, stream.get()>>>(
              fv, inv, d_sel, col->mutable_view().data<double>(), m_rows);
          }
        } else {
          if (is_f32) {
            raw_gather_kernel<float><<<grid, kBlock, 0, stream.get()>>>(
              reinterpret_cast<float const*>(d_pay), d_sel, col->mutable_view().data<float>(), m_rows);
          } else {
            raw_gather_kernel<double><<<grid, kBlock, 0, stream.get()>>>(
              reinterpret_cast<double const*>(d_pay), d_sel, col->mutable_view().data<double>(), m_rows);
          }
        }
      }
      out_cols.push_back(std::move(col));
    } else if (ctype == ColType::STRING && enc == Encoding::DICT_BITPACK) {
      bool const want_dict32 = (i < dict_as_dictionary32.size()) && (dict_as_dictionary32[i] != 0);
      DictPayloadHeader dh{};
      std::memcpy(&dh, host_src + plan.packed_payload_offsets[i], sizeof(dh));
      auto const* d_dict_offs = reinterpret_cast<int32_t const*>(d_pay + sizeof(dh));
      auto const* d_dict_ch   = reinterpret_cast<char const*>(
        d_pay + sizeof(dh) + (dh.dict_size + 1) * sizeof(int32_t));
      auto codes_view    = make_for_view(d_pay + dh.codes_offset, num_rows);
      cudf::size_type nc = 0;
      auto nbuf          = gather_null_buf(nc);
      if (want_dict32) {
        auto keys_offs = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                   static_cast<cudf::size_type>(dh.dict_size + 1),
                                                   cudf::mask_state::UNALLOCATED,
                                                   stream,
                                                   mr);
        CUDF_CUDA_TRY(cudaMemcpyAsync(keys_offs->mutable_view().data<int32_t>(),
                                      d_dict_offs,
                                      (dh.dict_size + 1) * sizeof(int32_t),
                                      cudaMemcpyDeviceToDevice,
                                      stream.get()));
        rmm::device_buffer keys_ch(dh.total_dict_bytes, stream, mr);
        if (dh.total_dict_bytes > 0) {
          CUDF_CUDA_TRY(cudaMemcpyAsync(
            keys_ch.data(), d_dict_ch, dh.total_dict_bytes, cudaMemcpyDeviceToDevice, stream.get()));
        }
        auto keys_col = cudf::make_strings_column(
          static_cast<cudf::size_type>(dh.dict_size),
          std::move(keys_offs),
          std::move(keys_ch),
          0,
          cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr));
        auto indices_col = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                     static_cast<cudf::size_type>(m_rows),
                                                     cudf::mask_state::UNALLOCATED,
                                                     stream,
                                                     mr);
        if (m_rows > 0) {
          for_bitpack_decode_selected_kernel<int32_t><<<grid, kBlock, 0, stream.get()>>>(
            codes_view, d_sel, indices_col->mutable_view().data<int32_t>(), m_rows);
        }
        out_cols.push_back(cudf::make_dictionary_column(
          std::move(keys_col), std::move(indices_col), std::move(nbuf), nc));
      } else {
        auto out_offs = cudf::make_numeric_column(cudf::data_type{cudf::type_id::INT32},
                                                  static_cast<cudf::size_type>(m_rows + 1),
                                                  cudf::mask_state::UNALLOCATED,
                                                  stream,
                                                  mr);
        auto* d_offs  = out_offs->mutable_view().data<int32_t>();
        uint32_t const grid_p1 = (m_rows + 1 + kBlock - 1) / kBlock;
        dict_lengths_selected_kernel<<<grid_p1, kBlock, 0, stream.get()>>>(
          codes_view, d_dict_offs, d_src_nm, d_sel, d_offs, m_rows);
        thrust::exclusive_scan(
          rmm::exec_policy_nosync(stream), d_offs, d_offs + m_rows + 1, d_offs);
        int32_t total_chars = 0;
        CUDF_CUDA_TRY(cudaMemcpyAsync(
          &total_chars, d_offs + m_rows, sizeof(int32_t), cudaMemcpyDeviceToHost, stream.get()));
        stream.sync();
        rmm::device_buffer out_chars(static_cast<size_t>(total_chars), stream, mr);
        if (m_rows > 0 && total_chars > 0) {
          dict_copy_chars_selected_kernel<<<grid, kBlock, 0, stream.get()>>>(
            codes_view,
            d_dict_offs,
            d_dict_ch,
            d_src_nm,
            d_sel,
            d_offs,
            static_cast<char*>(out_chars.data()),
            m_rows);
        }
        out_cols.push_back(cudf::make_strings_column(
          static_cast<cudf::size_type>(m_rows), std::move(out_offs), std::move(out_chars), nc, std::move(nbuf)));
      }
    } else {
      // Non-random-access encoding (DELTA_FOR_BITPACK / NVCOMP_LZ4_* / RAW_STRING): full col decode + gather
      ReadPlan single{};
      single.header = plan.header;
      single.metas.push_back(m);
      single.packed_payload_offsets.push_back(plan.packed_payload_offsets[i]);
      single.packed_null_offsets.push_back(plan.packed_null_offsets[i]);
      single.total_packed_bytes = plan.total_packed_bytes;
      std::vector<uint8_t> d32{i < dict_as_dictionary32.size() ? dict_as_dictionary32[i] : uint8_t{0}};
      auto full_tbl = decode_from_host(single, host_src, d32, stream, mr);
      auto gathered = cudf::gather(
        full_tbl->view(), selected_rows, cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
      auto cols = gathered->release();
      out_cols.push_back(std::move(cols[0]));
    }
  }

  stream.sync();
  return std::make_unique<cudf::table>(std::move(out_cols));
}

// Track 3: Single-H2D Unified Filter + Project / Decode LTO (eliminates Stage 2 re-open & 2nd H2D!)
std::unique_ptr<cudf::table> run_lto_filter_and_project(
  cudf::transform_program& filter_prog,
  cudf::transform_program* proj_prog,
  ReadPlan const& union_plan,
  uint8_t const* host_src,
  std::vector<int32_t> const& filter_col_map,
  std::vector<std::vector<std::string>> const& filter_dict_lits,
  std::vector<int32_t> const& proj_col_map,
  std::vector<std::vector<std::string>> const& proj_dict_lits,
  std::vector<uint8_t> const& out_dict32,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  uint32_t const num_rows = union_plan.header.num_rows;
  if (num_rows == 0) { return nullptr; }

  // ONE single H2D copy of the compressed union buffer!
  rmm::device_buffer d_packed(union_plan.total_packed_bytes, stream, mr);
  if (union_plan.total_packed_bytes > 0) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(d_packed.data(),
                                  host_src,
                                  union_plan.total_packed_bytes,
                                  cudaMemcpyHostToDevice,
                                  stream.get()));
  }
  auto const* d_base = static_cast<uint8_t const*>(d_packed.data());

  auto populate_desc = [&](std::vector<int32_t> const& cmap,
                           std::vector<std::vector<std::string>> const& dlits,
                           int32_t const* sel_ptr,
                           std::vector<rmm::device_buffer>& d_temps) -> LencLtoBatchDesc {
    LencLtoBatchDesc desc{};
    desc.selected_rows = sel_ptr;
    CUDF_EXPECTS(cmap.size() <= static_cast<size_t>(kMaxLtoCols), "lenc LTO: max 16 cols");
    for (size_t s = 0; s < cmap.size(); ++s) {
      size_t const ui     = static_cast<size_t>(cmap[s]);
      ColumnMeta const& m = union_plan.metas[ui];
      auto const* d_pay   = d_base + union_plan.packed_payload_offsets[ui];
      auto const enc      = static_cast<Encoding>(m.encoding);
      auto const ctype    = static_cast<ColType>(m.type);
      LencLtoColDesc& cd  = desc.cols[s];
      if (m.has_nulls && m.null_mask_size > 0) {
        cd.null_mask = reinterpret_cast<cudf::bitmask_type const*>(
          d_base + union_plan.packed_null_offsets[ui]);
      }
      if (enc == Encoding::FOR_BITPACK || enc == Encoding::SCALED_FLOAT_FOR) {
        auto fv               = make_for_view(d_pay, num_rows);
        cd.min_vals           = fv.min_vals;
        cd.bit_widths         = fv.bit_widths;
        cd.chunk_word_offsets = fv.chunk_word_offsets;
        cd.packed_words       = fv.packed_words;
        cd.inv_scale = (m.scale_exp10 == 0) ? 1.0 : ((m.scale_exp10 == 2) ? 0.01 : 0.0001);
      } else if (enc == Encoding::RLE_INT) {
        auto rv            = make_rle_view(d_pay, num_rows, m.extra);
        cd.rle_chunk_start = rv.chunk_start_run;
        cd.rle_run_ends    = rv.run_ends;
        cd.rle_run_vals    = rv.run_vals;
        cd.rle_num_runs    = rv.num_runs;
      } else if (enc == Encoding::DELTA_FOR_BITPACK) {
        uint32_t const nchunk = (num_rows + kChunkSize - 1) / kChunkSize;
        auto dv               = make_delta_for_view(d_pay, num_rows);
        bool const is64       = (ctype == ColType::INT64 || ctype == ColType::TIMESTAMP_US ||
                           ctype == ColType::DECIMAL64);
        rmm::device_buffer d_dec(static_cast<size_t>(num_rows) * (is64 ? 8 : 4), stream, mr);
        if (is64) {
          delta_for_decode_kernel<int64_t><<<nchunk, 256, 0, stream.get()>>>(
            dv, static_cast<int64_t*>(d_dec.data()), num_rows);
        } else {
          delta_for_decode_kernel<int32_t><<<nchunk, 256, 0, stream.get()>>>(
            dv, static_cast<int32_t*>(d_dec.data()), num_rows);
        }
        cd.raw_ptr = static_cast<uint8_t const*>(d_dec.data());
        d_temps.push_back(std::move(d_dec));
      } else if (enc == Encoding::NVCOMP_LZ4_FIXED) {
        Lz4PayloadHeader lh{};
        auto d_dec = nvcomp_lz4_decompress_device(
          host_src + union_plan.packed_payload_offsets[ui], d_pay, lh, stream, mr);
        cd.raw_ptr = static_cast<uint8_t const*>(d_dec.data());
        d_temps.push_back(std::move(d_dec));
      } else if (enc == Encoding::DICT_BITPACK) {
        DictPayloadHeader dh{};
        uint8_t const* h_pay = host_src + union_plan.packed_payload_offsets[ui];
        std::memcpy(&dh, h_pay, sizeof(dh));
        auto fv               = make_for_view(d_pay + dh.codes_offset, num_rows);
        cd.min_vals           = fv.min_vals;
        cd.bit_widths         = fv.bit_widths;
        cd.chunk_word_offsets = fv.chunk_word_offsets;
        cd.packed_words       = fv.packed_words;
        cd.dict_size          = dh.dict_size;
        std::vector<int32_t> h_remap(dh.dict_size, -1);
        if (s < dlits.size() && !dlits[s].empty()) {
          std::unordered_map<std::string_view, int32_t> lmap;
          lmap.reserve(dlits[s].size() * 2 + 1);
          for (size_t l = 0; l < dlits[s].size(); ++l) {
            lmap.emplace(std::string_view{dlits[s][l]}, static_cast<int32_t>(l));
          }
          auto const* dict_offs = reinterpret_cast<int32_t const*>(h_pay + sizeof(dh));
          auto const* dict_ch   = reinterpret_cast<char const*>(
            h_pay + sizeof(dh) + (dh.dict_size + 1) * sizeof(int32_t));
          for (uint32_t k = 0; k < dh.dict_size; ++k) {
            int32_t const st = dict_offs[k];
            int32_t const en = dict_offs[k + 1];
            auto it          = lmap.find(std::string_view(dict_ch + st, static_cast<size_t>(en - st)));
            if (it != lmap.end()) { h_remap[k] = it->second; }
          }
        }
        rmm::device_buffer d_rm(std::max<size_t>(dh.dict_size, 1) * sizeof(int32_t), stream, mr);
        if (dh.dict_size > 0) {
          CUDF_CUDA_TRY(cudaMemcpyAsync(d_rm.data(),
                                        h_remap.data(),
                                        dh.dict_size * sizeof(int32_t),
                                        cudaMemcpyHostToDevice,
                                        stream.get()));
        }
        cd.dict_remap = static_cast<int32_t const*>(d_rm.data());
        d_temps.push_back(std::move(d_rm));
      } else if (enc == Encoding::RAW_FIXED) {
        cd.raw_ptr = d_pay;
      }
    }
    return desc;
  };

  // Stage 1: Filter LTO in registers on already-resident d_packed
  std::vector<rmm::device_buffer> d_temps;
  auto f_desc = populate_desc(filter_col_map, filter_dict_lits, nullptr, d_temps);
  rmm::device_buffer d_fdesc(sizeof(LencLtoBatchDesc), stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    d_fdesc.data(), &f_desc, sizeof(LencLtoBatchDesc), cudaMemcpyHostToDevice, stream.get()));
  auto mask_tbl = filter_prog.run_with_user_data(
    d_fdesc.data(), static_cast<cudf::size_type>(num_rows), stream, mr);
  auto sel_col = compact_mask_indices(mask_tbl->view().column(0), stream, mr);
  if (sel_col->size() == 0) {
    stream.sync();
    return nullptr;
  }

  // Stage 2: Project LTO (or late-materialized decode) on the SAME d_packed buffer!
  if (proj_prog != nullptr) {
    auto p_desc = populate_desc(
      proj_col_map, proj_dict_lits, sel_col->view().data<int32_t>(), d_temps);
    rmm::device_buffer d_pdesc(sizeof(LencLtoBatchDesc), stream, mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      d_pdesc.data(), &p_desc, sizeof(LencLtoBatchDesc), cudaMemcpyHostToDevice, stream.get()));
    auto out_tbl = proj_prog->run_with_user_data(d_pdesc.data(), sel_col->size(), stream, mr);
    stream.sync();
    return out_tbl;
  }

  // Decode selected rows of proj_col_map from union_plan
  ReadPlan sub_plan{};
  sub_plan.header             = union_plan.header;
  sub_plan.total_packed_bytes = union_plan.total_packed_bytes;
  for (int32_t const ui : proj_col_map) {
    sub_plan.metas.push_back(union_plan.metas[static_cast<size_t>(ui)]);
    sub_plan.packed_payload_offsets.push_back(
      union_plan.packed_payload_offsets[static_cast<size_t>(ui)]);
    sub_plan.packed_null_offsets.push_back(union_plan.packed_null_offsets[static_cast<size_t>(ui)]);
  }
  return decode_from_host_selected(sub_plan, host_src, out_dict32, sel_col->view(), stream, mr);
}

}  // namespace rapids::jni::lenc

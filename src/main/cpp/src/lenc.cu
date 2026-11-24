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
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda_runtime_api.h>
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
        tid == cudf::type_id::TIMESTAMP_DAYS) {
      if (tid == cudf::type_id::INT32) {
        meta.type = static_cast<uint8_t>(ColType::INT32);
      } else if (tid == cudf::type_id::INT64) {
        meta.type = static_cast<uint8_t>(ColType::INT64);
      } else {
        meta.type = static_cast<uint8_t>(ColType::DATE32);
      }
      std::vector<int64_t> h_vals(num_rows, 0);
      if (num_rows > 0) {
        if (tid == cudf::type_id::INT64) {
          CUDF_CUDA_TRY(cudaMemcpyAsync(h_vals.data(),
                                        col.data<int64_t>() + col.offset(),
                                        num_rows * sizeof(int64_t),
                                        cudaMemcpyDeviceToHost,
                                        stream.get()));
          stream.sync();
        } else {
          std::vector<int32_t> h32(num_rows);
          CUDF_CUDA_TRY(cudaMemcpyAsync(h32.data(),
                                        col.data<int32_t>() + col.offset(),
                                        num_rows * sizeof(int32_t),
                                        cudaMemcpyDeviceToHost,
                                        stream.get()));
          stream.sync();
          for (uint32_t i = 0; i < num_rows; ++i) { h_vals[i] = static_cast<int64_t>(h32[i]); }
        }
      }
      auto for_buf      = encode_for_bitpack_i64(h_vals.data(), num_rows);
      uint32_t num_runs = 0;
      auto rle_buf      = encode_rle_i64(h_vals.data(), num_rows, num_runs);
      if (num_rows > 0 && rle_buf.size() * 2 < for_buf.size()) {
        meta.encoding = static_cast<uint8_t>(Encoding::RLE_INT);
        meta.extra    = num_runs;
        payloads[c]   = std::move(rle_buf);
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
          meta.encoding = static_cast<uint8_t>(Encoding::RAW_FIXED);
          payloads[c]   = std::move(raw);
        }
      } else {
        meta.encoding = static_cast<uint8_t>(Encoding::RAW_FIXED);
        payloads[c]   = std::move(raw);
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
      // Build dictionary (up to 65536 distinct strings)
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
      if (dict_ok) {
        auto codes_buf         = encode_for_bitpack_i64(codes.data(), num_rows);
        uint32_t const dsize   = static_cast<uint32_t>(dict_offs.size() - 1);
        size_t const off_doff  = sizeof(DictPayloadHeader);
        size_t const off_dchr  = off_doff + dict_offs.size() * sizeof(int32_t);
        size_t const off_codes = align_up(off_dchr + dict_chars.size(), 8);
        std::vector<uint8_t> buf(off_codes + codes_buf.size(), 0);
        DictPayloadHeader dh{dsize, static_cast<uint32_t>(dict_chars.size()), off_codes};
        std::memcpy(buf.data(), &dh, sizeof(dh));
        std::memcpy(buf.data() + off_doff, dict_offs.data(), dict_offs.size() * sizeof(int32_t));
        if (!dict_chars.empty()) {
          std::memcpy(buf.data() + off_dchr, dict_chars.data(), dict_chars.size());
        }
        std::memcpy(buf.data() + off_codes, codes_buf.data(), codes_buf.size());
        meta.encoding = static_cast<uint8_t>(Encoding::DICT_BITPACK);
        meta.extra    = dsize;
        payloads[c]   = std::move(buf);
      } else {
        // RAW_STRING fallback: [offsets(num_rows+1 int32)][chars(total_ch)]
        for (uint32_t i = 0; i <= num_rows; ++i) { h_offs[i] -= base_off; }
        size_t const off_bytes = (num_rows + 1) * sizeof(int32_t);
        std::vector<uint8_t> buf(off_bytes + static_cast<size_t>(total_ch), 0);
        std::memcpy(buf.data(), h_offs.data(), off_bytes);
        if (total_ch > 0) {
          std::memcpy(buf.data() + off_bytes, h_chars.data(), static_cast<size_t>(total_ch));
        }
        meta.encoding = static_cast<uint8_t>(Encoding::RAW_STRING);
        meta.extra    = static_cast<uint32_t>(total_ch);
        payloads[c]   = std::move(buf);
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
  ::close(fd);

  plan.col_indices = col_indices;
  plan.metas.reserve(col_indices.size());
  plan.packed_payload_offsets.reserve(col_indices.size());
  plan.packed_null_offsets.reserve(col_indices.size());

  uint64_t packed_off = 0;
  for (int32_t const ci : col_indices) {
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
  }
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

    if (ctype == ColType::INT32 || ctype == ColType::INT64 || ctype == ColType::DATE32) {
      cudf::data_type const dtype =
        (ctype == ColType::INT32)
          ? cudf::data_type{cudf::type_id::INT32}
          : (ctype == ColType::INT64 ? cudf::data_type{cudf::type_id::INT64}
                                     : cudf::data_type{cudf::type_id::TIMESTAMP_DAYS});
      auto col = cudf::make_fixed_width_column(
        dtype, static_cast<cudf::size_type>(num_rows), make_null_buf(), null_cnt, stream, mr);
      if (num_rows > 0) {
        if (enc == Encoding::FOR_BITPACK) {
          auto fv = make_for_view(d_pay, num_rows);
          if (ctype == ColType::INT64) {
            for_bitpack_decode_kernel<int64_t>
              <<<grid, kBlock, 0, stream.get()>>>(fv, col->mutable_view().data<int64_t>(), num_rows);
          } else {
            for_bitpack_decode_kernel<int32_t>
              <<<grid, kBlock, 0, stream.get()>>>(fv, col->mutable_view().data<int32_t>(), num_rows);
          }
        } else if (enc == Encoding::RLE_INT) {
          auto rv = make_rle_view(d_pay, num_rows, m.extra);
          if (ctype == ColType::INT64) {
            rle_decode_kernel<int64_t>
              <<<grid, kBlock, 0, stream.get()>>>(rv, col->mutable_view().data<int64_t>(), num_rows);
          } else {
            rle_decode_kernel<int32_t>
              <<<grid, kBlock, 0, stream.get()>>>(rv, col->mutable_view().data<int32_t>(), num_rows);
          }
        } else {
          CUDF_CUDA_TRY(cudaMemcpyAsync(col->mutable_view().head(),
                                        d_pay,
                                        num_rows * cudf::size_of(dtype),
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

}  // namespace rapids::jni::lenc

/*
 * Copyright (c) 2026, NVIDIA CORPORATION.
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
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace rapids::jni::lenc {

namespace {

constexpr int kMaxStarDims     = 6;
constexpr int kMaxExtraNotNull = 8;
constexpr uint64_t kMaxLutSpan = 2097152ULL;  // 8 MiB max direct-address LUT in L2 cache

struct DeviceKeyCol {
  void const* data;
  uint32_t const* mask;
  int32_t offset;
  uint8_t is_i64;  // 1 = INT64, 0 = INT32/DATE
};

__device__ __forceinline__ bool col_valid_dev(DeviceKeyCol const& c, int32_t row)
{
  if (c.mask == nullptr) { return true; }
  uint32_t const bit = static_cast<uint32_t>(row + c.offset);
  return ((c.mask[bit >> 5] >> (bit & 31u)) & 1u) != 0u;
}

__device__ __forceinline__ int64_t read_key_dev(DeviceKeyCol const& c, int32_t row)
{
  int32_t const idx = row + c.offset;
  if (c.is_i64) {
    return reinterpret_cast<int64_t const*>(c.data)[idx];
  }
  return static_cast<int64_t>(reinterpret_cast<int32_t const*>(c.data)[idx]);
}

__device__ __forceinline__ uint64_t mix64(uint64_t x)
{
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

struct DeviceBuildTable {
  int32_t const* lut;        // non-null => O(1) direct-address L2 lookup (unique keys!)
  int64_t min_key;
  uint64_t lut_span;         // max_key - min_key + 1
  int32_t const* head;       // hash bucket head (chained open-addressing)
  int32_t const* next;       // linked list per build row
  int64_t const* build_keys; // stored key per build row
  uint32_t hash_mask;        // cap - 1 (power of 2)
  uint8_t is_empty;
  uint8_t has_dup;
};

__device__ __forceinline__ int32_t probe_first_unique(DeviceBuildTable const& bt, int64_t k)
{
  if (bt.lut != nullptr) {
    uint64_t const d = static_cast<uint64_t>(k - bt.min_key);
    if (d >= bt.lut_span) { return -1; }
    return bt.lut[d];
  }
  uint32_t const slot = static_cast<uint32_t>(mix64(static_cast<uint64_t>(k))) & bt.hash_mask;
  int32_t cur         = bt.head[slot];
  while (cur >= 0) {
    if (bt.build_keys[cur] == k) { return cur; }
    cur = bt.next[cur];
  }
  return -1;
}

__device__ __forceinline__ int32_t probe_count_dup(DeviceBuildTable const& bt, int64_t k)
{
  if (bt.lut != nullptr) {
    uint64_t const d = static_cast<uint64_t>(k - bt.min_key);
    return (d < bt.lut_span && bt.lut[d] >= 0) ? 1 : 0;
  }
  uint32_t const slot = static_cast<uint32_t>(mix64(static_cast<uint64_t>(k))) & bt.hash_mask;
  int32_t cur         = bt.head[slot];
  int32_t cnt         = 0;
  while (cur >= 0) {
    if (bt.build_keys[cur] == k) { ++cnt; }
    cur = bt.next[cur];
  }
  return cnt;
}

// ---------------- Build-Side Kernels ----------------

struct BuildStats {
  int64_t min_k;
  int64_t max_k;
  uint32_t valid_cnt;
  uint32_t has_dup;
};

__global__ void build_hash_and_stats_kernel(
  DeviceKeyCol col,
  int32_t n,
  int32_t* head,
  int32_t* next,
  int64_t* build_keys,
  uint32_t hash_mask,
  BuildStats* stats)
{
  int32_t const row = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n) { return; }
  if (!col_valid_dev(col, row)) {
    next[row] = -1;
    return;
  }
  int64_t const k = read_key_dev(col, row);
  build_keys[row] = k;
  uint32_t const slot = static_cast<uint32_t>(mix64(static_cast<uint64_t>(k))) & hash_mask;
  int32_t const prev  = atomicExch(&head[slot], row);
  next[row]           = prev;

  // Update valid count & min/max key
  atomicAdd(&stats->valid_cnt, 1u);
  atomicMin(reinterpret_cast<unsigned long long*>(&stats->min_k),
            static_cast<unsigned long long>(k ^ (1ULL << 63)));
  atomicMax(reinterpret_cast<unsigned long long*>(&stats->max_k),
            static_cast<unsigned long long>(k ^ (1ULL << 63)));
}

__global__ void detect_dup_kernel(
  DeviceKeyCol col,
  int32_t n,
  int32_t const* head,
  int32_t const* next,
  int64_t const* build_keys,
  uint32_t hash_mask,
  uint32_t* has_dup)
{
  int32_t const row = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n || !col_valid_dev(col, row)) { return; }
  int64_t const k = build_keys[row];
  int32_t cur     = next[row];
  while (cur >= 0) {
    if (build_keys[cur] == k) {
      *has_dup = 1u;
      return;
    }
    cur = next[cur];
  }
}

__global__ void fill_lut_kernel(
  DeviceKeyCol col, int32_t n, int64_t min_key, int32_t* lut)
{
  int32_t const row = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= n || !col_valid_dev(col, row)) { return; }
  int64_t const k = read_key_dev(col, row);
  lut[static_cast<uint64_t>(k - min_key)] = row;
}

// ---------------- Probe Kernels ----------------

struct StarProbeArgs {
  int32_t n;
  int32_t num_dims;
  int32_t num_extra_nn;
  DeviceKeyCol probe_cols[kMaxStarDims];
  DeviceBuildTable dims[kMaxStarDims];
  DeviceKeyCol extra_nn[kMaxExtraNotNull];
  // For Mode C2: dim index -> output gather map index (-1 if dim emits 0 payload cols)
  int32_t dim_to_out_slot[kMaxStarDims];
  int32_t num_out_dim_slots;
};

__global__ void star_join_count_kernel(StarProbeArgs args, unsigned long long* out_count)
{
  __shared__ unsigned long long s_sum[8];  // 256 threads = 8 warps
  unsigned long long local_cnt = 0ULL;

  int32_t const tid    = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  int32_t const stride = static_cast<int32_t>(gridDim.x * blockDim.x);

  for (int32_t row = tid; row < args.n; row += stride) {
    bool ok = true;
    for (int e = 0; e < args.num_extra_nn; ++e) {
      if (!col_valid_dev(args.extra_nn[e], row)) {
        ok = false;
        break;
      }
    }
    if (!ok) { continue; }

    unsigned long long prod = 1ULL;
    for (int d = 0; d < args.num_dims; ++d) {
      if (!col_valid_dev(args.probe_cols[d], row)) {
        prod = 0ULL;
        break;
      }
      int64_t const k = read_key_dev(args.probe_cols[d], row);
      if (args.dims[d].has_dup) {
        int32_t const c = probe_count_dup(args.dims[d], k);
        if (c == 0) {
          prod = 0ULL;
          break;
        }
        prod *= static_cast<unsigned long long>(c);
      } else {
        if (probe_first_unique(args.dims[d], k) < 0) {
          prod = 0ULL;
          break;
        }
      }
    }
    local_cnt += prod;
  }

  // Warp reduction
  for (int off = 16; off > 0; off >>= 1) {
    local_cnt += __shfl_down_sync(0xFFFFFFFFu, local_cnt, off);
  }
  int const lane = threadIdx.x & 31;
  int const warp = threadIdx.x >> 5;
  if (lane == 0) { s_sum[warp] = local_cnt; }
  __syncthreads();
  if (warp == 0) {
    unsigned long long v = (lane < 8) ? s_sum[lane] : 0ULL;
    for (int off = 4; off > 0; off >>= 1) { v += __shfl_down_sync(0xFFFFFFFFu, v, off); }
    if (lane == 0 && v > 0ULL) { atomicAdd(out_count, v); }
  }
}

// Mode C2 Fast Path: ALL build tables have unique keys -> 1 single pass with warp-aggregated
// ballot compaction writing surviving fact_idx + only needed dim_idx[p] directly!
__global__ void star_join_probe_unique_kernel(
  StarProbeArgs args,
  int32_t* fact_idx_out,
  int32_t* dim_idx_out_0,
  int32_t* dim_idx_out_1,
  int32_t* dim_idx_out_2,
  int32_t* dim_idx_out_3,
  int32_t* dim_idx_out_4,
  int32_t* dim_idx_out_5,
  uint32_t* global_out_cnt)
{
  int32_t* dim_outs[kMaxStarDims] = {
    dim_idx_out_0, dim_idx_out_1, dim_idx_out_2, dim_idx_out_3, dim_idx_out_4, dim_idx_out_5};

  int32_t const tid    = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  int32_t const stride = static_cast<int32_t>(gridDim.x * blockDim.x);
  int const lane       = threadIdx.x & 31;

  // Round upper loop bound up to warp multiple so __ballot_sync always executes uniformly!
  int32_t const n_padded = ((args.n + 31) / 32) * 32;
  for (int32_t row = tid; row < n_padded; row += stride) {
    bool matched = (row < args.n);
    int32_t matched_rows[kMaxStarDims];
    if (matched) {
      for (int e = 0; e < args.num_extra_nn; ++e) {
        if (!col_valid_dev(args.extra_nn[e], row)) {
          matched = false;
          break;
        }
      }
    }
    if (matched) {
      for (int d = 0; d < args.num_dims; ++d) {
        if (!col_valid_dev(args.probe_cols[d], row)) {
          matched = false;
          break;
        }
        int64_t const k   = read_key_dev(args.probe_cols[d], row);
        int32_t const hit = probe_first_unique(args.dims[d], k);
        if (hit < 0) {
          matched = false;
          break;
        }
        matched_rows[d] = hit;
      }
    }

    unsigned const ballot = __ballot_sync(0xFFFFFFFFu, matched);
    if (ballot == 0u) { continue; }
    uint32_t base = 0u;
    if (lane == 0) {
      base = atomicAdd(global_out_cnt, static_cast<uint32_t>(__popc(ballot)));
    }
    base = __shfl_sync(0xFFFFFFFFu, base, 0);
    if (matched) {
      uint32_t const pos = base + static_cast<uint32_t>(__popc(ballot & ((1u << lane) - 1u)));
      if (fact_idx_out != nullptr) { fact_idx_out[pos] = row; }
      for (int d = 0; d < args.num_dims; ++d) {
        int32_t const slot = args.dim_to_out_slot[d];
        if (slot >= 0) { dim_outs[slot][pos] = matched_rows[d]; }
      }
    }
  }
}

// Mode C2 General Path (when any build table has duplicate keys):
__global__ void star_join_dup_count_kernel(StarProbeArgs args, int32_t* row_mults)
{
  int32_t const row = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= args.n) { return; }
  for (int e = 0; e < args.num_extra_nn; ++e) {
    if (!col_valid_dev(args.extra_nn[e], row)) {
      row_mults[row] = 0;
      return;
    }
  }
  int64_t prod = 1;
  for (int d = 0; d < args.num_dims; ++d) {
    if (!col_valid_dev(args.probe_cols[d], row)) {
      row_mults[row] = 0;
      return;
    }
    int64_t const k = read_key_dev(args.probe_cols[d], row);
    int32_t const c = probe_count_dup(args.dims[d], k);
    if (c == 0) {
      row_mults[row] = 0;
      return;
    }
    prod *= c;
  }
  row_mults[row] = static_cast<int32_t>(prod);
}

__global__ void star_join_dup_emit_kernel(
  StarProbeArgs args,
  int32_t const* row_mults,
  int32_t const* offsets,
  int32_t* fact_idx_out,
  int32_t* dim_idx_out_0,
  int32_t* dim_idx_out_1,
  int32_t* dim_idx_out_2,
  int32_t* dim_idx_out_3,
  int32_t* dim_idx_out_4,
  int32_t* dim_idx_out_5)
{
  int32_t const row = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= args.n || row_mults[row] == 0) { return; }

  int32_t* dim_outs[kMaxStarDims] = {
    dim_idx_out_0, dim_idx_out_1, dim_idx_out_2, dim_idx_out_3, dim_idx_out_4, dim_idx_out_5};

  int64_t pkeys[kMaxStarDims];
  int32_t firsts[kMaxStarDims];
  int32_t curs[kMaxStarDims];
  for (int d = 0; d < args.num_dims; ++d) {
    pkeys[d] = read_key_dev(args.probe_cols[d], row);
    if (args.dims[d].lut != nullptr) {
      firsts[d] = args.dims[d].lut[static_cast<uint64_t>(pkeys[d] - args.dims[d].min_key)];
    } else {
      uint32_t const slot =
        static_cast<uint32_t>(mix64(static_cast<uint64_t>(pkeys[d]))) & args.dims[d].hash_mask;
      int32_t c = args.dims[d].head[slot];
      while (c >= 0 && args.dims[d].build_keys[c] != pkeys[d]) { c = args.dims[d].next[c]; }
      firsts[d] = c;
    }
    curs[d] = firsts[d];
  }

  int32_t pos = offsets[row];
  while (true) {
    if (fact_idx_out != nullptr) { fact_idx_out[pos] = row; }
    for (int d = 0; d < args.num_dims; ++d) {
      int32_t const slot = args.dim_to_out_slot[d];
      if (slot >= 0) { dim_outs[slot][pos] = curs[d]; }
    }
    ++pos;

    // Advance odometer across dims
    int carry = args.num_dims - 1;
    while (carry >= 0) {
      if (args.dims[carry].lut != nullptr) {
        curs[carry] = firsts[carry];
        --carry;
      } else {
        int32_t c = args.dims[carry].next[curs[carry]];
        while (c >= 0 && args.dims[carry].build_keys[c] != pkeys[carry]) {
          c = args.dims[carry].next[c];
        }
        if (c >= 0) {
          curs[carry] = c;
          break;
        }
        curs[carry] = firsts[carry];
        --carry;
      }
    }
    if (carry < 0) { break; }
  }
}

DeviceKeyCol make_device_key_col(cudf::column_view const& cv)
{
  DeviceKeyCol c{};
  c.data   = cv.head<void>();
  c.mask   = cv.nullable() ? cv.null_mask() : nullptr;
  c.offset = cv.offset();
  auto id  = cv.type().id();
  if (id == cudf::type_id::INT64 || id == cudf::type_id::TIMESTAMP_MICROSECONDS) {
    c.is_i64 = 1u;
  } else if (id == cudf::type_id::INT32 || id == cudf::type_id::TIMESTAMP_DAYS) {
    c.is_i64 = 0u;
  } else {
    CUDF_FAIL("StarJoin only supports INT32/INT64/DATE join keys");
  }
  return c;
}

DeviceKeyCol make_device_nn_col(cudf::column_view const& cv)
{
  DeviceKeyCol c{};
  c.data   = nullptr;
  c.mask   = cv.nullable() ? cv.null_mask() : nullptr;
  c.offset = cv.offset();
  c.is_i64 = 0u;
  return c;
}

}  // namespace

struct StarJoinBuildTable {
  rmm::device_uvector<int32_t> head;
  rmm::device_uvector<int32_t> next;
  rmm::device_uvector<int64_t> build_keys;
  rmm::device_uvector<int32_t> lut;
  uint32_t hash_mask{0};
  int64_t min_key{0};
  uint64_t lut_span{0};
  int32_t valid_count{0};
  bool is_empty{true};
  bool has_dup{false};

  StarJoinBuildTable(cuda::stream_ref stream, rmm::device_async_resource_ref mr)
    : head(0, stream, mr), next(0, stream, mr), build_keys(0, stream, mr), lut(0, stream, mr)
  {
  }

  DeviceBuildTable device_view() const
  {
    DeviceBuildTable v{};
    v.lut        = lut.is_empty() ? nullptr : lut.data();
    v.min_key    = min_key;
    v.lut_span   = lut_span;
    v.head       = head.is_empty() ? nullptr : head.data();
    v.next       = next.is_empty() ? nullptr : next.data();
    v.build_keys = build_keys.is_empty() ? nullptr : build_keys.data();
    v.hash_mask  = hash_mask;
    v.is_empty   = is_empty ? 1u : 0u;
    v.has_dup    = has_dup ? 1u : 0u;
    return v;
  }
};

StarJoinBuildTable* build_star_join_table(
  cudf::column_view const& key_col,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto bt = std::make_unique<StarJoinBuildTable>(stream, mr);
  int32_t const n = key_col.size();
  if (n <= 0 || key_col.null_count() == n) {
    bt->is_empty = true;
    return bt.release();
  }

  uint32_t cap = 64u;
  while (cap < static_cast<uint32_t>(n) * 2u && cap < (1u << 26)) { cap <<= 1; }
  bt->hash_mask = cap - 1u;
  bt->head.resize(cap, stream);
  bt->next.resize(static_cast<size_t>(n), stream);
  bt->build_keys.resize(static_cast<size_t>(n), stream);
  CUDF_CUDA_TRY(cudaMemsetAsync(bt->head.data(), 0xFF, cap * sizeof(int32_t), stream.get()));

  rmm::device_uvector<BuildStats> d_stats(1, stream, mr);
  BuildStats init_stats{};
  init_stats.min_k     = static_cast<int64_t>(~0ULL);  // max unsigned when xor'd with sign bit
  init_stats.max_k     = 0;
  init_stats.valid_cnt = 0u;
  init_stats.has_dup   = 0u;
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    d_stats.data(), &init_stats, sizeof(BuildStats), cudaMemcpyHostToDevice, stream.get()));

  auto const dcol = make_device_key_col(key_col);
  int const blk   = 256;
  int const grid  = (n + blk - 1) / blk;
  build_hash_and_stats_kernel<<<grid, blk, 0, stream.get()>>>(
    dcol, n, bt->head.data(), bt->next.data(), bt->build_keys.data(), bt->hash_mask, d_stats.data());
  detect_dup_kernel<<<grid, blk, 0, stream.get()>>>(
    dcol,
    n,
    bt->head.data(),
    bt->next.data(),
    bt->build_keys.data(),
    bt->hash_mask,
    &d_stats.data()->has_dup);

  BuildStats h_stats{};
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    &h_stats, d_stats.data(), sizeof(BuildStats), cudaMemcpyDeviceToHost, stream.get()));
  stream.sync();

  if (h_stats.valid_cnt == 0u) {
    bt->is_empty = true;
    return bt.release();
  }

  bt->is_empty    = false;
  bt->valid_count = static_cast<int32_t>(h_stats.valid_cnt);
  bt->has_dup     = (h_stats.has_dup != 0u);
  bt->min_key     = static_cast<int64_t>(static_cast<uint64_t>(h_stats.min_k) ^ (1ULL << 63));
  int64_t const max_k =
    static_cast<int64_t>(static_cast<uint64_t>(h_stats.max_k) ^ (1ULL << 63));

  if (!bt->has_dup && max_k >= bt->min_key) {
    uint64_t const span = static_cast<uint64_t>(max_k - bt->min_key) + 1ULL;
    if (span <= kMaxLutSpan) {
      bt->lut_span = span;
      bt->lut.resize(span, stream);
      CUDF_CUDA_TRY(cudaMemsetAsync(bt->lut.data(), 0xFF, span * sizeof(int32_t), stream.get()));
      fill_lut_kernel<<<grid, blk, 0, stream.get()>>>(dcol, n, bt->min_key, bt->lut.data());
    }
  }
  return bt.release();
}

void destroy_star_join_table(StarJoinBuildTable* bt) { delete bt; }

int64_t run_star_join_count(
  cudf::table_view const& fact_table,
  std::vector<int32_t> const& probe_key_ords,
  std::vector<StarJoinBuildTable const*> const& build_tables,
  std::vector<int32_t> const& extra_nn_ords,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  int32_t const n = fact_table.num_rows();
  int32_t const k = static_cast<int32_t>(build_tables.size());
  if (n <= 0 || k <= 0) { return 0; }
  CUDF_EXPECTS(k <= kMaxStarDims, "StarJoin exceeds max dims");
  for (int i = 0; i < k; ++i) {
    if (build_tables[i] == nullptr || build_tables[i]->is_empty) { return 0; }
  }

  // Sort probe order by build valid_count ascending (most selective dimension first in L2!)
  std::vector<int> order(static_cast<size_t>(k));
  for (int i = 0; i < k; ++i) { order[i] = i; }
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return build_tables[a]->valid_count < build_tables[b]->valid_count;
  });

  StarProbeArgs args{};
  args.n        = n;
  args.num_dims = k;
  for (int i = 0; i < k; ++i) {
    int const orig      = order[i];
    args.probe_cols[i]  = make_device_key_col(fact_table.column(probe_key_ords[orig]));
    args.dims[i]        = build_tables[orig]->device_view();
  }
  int32_t const en  = std::min<int32_t>(static_cast<int32_t>(extra_nn_ords.size()), kMaxExtraNotNull);
  args.num_extra_nn = en;
  for (int e = 0; e < en; ++e) {
    args.extra_nn[e] = make_device_nn_col(fact_table.column(extra_nn_ords[e]));
  }

  rmm::device_uvector<unsigned long long> d_cnt(1, stream, mr);
  CUDF_CUDA_TRY(cudaMemsetAsync(d_cnt.data(), 0, sizeof(unsigned long long), stream.get()));

  int const blk  = 256;
  int const grid = std::min((n + blk - 1) / blk, 2048);
  star_join_count_kernel<<<grid, blk, 0, stream.get()>>>(args, d_cnt.data());

  unsigned long long h_cnt = 0ULL;
  CUDF_CUDA_TRY(cudaMemcpyAsync(
    &h_cnt, d_cnt.data(), sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream.get()));
  stream.sync();
  return static_cast<int64_t>(h_cnt);
}

std::unique_ptr<cudf::table> run_star_join_materialize(
  cudf::table_view const& fact_table,
  std::vector<int32_t> const& probe_key_ords,
  std::vector<StarJoinBuildTable const*> const& build_tables,
  std::vector<cudf::table_view> const& dim_payload_tables,
  std::vector<int32_t> const& extra_nn_ords,
  std::vector<int32_t> const& out_src_table,  // -1 = fact, 0..K-1 = dim index
  std::vector<int32_t> const& out_src_col,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  int32_t const n     = fact_table.num_rows();
  int32_t const k     = static_cast<int32_t>(build_tables.size());
  int32_t const n_out = static_cast<int32_t>(out_src_table.size());
  CUDF_EXPECTS(k > 0 && k <= kMaxStarDims, "StarJoin invalid dim count");

  auto make_empty_output = [&]() -> std::unique_ptr<cudf::table> {
    std::vector<std::unique_ptr<cudf::column>> empty_cols;
    empty_cols.reserve(static_cast<size_t>(n_out));
    for (int32_t o = 0; o < n_out; ++o) {
      auto const cv = (out_src_table[o] < 0)
                        ? fact_table.column(out_src_col[o])
                        : dim_payload_tables[out_src_table[o]].column(out_src_col[o]);
      empty_cols.push_back(cudf::empty_like(cv));
    }
    return std::make_unique<cudf::table>(std::move(empty_cols));
  };

  if (n <= 0) { return make_empty_output(); }
  bool any_dup = false;
  for (int i = 0; i < k; ++i) {
    if (build_tables[i] == nullptr || build_tables[i]->is_empty) { return make_empty_output(); }
    if (build_tables[i]->has_dup) { any_dup = true; }
  }

  // Determine which source tables (fact & dims) actually contribute output columns
  std::vector<int32_t> fact_needed_cols;
  std::vector<int32_t> fact_out_pos;
  std::vector<std::vector<int32_t>> dim_needed_cols(static_cast<size_t>(k));
  std::vector<std::vector<int32_t>> dim_out_pos(static_cast<size_t>(k));
  for (int32_t o = 0; o < n_out; ++o) {
    if (out_src_table[o] < 0) {
      fact_needed_cols.push_back(out_src_col[o]);
      fact_out_pos.push_back(o);
    } else {
      dim_needed_cols[out_src_table[o]].push_back(out_src_col[o]);
      dim_out_pos[out_src_table[o]].push_back(o);
    }
  }

  // Sort probe order by build valid_count ascending (most selective first!)
  std::vector<int> order(static_cast<size_t>(k));
  for (int i = 0; i < k; ++i) { order[i] = i; }
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return build_tables[a]->valid_count < build_tables[b]->valid_count;
  });

  StarProbeArgs args{};
  args.n        = n;
  args.num_dims = k;
  std::vector<int> out_slot_to_orig_dim;
  for (int i = 0; i < k; ++i) {
    int const orig     = order[i];
    args.probe_cols[i] = make_device_key_col(fact_table.column(probe_key_ords[orig]));
    args.dims[i]       = build_tables[orig]->device_view();
    if (!dim_needed_cols[orig].empty()) {
      args.dim_to_out_slot[i] = static_cast<int32_t>(out_slot_to_orig_dim.size());
      out_slot_to_orig_dim.push_back(orig);
    } else {
      args.dim_to_out_slot[i] = -1;
    }
  }
  int32_t const n_dim_slots = static_cast<int32_t>(out_slot_to_orig_dim.size());
  args.num_out_dim_slots    = n_dim_slots;
  int32_t const en = std::min<int32_t>(static_cast<int32_t>(extra_nn_ords.size()), kMaxExtraNotNull);
  args.num_extra_nn = en;
  for (int e = 0; e < en; ++e) {
    args.extra_nn[e] = make_device_nn_col(fact_table.column(extra_nn_ords[e]));
  }

  bool const need_fact = !fact_needed_cols.empty();
  int32_t out_rows     = 0;
  rmm::device_uvector<int32_t> fact_idx(0, stream, mr);
  std::vector<rmm::device_uvector<int32_t>> dim_idxs;
  dim_idxs.reserve(static_cast<size_t>(n_dim_slots));

  if (!any_dup) {
    if (need_fact) { fact_idx.resize(static_cast<size_t>(n), stream); }
    for (int s = 0; s < n_dim_slots; ++s) {
      dim_idxs.emplace_back(static_cast<size_t>(n), stream, mr);
    }
    int32_t* d_ptrs[kMaxStarDims] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    for (int s = 0; s < n_dim_slots; ++s) { d_ptrs[s] = dim_idxs[s].data(); }

    rmm::device_uvector<uint32_t> d_cnt(1, stream, mr);
    CUDF_CUDA_TRY(cudaMemsetAsync(d_cnt.data(), 0, sizeof(uint32_t), stream.get()));
    int const blk  = 256;
    int const grid = std::min(((n + 31) / 32 * 32 + blk - 1) / blk, 2048);
    star_join_probe_unique_kernel<<<grid, blk, 0, stream.get()>>>(
      args,
      need_fact ? fact_idx.data() : nullptr,
      d_ptrs[0],
      d_ptrs[1],
      d_ptrs[2],
      d_ptrs[3],
      d_ptrs[4],
      d_ptrs[5],
      d_cnt.data());

    uint32_t h_cnt = 0u;
    CUDF_CUDA_TRY(
      cudaMemcpyAsync(&h_cnt, d_cnt.data(), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream.get()));
    stream.sync();
    out_rows = static_cast<int32_t>(h_cnt);
  } else {
    rmm::device_uvector<int32_t> row_mults(static_cast<size_t>(n), stream, mr);
    rmm::device_uvector<int32_t> offsets(static_cast<size_t>(n), stream, mr);
    int const blk  = 256;
    int const grid = (n + blk - 1) / blk;
    star_join_dup_count_kernel<<<grid, blk, 0, stream.get()>>>(args, row_mults.data());
    thrust::exclusive_scan(
      thrust::cuda::par.on(stream.get()),
      row_mults.data(),
      row_mults.data() + n,
      offsets.data());
    int32_t last_off = 0, last_m = 0;
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      &last_off, offsets.data() + (n - 1), sizeof(int32_t), cudaMemcpyDeviceToHost, stream.get()));
    CUDF_CUDA_TRY(cudaMemcpyAsync(
      &last_m, row_mults.data() + (n - 1), sizeof(int32_t), cudaMemcpyDeviceToHost, stream.get()));
    stream.sync();
    out_rows = last_off + last_m;
    if (out_rows > 0) {
      if (need_fact) { fact_idx.resize(static_cast<size_t>(out_rows), stream); }
      for (int s = 0; s < n_dim_slots; ++s) {
        dim_idxs.emplace_back(static_cast<size_t>(out_rows), stream, mr);
      }
      int32_t* d_ptrs[kMaxStarDims] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
      for (int s = 0; s < n_dim_slots; ++s) { d_ptrs[s] = dim_idxs[s].data(); }
      star_join_dup_emit_kernel<<<grid, blk, 0, stream.get()>>>(
        args,
        row_mults.data(),
        offsets.data(),
        need_fact ? fact_idx.data() : nullptr,
        d_ptrs[0],
        d_ptrs[1],
        d_ptrs[2],
        d_ptrs[3],
        d_ptrs[4],
        d_ptrs[5]);
    }
  }

  if (out_rows <= 0) { return make_empty_output(); }

  // Final ONE-SHOT gather ONLY on surviving rows & ONLY on needed output columns!
  std::vector<std::unique_ptr<cudf::column>> final_cols(static_cast<size_t>(n_out));
  if (need_fact) {
    std::vector<cudf::column_view> f_views;
    f_views.reserve(fact_needed_cols.size());
    for (int32_t c : fact_needed_cols) { f_views.push_back(fact_table.column(c)); }
    cudf::column_view gm(cudf::data_type{cudf::type_id::INT32}, out_rows, fact_idx.data(), nullptr, 0);
    auto gathered = cudf::gather(
      cudf::table_view{f_views}, gm, cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
    auto gcols = gathered->release();
    for (size_t i = 0; i < fact_out_pos.size(); ++i) {
      final_cols[static_cast<size_t>(fact_out_pos[i])] = std::move(gcols[i]);
    }
  }
  for (int s = 0; s < n_dim_slots; ++s) {
    int const orig = out_slot_to_orig_dim[s];
    std::vector<cudf::column_view> d_views;
    d_views.reserve(dim_needed_cols[orig].size());
    for (int32_t c : dim_needed_cols[orig]) { d_views.push_back(dim_payload_tables[orig].column(c)); }
    cudf::column_view gm(
      cudf::data_type{cudf::type_id::INT32}, out_rows, dim_idxs[s].data(), nullptr, 0);
    auto gathered = cudf::gather(
      cudf::table_view{d_views}, gm, cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
    auto gcols = gathered->release();
    for (size_t i = 0; i < dim_out_pos[orig].size(); ++i) {
      final_cols[static_cast<size_t>(dim_out_pos[orig][i])] = std::move(gcols[i]);
    }
  }
  return std::make_unique<cudf::table>(std::move(final_cols));
}

}  // namespace rapids::jni::lenc

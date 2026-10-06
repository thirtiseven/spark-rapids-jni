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

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>

// Process-global gen-once input cache shared by EVERY get_json_object benchmark linked into
// this binary: each distinct input is generated exactly once per process and uploaded to the
// device exactly once (single H2D), no matter how many benchmark cells read it. nvbench runs
// benchmark invocations serially in one thread, so the map needs no locking.
namespace spark_rapids_jni {
namespace bench {

using input_cache_key = std::uint64_t;

/**
 * @brief Pack (scenario, size_bytes, seed) into a single cache key.
 *
 * `size_bytes` must fit 32 bits (max ~4 GiB); `scenario` and `seed` must fit 16 bits.
 * `scenario` disambiguates different input shapes for the same byte size (e.g. a row kind or
 * a data_factory `max_depth`), `seed` different generator seeds.
 */
constexpr input_cache_key make_input_cache_key(std::uint64_t scenario,
                                               std::uint64_t size_bytes,
                                               std::uint64_t seed)
{
  return (scenario << 48) | ((size_bytes & 0xFFFF'FFFFULL) << 16) | (seed & 0xFFFFULL);
}

/**
 * @brief Return a view of the process-cached input column for `key`, generating it exactly
 * once.
 *
 * Ownership: the map stores every cached column as `std::unique_ptr<cudf::column>`; view types
 * are passed BY VALUE; a borrowed view is valid only while the owning column lives — the
 * never-destroyed cache owns every cached column for the whole process.
 *
 * On the first call `generate()` runs and its column is stored; every later call with the
 * same key returns that stored column, so the generator must be deterministic in the key.
 *
 * The map is heap-allocated and deliberately never destroyed: freeing the cached columns'
 * device memory during static teardown crashes (cudaFreeAsync after the CUDA runtime has
 * started exiting — observed as a SIGSEGV from `cudaEventRecord` under cuda-gdb), so the
 * columns are intentionally leaked and reclaimed by the driver at process exit.
 *
 * Memory growth is bounded by the number of distinct keys actually generated (worst case:
 * every benchmark's full input set, a few hundred MiB), not by cell count or run time.
 */
// One shared map instance: an inline function's static local is unique per FUNCTION, so every
// cache consumer must reach the cache through THIS accessor or its statics would be distinct
// maps (a split cache means clear/inspect logic silently no-ops).
inline std::unordered_map<input_cache_key, std::unique_ptr<cudf::column>>& input_cache()
{
  static auto* cache = new std::unordered_map<input_cache_key, std::unique_ptr<cudf::column>>();
  return *cache;
}

template <typename Generate>
cudf::column_view get_or_generate_input(input_cache_key key, Generate&& generate)
{
  auto& cache = input_cache();
  auto it     = cache.find(key);
  if (it == cache.end()) { it = cache.try_emplace(key, generate()).first; }
  return it->second->view();
}

}  // namespace bench
}  // namespace spark_rapids_jni

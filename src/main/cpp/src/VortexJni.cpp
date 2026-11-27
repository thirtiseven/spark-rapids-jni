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

#include "cudf_jni_apis.hpp"

#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/filling.hpp>
#include <cudf/interop.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <atomic>
#include <bit>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE
struct ArrowSchema {
  char const* format;
  char const* name;
  char const* metadata;
  int64_t flags;
  int64_t n_children;
  struct ArrowSchema** children;
  struct ArrowSchema* dictionary;
  void (*release)(struct ArrowSchema*);
  void* private_data;
};
struct ArrowArray {
  int64_t length;
  int64_t null_count;
  int64_t offset;
  int64_t n_buffers;
  int64_t n_children;
  void const** buffers;
  struct ArrowArray** children;
  struct ArrowArray* dictionary;
  void (*release)(struct ArrowArray*);
  void* private_data;
};
#endif

#ifndef ARROW_C_DEVICE_DATA_INTERFACE
#define ARROW_C_DEVICE_DATA_INTERFACE
typedef int32_t ArrowDeviceType;
struct ArrowDeviceArray {
  struct ArrowArray array;
  int64_t device_id;
  ArrowDeviceType device_type;
  void* sync_event;
  int64_t reserved[3];
};
#endif

namespace {

struct vx_session;
struct vx_error;
struct vx_dtype;
struct vx_array;
struct vx_array_sink;
struct vx_cuda_device_batches;

struct vx_view {
  char const* ptr;
  size_t len;
};

struct vx_cuda_scan_options {
  uint32_t flags;
  size_t batch_rows;
};

// Forward declaration so vortex_rmm_alloc_cb can drain pending async zero-copy owners on OOM retry!
void drain_arrow_owner_reaper() noexcept;

// Unified RMM allocator callbacks injected into vortex-cuda so Vortex and Spark RAPIDS
// share ONE unified spillable RMM GPU pool (eliminates allocFraction=0.72 fragmentation & OOM!).
constexpr size_t kVortexRmmAlign = 256u;
static std::atomic<int64_t> g_rmm_alloc_bytes{0};
static std::atomic<int64_t> g_rmm_free_bytes{0};
static std::atomic<int64_t> g_owners_created{0};
static std::atomic<int64_t> g_owners_freed{0};

extern "C" uint64_t vortex_rmm_alloc_cb(size_t bytes, void* cu_stream) noexcept
{
  auto try_alloc = [&]() -> uint64_t {
    auto mr = cudf::get_current_device_resource_ref();
    void* p = mr.allocate(
      cuda::stream_ref(static_cast<cudaStream_t>(cu_stream)), bytes, kVortexRmmAlign);
    g_rmm_alloc_bytes.fetch_add(static_cast<int64_t>(bytes), std::memory_order_relaxed);
    return reinterpret_cast<uint64_t>(p);
  };
  try {
    return try_alloc();
  } catch (...) {
    // Reclaim any completed/in-flight ZeroCopyArrowOwners sitting in the async reaper, then retry!
    drain_arrow_owner_reaper();
    try {
      return try_alloc();
    } catch (std::exception const& e) {
      fprintf(stderr,
              "VortexJni: vortex_rmm_alloc_cb(%zu) failed after reaper drain: %s "
              "[vortex_live_MB=%.1f alloc_MB=%.1f free_MB=%.1f owners=%lld/%lld]\n",
              bytes,
              e.what(),
              (g_rmm_alloc_bytes.load() - g_rmm_free_bytes.load()) / (1024.0 * 1024.0),
              g_rmm_alloc_bytes.load() / (1024.0 * 1024.0),
              g_rmm_free_bytes.load() / (1024.0 * 1024.0),
              static_cast<long long>(g_owners_created.load()),
              static_cast<long long>(g_owners_freed.load()));
      return 0ULL;
    } catch (...) {
      return 0ULL;
    }
  }
}

extern "C" void vortex_rmm_free_cb(uint64_t ptr, size_t bytes, void* cu_stream) noexcept
{
  if (ptr == 0ULL) { return; }
  try {
    auto mr = cudf::get_current_device_resource_ref();
    mr.deallocate(cuda::stream_ref(static_cast<cudaStream_t>(cu_stream)),
                  reinterpret_cast<void*>(ptr),
                  bytes,
                  kVortexRmmAlign);
    g_rmm_free_bytes.fetch_add(static_cast<int64_t>(bytes), std::memory_order_relaxed);
  } catch (std::exception const& e) {
    fprintf(stderr, "VortexJni: vortex_rmm_free_cb(%zu) exception: %s\n", bytes, e.what());
  } catch (...) {
  }
}

struct VortexCudaApi {
  void* dl{nullptr};
  void (*set_external_allocator)(uint64_t (*)(size_t, void*), void (*)(uint64_t, size_t, void*)) =
    nullptr;
  vx_session* (*session_new)(vx_error**)                                                    = nullptr;
  void (*session_free)(vx_session const*)                                                   = nullptr;
  void (*error_free)(vx_error const*)                                                       = nullptr;
  vx_view (*error_message)(vx_error const*)                                                 = nullptr;
  vx_dtype const* (*dtype_from_arrow)(vx_session const*, ArrowSchema*, vx_error**)          = nullptr;
  void (*dtype_free)(vx_dtype const*)                                                       = nullptr;
  vx_array const* (*array_from_arrow)(vx_session const*, ArrowArray*, ArrowSchema*, bool, vx_error**) =
    nullptr;
  void (*array_free)(vx_array const*)                                                       = nullptr;
  vx_array_sink* (*sink_open_file)(vx_session const*, vx_view, vx_dtype const*, vx_error**) = nullptr;
  void (*sink_push)(vx_array_sink*, vx_array const*, vx_error**)                            = nullptr;
  void (*sink_close)(vx_array_sink*, vx_error**)                                            = nullptr;
  vx_cuda_device_batches* (*scan_device_batches)(vx_session const*,
                                                 vx_view,
                                                 vx_cuda_scan_options const*,
                                                 vx_view const*,
                                                 size_t,
                                                 vx_error**)                                = nullptr;
  vx_cuda_device_batches* (*scan_paths_prefetch_batches)(vx_session const*,
                                                         vx_view const*,
                                                         size_t,
                                                         vx_cuda_scan_options const*,
                                                         vx_view const*,
                                                         size_t,
                                                         size_t,
                                                         vx_error**)                        = nullptr;
  vx_cuda_device_batches* (*scan_device_batches_with_rows)(vx_session const*,
                                                           vx_view,
                                                           vx_cuda_scan_options const*,
                                                           vx_view const*,
                                                           size_t,
                                                           uint64_t const*,
                                                           size_t,
                                                           vx_error**)                      = nullptr;
  int (*device_batches_next)(
    vx_cuda_device_batches*, ArrowSchema*, ArrowDeviceArray*, vx_error**)                   = nullptr;
  int (*device_batches_next_merged)(vx_cuda_device_batches*,
                                    size_t,
                                    size_t,
                                    size_t,
                                    size_t,
                                    ArrowSchema*,
                                    ArrowDeviceArray*,
                                    int32_t*,
                                    int32_t*,
                                    size_t,
                                    size_t*,
                                    uint8_t*,
                                    vx_error**)                                             = nullptr;
  size_t (*device_batches_last_file_idx)(vx_cuda_device_batches const*)                     = nullptr;
  void (*device_batches_free)(vx_cuda_device_batches*)                                      = nullptr;
  vx_session* shared_session{nullptr};

  template <typename Fn>
  void load(Fn& out, char const* sym)
  {
    void* p = dlsym(dl, sym);
    CUDF_EXPECTS(p != nullptr, std::string("VortexJni: missing symbol ") + sym);
    out = reinterpret_cast<Fn>(p);
  }

  std::string consume_error(vx_error* err) const
  {
    if (err == nullptr) { return "unknown vortex error"; }
    vx_view v = error_message(err);
    std::string msg(v.ptr ? v.ptr : "", v.len);
    error_free(err);
    return msg;
  }

  static VortexCudaApi& get()
  {
    static VortexCudaApi api = []() {
      VortexCudaApi a{};
      char const* env_path = std::getenv("VORTEX_CUDA_FFI_LIB");
      std::string lib_path =
        (env_path != nullptr && env_path[0] != '\0')
          ? std::string(env_path)
          : std::string("/disk2/haoyangl/plugin_perf/light-decode-jit/vortex-target/release/libvortex_cuda_ffi.so");
      a.dl = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
      CUDF_EXPECTS(a.dl != nullptr,
                   "VortexJni: dlopen failed for " + lib_path + ": " +
                     (dlerror() ? dlerror() : "unknown"));
      a.load(a.set_external_allocator, "vx_cuda_set_external_allocator");
      a.load(a.session_new, "vx_cuda_session_new");
      a.load(a.session_free, "vx_session_free");
      a.load(a.error_free, "vx_error_free");
      a.load(a.error_message, "vx_error_message");
      a.load(a.dtype_from_arrow, "vx_dtype_from_arrow_schema");
      a.load(a.dtype_free, "vx_dtype_free");
      a.load(a.array_from_arrow, "vx_array_from_arrow");
      a.load(a.array_free, "vx_array_free");
      a.load(a.sink_open_file, "vx_cuda_array_sink_open_file");
      a.load(a.sink_push, "vx_array_sink_push");
      a.load(a.sink_close, "vx_array_sink_close");
      a.load(a.scan_device_batches, "vx_cuda_scan_path_device_batches");
      a.load(a.scan_paths_prefetch_batches, "vx_cuda_scan_paths_prefetch_batches");
      a.load(a.scan_device_batches_with_rows, "vx_cuda_scan_path_device_batches_with_rows");
      a.load(a.device_batches_next, "vx_cuda_device_batches_next");
      a.load(a.device_batches_next_merged, "vx_cuda_device_batches_next_merged");
      a.load(a.device_batches_last_file_idx, "vx_cuda_device_batches_last_file_idx");
      a.load(a.device_batches_free, "vx_cuda_device_batches_free");

      // Inject Spark RAPIDS RMM pool into vortex-cuda BEFORE creating the shared session!
      a.set_external_allocator(&vortex_rmm_alloc_cb, &vortex_rmm_free_cb);

      vx_error* err    = nullptr;
      a.shared_session = a.session_new(&err);
      CUDF_EXPECTS(a.shared_session != nullptr && err == nullptr,
                   "VortexJni: vx_cuda_session_new failed: " + a.consume_error(err));
      return a;
    }();
    return api;
  }
};

struct RawArrowBatch {
  ArrowSchema schema{};
  ArrowDeviceArray dev_arr{};
  int32_t file_idx{0};

  RawArrowBatch() = default;
  RawArrowBatch(RawArrowBatch&& o) noexcept : schema(o.schema), dev_arr(o.dev_arr), file_idx(o.file_idx)
  {
    o.schema.release        = nullptr;
    o.dev_arr.array.release = nullptr;
  }
  RawArrowBatch& operator=(RawArrowBatch&& o) noexcept
  {
    if (this != &o) {
      release();
      schema                  = o.schema;
      dev_arr                 = o.dev_arr;
      file_idx                = o.file_idx;
      o.schema.release        = nullptr;
      o.dev_arr.array.release = nullptr;
    }
    return *this;
  }
  RawArrowBatch(RawArrowBatch const&)            = delete;
  RawArrowBatch& operator=(RawArrowBatch const&) = delete;

  void release() noexcept
  {
    if (dev_arr.array.release != nullptr) {
      dev_arr.array.release(&dev_arr.array);
      dev_arr.array.release = nullptr;
    }
    if (schema.release != nullptr) {
      schema.release(&schema);
      schema.release = nullptr;
    }
  }
  ~RawArrowBatch() { release(); }
};

// Recursively patches ArrowDeviceArray nodes whose buffers[0] == nullptr while null_count != 0
// (e.g. all-null Vortex columns or dictionary keys in TPC-DS), restoring nullptr before release.
// Also temporarily patches Arrow UNKNOWN_NULL_COUNT (-1) with a non-null mask to 1 so
// cudf::from_arrow_device keeps the validity bitmask intact for fused cudf::concatenate popcount!
struct ArrowNullMaskPatcher {
  std::vector<rmm::device_buffer> masks;
  std::vector<ArrowArray*> patched_bufs;
  std::vector<ArrowArray*> patched_unknown_nc;

  bool patch_recursive(ArrowArray* a, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
  {
    bool saw_unknown = false;
    if (a == nullptr) { return false; }
    if (a->n_buffers > 0 && a->buffers != nullptr) {
      if (a->buffers[0] == nullptr) {
        if (a->null_count > 0 && a->length > 0) {
          size_t const mbytes = cudf::bitmask_allocation_size_bytes(
            static_cast<cudf::size_type>(a->length + a->offset));
          rmm::device_buffer mask(mbytes, stream, mr);
          CUDF_CUDA_TRY(cudaMemsetAsync(mask.data(), 0, mbytes, stream.get()));
          const_cast<void const**>(a->buffers)[0] = mask.data();
          masks.push_back(std::move(mask));
          patched_bufs.push_back(a);
        } else if (a->null_count < 0) {
          a->null_count = 0;
        }
      } else if (a->null_count < 0) {
        a->null_count = 1;
        patched_unknown_nc.push_back(a);
        saw_unknown = true;
      }
    }
    if (a->dictionary != nullptr) {
      saw_unknown = patch_recursive(a->dictionary, stream, mr) || saw_unknown;
    }
    if (a->children != nullptr) {
      for (int64_t i = 0; i < a->n_children; ++i) {
        saw_unknown = patch_recursive(a->children[i], stream, mr) || saw_unknown;
      }
    }
    return saw_unknown;
  }

  void restore() noexcept
  {
    for (auto* a : patched_bufs) {
      if (a != nullptr && a->buffers != nullptr && a->n_buffers > 0) {
        const_cast<void const**>(a->buffers)[0] = nullptr;
      }
    }
    for (auto* a : patched_unknown_nc) {
      if (a != nullptr) { a->null_count = -1; }
    }
    patched_bufs.clear();
    patched_unknown_nc.clear();
    masks.clear();
  }

  ~ArrowNullMaskPatcher() { restore(); }
};

// Async release payload so materialize_raw_batches NEVER blocks the Spark thread on stream.sync()!
struct MaterializedInputsOwner {
  std::vector<RawArrowBatch> raw_batches;
  ArrowNullMaskPatcher patcher;
  std::vector<cudf::unique_table_view_t> holders;
  std::vector<std::vector<std::unique_ptr<cudf::column>>> decoded_temps;

  ~MaterializedInputsOwner()
  {
    decoded_temps.clear();
    holders.clear();
    patcher.restore();
    raw_batches.clear();
  }
};

void enqueue_materialized_inputs_async(MaterializedInputsOwner* payload, cudaStream_t stream);

// Single-pass zero-double-copy materializer: takes 1+ raw ArrowDeviceArray batches,
// imports zero-copy via cudf::from_arrow_device, and runs ONE GPU concatenate with
// fused __popc exact null_count + async off-thread input cleanup (ZERO stream.sync()!)
std::unique_ptr<cudf::table> materialize_raw_batches(
  std::vector<RawArrowBatch>& raw_batches,
  std::vector<uint8_t> const& keep_dict32,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto payload         = std::make_unique<MaterializedInputsOwner>();
  payload->raw_batches = std::move(raw_batches);
  payload->holders.reserve(payload->raw_batches.size());
  payload->decoded_temps.reserve(payload->raw_batches.size());

  std::vector<cudf::table_view> norm_views;
  norm_views.reserve(payload->raw_batches.size());
  std::vector<uint8_t> col_unknown_nc;

  for (auto& rb : payload->raw_batches) {
    if (rb.dev_arr.array.n_children > 0 && rb.dev_arr.array.children != nullptr) {
      if (col_unknown_nc.size() < static_cast<size_t>(rb.dev_arr.array.n_children)) {
        col_unknown_nc.resize(static_cast<size_t>(rb.dev_arr.array.n_children), 0u);
      }
      for (int64_t c = 0; c < rb.dev_arr.array.n_children; ++c) {
        if (payload->patcher.patch_recursive(rb.dev_arr.array.children[c], stream, mr)) {
          col_unknown_nc[static_cast<size_t>(c)] = 1u;
        }
      }
    } else {
      payload->patcher.patch_recursive(&rb.dev_arr.array, stream, mr);
    }
    auto h  = cudf::from_arrow_device(&rb.schema, &rb.dev_arr, stream, mr);
    auto tv = *h;
    std::vector<cudf::column_view> cols;
    cols.reserve(static_cast<size_t>(tv.num_columns()));
    std::vector<std::unique_ptr<cudf::column>> temps;
    for (cudf::size_type c = 0; c < tv.num_columns(); ++c) {
      auto cv = tv.column(c);
      bool const want_d32 =
        (static_cast<size_t>(c) < keep_dict32.size() && keep_dict32[static_cast<size_t>(c)] != 0);
      if (cv.type().id() == cudf::type_id::DICTIONARY32 && !want_d32) {
        auto dec = cudf::dictionary::decode(cudf::dictionary_column_view(cv), stream, mr);
        cols.push_back(dec->view());
        temps.push_back(std::move(dec));
      } else {
        cols.push_back(cv);
      }
    }
    payload->holders.push_back(std::move(h));
    payload->decoded_temps.push_back(std::move(temps));
    norm_views.emplace_back(cols);
  }

  std::unique_ptr<cudf::table> out_tbl;
  if (norm_views.size() == 1) {
    auto const& tv = norm_views[0];
    std::vector<std::unique_ptr<cudf::column>> owned_cols;
    owned_cols.reserve(static_cast<size_t>(tv.num_columns()));
    size_t tmp_idx = 0;
    auto& temps0   = payload->decoded_temps[0];
    for (cudf::size_type c = 0; c < tv.num_columns(); ++c) {
      auto cv = tv.column(c);
      std::unique_ptr<cudf::column> col;
      if (tmp_idx < temps0.size() && temps0[tmp_idx]->view().head() == cv.head() &&
          temps0[tmp_idx]->type() == cv.type()) {
        col = std::move(temps0[tmp_idx++]);
      } else {
        col = std::make_unique<cudf::column>(cv, stream, mr);
      }
      if (static_cast<size_t>(c) < col_unknown_nc.size() && col_unknown_nc[static_cast<size_t>(c)] != 0 &&
          col->nullable()) {
        auto nc = cudf::null_count(col->view().null_mask(), 0, col->size(), stream);
        col->set_null_count(nc);
      }
      owned_cols.push_back(std::move(col));
    }
    out_tbl = std::make_unique<cudf::table>(std::move(owned_cols));
  } else {
    // cudf::concatenate computes exact null_count for all K merged files inside its fused
    // kernel (or per-column bitmask concat), eliminating K-1 per-file DtoH sync stalls!
    out_tbl = cudf::concatenate(norm_views, stream, mr);
  }
  enqueue_materialized_inputs_async(payload.release(), stream.get());
  return out_tbl;
}

// Multi-file scan state backed by pure-CPU off-semaphore NVMe prefetcher + single reused CudaExecutionCtx!
struct VortexMultiFileScanState {
  std::vector<std::string> files;
  std::vector<std::string> col_names;
  size_t batch_rows{0};
  size_t prefetch_depth{3};
  uint32_t scan_flags{0};
  vx_cuda_device_batches* active{nullptr};
  bool started{false};
  int32_t last_file_idx{0};
  std::deque<RawArrowBatch> pushback;

  void start_prefetch()
  {
    if (started || files.empty()) { return; }
    started   = true;
    auto& api = VortexCudaApi::get();
    std::vector<vx_view> path_views;
    path_views.reserve(files.size());
    for (auto const& f : files) { path_views.push_back(vx_view{f.data(), f.size()}); }
    std::vector<vx_view> col_views;
    col_views.reserve(col_names.size());
    for (auto const& s : col_names) { col_views.push_back(vx_view{s.data(), s.size()}); }
    vx_cuda_scan_options opts{};
    opts.flags      = scan_flags;
    opts.batch_rows = batch_rows;
    vx_error* err   = nullptr;
    active          = api.scan_paths_prefetch_batches(api.shared_session,
                                             path_views.data(),
                                             path_views.size(),
                                             &opts,
                                             col_views.empty() ? nullptr : col_views.data(),
                                             col_views.size(),
                                             prefetch_depth > 0 ? prefetch_depth : 3u,
                                             &err);
    CUDF_EXPECTS(active != nullptr && err == nullptr,
                 "VortexJni: scan_paths_prefetch_batches failed: " + api.consume_error(err));
  }

  bool pop_next(RawArrowBatch& out)
  {
    if (!pushback.empty()) {
      out           = std::move(pushback.front());
      last_file_idx = out.file_idx;
      pushback.pop_front();
      return true;
    }
    if (!started) { start_prefetch(); }
    if (active == nullptr) { return false; }
    auto& api = VortexCudaApi::get();
    while (active != nullptr) {
      RawArrowBatch item;
      vx_error* err = nullptr;
      int rc        = api.device_batches_next(active, &item.schema, &item.dev_arr, &err);
      if (rc != 0 || err != nullptr) {
        api.device_batches_free(active);
        active = nullptr;
        CUDF_FAIL("VortexJni: device_batches_next failed: " + api.consume_error(err));
      }
      if (item.dev_arr.array.release != nullptr || item.schema.release != nullptr) {
        if (item.dev_arr.array.length == 0) { continue; }
        item.file_idx = static_cast<int32_t>(api.device_batches_last_file_idx(active));
        last_file_idx = item.file_idx;
        out           = std::move(item);
        return true;
      }
      api.device_batches_free(active);
      active = nullptr;
    }
    return false;
  }

  ~VortexMultiFileScanState()
  {
    pushback.clear();
    if (active != nullptr) {
      VortexCudaApi::get().device_batches_free(active);
      active = nullptr;
    }
  }
};

// True Zero-Copy ArrowDeviceArray Owner: keeps Vortex's GPU buffers & cuDF table_view holder
// alive across all ColumnVectors in the batch until the last GpuColumnVector is closed!
struct ZeroCopyArrowOwner {
  RawArrowBatch raw;
  ArrowNullMaskPatcher null_patcher;
  std::optional<cudf::unique_table_view_t> holder;
  std::vector<std::unique_ptr<cudf::column>> decoded_dict_cols;

  ~ZeroCopyArrowOwner()
  {
    g_owners_freed.fetch_add(1, std::memory_order_relaxed);
    decoded_dict_cols.clear();
    holder.reset();
    null_patcher.restore();
    raw.release();
  }
};

// Non-blocking background reaper: records a CUDA event on the hot stream and releases
// ZeroCopyArrowOwner off the Spark task thread once the event completes (eliminates
// per-batch cudaStreamSynchronize stalls on Q01's 200 small zero-copy batches!).
struct ArrowOwnerReaper {
  struct Pending {
    cudaEvent_t ev;
    ZeroCopyArrowOwner* owner;
    MaterializedInputsOwner* mat_owner;
  };
  std::mutex mtx;
  std::mutex drain_mtx;
  std::condition_variable cv;
  std::deque<Pending> queue;
  bool stop{false};
  int dev{0};
  std::thread worker;

  ArrowOwnerReaper()
  {
    cudaGetDevice(&dev);
    worker = std::thread([this]() {
      cudaSetDevice(dev);
      while (true) {
        Pending p{nullptr, nullptr, nullptr};
        {
          std::unique_lock<std::mutex> lk(mtx);
          cv.wait(lk, [&]() { return !queue.empty() || stop; });
          if (queue.empty() && stop) { break; }
          p = queue.front();
          queue.pop_front();
        }
        if (p.ev != nullptr) {
          cudaEventSynchronize(p.ev);
          cudaEventDestroy(p.ev);
        }
        delete p.owner;
        delete p.mat_owner;
      }
    });
  }

  void enqueue_any(ZeroCopyArrowOwner* owner, MaterializedInputsOwner* mat_owner, cudaStream_t stream)
  {
    cudaEvent_t ev = nullptr;
    if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) == cudaSuccess &&
        cudaEventRecord(ev, stream) == cudaSuccess) {
      std::vector<Pending> overflow;
      {
        std::lock_guard<std::mutex> lk(mtx);
        if (!stop) {
          queue.push_back(Pending{ev, owner, mat_owner});
          while (queue.size() > 8u) {
            overflow.push_back(queue.front());
            queue.pop_front();
          }
          cv.notify_one();
          ev        = nullptr;
          owner     = nullptr;
          mat_owner = nullptr;
        }
      }
      for (auto& p : overflow) {
        if (p.ev != nullptr) {
          cudaEventSynchronize(p.ev);
          cudaEventDestroy(p.ev);
        }
        delete p.owner;
        delete p.mat_owner;
      }
      if (owner == nullptr && mat_owner == nullptr) { return; }
      cudaEventSynchronize(ev);
      cudaEventDestroy(ev);
    } else {
      if (ev != nullptr) { cudaEventDestroy(ev); }
      cudaStreamSynchronize(stream);
    }
    delete owner;
    delete mat_owner;
  }

  void enqueue(ZeroCopyArrowOwner* owner, cudaStream_t stream)
  {
    enqueue_any(owner, nullptr, stream);
  }

  void enqueue_mat(MaterializedInputsOwner* mat_owner, cudaStream_t stream)
  {
    enqueue_any(nullptr, mat_owner, stream);
  }

  void drain_all()
  {
    // Serialize concurrent OOM callers so thread B waits until thread A finishes freeing `local`!
    std::lock_guard<std::mutex> dlk(drain_mtx);
    std::deque<Pending> local;
    {
      std::lock_guard<std::mutex> lk(mtx);
      local.swap(queue);
    }
    for (auto& p : local) {
      if (p.ev != nullptr) {
        cudaEventSynchronize(p.ev);
        cudaEventDestroy(p.ev);
      }
      delete p.owner;
      delete p.mat_owner;
    }
  }

  ~ArrowOwnerReaper()
  {
    {
      std::lock_guard<std::mutex> lk(mtx);
      stop = true;
    }
    cv.notify_all();
    if (worker.joinable()) { worker.join(); }
    drain_all();
  }

  static ArrowOwnerReaper& get()
  {
    static ArrowOwnerReaper inst;
    return inst;
  }
};

void enqueue_materialized_inputs_async(MaterializedInputsOwner* payload, cudaStream_t stream)
{
  ArrowOwnerReaper::get().enqueue_mat(payload, stream);
}

void drain_arrow_owner_reaper() noexcept
{
  try {
    ArrowOwnerReaper::get().drain_all();
  } catch (...) {
  }
}

}  // namespace

extern "C" {

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_VortexNative_writeTableNative(
  JNIEnv* env, jclass, jlong j_table_view, jobjectArray j_col_names, jstring j_path)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto& api      = VortexCudaApi::get();
    auto const* tv = reinterpret_cast<cudf::table_view const*>(j_table_view);
    cudf::jni::native_jstring path(env, j_path);

    jsize const ncols = env->GetArrayLength(j_col_names);
    std::vector<cudf::column_metadata> metas;
    metas.reserve(static_cast<size_t>(ncols));
    for (jsize i = 0; i < ncols; ++i) {
      auto js = static_cast<jstring>(env->GetObjectArrayElement(j_col_names, i));
      cudf::jni::native_jstring ns(env, js);
      metas.emplace_back(ns.get());
      env->DeleteLocalRef(js);
    }

    auto stream   = cudf::get_default_stream();
    auto schema_d = cudf::to_arrow_schema(*tv, metas);
    auto schema_a = cudf::to_arrow_schema(*tv, metas);
    auto host_arr = cudf::to_arrow_host(*tv, stream);
    stream.sync();

    vx_error* err       = nullptr;
    vx_dtype const* vdt = api.dtype_from_arrow(api.shared_session, schema_d.get(), &err);
    CUDF_EXPECTS(vdt != nullptr && err == nullptr,
                 "VortexJni: vx_dtype_from_arrow_schema failed: " + api.consume_error(err));

    std::string const path_str(path.get());
    vx_view pview{path_str.data(), path_str.size()};
    vx_array_sink* sink = api.sink_open_file(api.shared_session, pview, vdt, &err);
    api.dtype_free(vdt);
    CUDF_EXPECTS(sink != nullptr && err == nullptr,
                 "VortexJni: vx_cuda_array_sink_open_file failed: " + api.consume_error(err));

    ArrowArray raw_arr      = host_arr->array;
    host_arr->array.release = nullptr;
    vx_array const* vx_arr =
      api.array_from_arrow(api.shared_session, &raw_arr, schema_a.get(), false, &err);
    if (vx_arr == nullptr || err != nullptr) {
      if (raw_arr.release != nullptr) { raw_arr.release(&raw_arr); }
      vx_error* cerr = nullptr;
      api.sink_close(sink, &cerr);
      if (cerr != nullptr) { api.error_free(cerr); }
      CUDF_FAIL("VortexJni: vx_array_from_arrow failed: " + api.consume_error(err));
    }

    api.sink_push(sink, vx_arr, &err);
    api.array_free(vx_arr);
    if (err != nullptr) {
      std::string msg = api.consume_error(err);
      vx_error* cerr  = nullptr;
      api.sink_close(sink, &cerr);
      if (cerr != nullptr) { api.error_free(cerr); }
      CUDF_FAIL("VortexJni: vx_array_sink_push failed: " + msg);
    }

    api.sink_close(sink, &err);
    CUDF_EXPECTS(err == nullptr, "VortexJni: vx_array_sink_close failed: " + api.consume_error(err));
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_VortexNative_openMultiScanNative(
  JNIEnv* env, jclass, jobjectArray j_paths, jobjectArray j_col_names, jlong j_batch_rows, jint j_prefetch)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    (void)VortexCudaApi::get();

    auto* st       = new VortexMultiFileScanState();
    st->batch_rows = (j_batch_rows > 0) ? static_cast<size_t>(j_batch_rows) : 0u;
    if (j_prefetch < 0) {
      // Negative prefetch requests fused GPU dict decode + lazy null count for batched multi-file decode!
      st->scan_flags     = (1u << 1) | (1u << 2);  // DECODE_DICT | LAZY_NULL_COUNT
      st->prefetch_depth = static_cast<size_t>(-j_prefetch);
    } else {
      st->scan_flags     = 0u;
      st->prefetch_depth = (j_prefetch > 0) ? static_cast<size_t>(j_prefetch) : 3u;
    }

    jsize const nf = env->GetArrayLength(j_paths);
    st->files.reserve(static_cast<size_t>(nf));
    for (jsize i = 0; i < nf; ++i) {
      auto js = static_cast<jstring>(env->GetObjectArrayElement(j_paths, i));
      cudf::jni::native_jstring ns(env, js);
      st->files.emplace_back(ns.get());
      env->DeleteLocalRef(js);
    }

    jsize const nc = (j_col_names != nullptr) ? env->GetArrayLength(j_col_names) : 0;
    st->col_names.reserve(static_cast<size_t>(nc));
    for (jsize i = 0; i < nc; ++i) {
      auto js = static_cast<jstring>(env->GetObjectArrayElement(j_col_names, i));
      cudf::jni::native_jstring ns(env, js);
      st->col_names.emplace_back(ns.get());
      env->DeleteLocalRef(js);
    }

    // Start pure-CPU off-semaphore NVMe compressed-segment prefetcher immediately!
    st->start_prefetch();

    return std::bit_cast<jlong>(st);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_VortexNative_nextCoalescedBatchNative(
  JNIEnv* env, jclass, jlong j_handle, jbyteArray j_dict32_mask, jlong j_target_bytes)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* st    = std::bit_cast<VortexMultiFileScanState*>(j_handle);
    auto stream = cudf::get_default_stream();
    auto mr     = cudf::get_current_device_resource_ref();

    std::vector<uint8_t> keep_dict32;
    if (j_dict32_mask != nullptr) {
      cudf::jni::native_jbyteArray nb(env, j_dict32_mask);
      keep_dict32.resize(static_cast<size_t>(nb.size()));
      for (size_t i = 0; i < keep_dict32.size(); ++i) {
        keep_dict32[i] = static_cast<uint8_t>(nb[i]);
      }
    }

    RawArrowBatch first;
    if (!st->pop_next(first)) { return nullptr; }

    auto has_kept_d32 = [&](ArrowSchema const& s) -> bool {
      if (keep_dict32.empty()) { return false; }
      for (int64_t c = 0; c < s.n_children && static_cast<size_t>(c) < keep_dict32.size(); ++c) {
        if (keep_dict32[static_cast<size_t>(c)] != 0 && s.children[c]->dictionary != nullptr) {
          return true;
        }
      }
      return false;
    };

    std::vector<RawArrowBatch> raw_batches;
    bool const first_d32 = has_kept_d32(first.schema);
    int64_t acc_rows     = first.dev_arr.array.length;
    size_t acc_bytes     = static_cast<size_t>(acc_rows) * std::max<size_t>(st->col_names.size(), 1u) * 8u;
    raw_batches.push_back(std::move(first));

    size_t const target_b = (j_target_bytes > 0) ? static_cast<size_t>(j_target_bytes) : 0u;
    if (target_b > 0 && !first_d32) {
      while (acc_bytes < target_b && acc_rows < 80000000LL) {
        RawArrowBatch nxt;
        if (!st->pop_next(nxt)) { break; }
        if (has_kept_d32(nxt.schema)) {
          st->pushback.push_front(std::move(nxt));
          break;
        }
        acc_rows += nxt.dev_arr.array.length;
        acc_bytes += static_cast<size_t>(nxt.dev_arr.array.length) *
                     std::max<size_t>(st->col_names.size(), 1u) * 8u;
        raw_batches.push_back(std::move(nxt));
      }
    }

    auto out_tbl = materialize_raw_batches(raw_batches, keep_dict32, stream, mr);
    return cudf::jni::convert_table_for_return(env, std::move(out_tbl));
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_VortexNative_closeMultiScanNative(
  JNIEnv* env, jclass, jlong j_handle)
{
  JNI_TRY
  {
    auto* st = std::bit_cast<VortexMultiFileScanState*>(j_handle);
    delete st;
  }
  JNI_CATCH(env, );
}

// Opt V2: GPU 2-Phase Late Materialization helpers!
// 1. Scan one file completely (or with strictly-sorted u64 row indices via vx_cuda_scan_path_device_batches_with_rows!)
JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_VortexNative_scanFileWithRowsNative(
  JNIEnv* env,
  jclass,
  jstring j_path,
  jobjectArray j_col_names,
  jlongArray j_row_indices,
  jbyteArray j_dict32_mask)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto& api   = VortexCudaApi::get();
    auto stream = cudf::get_default_stream();
    auto mr     = cudf::get_current_device_resource_ref();
    cudf::jni::native_jstring path(env, j_path);
    std::string const path_str(path.get());

    jsize const nc = (j_col_names != nullptr) ? env->GetArrayLength(j_col_names) : 0;
    std::vector<std::string> col_strs;
    std::vector<vx_view> col_views;
    col_strs.reserve(static_cast<size_t>(nc));
    col_views.reserve(static_cast<size_t>(nc));
    for (jsize i = 0; i < nc; ++i) {
      auto js = static_cast<jstring>(env->GetObjectArrayElement(j_col_names, i));
      cudf::jni::native_jstring ns(env, js);
      col_strs.emplace_back(ns.get());
      env->DeleteLocalRef(js);
    }
    for (auto const& s : col_strs) { col_views.push_back(vx_view{s.data(), s.size()}); }

    std::vector<uint8_t> keep_dict32;
    if (j_dict32_mask != nullptr) {
      cudf::jni::native_jbyteArray nb(env, j_dict32_mask);
      keep_dict32.resize(static_cast<size_t>(nb.size()));
      for (size_t i = 0; i < keep_dict32.size(); ++i) {
        keep_dict32[i] = static_cast<uint8_t>(nb[i]);
      }
    }

    std::vector<uint64_t> row_idx;
    uint64_t const* rptr = nullptr;
    size_t rlen          = 0;
    if (j_row_indices != nullptr) {
      cudf::jni::native_jlongArray nr(env, j_row_indices);
      rlen = static_cast<size_t>(nr.size());
      if (rlen == 0) { return nullptr; }
      row_idx.resize(rlen);
      for (size_t i = 0; i < rlen; ++i) { row_idx[i] = static_cast<uint64_t>(nr[i]); }
      rptr = row_idx.data();
    }

    vx_cuda_scan_options opts{};
    opts.flags      = 0u;
    opts.batch_rows = 0u;
    vx_error* err   = nullptr;
    vx_view pview{path_str.data(), path_str.size()};
    vx_cuda_device_batches* batches =
      (rptr != nullptr)
        ? api.scan_device_batches_with_rows(api.shared_session,
                                            pview,
                                            &opts,
                                            col_views.empty() ? nullptr : col_views.data(),
                                            col_views.size(),
                                            rptr,
                                            rlen,
                                            &err)
        : api.scan_device_batches(api.shared_session,
                                  pview,
                                  &opts,
                                  col_views.empty() ? nullptr : col_views.data(),
                                  col_views.size(),
                                  &err);
    CUDF_EXPECTS(batches != nullptr && err == nullptr,
                 "VortexJni: scanFileWithRows failed on " + path_str + ": " + api.consume_error(err));

    std::vector<RawArrowBatch> raw_batches;
    while (true) {
      RawArrowBatch item;
      vx_error* berr = nullptr;
      int rc         = api.device_batches_next(batches, &item.schema, &item.dev_arr, &berr);
      if (rc != 0 || berr != nullptr) {
        api.device_batches_free(batches);
        CUDF_FAIL("VortexJni: scanFileWithRows next failed: " + api.consume_error(berr));
      }
      if (item.dev_arr.array.release == nullptr && item.schema.release == nullptr) { break; }
      if (item.dev_arr.array.length == 0) { continue; }
      raw_batches.push_back(std::move(item));
    }
    api.device_batches_free(batches);
    if (raw_batches.empty()) { return nullptr; }

    auto out_tbl = materialize_raw_batches(raw_batches, keep_dict32, stream, mr);
    return cudf::jni::convert_table_for_return(env, std::move(out_tbl));
  }
  JNI_CATCH(env, nullptr);
}

// 2. Compact a BOOL8 GPU mask column into strictly-sorted int64_t surviving row indices on host!
JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_VortexNative_maskToRowIndicesNative(
  JNIEnv* env, jclass, jlong j_mask_col_view)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto stream    = cudf::get_default_stream();
    auto mr        = cudf::get_current_device_resource_ref();
    auto const* mv = reinterpret_cast<cudf::column_view const*>(j_mask_col_view);
    auto const n   = mv->size();
    if (n == 0) {
      cudf::jni::native_jlongArray empty(env, 0);
      return empty.get_jArray();
    }
    cudf::numeric_scalar<int64_t> zero(0, true, stream);
    auto seq     = cudf::sequence(n, zero, stream, mr);
    auto idx_tbl = cudf::apply_retention_mask(cudf::table_view({seq->view()}), *mv, stream, mr);
    auto const m = idx_tbl->num_rows();
    cudf::jni::native_jlongArray ret(env, m);
    if (m > 0) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(ret.data(),
                                    idx_tbl->get_column(0).view().data<int64_t>(),
                                    static_cast<size_t>(m) * sizeof(int64_t),
                                    cudaMemcpyDeviceToHost,
                                    stream.get()));
      stream.sync();
    }
    return ret.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

// True Zero-Copy: returns [ownerPtr, approxBytes, colViewHandle_0, ..., colViewHandle_{nc-1}]
// ZERO GPU D2D copy and ZERO CPU stream.sync() on the hot read path!
JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_VortexNative_nextZeroCopyBatchNative(
  JNIEnv* env, jclass, jlong j_handle, jbyteArray j_dict32_mask)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* st    = std::bit_cast<VortexMultiFileScanState*>(j_handle);
    auto stream = cudf::get_default_stream();
    auto mr     = cudf::get_current_device_resource_ref();

    std::vector<uint8_t> keep_dict32;
    if (j_dict32_mask != nullptr) {
      cudf::jni::native_jbyteArray nb(env, j_dict32_mask);
      keep_dict32.resize(static_cast<size_t>(nb.size()));
      for (size_t i = 0; i < keep_dict32.size(); ++i) {
        keep_dict32[i] = static_cast<uint8_t>(nb[i]);
      }
    }

    auto owner = std::make_unique<ZeroCopyArrowOwner>();
    if (!st->pop_next(owner->raw)) { return nullptr; }
    g_owners_created.fetch_add(1, std::memory_order_relaxed);

    // Recursively patch any all-null Arrow children, or lazy UNKNOWN_NULL_COUNT (-1) with bitmask!
    std::vector<uint8_t> col_unknown_nc;
    if (owner->raw.dev_arr.array.n_children > 0 && owner->raw.dev_arr.array.children != nullptr) {
      col_unknown_nc.resize(static_cast<size_t>(owner->raw.dev_arr.array.n_children), 0u);
      for (int64_t c = 0; c < owner->raw.dev_arr.array.n_children; ++c) {
        if (owner->null_patcher.patch_recursive(owner->raw.dev_arr.array.children[c], stream, mr)) {
          col_unknown_nc[static_cast<size_t>(c)] = 1u;
        }
      }
    } else {
      owner->null_patcher.patch_recursive(&owner->raw.dev_arr.array, stream, mr);
    }

    // Zero-copy Arrow C Device import (automatically calls cudaStreamWaitEvent on stream!)
    owner->holder.emplace(cudf::from_arrow_device(&owner->raw.schema, &owner->raw.dev_arr, stream, mr));
    auto tv       = **owner->holder;
    auto const nc = tv.num_columns();

    cudf::jni::native_jlongArray ret(env, static_cast<jsize>(2 + nc));
    size_t approx_bytes = 0;
    std::vector<cudf::column_view*> allocated_views;
    allocated_views.reserve(static_cast<size_t>(nc));
    try {
      for (cudf::size_type c = 0; c < nc; ++c) {
        auto cv = tv.column(c);
        if (static_cast<size_t>(c) < col_unknown_nc.size() &&
            col_unknown_nc[static_cast<size_t>(c)] != 0 && cv.nullable()) {
          auto exact_nc = cudf::null_count(cv.null_mask(), cv.offset(), cv.offset() + cv.size(), stream);
          cv = cudf::column_view(
            cv.type(), cv.size(), cv.head(), cv.null_mask(), exact_nc, cv.offset(),
            std::vector<cudf::column_view>(cv.child_begin(), cv.child_end()));
        }
        bool const want_d32 =
          (static_cast<size_t>(c) < keep_dict32.size() && keep_dict32[static_cast<size_t>(c)] != 0);
        if (cv.type().id() == cudf::type_id::DICTIONARY32 && !want_d32) {
          auto dec = cudf::dictionary::decode(cudf::dictionary_column_view(cv), stream, mr);
          auto* v  = new cudf::column_view(dec->view());
          allocated_views.push_back(v);
          owner->decoded_dict_cols.push_back(std::move(dec));
          approx_bytes += static_cast<size_t>(tv.num_rows()) * 16u;
        } else {
          auto* v = new cudf::column_view(cv);
          allocated_views.push_back(v);
          auto id = cv.type().id();
          if (id == cudf::type_id::STRING) {
            approx_bytes += static_cast<size_t>(cv.size()) * 16u;
          } else if (id == cudf::type_id::DICTIONARY32) {
            approx_bytes += static_cast<size_t>(cv.size()) * 4u;
          } else {
            approx_bytes += static_cast<size_t>(cv.size()) * cudf::size_of(cv.type());
          }
        }
      }
    } catch (...) {
      for (auto* v : allocated_views) { delete v; }
      throw;
    }

    ret[0] = std::bit_cast<jlong>(owner.release());
    ret[1] = static_cast<jlong>(std::max<size_t>(approx_bytes, 64u));
    for (cudf::size_type c = 0; c < nc; ++c) {
      ret[2 + c] = std::bit_cast<jlong>(allocated_views[static_cast<size_t>(c)]);
    }
    return ret.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

// Path A-Deep: Multi-stream pre-allocated direct-into-output multi-file GPU decode!
// Calls vx_cuda_device_batches_next_merged to decode up to max_batches small files concurrently
// across up to 8 CUDA streams into ONE pre-allocated ArrowDeviceArray + ONE SyncEvent, and imports
// TRUE ZERO-COPY via ZeroCopyArrowOwner (ZERO cudf::concatenate, ZERO cudf::column copy!).
// Packed return format: [ownerPtr, approxBytes, ncols, colPtr_0..colPtr_{nc-1}, numRuns, fIdx_0, rows_0, ...]
// (ownerPtr != 0 => colPtr_i are zero-copy cudf::column_view* owned by ownerPtr;
//  ownerPtr == 0 => colPtr_i are owned cudf::column* from string/non-fixed-width fallback).
JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_VortexNative_nextBatchedMultiFileNative(
  JNIEnv* env, jclass, jlong j_handle, jint j_max_batches, jlong j_max_rows, jlong j_target_bytes)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* st    = std::bit_cast<VortexMultiFileScanState*>(j_handle);
    auto stream = cudf::get_default_stream();
    auto mr     = cudf::get_current_device_resource_ref();

    st->start_prefetch();
    if (st->active == nullptr) { return nullptr; }

    size_t const max_b     = (j_max_batches > 0) ? static_cast<size_t>(j_max_batches) : 64u;
    size_t const max_r     = (j_max_rows > 0) ? static_cast<size_t>(j_max_rows) : 32000000u;
    size_t const target_b  = (j_target_bytes > 0) ? static_cast<size_t>(j_target_bytes) : (256u << 20);
    size_t const est_row_b = std::max<size_t>(st->col_names.size(), 1u) * 8u;

    auto& api = VortexCudaApi::get();
    // If pushback is empty (normal case), invoke Path A-Deep directly!
    if (st->pushback.empty() && api.device_batches_next_merged != nullptr) {
      auto owner = std::make_unique<ZeroCopyArrowOwner>();
      std::vector<int32_t> f_idx(max_b, 0);
      std::vector<int32_t> r_cnt(max_b, 0);
      size_t n_runs      = 0;
      uint8_t was_merged = 0;
      vx_error* err      = nullptr;
      int const rc       = api.device_batches_next_merged(st->active,
                                                    max_b,
                                                    max_r,
                                                    target_b,
                                                    est_row_b,
                                                    &owner->raw.schema,
                                                    &owner->raw.dev_arr,
                                                    f_idx.data(),
                                                    r_cnt.data(),
                                                    max_b,
                                                    &n_runs,
                                                    &was_merged,
                                                    &err);
      if (rc != 0) { CUDF_FAIL("Vortex next_merged failed: " + api.consume_error(err)); }
      if (owner->raw.dev_arr.array.release == nullptr) {
        api.device_batches_free(st->active);
        st->active = nullptr;
        return nullptr;
      }

      // If Path A-Deep merged multiple fixed-width files OR this is a single-batch return (e.g.
      // 1 large file or string table where was_merged==0 and n_runs==1):
      // Wait: if was_merged==0 and n_runs==1 and rows is tiny (string multi-file table), coalesce
      // via existing materialize_raw_batches; otherwise wrap TRUE ZERO-COPY right here!
      int64_t const nrows = owner->raw.dev_arr.array.length;
      bool const small_unmerged_string =
        (was_merged == 0u && max_b > 1u && nrows > 0 &&
         static_cast<size_t>(nrows) * est_row_b < target_b / 2u &&
         static_cast<size_t>(nrows) < max_r);

      if (!small_unmerged_string) {
        g_owners_created.fetch_add(1, std::memory_order_relaxed);
        std::vector<uint8_t> col_unknown_nc;
        if (owner->raw.dev_arr.array.n_children > 0 && owner->raw.dev_arr.array.children != nullptr) {
          col_unknown_nc.resize(static_cast<size_t>(owner->raw.dev_arr.array.n_children), 0u);
          for (int64_t c = 0; c < owner->raw.dev_arr.array.n_children; ++c) {
            if (owner->null_patcher.patch_recursive(owner->raw.dev_arr.array.children[c], stream, mr)) {
              col_unknown_nc[static_cast<size_t>(c)] = 1u;
            }
          }
        } else {
          owner->null_patcher.patch_recursive(&owner->raw.dev_arr.array, stream, mr);
        }

        owner->holder.emplace(
          cudf::from_arrow_device(&owner->raw.schema, &owner->raw.dev_arr, stream, mr));
        auto tv       = **owner->holder;
        auto const nc = tv.num_columns();

        size_t approx_bytes = 0;
        std::vector<cudf::column_view*> allocated_views;
        allocated_views.reserve(static_cast<size_t>(nc));
        try {
          for (cudf::size_type c = 0; c < nc; ++c) {
            auto cv = tv.column(c);
            if (static_cast<size_t>(c) < col_unknown_nc.size() &&
                col_unknown_nc[static_cast<size_t>(c)] != 0 && cv.nullable()) {
              auto exact_nc =
                cudf::null_count(cv.null_mask(), cv.offset(), cv.offset() + cv.size(), stream);
              cv = cudf::column_view(
                cv.type(), cv.size(), cv.head(), cv.null_mask(), exact_nc, cv.offset(),
                std::vector<cudf::column_view>(cv.child_begin(), cv.child_end()));
            }
            if (cv.type().id() == cudf::type_id::DICTIONARY32) {
              auto dec = cudf::dictionary::decode(cudf::dictionary_column_view(cv), stream, mr);
              approx_bytes += static_cast<size_t>(dec->size()) * 8u;
              allocated_views.push_back(new cudf::column_view(dec->view()));
              owner->decoded_dict_cols.push_back(std::move(dec));
            } else {
              approx_bytes += static_cast<size_t>(cv.size()) * 8u;
              allocated_views.push_back(new cudf::column_view(cv));
            }
          }
        } catch (...) {
          for (auto* vp : allocated_views) { delete vp; }
          throw;
        }
        auto const nr = static_cast<jsize>(n_runs);
        cudf::jni::native_jlongArray ret(env, 4 + nc + 2 * nr);
        ret[0] = std::bit_cast<jlong>(owner.release());
        ret[1] = static_cast<jlong>(std::max<size_t>(approx_bytes, 64u));
        ret[2] = static_cast<jlong>(nc);
        for (cudf::size_type c = 0; c < nc; ++c) {
          ret[3 + c] = reinterpret_cast<jlong>(allocated_views[static_cast<size_t>(c)]);
        }
        ret[3 + nc] = static_cast<jlong>(nr);
        for (jsize r = 0; r < nr; ++r) {
          ret[4 + nc + 2 * r]     = static_cast<jlong>(f_idx[static_cast<size_t>(r)]);
          ret[4 + nc + 2 * r + 1] = static_cast<jlong>(r_cnt[static_cast<size_t>(r)]);
        }
        return ret.get_jArray();
      } else {
        // String / non-fixed-width small file fallback: coalesce up to max_b via materialize_raw_batches
        owner->raw.file_idx = (n_runs > 0) ? f_idx[0] : 0;
        st->pushback.push_front(std::move(owner->raw));
      }
    }

    RawArrowBatch first;
    if (!st->pop_next(first)) { return nullptr; }

    std::vector<RawArrowBatch> raw_batches;
    raw_batches.reserve(std::min<size_t>(max_b, 64u));
    std::vector<std::pair<int32_t, int64_t>> runs;
    runs.reserve(std::min<size_t>(max_b, 64u));

    int64_t acc_rows = first.dev_arr.array.length;
    size_t acc_bytes = static_cast<size_t>(acc_rows) * est_row_b;
    runs.emplace_back(first.file_idx, first.dev_arr.array.length);
    raw_batches.push_back(std::move(first));

    while (raw_batches.size() < max_b && acc_bytes < target_b &&
           acc_rows < static_cast<int64_t>(max_r)) {
      RawArrowBatch nxt;
      if (!st->pop_next(nxt)) { break; }
      int64_t const nr_rows = nxt.dev_arr.array.length;
      if (!runs.empty() && runs.back().first == nxt.file_idx) {
        runs.back().second += nr_rows;
      } else {
        runs.emplace_back(nxt.file_idx, nr_rows);
      }
      acc_rows += nr_rows;
      acc_bytes += static_cast<size_t>(nr_rows) * est_row_b;
      raw_batches.push_back(std::move(nxt));
    }

    auto out_tbl  = materialize_raw_batches(raw_batches, {}, stream, mr);
    auto cols     = out_tbl->release();
    auto const nc = static_cast<jsize>(cols.size());
    auto const nr = static_cast<jsize>(runs.size());
    cudf::jni::native_jlongArray ret(env, 4 + nc + 2 * nr);
    ret[0] = 0LL;  // ownerPtr == 0 => owned cudf::column* handles
    ret[1] = 0LL;
    ret[2] = static_cast<jlong>(nc);
    for (jsize c = 0; c < nc; ++c) {
      ret[3 + c] = std::bit_cast<jlong>(cols[static_cast<size_t>(c)].release());
    }
    ret[3 + nc] = static_cast<jlong>(nr);
    for (jsize r = 0; r < nr; ++r) {
      ret[4 + nc + 2 * r]     = static_cast<jlong>(runs[static_cast<size_t>(r)].first);
      ret[4 + nc + 2 * r + 1] = static_cast<jlong>(runs[static_cast<size_t>(r)].second);
    }
    return ret.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jint JNICALL Java_com_nvidia_spark_rapids_jni_VortexNative_getLastFileIndexNative(
  JNIEnv* env, jclass, jlong j_handle)
{
  JNI_TRY
  {
    auto* st = std::bit_cast<VortexMultiFileScanState*>(j_handle);
    return (st != nullptr) ? static_cast<jint>(st->last_file_idx) : 0;
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_VortexNative_releaseArrowOwnerNative(
  JNIEnv* env, jclass, jlong j_owner)
{
  JNI_TRY
  {
    auto* owner = std::bit_cast<ZeroCopyArrowOwner*>(j_owner);
    if (owner != nullptr) {
      ArrowOwnerReaper::get().enqueue(owner, cudf::get_default_stream().get());
    }
  }
  JNI_CATCH(env, );
}

}  // extern "C"

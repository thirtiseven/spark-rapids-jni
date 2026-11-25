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
#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/interop.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
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

// Minimal C-ABI declarations matching vortex/vortex-ffi/cinclude/vortex.h & vortex_cuda.h
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

struct VortexCudaApi {
  void* dl{nullptr};
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
  int (*device_batches_next)(
    vx_cuda_device_batches*, ArrowSchema*, ArrowDeviceArray*, vx_error**)                   = nullptr;
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
      a.load(a.device_batches_next, "vx_cuda_device_batches_next");
      a.load(a.device_batches_free, "vx_cuda_device_batches_free");

      vx_error* err    = nullptr;
      a.shared_session = a.session_new(&err);
      CUDF_EXPECTS(a.shared_session != nullptr && err == nullptr,
                   "VortexJni: vx_cuda_session_new failed: " + a.consume_error(err));
      return a;
    }();
    return api;
  }
};

struct VortexScanState {
  vx_cuda_device_batches* batches{nullptr};
};

}  // namespace

extern "C" {

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_VortexNative_writeTableNative(
  JNIEnv* env, jclass, jlong j_table_view, jobjectArray j_col_names, jstring j_path)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto& api = VortexCudaApi::get();
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

    vx_error* err        = nullptr;
    vx_dtype const* vdt  = api.dtype_from_arrow(api.shared_session, schema_d.get(), &err);
    CUDF_EXPECTS(vdt != nullptr && err == nullptr,
                 "VortexJni: vx_dtype_from_arrow_schema failed: " + api.consume_error(err));

    std::string const path_str(path.get());
    vx_view pview{path_str.data(), path_str.size()};
    vx_array_sink* sink = api.sink_open_file(api.shared_session, pview, vdt, &err);
    api.dtype_free(vdt);
    CUDF_EXPECTS(sink != nullptr && err == nullptr,
                 "VortexJni: vx_cuda_array_sink_open_file failed: " + api.consume_error(err));

    // Transfer host ArrowArray + ArrowSchema into vx_array
    ArrowArray raw_arr = host_arr->array;
    host_arr->array.release = nullptr;  // ownership transferred to raw_arr
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

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_VortexNative_openScanNative(
  JNIEnv* env, jclass, jstring j_path, jobjectArray j_col_names, jlong j_batch_rows)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto& api = VortexCudaApi::get();
    cudf::jni::native_jstring path(env, j_path);
    std::string const path_str(path.get());

    jsize const ncols = (j_col_names != nullptr) ? env->GetArrayLength(j_col_names) : 0;
    std::vector<std::string> col_strs;
    std::vector<vx_view> col_views;
    col_strs.reserve(static_cast<size_t>(ncols));
    col_views.reserve(static_cast<size_t>(ncols));
    for (jsize i = 0; i < ncols; ++i) {
      auto js = static_cast<jstring>(env->GetObjectArrayElement(j_col_names, i));
      cudf::jni::native_jstring ns(env, js);
      col_strs.emplace_back(ns.get());
      env->DeleteLocalRef(js);
    }
    for (auto const& s : col_strs) { col_views.push_back(vx_view{s.data(), s.size()}); }

    vx_cuda_scan_options opts{};
    opts.flags      = 0u;
    opts.batch_rows = (j_batch_rows > 0) ? static_cast<size_t>(j_batch_rows) : 0u;

    vx_error* err = nullptr;
    vx_view pview{path_str.data(), path_str.size()};
    auto* batches = api.scan_device_batches(api.shared_session,
                                            pview,
                                            &opts,
                                            col_views.empty() ? nullptr : col_views.data(),
                                            col_views.size(),
                                            &err);
    CUDF_EXPECTS(batches != nullptr && err == nullptr,
                 "VortexJni: vx_cuda_scan_path_device_batches failed: " + api.consume_error(err));
    auto* st = new VortexScanState{batches};
    return std::bit_cast<jlong>(st);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_VortexNative_nextBatchNative(
  JNIEnv* env, jclass, jlong j_handle, jbyteArray j_dict32_mask)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto& api     = VortexCudaApi::get();
    auto* st      = std::bit_cast<VortexScanState*>(j_handle);
    auto stream   = cudf::get_default_stream();
    auto mr       = cudf::get_current_device_resource_ref();

    std::vector<uint8_t> keep_dict32;
    if (j_dict32_mask != nullptr) {
      cudf::jni::native_jbyteArray nb(env, j_dict32_mask);
      keep_dict32.resize(static_cast<size_t>(nb.size()));
      for (size_t i = 0; i < keep_dict32.size(); ++i) {
        keep_dict32[i] = static_cast<uint8_t>(nb[i]);
      }
    }

    while (true) {
      ArrowSchema schema{};
      ArrowDeviceArray dev_arr{};
      vx_error* err = nullptr;
      int rc        = api.device_batches_next(st->batches, &schema, &dev_arr, &err);
      if (rc != 0 || err != nullptr) {
        if (dev_arr.array.release != nullptr) { dev_arr.array.release(&dev_arr.array); }
        if (schema.release != nullptr) { schema.release(&schema); }
        CUDF_FAIL("VortexJni: vx_cuda_device_batches_next failed: " + api.consume_error(err));
      }
      // EOS: both release callbacks are null
      if (dev_arr.array.release == nullptr && schema.release == nullptr) { return nullptr; }

      struct ArrowGuard {
        ArrowSchema* s;
        ArrowDeviceArray* a;
        ~ArrowGuard()
        {
          if (a->array.release != nullptr) { a->array.release(&a->array); }
          if (s->release != nullptr) { s->release(s); }
        }
      } guard{&schema, &dev_arr};

      if (dev_arr.array.length == 0) { continue; }

      // Zero-copy Arrow C Device import into cuDF (waits on dev_arr.sync_event on `stream`!)
      auto tv_holder = cudf::from_arrow_device(&schema, &dev_arr, stream, mr);
      auto tv        = *tv_holder;

      // Materialize RMM-owned columns (decoding DICTIONARY32 unless requested in keep_dict32)
      // and sync stream before releasing Vortex's device buffers!
      std::vector<std::unique_ptr<cudf::column>> owned_cols;
      owned_cols.reserve(static_cast<size_t>(tv.num_columns()));
      for (cudf::size_type c = 0; c < tv.num_columns(); ++c) {
        auto cv = tv.column(c);
        bool const want_d32 =
          (static_cast<size_t>(c) < keep_dict32.size() && keep_dict32[static_cast<size_t>(c)] != 0);
        if (cv.type().id() == cudf::type_id::DICTIONARY32 && !want_d32) {
          owned_cols.push_back(cudf::dictionary::decode(cudf::dictionary_column_view(cv), stream, mr));
        } else {
          owned_cols.push_back(std::make_unique<cudf::column>(cv, stream, mr));
        }
      }
      stream.sync();
      auto tbl = std::make_unique<cudf::table>(std::move(owned_cols));
      return cudf::jni::convert_table_for_return(env, std::move(tbl));
    }
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_VortexNative_closeScanNative(
  JNIEnv* env, jclass, jlong j_handle)
{
  JNI_TRY
  {
    auto* st = std::bit_cast<VortexScanState*>(j_handle);
    if (st != nullptr) {
      if (st->batches != nullptr) { VortexCudaApi::get().device_batches_free(st->batches); }
      delete st;
    }
  }
  JNI_CATCH(env, );
}

}  // extern "C"

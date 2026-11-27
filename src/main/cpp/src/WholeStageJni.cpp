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

#include "cudf_jni_apis.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/whole_stage_jit.hpp>

#include <bit>
#include <cstdint>
#include <vector>

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_WholeStageNative_compileProgramNative(
  JNIEnv* env, jclass, jstring j_body_src, jstring j_kernel_name)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jstring src(env, j_body_src);
    cudf::jni::native_jstring kn(env, j_kernel_name);
    auto* prog = cudf::whole_stage::compile_whole_stage_program(src.get(), kn.get());
    return std::bit_cast<jlong>(prog);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_WholeStageNative_destroyProgramNative(
  JNIEnv* env, jclass, jlong j_prog)
{
  JNI_TRY
  {
    auto* prog = std::bit_cast<cudf::whole_stage::whole_stage_program*>(j_prog);
    cudf::whole_stage::destroy_whole_stage_program(prog);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_WholeStageNative_buildProbeTableNative(
  JNIEnv* env, jclass, jlong j_key_col_view)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const* cv = reinterpret_cast<cudf::column_view const*>(j_key_col_view);
    auto* tbl      = cudf::whole_stage::build_broadcast_probe_table(*cv);
    return std::bit_cast<jlong>(tbl);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_WholeStageNative_destroyProbeTableNative(
  JNIEnv* env, jclass, jlong j_probe_tbl)
{
  JNI_TRY
  {
    auto* tbl = std::bit_cast<cudf::whole_stage::broadcast_probe_table*>(j_probe_tbl);
    cudf::whole_stage::destroy_broadcast_probe_table(tbl);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_WholeStageNative_openHashAggTaskNative(
  JNIEnv* env,
  jclass,
  jlong j_prog,
  jintArray j_key_type_ids,
  jintArray j_key_scales,
  jbyteArray j_agg_kinds,
  jint j_init_cap)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* prog = std::bit_cast<cudf::whole_stage::whole_stage_program*>(j_prog);
    std::vector<cudf::data_type> key_dts;
    if (j_key_type_ids != nullptr) {
      cudf::jni::native_jintArray tids(env, j_key_type_ids);
      cudf::jni::native_jintArray scs(env, j_key_scales);
      key_dts.reserve(static_cast<size_t>(tids.size()));
      for (int i = 0; i < tids.size(); ++i) {
        auto tid = static_cast<cudf::type_id>(tids[i]);
        if (tid == cudf::type_id::DECIMAL32 || tid == cudf::type_id::DECIMAL64 ||
            tid == cudf::type_id::DECIMAL128) {
          key_dts.emplace_back(tid, scs[i]);
        } else {
          key_dts.emplace_back(tid);
        }
      }
    }
    cudf::jni::native_jbyteArray aks(env, j_agg_kinds);
    std::vector<cudf::whole_stage::agg_kind> agg_ks;
    agg_ks.reserve(static_cast<size_t>(aks.size()));
    for (int i = 0; i < aks.size(); ++i) {
      agg_ks.push_back(static_cast<cudf::whole_stage::agg_kind>(aks[i]));
    }
    auto* st = cudf::whole_stage::open_hash_agg_task(
      *prog, key_dts, agg_ks, static_cast<uint32_t>(std::max<jint>(j_init_cap, 64)));
    return std::bit_cast<jlong>(st);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_WholeStageNative_addBatchHashAggNative(
  JNIEnv* env,
  jclass,
  jlong j_state,
  jlongArray j_fact_col_views,
  jint j_num_rows,
  jlongArray j_probe_tables,
  jlongArray j_dim_col_views)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* st = std::bit_cast<cudf::whole_stage::hash_agg_task_state*>(j_state);

    std::vector<cudf::whole_stage::col_source_desc> fact_srcs;
    if (j_fact_col_views != nullptr) {
      cudf::jni::native_jlongArray fc(env, j_fact_col_views);
      fact_srcs.reserve(static_cast<size_t>(fc.size()));
      for (int i = 0; i < fc.size(); ++i) {
        auto const* cv = reinterpret_cast<cudf::column_view const*>(fc[i]);
        fact_srcs.push_back(cudf::whole_stage::make_raw_source(*cv));
      }
    }

    std::vector<cudf::whole_stage::broadcast_probe_table const*> probes;
    if (j_probe_tables != nullptr) {
      cudf::jni::native_jlongArray pt(env, j_probe_tables);
      probes.reserve(static_cast<size_t>(pt.size()));
      for (int i = 0; i < pt.size(); ++i) {
        probes.push_back(std::bit_cast<cudf::whole_stage::broadcast_probe_table const*>(pt[i]));
      }
    }

    std::vector<cudf::whole_stage::col_source_desc> dim_srcs;
    if (j_dim_col_views != nullptr) {
      cudf::jni::native_jlongArray dc(env, j_dim_col_views);
      dim_srcs.reserve(static_cast<size_t>(dc.size()));
      for (int i = 0; i < dc.size(); ++i) {
        auto const* cv = reinterpret_cast<cudf::column_view const*>(dc[i]);
        dim_srcs.push_back(cudf::whole_stage::make_raw_source(*cv));
      }
    }

    cudf::whole_stage::add_batch_whole_stage_agg(
      *st, fact_srcs, static_cast<uint32_t>(j_num_rows), probes, dim_srcs);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_WholeStageNative_finishHashAggTaskNative(
  JNIEnv* env, jclass, jlong j_state)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* st = std::bit_cast<cudf::whole_stage::hash_agg_task_state*>(j_state);
    auto tbl = cudf::whole_stage::finish_hash_agg_task(*st);
    return cudf::jni::convert_table_for_return(env, std::move(tbl));
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_WholeStageNative_destroyHashAggTaskNative(
  JNIEnv* env, jclass, jlong j_state)
{
  JNI_TRY
  {
    auto* st = std::bit_cast<cudf::whole_stage::hash_agg_task_state*>(j_state);
    cudf::whole_stage::destroy_hash_agg_task(st);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_WholeStageNative_encodeDimStringDictNative(
  JNIEnv* env, jclass, jlong j_col_view, jint j_substr_start0, jint j_substr_len)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const* cv = reinterpret_cast<cudf::column_view const*>(j_col_view);
    auto stream    = cudf::get_default_stream();
    auto mr        = cudf::get_current_device_resource_ref();
    std::unique_ptr<cudf::column> sliced_owner;
    cudf::column_view src_view = *cv;
    if (j_substr_len >= 0) {
      sliced_owner = cudf::strings::slice_strings(
        cudf::strings_column_view(*cv),
        static_cast<cudf::size_type>(j_substr_start0),
        static_cast<cudf::size_type>(j_substr_start0 + j_substr_len),
        1,
        stream,
        mr);
      src_view = sliced_owner->view();
    }
    auto enc = cudf::dictionary::encode(src_view, cudf::data_type{cudf::type_id::INT32}, stream, mr);
    cudf::dictionary_column_view const dcv(enc->view());
    auto idx_col = std::make_unique<cudf::column>(dcv.indices(), stream, mr);
    if (src_view.nullable() && src_view.null_count() > 0) {
      idx_col->set_null_mask(cudf::copy_bitmask(src_view, stream, mr), src_view.null_count());
    }
    auto keys_col = std::make_unique<cudf::column>(dcv.keys(), stream, mr);
    jlongArray ret = env->NewLongArray(2);
    jlong ptrs[2]  = {std::bit_cast<jlong>(idx_col.release()), std::bit_cast<jlong>(keys_col.release())};
    env->SetLongArrayRegion(ret, 0, 2, ptrs);
    return ret;
  }
  JNI_CATCH(env, nullptr);
}

}  // extern "C"

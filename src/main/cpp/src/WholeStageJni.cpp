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
  JNIEnv* env, jclass, jstring j_body_src, jstring j_kernel_name, jint j_preamble_flags)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jstring src(env, j_body_src);
    cudf::jni::native_jstring kn(env, j_kernel_name);
    auto* prog = cudf::whole_stage::compile_whole_stage_program(
      src.get(), kn.get(), static_cast<uint32_t>(j_preamble_flags));
    return std::bit_cast<jlong>(prog);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_WholeStageNative_warmupLtoHelpersNative(
  JNIEnv* env, jclass, jstring j_arch)
{
  JNI_TRY
  {
    std::string arch = "compute_75";
    if (j_arch != nullptr) {
      char const* a_c = env->GetStringUTFChars(j_arch, nullptr);
      arch = a_c;
      env->ReleaseStringUTFChars(j_arch, a_c);
    }
    cudf::whole_stage::warmup_whole_stage_lto_helpers(arch);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jbyteArray JNICALL Java_com_nvidia_spark_rapids_jni_WholeStageNative_compileToPtxCpuOnlyNative(
  JNIEnv* env, jclass, jstring j_body_src, jstring j_kernel_name, jint j_preamble_flags, jstring j_arch)
{
  JNI_TRY
  {
    // Pure CPU NVRTC compile: intentionally NO auto_set_device / NO CUDA driver call!
    cudf::jni::native_jstring src(env, j_body_src);
    cudf::jni::native_jstring kn(env, j_kernel_name);
    cudf::jni::native_jstring arch(env, j_arch);
    std::string ptx = cudf::whole_stage::compile_whole_stage_to_ptx_cpu_only(
      src.get(), kn.get(), static_cast<uint32_t>(j_preamble_flags), arch.get());
    jbyteArray arr = env->NewByteArray(static_cast<jsize>(ptx.size()));
    if (arr != nullptr && !ptx.empty()) {
      env->SetByteArrayRegion(arr, 0, static_cast<jsize>(ptx.size()), reinterpret_cast<jbyte const*>(ptx.data()));
    }
    return arr;
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_WholeStageNative_loadProgramFromPtxNative(
  JNIEnv* env, jclass, jbyteArray j_ptx, jstring j_kernel_name)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jbyteArray ptx_arr(env, j_ptx);
    cudf::jni::native_jstring kn(env, j_kernel_name);
    std::string ptx(reinterpret_cast<char const*>(ptx_arr.data()), static_cast<size_t>(ptx_arr.size()));
    auto* prog = cudf::whole_stage::load_whole_stage_program_from_ptx(ptx, kn.get());
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
  jint j_init_cap,
  jlongArray j_const_bits)
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
    std::vector<uint64_t> cbits;
    if (j_const_bits != nullptr) {
      cudf::jni::native_jlongArray cb(env, j_const_bits);
      cbits.reserve(static_cast<size_t>(cb.size()));
      for (int i = 0; i < cb.size(); ++i) { cbits.push_back(static_cast<uint64_t>(cb[i])); }
    }
    auto* st = cudf::whole_stage::open_hash_agg_task(
      *prog, key_dts, agg_ks, static_cast<uint32_t>(std::max<jint>(j_init_cap, 64)), cbits);
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

JNIEXPORT jboolean JNICALL Java_com_nvidia_spark_rapids_jni_WholeStageNative_tryAddLshfFrameToAggNative(
  JNIEnv* env,
  jclass,
  jlong j_state,
  jlong j_host_addr,
  jlong j_host_len,
  jlong j_dev_addr,
  jlong j_dev_len,
  jintArray j_sel_cols,
  jlongArray j_probe_tables,
  jlongArray j_dim_col_views)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* st = std::bit_cast<cudf::whole_stage::hash_agg_task_state*>(j_state);
    cudf::jni::native_jintArray sc(env, j_sel_cols);
    std::vector<int32_t> sel_cols(sc.begin(), sc.end());

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
    bool const ok = cudf::whole_stage::try_add_lshf_frame_whole_stage_agg(
      *st,
      reinterpret_cast<uint8_t const*>(j_host_addr),
      static_cast<size_t>(j_host_len),
      reinterpret_cast<uint8_t const*>(j_dev_addr),
      static_cast<size_t>(j_dev_len),
      sel_cols,
      probes,
      dim_srcs);
    return ok ? JNI_TRUE : JNI_FALSE;
  }
  JNI_CATCH(env, JNI_FALSE);
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_WholeStageNative_runStagePartitionScatterNative(
  JNIEnv* env,
  jclass,
  jlong j_count_prog,
  jlong j_scatter_prog,
  jlongArray j_fact_col_views,
  jint j_num_rows,
  jlongArray j_probe_tables,
  jlongArray j_dim_col_views,
  jintArray j_out_tids,
  jintArray j_out_scales,
  jint j_num_parts,
  jintArray j_out_splits,
  jlongArray j_const_bits)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* cprog = std::bit_cast<cudf::whole_stage::whole_stage_program*>(j_count_prog);
    auto* sprog = std::bit_cast<cudf::whole_stage::whole_stage_program*>(j_scatter_prog);

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
    cudf::jni::native_jintArray tids(env, j_out_tids);
    cudf::jni::native_jintArray scs(env, j_out_scales);
    std::vector<cudf::data_type> out_dts;
    out_dts.reserve(static_cast<size_t>(tids.size()));
    for (int i = 0; i < tids.size(); ++i) {
      auto tid = static_cast<cudf::type_id>(tids[i]);
      if (tid == cudf::type_id::DECIMAL32 || tid == cudf::type_id::DECIMAL64 ||
          tid == cudf::type_id::DECIMAL128) {
        out_dts.emplace_back(tid, scs[i]);
      } else {
        out_dts.emplace_back(tid);
      }
    }
    std::vector<uint64_t> cbits;
    if (j_const_bits != nullptr) {
      cudf::jni::native_jlongArray cb(env, j_const_bits);
      cbits.reserve(static_cast<size_t>(cb.size()));
      for (int i = 0; i < cb.size(); ++i) { cbits.push_back(static_cast<uint64_t>(cb[i])); }
    }

    auto res = cudf::whole_stage::run_whole_stage_partition_scatter(
      *cprog, *sprog, fact_srcs, static_cast<uint32_t>(j_num_rows), probes, dim_srcs, out_dts, j_num_parts, cbits);
    env->SetIntArrayRegion(j_out_splits, 0, j_num_parts, reinterpret_cast<jint const*>(res.splits.data()));
    return cudf::jni::convert_table_for_return(env, std::move(res.partitioned_table));
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_WholeStageNative_runStageMaterializeNative(
  JNIEnv* env,
  jclass,
  jlong j_count_prog,
  jlong j_scatter_prog,
  jlongArray j_fact_col_views,
  jint j_num_rows,
  jlongArray j_probe_tables,
  jlongArray j_dim_col_views,
  jintArray j_out_tids,
  jintArray j_out_scales,
  jlongArray j_const_bits)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* cprog = std::bit_cast<cudf::whole_stage::whole_stage_program*>(j_count_prog);
    auto* sprog = std::bit_cast<cudf::whole_stage::whole_stage_program*>(j_scatter_prog);

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
    cudf::jni::native_jintArray tids(env, j_out_tids);
    cudf::jni::native_jintArray scs(env, j_out_scales);
    std::vector<cudf::data_type> out_dts;
    out_dts.reserve(static_cast<size_t>(tids.size()));
    for (int i = 0; i < tids.size(); ++i) {
      auto tid = static_cast<cudf::type_id>(tids[i]);
      if (tid == cudf::type_id::DECIMAL32 || tid == cudf::type_id::DECIMAL64 ||
          tid == cudf::type_id::DECIMAL128) {
        out_dts.emplace_back(tid, scs[i]);
      } else {
        out_dts.emplace_back(tid);
      }
    }
    std::vector<uint64_t> cbits;
    if (j_const_bits != nullptr) {
      cudf::jni::native_jlongArray cb(env, j_const_bits);
      cbits.reserve(static_cast<size_t>(cb.size()));
      for (int i = 0; i < cb.size(); ++i) { cbits.push_back(static_cast<uint64_t>(cb[i])); }
    }
    auto tbl = cudf::whole_stage::run_whole_stage_materialize(
      *cprog, *sprog, fact_srcs, static_cast<uint32_t>(j_num_rows), probes, dim_srcs, out_dts, cbits);
    return cudf::jni::convert_table_for_return(env, std::move(tbl));
  }
  JNI_CATCH(env, nullptr);
}

}  // extern "C"

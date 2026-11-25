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
#include "lenc.hpp"

#include <bit>
#include <cstdint>
#include <string>
#include <vector>

namespace {

std::vector<std::vector<std::string>> parse_dict_lits_2d(JNIEnv* env, jobjectArray j_lits_2d)
{
  std::vector<std::vector<std::string>> out;
  if (j_lits_2d == nullptr) { return out; }
  jsize const n = env->GetArrayLength(j_lits_2d);
  out.resize(static_cast<size_t>(n));
  for (jsize i = 0; i < n; ++i) {
    auto inner = static_cast<jobjectArray>(env->GetObjectArrayElement(j_lits_2d, i));
    if (inner != nullptr) {
      jsize const m = env->GetArrayLength(inner);
      out[static_cast<size_t>(i)].reserve(static_cast<size_t>(m));
      for (jsize j = 0; j < m; ++j) {
        auto js = static_cast<jstring>(env->GetObjectArrayElement(inner, j));
        cudf::jni::native_jstring ns(env, js);
        out[static_cast<size_t>(i)].emplace_back(ns.get());
        env->DeleteLocalRef(js);
      }
      env->DeleteLocalRef(inner);
    }
  }
  return out;
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_LencNative_decodeDictionaryNative(
  JNIEnv* env, jclass, jlong j_col_view)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const* cv = reinterpret_cast<cudf::column_view const*>(j_col_view);
    auto decoded   = rapids::jni::lenc::decode_dictionary(*cv);
    return cudf::jni::release_as_jlong(decoded);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL
Java_com_nvidia_spark_rapids_jni_LencNative_remapDictionaryToCanonicalNative(JNIEnv* env,
                                                                             jclass,
                                                                             jlong j_col_view,
                                                                             jobjectArray j_lits)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const* cv = reinterpret_cast<cudf::column_view const*>(j_col_view);
    std::vector<std::string> lits;
    jsize const n = env->GetArrayLength(j_lits);
    lits.reserve(static_cast<size_t>(n));
    for (jsize i = 0; i < n; ++i) {
      auto js = static_cast<jstring>(env->GetObjectArrayElement(j_lits, i));
      cudf::jni::native_jstring ns(env, js);
      lits.emplace_back(ns.get());
      env->DeleteLocalRef(js);
    }
    auto remapped = rapids::jni::lenc::remap_dictionary_to_canonical(*cv, lits);
    return cudf::jni::release_as_jlong(remapped);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_LencNative_splitAndEncodeShuffleNative(JNIEnv* env,
                                                                        jclass,
                                                                        jlong j_table_view,
                                                                        jintArray j_splits)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const* tv = reinterpret_cast<cudf::table_view const*>(j_table_view);
    cudf::jni::native_jintArray sp(env, j_splits);
    std::vector<cudf::size_type> splits(sp.data(), sp.data() + sp.size());

    auto [data_buf, offs_buf] = rapids::jni::lenc::split_and_encode_shuffle_to_device(*tv, splits);
    auto* data_ptr            = new rmm::device_buffer(std::move(data_buf));
    auto* offs_ptr            = new rmm::device_buffer(std::move(offs_buf));

    cudf::jni::native_jlongArray ret(env, 6);
    ret[0] = reinterpret_cast<jlong>(data_ptr->data());
    ret[1] = static_cast<jlong>(data_ptr->size());
    ret[2] = reinterpret_cast<jlong>(data_ptr);
    ret[3] = reinterpret_cast<jlong>(offs_ptr->data());
    ret[4] = static_cast<jlong>(offs_ptr->size());
    ret[5] = reinterpret_cast<jlong>(offs_ptr);
    return ret.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_LencNative_assembleAndDecodeShuffleNative(JNIEnv* env,
                                                                           jclass,
                                                                           jlong j_host_addr,
                                                                           jlong j_data_addr,
                                                                           jlong j_data_len,
                                                                           jlongArray j_offsets,
                                                                           jintArray j_type_ids,
                                                                           jintArray j_scales,
                                                                           jbyteArray j_dict32)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const* h_data = std::bit_cast<uint8_t const*>(j_host_addr);
    auto const* d_data = std::bit_cast<uint8_t const*>(j_data_addr);
    cudf::jni::native_jlongArray offs(env, j_offsets);
    size_t const n_tables = (offs.size() > 0) ? static_cast<size_t>(offs.size() - 1) : 0;

    cudf::jni::native_jintArray tids(env, j_type_ids);
    cudf::jni::native_jintArray scs(env, j_scales);
    std::vector<cudf::data_type> dtypes;
    dtypes.reserve(static_cast<size_t>(tids.size()));
    for (int i = 0; i < tids.size(); ++i) {
      auto tid = static_cast<cudf::type_id>(tids[i]);
      if (tid == cudf::type_id::DECIMAL32 || tid == cudf::type_id::DECIMAL64 ||
          tid == cudf::type_id::DECIMAL128) {
        dtypes.emplace_back(tid, scs[i]);
      } else {
        dtypes.emplace_back(tid);
      }
    }
    std::vector<uint8_t> d32;
    if (j_dict32 != nullptr) {
      cudf::jni::native_jbyteArray nb(env, j_dict32);
      d32.resize(static_cast<size_t>(nb.size()));
      for (size_t i = 0; i < d32.size(); ++i) { d32[i] = static_cast<uint8_t>(nb[i]); }
    }
    auto tbl = rapids::jni::lenc::assemble_and_decode_shuffle_from_device(
      h_data, d_data, static_cast<size_t>(j_data_len), offs.data(), n_tables, dtypes, d32);
    return cudf::jni::convert_table_for_return(env, std::move(tbl));
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_LencNative_compileJitReduceNative(
  JNIEnv* env, jclass, jstring j_cuda_src, jstring j_kernel_name)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jstring src(env, j_cuda_src);
    cudf::jni::native_jstring kn(env, j_kernel_name);
    auto* prog = rapids::jni::lenc::compile_jit_reduce_program(src.get(), kn.get());
    return std::bit_cast<jlong>(prog);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_LencNative_destroyJitReduceNative(
  JNIEnv* env, jclass, jlong j_prog)
{
  JNI_TRY
  {
    auto* prog = std::bit_cast<rapids::jni::lenc::JitReduceProgram*>(j_prog);
    rapids::jni::lenc::destroy_jit_reduce_program(prog);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_LencNative_runJitReduceTableNative(
  JNIEnv* env,
  jclass,
  jlong j_prog,
  jlong j_table_view,
  jobjectArray j_dict_lits_2d,
  jintArray j_gk_slots,
  jint j_max_groups,
  jbyteArray j_agg_kinds)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* prog     = std::bit_cast<rapids::jni::lenc::JitReduceProgram*>(j_prog);
    auto const* tv = reinterpret_cast<cudf::table_view const*>(j_table_view);
    auto dict_lits = parse_dict_lits_2d(env, j_dict_lits_2d);
    cudf::jni::native_jintArray ng(env, j_gk_slots);
    std::vector<int32_t> gk_slots(ng.begin(), ng.end());
    cudf::jni::native_jbyteArray na(env, j_agg_kinds);
    std::vector<uint8_t> agg_kinds(static_cast<size_t>(na.size()));
    for (size_t i = 0; i < agg_kinds.size(); ++i) { agg_kinds[i] = static_cast<uint8_t>(na[i]); }
    auto tbl = rapids::jni::lenc::run_jit_reduce_table(
      *prog, *tv, dict_lits, gk_slots, static_cast<int32_t>(j_max_groups), agg_kinds);
    return cudf::jni::convert_table_for_return(env, std::move(tbl));
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_LencNative_runJitReduceHighCardNative(
  JNIEnv* env,
  jclass,
  jlong j_prog,
  jlong j_key_table,
  jlong j_val_table,
  jobjectArray j_dict_lits_2d,
  jbyteArray j_agg_kinds)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* prog      = std::bit_cast<rapids::jni::lenc::JitReduceProgram*>(j_prog);
    auto const* ktv = reinterpret_cast<cudf::table_view const*>(j_key_table);
    auto const* vtv = reinterpret_cast<cudf::table_view const*>(j_val_table);
    auto dict_lits  = parse_dict_lits_2d(env, j_dict_lits_2d);
    cudf::jni::native_jbyteArray na(env, j_agg_kinds);
    std::vector<uint8_t> agg_kinds(static_cast<size_t>(na.size()));
    for (size_t i = 0; i < agg_kinds.size(); ++i) { agg_kinds[i] = static_cast<uint8_t>(na[i]); }
    auto tbl = rapids::jni::lenc::run_jit_reduce_high_card(
      *prog, *ktv, *vtv, dict_lits, agg_kinds);
    return cudf::jni::convert_table_for_return(env, std::move(tbl));
  }
  JNI_CATCH(env, nullptr);
}

}  // extern "C"

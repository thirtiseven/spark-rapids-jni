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
#include "jni_compiled_expr.hpp"
#include "jni_utils.hpp"
#include "lenc.hpp"

#include <bit>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

extern "C" {

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_LencNative_writeTableNative(
  JNIEnv* env, jclass, jlong j_table_view, jstring j_path)
{
  JNI_NULL_CHECK(env, j_table_view, "table view is null", );
  JNI_NULL_CHECK(env, j_path, "path is null", );
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const* table = reinterpret_cast<cudf::table_view const*>(j_table_view);
    cudf::jni::native_jstring path(env, j_path);
    rapids::jni::lenc::write_table(*table, std::string{path.get()});
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_LencNative_planReadNative(
  JNIEnv* env, jclass, jstring j_path, jintArray j_col_indices)
{
  JNI_NULL_CHECK(env, j_path, "path is null", 0);
  JNI_NULL_CHECK(env, j_col_indices, "col indices is null", 0);
  JNI_TRY
  {
    cudf::jni::native_jstring path(env, j_path);
    auto const cols = cudf::jni::native_jintArray(env, j_col_indices).to_vector<int32_t>();
    auto plan =
      std::make_unique<rapids::jni::lenc::ReadPlan>(rapids::jni::lenc::plan_read(std::string{path.get()}, cols));
    return std::bit_cast<jlong>(plan.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_LencNative_planTotalBytesNative(
  JNIEnv* env, jclass, jlong j_plan)
{
  JNI_NULL_CHECK(env, j_plan, "plan is null", 0);
  JNI_TRY
  {
    auto const* plan = std::bit_cast<rapids::jni::lenc::ReadPlan const*>(j_plan);
    return static_cast<jlong>(plan->total_packed_bytes);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jint JNICALL Java_com_nvidia_spark_rapids_jni_LencNative_planNumRowsNative(
  JNIEnv* env, jclass, jlong j_plan)
{
  JNI_NULL_CHECK(env, j_plan, "plan is null", 0);
  JNI_TRY
  {
    auto const* plan = std::bit_cast<rapids::jni::lenc::ReadPlan const*>(j_plan);
    return static_cast<jint>(plan->header.num_rows);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_LencNative_readToHostNative(
  JNIEnv* env, jclass, jstring j_path, jlong j_plan, jlong j_host_addr)
{
  JNI_NULL_CHECK(env, j_path, "path is null", );
  JNI_NULL_CHECK(env, j_plan, "plan is null", );
  JNI_TRY
  {
    cudf::jni::native_jstring path(env, j_path);
    auto const* plan = std::bit_cast<rapids::jni::lenc::ReadPlan const*>(j_plan);
    auto* dst        = std::bit_cast<uint8_t*>(j_host_addr);
    rapids::jni::lenc::read_to_host(std::string{path.get()}, *plan, dst);
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_LencNative_decodeFromHostNative(
  JNIEnv* env, jclass, jlong j_plan, jlong j_host_addr, jbyteArray j_dict32_mask)
{
  JNI_NULL_CHECK(env, j_plan, "plan is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const* plan = std::bit_cast<rapids::jni::lenc::ReadPlan const*>(j_plan);
    auto const* src  = std::bit_cast<uint8_t const*>(j_host_addr);
    std::vector<uint8_t> dict_mask;
    if (j_dict32_mask != nullptr) {
      cudf::jni::native_jbyteArray arr(env, j_dict32_mask);
      dict_mask.resize(arr.size());
      for (int i = 0; i < arr.size(); ++i) { dict_mask[i] = static_cast<uint8_t>(arr[i]); }
    }
    auto tbl = rapids::jni::lenc::decode_from_host(*plan, src, dict_mask);
    return cudf::jni::convert_table_for_return(env, std::move(tbl));
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_LencNative_decodeAndEvalFusedNative(
  JNIEnv* env, jclass, jlong j_plan, jlong j_host_addr, jint j_expr_kind, jlong j_param0)
{
  JNI_NULL_CHECK(env, j_plan, "plan is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const* plan = std::bit_cast<rapids::jni::lenc::ReadPlan const*>(j_plan);
    auto const* src  = std::bit_cast<uint8_t const*>(j_host_addr);
    auto tbl = rapids::jni::lenc::decode_and_eval_fused(
      *plan, src, static_cast<int32_t>(j_expr_kind), static_cast<int64_t>(j_param0));
    return cudf::jni::convert_table_for_return(env, std::move(tbl));
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_LencNative_destroyPlanNative(
  JNIEnv* env, jclass, jlong j_plan)
{
  JNI_TRY { delete std::bit_cast<rapids::jni::lenc::ReadPlan*>(j_plan); }
  JNI_CATCH(env, );
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_LencNative_decodeDictionaryNative(
  JNIEnv* env, jclass, jlong j_col_view)
{
  JNI_NULL_CHECK(env, j_col_view, "dict column view is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const* cv = reinterpret_cast<cudf::column_view const*>(j_col_view);
    return cudf::jni::release_as_jlong(rapids::jni::lenc::decode_dictionary(*cv));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_LencNative_remapDictionaryToCanonicalNative(
  JNIEnv* env, jclass, jlong j_col_view, jobjectArray j_literals)
{
  JNI_NULL_CHECK(env, j_col_view, "dict column view is null", 0);
  JNI_NULL_CHECK(env, j_literals, "query literals is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const* cv = reinterpret_cast<cudf::column_view const*>(j_col_view);
    cudf::jni::native_jstringArray n_lits(env, j_literals);
    return cudf::jni::release_as_jlong(
      rapids::jni::lenc::remap_dictionary_to_canonical(*cv, n_lits.as_cpp_vector()));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jbyteArray JNICALL Java_com_nvidia_spark_rapids_jni_LencNative_planColumnEncodingsNative(
  JNIEnv* env, jclass, jlong j_plan)
{
  JNI_NULL_CHECK(env, j_plan, "plan is null", nullptr);
  JNI_TRY
  {
    auto const* plan = std::bit_cast<rapids::jni::lenc::ReadPlan const*>(j_plan);
    jbyteArray arr   = env->NewByteArray(static_cast<jsize>(plan->metas.size()));
    std::vector<jbyte> tmp(plan->metas.size());
    for (size_t i = 0; i < plan->metas.size(); ++i) {
      tmp[i] = static_cast<jbyte>(plan->metas[i].encoding);
    }
    if (!tmp.empty()) { env->SetByteArrayRegion(arr, 0, static_cast<jsize>(tmp.size()), tmp.data()); }
    return arr;
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jbyteArray JNICALL Java_com_nvidia_spark_rapids_jni_LencNative_planColumnTypesNative(
  JNIEnv* env, jclass, jlong j_plan)
{
  JNI_NULL_CHECK(env, j_plan, "plan is null", nullptr);
  JNI_TRY
  {
    auto const* plan = std::bit_cast<rapids::jni::lenc::ReadPlan const*>(j_plan);
    jbyteArray arr   = env->NewByteArray(static_cast<jsize>(plan->metas.size()));
    std::vector<jbyte> tmp(plan->metas.size());
    for (size_t i = 0; i < plan->metas.size(); ++i) {
      tmp[i] = static_cast<jbyte>(plan->metas[i].type);
    }
    if (!tmp.empty()) { env->SetByteArrayRegion(arr, 0, static_cast<jsize>(tmp.size()), tmp.data()); }
    return arr;
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_LencNative_compileLtoProgramNative(
  JNIEnv* env, jclass, jlongArray j_asts)
{
  JNI_NULL_CHECK(env, j_asts, "compiled AST array is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jlongArray ast_handles(env, j_asts);
    std::vector<std::reference_wrapper<cudf::ast::expression const>> exprs;
    exprs.reserve(ast_handles.size());
    bool has_lits = false;
    for (auto const h : ast_handles) {
      auto const* ce = reinterpret_cast<cudf::jni::ast::compiled_expr const*>(h);
      exprs.emplace_back(ce->get_jit_top_expression());
      has_lits |= ce->has_jit_literals();
    }
    ast_handles.cancel();
    auto prog = rapids::jni::lenc::compile_lto_program(exprs);
    if (has_lits) { cudf::get_default_stream().sync(); }
    return std::bit_cast<jlong>(prog.release());
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_LencNative_runLtoProgramNative(
  JNIEnv* env,
  jclass,
  jlong j_prog,
  jlong j_plan,
  jlong j_host_addr,
  jobjectArray j_dict_lits,
  jlong j_selected_rows_view)
{
  JNI_NULL_CHECK(env, j_prog, "lto program is null", nullptr);
  JNI_NULL_CHECK(env, j_plan, "plan is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* prog       = std::bit_cast<cudf::transform_program*>(j_prog);
    auto const* plan = std::bit_cast<rapids::jni::lenc::ReadPlan const*>(j_plan);
    auto const* src  = std::bit_cast<uint8_t const*>(j_host_addr);
    auto const* sel  = (j_selected_rows_view != 0)
                         ? reinterpret_cast<cudf::column_view const*>(j_selected_rows_view)
                         : nullptr;
    std::vector<std::vector<std::string>> dict_lits;
    if (j_dict_lits != nullptr) {
      jsize const n = env->GetArrayLength(j_dict_lits);
      dict_lits.resize(static_cast<size_t>(n));
      for (jsize i = 0; i < n; ++i) {
        auto inner = static_cast<jobjectArray>(env->GetObjectArrayElement(j_dict_lits, i));
        if (inner != nullptr) {
          cudf::jni::native_jstringArray sarr(env, inner);
          dict_lits[static_cast<size_t>(i)] = sarr.as_cpp_vector();
          env->DeleteLocalRef(inner);
        }
      }
    }
    auto tbl = rapids::jni::lenc::run_lto_program(*prog, *plan, src, dict_lits, sel);
    return cudf::jni::convert_table_for_return(env, std::move(tbl));
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_LencNative_compactMaskIndicesNative(
  JNIEnv* env, jclass, jlong j_mask_view)
{
  JNI_NULL_CHECK(env, j_mask_view, "mask view is null", 0);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const* cv = reinterpret_cast<cudf::column_view const*>(j_mask_view);
    return cudf::jni::release_as_jlong(rapids::jni::lenc::compact_mask_indices(*cv));
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL Java_com_nvidia_spark_rapids_jni_LencNative_decodeFromHostSelectedNative(
  JNIEnv* env,
  jclass,
  jlong j_plan,
  jlong j_host_addr,
  jbyteArray j_dict32,
  jlong j_selected_rows_view)
{
  JNI_NULL_CHECK(env, j_plan, "plan is null", nullptr);
  JNI_NULL_CHECK(env, j_selected_rows_view, "selected rows view is null", nullptr);
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto const* plan = std::bit_cast<rapids::jni::lenc::ReadPlan const*>(j_plan);
    auto const* src  = std::bit_cast<uint8_t const*>(j_host_addr);
    auto const* sel  = reinterpret_cast<cudf::column_view const*>(j_selected_rows_view);
    std::vector<uint8_t> d32;
    if (j_dict32 != nullptr) {
      cudf::jni::native_jbyteArray nb(env, j_dict32);
      d32.resize(static_cast<size_t>(nb.size()));
      for (size_t i = 0; i < d32.size(); ++i) { d32[i] = static_cast<uint8_t>(nb[i]); }
    }
    auto tbl = rapids::jni::lenc::decode_from_host_selected(*plan, src, d32, *sel);
    return cudf::jni::convert_table_for_return(env, std::move(tbl));
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_LencNative_destroyLtoProgramNative(
  JNIEnv* env, jclass, jlong j_prog)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    delete std::bit_cast<cudf::transform_program*>(j_prog);
  }
  JNI_CATCH(env, );
}

// ---------------- Track 3: Min-Max Stats & Single-H2D Unified Filter+Project LTO ----------------

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_LencNative_planColMinsNative(JNIEnv* env, jclass, jlong j_plan)
{
  JNI_TRY
  {
    auto const* plan = std::bit_cast<rapids::jni::lenc::ReadPlan const*>(j_plan);
    cudf::jni::native_jlongArray out(env, static_cast<jsize>(plan->col_mins.size()));
    for (size_t i = 0; i < plan->col_mins.size(); ++i) { out[i] = plan->col_mins[i]; }
    return out.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_LencNative_planColMaxsNative(JNIEnv* env, jclass, jlong j_plan)
{
  JNI_TRY
  {
    auto const* plan = std::bit_cast<rapids::jni::lenc::ReadPlan const*>(j_plan);
    cudf::jni::native_jlongArray out(env, static_cast<jsize>(plan->col_maxs.size()));
    for (size_t i = 0; i < plan->col_maxs.size(); ++i) { out[i] = plan->col_maxs[i]; }
    return out.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jbyteArray JNICALL Java_com_nvidia_spark_rapids_jni_LencNative_planColHasMinMaxNative(
  JNIEnv* env, jclass, jlong j_plan)
{
  JNI_TRY
  {
    auto const* plan = std::bit_cast<rapids::jni::lenc::ReadPlan const*>(j_plan);
    cudf::jni::native_jbyteArray out(env, static_cast<jsize>(plan->col_has_minmax.size()));
    for (size_t i = 0; i < plan->col_has_minmax.size(); ++i) {
      out[i] = static_cast<jbyte>(plan->col_has_minmax[i]);
    }
    return out.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jbyteArray JNICALL
Java_com_nvidia_spark_rapids_jni_LencNative_planColScaleExp10Native(JNIEnv* env, jclass, jlong j_plan)
{
  JNI_TRY
  {
    auto const* plan = std::bit_cast<rapids::jni::lenc::ReadPlan const*>(j_plan);
    cudf::jni::native_jbyteArray out(env, static_cast<jsize>(plan->metas.size()));
    for (size_t i = 0; i < plan->metas.size(); ++i) {
      out[i] = static_cast<jbyte>(plan->metas[i].scale_exp10);
    }
    return out.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

static std::vector<std::vector<std::string>> unpack_dict_lits(JNIEnv* env, jobjectArray j_lits)
{
  std::vector<std::vector<std::string>> dlits;
  if (j_lits == nullptr) { return dlits; }
  jsize const ns = env->GetArrayLength(j_lits);
  dlits.resize(static_cast<size_t>(ns));
  for (jsize s = 0; s < ns; ++s) {
    auto inner = static_cast<jobjectArray>(env->GetObjectArrayElement(j_lits, s));
    if (inner != nullptr) {
      jsize const nl = env->GetArrayLength(inner);
      dlits[static_cast<size_t>(s)].reserve(static_cast<size_t>(nl));
      for (jsize l = 0; l < nl; ++l) {
        auto jstr = static_cast<jstring>(env->GetObjectArrayElement(inner, l));
        cudf::jni::native_jstring ns_str(env, jstr);
        dlits[static_cast<size_t>(s)].emplace_back(ns_str.get());
        env->DeleteLocalRef(jstr);
      }
      env->DeleteLocalRef(inner);
    }
  }
  return dlits;
}

// ---------------- Item 1: S3A/HDFS Distributed Plan-From-Buffers + Read Ranges ----------------

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_LencNative_planReadFromBuffersNative(
  JNIEnv* env, jclass, jbyteArray j_hdr, jbyteArray j_ftr, jintArray j_col_indices)
{
  JNI_TRY
  {
    cudf::jni::native_jbyteArray hdr(env, j_hdr);
    cudf::jni::native_jbyteArray ftr(env, j_ftr);
    cudf::jni::native_jintArray ci(env, j_col_indices);
    std::vector<int32_t> cols(ci.data(), ci.data() + ci.size());
    auto plan = rapids::jni::lenc::plan_read_from_buffers(
      reinterpret_cast<uint8_t const*>(hdr.data()),
      static_cast<size_t>(hdr.size()),
      reinterpret_cast<uint8_t const*>(ftr.data()),
      static_cast<size_t>(ftr.size()),
      cols);
    auto* ptr = new rapids::jni::lenc::ReadPlan(std::move(plan));
    return std::bit_cast<jlong>(ptr);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_LencNative_planGetReadRangesNative(JNIEnv* env, jclass, jlong j_plan)
{
  JNI_TRY
  {
    auto const* plan = std::bit_cast<rapids::jni::lenc::ReadPlan const*>(j_plan);
    auto ranges      = rapids::jni::lenc::plan_get_read_ranges(*plan);
    cudf::jni::native_jlongArray out(env, static_cast<jsize>(ranges.size()));
    for (size_t i = 0; i < ranges.size(); ++i) { out[i] = ranges[i]; }
    return out.get_jArray();
  }
  JNI_CATCH(env, nullptr);
}

static std::vector<rapids::jni::lenc::ChunkRangePred> unpack_chunk_preds(JNIEnv* env,
                                                                         jintArray j_slots,
                                                                         jintArray j_ops,
                                                                         jbyteArray j_is_f,
                                                                         jlongArray j_ib,
                                                                         jdoubleArray j_db)
{
  std::vector<rapids::jni::lenc::ChunkRangePred> out;
  if (j_slots == nullptr) { return out; }
  cudf::jni::native_jintArray sl(env, j_slots);
  cudf::jni::native_jintArray op(env, j_ops);
  cudf::jni::native_jbyteArray isf(env, j_is_f);
  cudf::jni::native_jlongArray ib(env, j_ib);
  cudf::jni::native_jdoubleArray db(env, j_db);
  out.reserve(static_cast<size_t>(sl.size()));
  for (int i = 0; i < sl.size(); ++i) {
    out.push_back(rapids::jni::lenc::ChunkRangePred{
      sl[i], op[i], static_cast<uint8_t>(isf[i]), ib[i], db[i]});
  }
  return out;
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_LencNative_runLtoFilterAndProjectNative(JNIEnv* env,
                                                                         jclass,
                                                                         jlong j_filter_prog,
                                                                         jlong j_proj_prog,
                                                                         jlong j_union_plan,
                                                                         jlong j_host_addr,
                                                                         jintArray j_filter_cmap,
                                                                         jobjectArray j_filter_dlits,
                                                                         jintArray j_proj_cmap,
                                                                         jobjectArray j_proj_dlits,
                                                                         jbyteArray j_out_dict32,
                                                                         jintArray j_cp_slots,
                                                                         jintArray j_cp_ops,
                                                                         jbyteArray j_cp_isf,
                                                                         jlongArray j_cp_ib,
                                                                         jdoubleArray j_cp_db)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* fprog = std::bit_cast<cudf::transform_program*>(j_filter_prog);
    auto* pprog =
      (j_proj_prog != 0) ? std::bit_cast<cudf::transform_program*>(j_proj_prog) : nullptr;
    auto const* plan = std::bit_cast<rapids::jni::lenc::ReadPlan const*>(j_union_plan);
    auto const* src  = std::bit_cast<uint8_t const*>(j_host_addr);

    cudf::jni::native_jintArray fc(env, j_filter_cmap);
    std::vector<int32_t> fcmap(fc.data(), fc.data() + fc.size());
    cudf::jni::native_jintArray pc(env, j_proj_cmap);
    std::vector<int32_t> pcmap(pc.data(), pc.data() + pc.size());

    auto fdlits = unpack_dict_lits(env, j_filter_dlits);
    auto pdlits = unpack_dict_lits(env, j_proj_dlits);
    std::vector<uint8_t> d32;
    if (j_out_dict32 != nullptr) {
      cudf::jni::native_jbyteArray nb(env, j_out_dict32);
      d32.resize(static_cast<size_t>(nb.size()));
      for (size_t i = 0; i < d32.size(); ++i) { d32[i] = static_cast<uint8_t>(nb[i]); }
    }
    auto cpreds = unpack_chunk_preds(env, j_cp_slots, j_cp_ops, j_cp_isf, j_cp_ib, j_cp_db);
    auto tbl    = rapids::jni::lenc::run_lto_filter_and_project(
      *fprog, pprog, *plan, src, fcmap, fdlits, pcmap, pdlits, d32, cpreds);
    if (!tbl) { return nullptr; }
    return cudf::jni::convert_table_for_return(env, std::move(tbl));
  }
  JNI_CATCH(env, nullptr);
}

// ---------------- Track 1: GPU-Native Lightweight-Encoded Shuffle ----------------

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

// ---------------- Item 2: Generic Compiler-Level NVRTC/LTO Reduce + Conditional GroupBy ---------

JNIEXPORT jlong JNICALL Java_com_nvidia_spark_rapids_jni_LencNative_compileJitReduceNative(
  JNIEnv* env, jclass, jstring j_cuda_src, jstring j_kernel_name)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    cudf::jni::native_jstring src(env, j_cuda_src);
    cudf::jni::native_jstring kname(env, j_kernel_name);
    auto* prog = rapids::jni::lenc::compile_jit_reduce_program(src.get(), kname.get());
    return std::bit_cast<jlong>(prog);
  }
  JNI_CATCH(env, 0);
}

JNIEXPORT void JNICALL Java_com_nvidia_spark_rapids_jni_LencNative_destroyJitReduceNative(
  JNIEnv* env, jclass, jlong j_prog)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    rapids::jni::lenc::destroy_jit_reduce_program(
      std::bit_cast<rapids::jni::lenc::JitReduceProgram*>(j_prog));
  }
  JNI_CATCH(env, );
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_LencNative_runJitReduceLencNative(JNIEnv* env,
                                                                   jclass,
                                                                   jlong j_prog,
                                                                   jlong j_plan,
                                                                   jlong j_host_addr,
                                                                   jobjectArray j_dlits,
                                                                   jintArray j_gk_slots,
                                                                   jint j_max_groups,
                                                                   jbyteArray j_agg_kinds,
                                                                   jintArray j_cp_slots,
                                                                   jintArray j_cp_ops,
                                                                   jbyteArray j_cp_isf,
                                                                   jlongArray j_cp_ib,
                                                                   jdoubleArray j_cp_db)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* prog       = std::bit_cast<rapids::jni::lenc::JitReduceProgram*>(j_prog);
    auto const* plan = std::bit_cast<rapids::jni::lenc::ReadPlan const*>(j_plan);
    auto const* src  = std::bit_cast<uint8_t const*>(j_host_addr);
    auto dlits       = unpack_dict_lits(env, j_dlits);
    std::vector<int32_t> gk_slots;
    if (j_gk_slots != nullptr) {
      cudf::jni::native_jintArray gka(env, j_gk_slots);
      gk_slots.assign(gka.data(), gka.data() + gka.size());
    }
    cudf::jni::native_jbyteArray ak(env, j_agg_kinds);
    std::vector<uint8_t> agg_kinds(static_cast<size_t>(ak.size()));
    for (size_t i = 0; i < agg_kinds.size(); ++i) { agg_kinds[i] = static_cast<uint8_t>(ak[i]); }
    auto cpreds = unpack_chunk_preds(env, j_cp_slots, j_cp_ops, j_cp_isf, j_cp_ib, j_cp_db);
    auto tbl    = rapids::jni::lenc::run_jit_reduce_lenc(
      *prog, *plan, src, dlits, gk_slots, j_max_groups, agg_kinds, cpreds);
    return cudf::jni::convert_table_for_return(env, std::move(tbl));
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_LencNative_runJitReduceTableNative(JNIEnv* env,
                                                                    jclass,
                                                                    jlong j_prog,
                                                                    jlong j_table_view,
                                                                    jobjectArray j_dlits,
                                                                    jintArray j_gk_slots,
                                                                    jint j_max_groups,
                                                                    jbyteArray j_agg_kinds)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* prog     = std::bit_cast<rapids::jni::lenc::JitReduceProgram*>(j_prog);
    auto const* tv = reinterpret_cast<cudf::table_view const*>(j_table_view);
    auto dlits     = unpack_dict_lits(env, j_dlits);
    std::vector<int32_t> gk_slots;
    if (j_gk_slots != nullptr) {
      cudf::jni::native_jintArray gka(env, j_gk_slots);
      gk_slots.assign(gka.data(), gka.data() + gka.size());
    }
    cudf::jni::native_jbyteArray ak(env, j_agg_kinds);
    std::vector<uint8_t> agg_kinds(static_cast<size_t>(ak.size()));
    for (size_t i = 0; i < agg_kinds.size(); ++i) { agg_kinds[i] = static_cast<uint8_t>(ak[i]); }
    auto tbl = rapids::jni::lenc::run_jit_reduce_table(
      *prog, *tv, dlits, gk_slots, j_max_groups, agg_kinds);
    return cudf::jni::convert_table_for_return(env, std::move(tbl));
  }
  JNI_CATCH(env, nullptr);
}

JNIEXPORT jlongArray JNICALL
Java_com_nvidia_spark_rapids_jni_LencNative_runJitReduceHighCardNative(JNIEnv* env,
                                                                       jclass,
                                                                       jlong j_prog,
                                                                       jlong j_gk_tv,
                                                                       jlong j_val_tv,
                                                                       jobjectArray j_dlits,
                                                                       jbyteArray j_agg_kinds)
{
  JNI_TRY
  {
    cudf::jni::auto_set_device(env);
    auto* prog     = std::bit_cast<rapids::jni::lenc::JitReduceProgram*>(j_prog);
    auto const* gk = reinterpret_cast<cudf::table_view const*>(j_gk_tv);
    auto const* vc = reinterpret_cast<cudf::table_view const*>(j_val_tv);
    auto dlits     = unpack_dict_lits(env, j_dlits);
    cudf::jni::native_jbyteArray ak(env, j_agg_kinds);
    std::vector<uint8_t> agg_kinds(static_cast<size_t>(ak.size()));
    for (size_t i = 0; i < agg_kinds.size(); ++i) { agg_kinds[i] = static_cast<uint8_t>(ak[i]); }
    auto tbl = rapids::jni::lenc::run_jit_reduce_highcard_groupby(*prog, *gk, *vc, dlits, agg_kinds);
    return cudf::jni::convert_table_for_return(env, std::move(tbl));
  }
  JNI_CATCH(env, nullptr);
}

}  // extern "C"

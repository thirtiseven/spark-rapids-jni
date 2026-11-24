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

}  // extern "C"

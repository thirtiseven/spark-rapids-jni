/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>

#include <get_json_object.hpp>

struct GetJsonObjectTests : public cudf::test::BaseFixture {};
using spark_rapids_jni::json_parser;
using spark_rapids_jni::json_parser_options;
using spark_rapids_jni::detail::path_instruction;
using spark_rapids_jni::detail::path_instruction_type;

template <int max_json_depth = spark_rapids_jni::curr_max_json_nesting_depth>
spark_rapids_jni::json_parser<max_json_depth> get_parser(
  spark_rapids_jni::json_parser_options& options, std::string const& json_str)
{
  options.set_allow_single_quotes(true);
  options.set_allow_unescaped_control_chars(true);
  options.set_allow_tailing_sub_string(true);
  options.set_max_string_len(20 * 1024 * 1024);
  options.set_max_num_len(1000);
  return json_parser<max_json_depth>(options, json_str.data(), json_str.size());
}

template <int max_json_depth = spark_rapids_jni::curr_max_json_nesting_depth>
spark_rapids_jni::detail::json_generator<max_json_depth> get_generator(char* buf)
{
  return spark_rapids_jni::detail::json_generator<max_json_depth>(buf);
}

template <int max_json_depth = spark_rapids_jni::curr_max_json_nesting_depth>
bool eval_path(spark_rapids_jni::json_parser<max_json_depth>& p,
               spark_rapids_jni::detail::json_generator<max_json_depth>& g,
               spark_rapids_jni::detail::path_instruction const* path_ptr,
               int path_size)
{
  return spark_rapids_jni::detail::path_evaluator::evaluate_path<max_json_depth>(
    p, g, true, spark_rapids_jni::detail::write_style::raw_style, path_ptr, path_size);
}

path_instruction get_path(path_instruction_type type, cudf::string_view name, int index)
{
  path_instruction p(type);
  p.name  = name;
  p.index = index;
  return p;
}

void clear_buff(char buf[], std::size_t size);
void assert_start_with(char* buf, std::size_t buf_size, const std::string& prefix);

TEST_F(GetJsonObjectTests, NormalTest)
{
  constexpr int buf_len = 1024;
  char buf[buf_len];

  json_parser_options options;
  std::string json = " {  'k'  :  1   }  ";
  auto p           = get_parser(options, json);
  auto g           = get_generator(buf);
  p.next_token();

  // cudf::string_view empty_name("", 0);
  // cudf::string_view name("k", 1);
  // auto p1 = get_path(path_instruction_type::key, empty_name, -1);
  // auto p2 = get_path(path_instruction_type::named, name, -1);
  // path_instruction paths[]{p1, p2};
  // ASSERT_TRUE(eval_path(p, g, paths, 2));
  clear_buff(buf, buf_len);
  ASSERT_TRUE(eval_path(p, g, nullptr, 0));
  // std::string expect = "{\"k\":1}"
  // assert_start_with(buf, buf_len, expect);
}

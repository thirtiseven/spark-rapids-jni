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

// defined in json_parser_tests.cpp
void clear_buff(char buf[], std::size_t size);
void assert_start_with(char* buf, std::size_t buf_size, const std::string& prefix);

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

path_instruction get_subscript_path() { return path_instruction(path_instruction_type::subscript); }

path_instruction get_wildcard_path() { return path_instruction(path_instruction_type::wildcard); }

path_instruction get_key_path() { return path_instruction(path_instruction_type::key); }

path_instruction get_index_path(int index)
{
  auto p  = path_instruction(path_instruction_type::index);
  p.index = index;
  return p;
}

path_instruction get_named_path(std::string name)
{
  auto p = path_instruction(path_instruction_type::named);
  p.name = cudf::string_view(name.data(), name.size());
  return p;
}

void test_get_json_object(std::string json,
                          std::vector<path_instruction> paths,
                          std::string expected)
{
  size_t buf_len = 100 * 1024;
  char buf[buf_len];
  clear_buff(buf, buf_len);

  json_parser_options options;
  auto p = get_parser(options, json);
  auto g = get_generator(buf);
  p.next_token();

  ASSERT_TRUE(eval_path(p, g, paths.data(), paths.size()));
  assert_start_with(buf, buf_len, expected);
}

void test_get_json_object_fail(std::string json, std::vector<path_instruction> paths)
{
  size_t buf_len = 100 * 1024;
  char buf[buf_len];
  clear_buff(buf, buf_len);

  json_parser_options options;
  auto p = get_parser(options, json);
  auto g = get_generator(buf);
  p.next_token();

  ASSERT_FALSE(eval_path(p, g, paths.data(), paths.size()));
}

void test_get_json_object(std::string json, std::string expected)
{
  size_t buf_len = 100 * 1024;
  char buf[buf_len];
  clear_buff(buf, buf_len);

  json_parser_options options;
  auto p = get_parser(options, json);
  auto g = get_generator(buf);
  p.next_token();

  ASSERT_TRUE(eval_path(p, g, nullptr, 0));
  assert_start_with(buf, buf_len, expected);
}

static const std::string json_for_test = R"(
{"store":{"fruit":[{"weight":8,"type":"apple"},{"weight":9,"type":"pear"}],
"basket":[[1,2,{"b":"y","a":"x"}],[3,4],[5,6]],"book":[{"author":"Nigel Rees",
"title":"Sayings of the Century","category":"reference","price":8.95},
{"author":"Herman Melville","title":"Moby Dick","category":"fiction","price":8.99,
"isbn":"0-553-21311-3"},{"author":"J. R. R. Tolkien","title":"The Lord of the Rings",
"category":"fiction","reader":[{"age":25,"name":"bob"},{"age":26,"name":"jack"}],
"price":22.99,"isbn":"0-395-19395-8"}],"bicycle":{"price":19.95,"color":"red"}},
"email":"amy@only_for_json_udf_test.net","owner":"amy","zip code":"94025",
"fb:testid":"1234"}
)";

/**
 * Tests from Spark JsonExpressionsSuite
 */
TEST_F(GetJsonObjectTests, NormalTest)
{
  test_get_json_object(" {  'k'  :  [1, [21, 22, 23], 3]   }  ",
                       std::vector<path_instruction>{get_key_path(), get_named_path("k")},
                       "[1,[21,22,23],3]");
  test_get_json_object(" {  'k'  :  [1, [21, 22, 23], 3]   }  ", R"({"k":[1,[21,22,23],3]})");
  test_get_json_object(
    " {  'k'  :  [1, [21, 22, 23], 3]   }  ",
    std::vector<path_instruction>{
      get_key_path(), get_named_path("k"), get_subscript_path(), get_wildcard_path()},
    R"([1,[21,22,23],3])");
  test_get_json_object(" {  'k'  :  [1, [21, 22, 23], 3]   }  ",
                       std::vector<path_instruction>{get_key_path(),
                                                     get_named_path("k"),
                                                     get_subscript_path(),
                                                     get_wildcard_path(),
                                                     get_subscript_path(),
                                                     get_wildcard_path()},
                       R"([1,21,22,23,3])");
  test_get_json_object(" {  'k'  :  [1, [21, 22, 23], 3]   }  ",
                       std::vector<path_instruction>{get_key_path(),
                                                     get_named_path("k"),
                                                     get_subscript_path(),
                                                     get_wildcard_path(),
                                                     get_subscript_path(),
                                                     get_index_path(0)},
                       R"(21)");
  test_get_json_object(
    " [[11,12,13], [21, 22, 23], [31, 32, 33]]  ",
    std::vector<path_instruction>{
      get_subscript_path(), get_wildcard_path(), get_subscript_path(), get_index_path(0)},
    R"([11,21,31])");
  test_get_json_object(
    " [[11,12,13]]  ",
    std::vector<path_instruction>{
      get_subscript_path(), get_wildcard_path(), get_subscript_path(), get_index_path(0)},
    R"(11)");

  test_get_json_object(
    " [[11,12,13]]  ",
    std::vector<path_instruction>{
      get_subscript_path(), get_wildcard_path(), get_subscript_path(), get_index_path(0)},
    R"(11)");

  // tests from Spark unit test cases
  test_get_json_object(
    json_for_test,
    std::vector<path_instruction>{
      get_key_path(), get_named_path("store"), get_key_path(), get_named_path("bicycle")},
    R"({"price":19.95,"color":"red"})");

  test_get_json_object(
    R"({ "key with spaces": "it works" })",
    std::vector<path_instruction>{get_key_path(), get_named_path("key with spaces")},
    R"(it works)");

  std::string e1 =
    R"([{"author":"Nigel Rees","title":"Sayings of the Century","category":"reference",)";
  e1 += R"("price":8.95},{"author":"Herman Melville","title":"Moby Dick","category":"fiction",)";
  e1 += R"("price":8.99,"isbn":"0-553-21311-3"},{"author":"J. R. R. Tolkien","title":)";
  e1 += R"("The Lord of the Rings","category":"fiction","reader":[{"age":25,"name":"bob"},)";
  e1 += R"({"age":26,"name":"jack"}],"price":22.99,"isbn":"0-395-19395-8"}])";

  test_get_json_object(
    json_for_test,
    std::vector<path_instruction>{
      get_key_path(), get_named_path("store"), get_key_path(), get_named_path("book")},
    e1);

  std::string e2 = R"({"author":"Nigel Rees","title":"Sayings of the Century",)";
  e2 += R"("category":"reference","price":8.95})";
  test_get_json_object(json_for_test,
                       std::vector<path_instruction>{get_key_path(),
                                                     get_named_path("store"),
                                                     get_key_path(),
                                                     get_named_path("book"),
                                                     get_subscript_path(),
                                                     get_index_path(0)},
                       e2);

  test_get_json_object(json_for_test,
                       std::vector<path_instruction>{get_key_path(),
                                                     get_named_path("store"),
                                                     get_key_path(),
                                                     get_named_path("book"),
                                                     get_subscript_path(),
                                                     get_wildcard_path()},
                       e1);

  auto e3 = json_for_test;
  e3.erase(std::remove(e3.begin(), e3.end(), '\n'), e3.end());
  test_get_json_object(json_for_test, e3);

  test_get_json_object(json_for_test,
                       std::vector<path_instruction>{get_key_path(),
                                                     get_named_path("store"),
                                                     get_key_path(),
                                                     get_named_path("book"),
                                                     get_subscript_path(),
                                                     get_index_path(0),
                                                     get_key_path(),
                                                     get_named_path("category")},
                       "reference");

  test_get_json_object(json_for_test,
                       std::vector<path_instruction>{get_key_path(),
                                                     get_named_path("store"),
                                                     get_key_path(),
                                                     get_named_path("book"),
                                                     get_subscript_path(),
                                                     get_wildcard_path(),
                                                     get_key_path(),
                                                     get_named_path("category")},
                       R"(["reference","fiction","fiction"])");

  test_get_json_object(json_for_test,
                       std::vector<path_instruction>{get_key_path(),
                                                     get_named_path("store"),
                                                     get_key_path(),
                                                     get_named_path("book"),
                                                     get_subscript_path(),
                                                     get_wildcard_path(),
                                                     get_key_path(),
                                                     get_named_path("isbn")},
                       R"(["0-553-21311-3","0-395-19395-8"])");

  test_get_json_object(json_for_test,
                       std::vector<path_instruction>{get_key_path(),
                                                     get_named_path("store"),
                                                     get_key_path(),
                                                     get_named_path("book"),
                                                     get_subscript_path(),
                                                     get_wildcard_path(),
                                                     get_key_path(),
                                                     get_named_path("reader")},
                       R"([{"age":25,"name":"bob"},{"age":26,"name":"jack"}])");

  test_get_json_object(json_for_test,
                       std::vector<path_instruction>{get_key_path(),
                                                     get_named_path("store"),
                                                     get_key_path(),
                                                     get_named_path("basket"),
                                                     get_subscript_path(),
                                                     get_index_path(0),
                                                     get_subscript_path(),
                                                     get_index_path(1)},
                       R"(2)");

  test_get_json_object(json_for_test,
                       std::vector<path_instruction>{get_key_path(),
                                                     get_named_path("store"),
                                                     get_key_path(),
                                                     get_named_path("basket"),
                                                     get_subscript_path(),
                                                     get_wildcard_path()},
                       R"([[1,2,{"b":"y","a":"x"}],[3,4],[5,6]])");

  test_get_json_object(json_for_test,
                       std::vector<path_instruction>{get_key_path(),
                                                     get_named_path("store"),
                                                     get_key_path(),
                                                     get_named_path("basket"),
                                                     get_subscript_path(),
                                                     get_wildcard_path(),
                                                     get_subscript_path(),
                                                     get_index_path(0)},
                       R"([1,3,5])");

  test_get_json_object(json_for_test,
                       std::vector<path_instruction>{get_key_path(),
                                                     get_named_path("store"),
                                                     get_key_path(),
                                                     get_named_path("basket"),
                                                     get_subscript_path(),
                                                     get_index_path(0),
                                                     get_subscript_path(),
                                                     get_wildcard_path()},
                       R"([1,2,{"b":"y","a":"x"}])");

  test_get_json_object(json_for_test,
                       std::vector<path_instruction>{get_key_path(),
                                                     get_named_path("store"),
                                                     get_key_path(),
                                                     get_named_path("basket"),
                                                     get_subscript_path(),
                                                     get_wildcard_path(),
                                                     get_subscript_path(),
                                                     get_wildcard_path()},
                       R"([1,2,{"b":"y","a":"x"},3,4,5,6])");

  test_get_json_object(json_for_test,
                       std::vector<path_instruction>{get_key_path(),
                                                     get_named_path("store"),
                                                     get_key_path(),
                                                     get_named_path("basket"),
                                                     get_subscript_path(),
                                                     get_index_path(0),
                                                     get_subscript_path(),
                                                     get_index_path(2),
                                                     get_key_path(),
                                                     get_named_path("b")},
                       R"(y)");

  test_get_json_object(json_for_test,
                       std::vector<path_instruction>{get_key_path(),
                                                     get_named_path("store"),
                                                     get_key_path(),
                                                     get_named_path("basket"),
                                                     get_subscript_path(),
                                                     get_index_path(0),
                                                     get_subscript_path(),
                                                     get_wildcard_path(),
                                                     get_key_path(),
                                                     get_named_path("b")},
                       R"(["y"])");

  test_get_json_object(json_for_test,
                       std::vector<path_instruction>{get_key_path(), get_named_path("zip code")},
                       R"(94025)");

  test_get_json_object(json_for_test,
                       std::vector<path_instruction>{get_key_path(), get_named_path("fb:testid")},
                       R"(1234)");

  test_get_json_object(
    R"({"a":"b\nc"})", std::vector<path_instruction>{get_key_path(), get_named_path("a")}, "b\nc");

  test_get_json_object(
    R"({"a":"b\"c"})", std::vector<path_instruction>{get_key_path(), get_named_path("a")}, "b\"c");

  test_get_json_object_fail(
    json_for_test, std::vector<path_instruction>{get_key_path(), get_named_path("non_exist_key")});

  test_get_json_object_fail(json_for_test,
                            std::vector<path_instruction>{get_key_path(),
                                                          get_named_path("store"),
                                                          get_key_path(),
                                                          get_named_path("book"),
                                                          get_subscript_path(),
                                                          get_index_path(10)});

  test_get_json_object_fail(json_for_test,
                            std::vector<path_instruction>{get_key_path(),
                                                          get_named_path("store"),
                                                          get_key_path(),
                                                          get_named_path("book"),
                                                          get_subscript_path(),
                                                          get_index_path(0),
                                                          get_key_path(),
                                                          get_named_path("non_exist_key")});

  test_get_json_object_fail(json_for_test,
                            std::vector<path_instruction>{get_key_path(),
                                                          get_named_path("store"),
                                                          get_key_path(),
                                                          get_named_path("basket"),
                                                          get_subscript_path(),
                                                          get_wildcard_path(),
                                                          get_key_path(),
                                                          get_named_path("non_exist_key")});

  std::string bad_json = "\u0000\u0000\u0000A\u0001AAA";
  test_get_json_object_fail(bad_json,
                            std::vector<path_instruction>{get_key_path(), get_named_path("a")});
}

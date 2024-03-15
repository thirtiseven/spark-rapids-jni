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

#include "json_parser.hpp"

#include <cudf/strings/string_view.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <thrust/optional.h>
#include <thrust/pair.h>
#include <thrust/scan.h>
#include <thrust/tuple.h>

#include <memory>

namespace spark_rapids_jni {

namespace detail {

/**
 * write JSON style
 */
enum class write_style { raw_style, quoted_style, flatten_style };

/**
 * path instruction type
 */
enum class path_instruction_type { subscript, wildcard, key, index, named };

/**
 * path instruction
 */
struct path_instruction {
  CUDF_HOST_DEVICE inline path_instruction(path_instruction_type _type) : type(_type) {}

  path_instruction_type type;

  // used when type is named type
  cudf::string_view name;

  // used when type is index
  int index{-1};
};

// TODO parse JSON path
thrust::optional<rmm::device_uvector<path_instruction>> parse_path(
  cudf::string_scalar const& json_path)
{
  return thrust::nullopt;
}

/**
 * JSON generator is used to write out JSON string.
 * It's not a full featured JSON generator, because get json object
 * outputs an array or single item. JSON object is wroten as a whole item.
 */
template <int max_json_nesting_depth = curr_max_json_nesting_depth>
class json_generator {
 public:
  CUDF_HOST_DEVICE json_generator(char* _output, size_t _output_len)
    : output(_output), output_len(_output_len)
  {
  }
  CUDF_HOST_DEVICE json_generator() : output(nullptr), output_len(0) {}

  // create a nested child generator based on this parent generator
  // child generator is a view
  CUDF_HOST_DEVICE json_generator new_child_generator()
  {
    if (nullptr == output) {
      return json_generator();
    } else {
      return json_generator(output + output_len, 0);
    }
  }

  CUDF_HOST_DEVICE void write_start_array()
  {
    if (output) { *(output + output_len) = '['; }
    output_len++;
    is_first_item[array_depth] = true;
    array_depth++;
  }

  /**
   * only update the internal state, not actually write to underlying buffer
   */
  CUDF_HOST_DEVICE void write_start_array_fake()
  {
    output_len++;
    is_first_item[array_depth] = true;
    array_depth++;
  }

  CUDF_HOST_DEVICE void write_end_array()
  {
    if (output) { *(output + output_len) = ']'; }
    output_len++;
    array_depth--;
  }

  CUDF_HOST_DEVICE void write_end_array_fake()
  {
    output_len++;
    array_depth--;
  }

  CUDF_HOST_DEVICE bool need_comma() { return (array_depth > 0 && is_first_item[array_depth - 1]); }

  /**
   * write comma accroding to current generator state
   */
  CUDF_HOST_DEVICE void try_write_comma()
  {
    if (need_comma()) {
      // in array context and writes first item
      *(output + output_len) = ',';
      output_len++;
    }
  }

  /**
   * copy current structure when parsing. If current token is start
   * object/array, then copy to corresponding matched end object/array. return
   * false if JSON format is invalid return true if JSON format is valid
   */
  CUDF_HOST_DEVICE bool copy_current_structure(json_parser<max_json_nesting_depth>& parser)
  {
    // first try add comma
    try_write_comma();

    is_first_item[array_depth - 1] = false;

    if (nullptr != output) {
      auto copy_to       = output + output_len;
      auto [b, copy_len] = parser.copy_current_structure(copy_to);
      output_len += copy_len;
      return b;
    } else {
      char* copy_to      = nullptr;
      auto [b, copy_len] = parser.copy_current_structure(copy_to);
      output_len += copy_len;
      return b;
    }
  }

  /**
   * Get current text from JSON parser and then write the text
   * Note: Because JSON strings contains '\' to do escape,
   * JSON parser should do unescape to remove '\' and JSON parser
   * then can not return a pointer and length pair (char *, len),
   * For number token, JSON parser can return a pair (char *, len)
   */
  CUDF_HOST_DEVICE void write_raw(json_parser<max_json_nesting_depth>& parser)
  {
    is_first_item[array_depth - 1] = false;

    if (nullptr != output) {
      auto copied = parser.write_unescaped_text(output + output_len);
      output_len += copied;
    } else {
      auto len = parser.compute_unescaped_len();
      output_len += len;
    }
  }

  /**
   * write child raw value
   * e.g.:
   *
   * write_array_tokens = false
   * need_comma = true
   * [1,2,3]1,2,3
   *        ^
   *        |
   *    child pointer
   * ==>>
   * [1,2,3],1,2,3
   *
   *
   * write_array_tokens = true
   * need_comma = true
   *   [12,3,4
   *     ^
   *     |
   * child pointer
   * ==>>
   *   [1,[2,3,4]
   *
   * @param child_block_begin
   * @param child_block_len
   */
  CUDF_HOST_DEVICE void write_child_raw_value(char* child_block_begin,
                                              size_t child_block_len,
                                              bool write_outer_array_tokens)
  {
    bool insert_comma = need_comma();

    is_first_item[array_depth - 1] = false;

    if (nullptr != output) {
      if (write_outer_array_tokens) {
        if (insert_comma) {
          *(child_block_begin + child_block_len + 2) = ']';
          move_forward(child_block_begin, child_block_len, 2);
          *(child_block_begin + 1) = '[';
          *(child_block_begin)     = ',';
        } else {
          *(child_block_begin + child_block_len + 1) = ']';
          move_forward(child_block_begin, child_block_len, 1);
          *(child_block_begin) = '[';
        }
      } else {
        if (insert_comma) {
          move_forward(child_block_begin, child_block_len, 1);
          *(child_block_begin) = ',';
        } else {
          // do not need comma && do not need write outer array tokens
          // do nothing, because child generator buff is directly after the parent generator
        }
      }
    }

    // update length
    if (insert_comma) { output_len++; }
    if (write_outer_array_tokens) { output_len += 2; }
    output_len += child_block_len;
  }

  CUDF_HOST_DEVICE void move_forward(char* begin, size_t len, int forward)
  {
    char* pos = begin + len + forward - 1;
    char* e   = begin + forward - 1;
    // should add outer array tokens
    // First move chars [2, end) a byte forward
    while (pos > e) {
      *pos = *(pos - 1);
      pos--;
    }
  }

  CUDF_HOST_DEVICE inline size_t get_output_len() const { return output_len; }
  CUDF_HOST_DEVICE inline char* get_output_start_position() const { return output; }
  CUDF_HOST_DEVICE inline char* get_current_output_position() const { return output + output_len; }

 private:
  char* output;
  size_t output_len;

  bool is_first_item[max_json_nesting_depth];
  int array_depth = 0;
};

// declaration
template <int max_json_nesting_depth = curr_max_json_nesting_depth>
CUDF_HOST_DEVICE bool evaluate_path(json_parser<max_json_nesting_depth>& p,
                                    json_generator<max_json_nesting_depth>& g,
                                    bool g_contains_outer_array_pairs,
                                    write_style style,
                                    path_instruction const* path_ptr,
                                    int path_size);

}  // namespace detail

/**
 * Extracts json object from a json string based on json path specified, and
 * returns json string of the extracted json object. It will return null if the
 * input json string is invalid.
 */
std::unique_ptr<cudf::column> get_json_object(
  cudf::strings_column_view const& col,
  cudf::string_scalar const& json_path,
  spark_rapids_jni::json_parser_options options,
  rmm::cuda_stream_view stream        = cudf::get_default_stream(),
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace spark_rapids_jni

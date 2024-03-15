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

#include "get_json_object.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/json/json.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

namespace spark_rapids_jni {

namespace detail {

/**
 * @brief Result of calling a parse function.
 *
 * The primary use of this is to distinguish between "success" and
 * "success but no data" return cases.  For example, if you are reading the
 * values of an array you might call a parse function in a while loop. You
 * would want to continue doing this until you either encounter an error
 * (parse_result::ERROR) or you get nothing back (parse_result::EMPTY)
 */
enum class parse_result { ERROR, SUCCESS };

CUDF_HOST_DEVICE inline bool path_is_empty(size_t path_size) { return path_size == 0; }

CUDF_HOST_DEVICE inline bool path_match_element(path_instruction const* path_ptr,
                                                size_t path_size,
                                                path_instruction_type path_type0)
{
  if (path_size < 1) { return false; }
  return path_ptr[0].type == path_type0;
}

CUDF_HOST_DEVICE inline bool path_match_elements(path_instruction const* path_ptr,
                                                 size_t path_size,
                                                 path_instruction_type path_type0,
                                                 path_instruction_type path_type1)
{
  if (path_size < 2) { return false; }
  return path_ptr[0].type == path_type0 && path_ptr[1].type == path_type1;
}

CUDF_HOST_DEVICE inline bool path_match_elements(path_instruction const* path_ptr,
                                                 size_t path_size,
                                                 path_instruction_type path_type0,
                                                 path_instruction_type path_type1,
                                                 path_instruction_type path_type2,
                                                 path_instruction_type path_type3)
{
  if (path_size < 4) { return false; }
  return path_ptr[0].type == path_type0 && path_ptr[1].type == path_type1 &&
         path_ptr[2].type == path_type2 && path_ptr[3].type == path_type3;
}

CUDF_HOST_DEVICE inline thrust::tuple<bool, int> path_match_subscript_index(
  path_instruction const* path_ptr, size_t path_size)
{
  auto match = path_match_elements(
    path_ptr, path_size, path_instruction_type::subscript, path_instruction_type::index);
  if (match) {
    return thrust::make_tuple(true, path_ptr[1].index);
  } else {
    return thrust::make_tuple(false, 0);
  }
}

CUDF_HOST_DEVICE inline thrust::tuple<bool, cudf::string_view> path_match_named(
  path_instruction const* path_ptr, size_t path_size)
{
  auto match = path_match_element(path_ptr, path_size, path_instruction_type::named);
  if (match) {
    return thrust::make_tuple(true, path_ptr[0].name);
  } else {
    return thrust::make_tuple(false, cudf::string_view());
  }
}

CUDF_HOST_DEVICE inline thrust::tuple<bool, int> path_match_subscript_index_subscript_wildcard(
  path_instruction const* path_ptr, size_t path_size)
{
  auto match = path_match_elements(path_ptr,
                                   path_size,
                                   path_instruction_type::subscript,
                                   path_instruction_type::index,
                                   path_instruction_type::subscript,
                                   path_instruction_type::wildcard);
  if (match) {
    return thrust::make_tuple(true, path_ptr[1].index);
  } else {
    return thrust::make_tuple(false, 0);
  }
}

template <int max_json_nesting_depth>
CUDF_HOST_DEVICE bool evaluate_path(json_parser<max_json_nesting_depth>& p,
                                    json_generator<max_json_nesting_depth>& g,
                                    bool g_contains_outer_array_pairs,
                                    write_style style,
                                    path_instruction const* path_ptr,
                                    int path_size)
{
  auto token = p.get_current_token();

  // case (VALUE_STRING, Nil) if style == RawStyle
  if (json_token::VALUE_STRING == token && path_is_empty(path_size) &&
      style == write_style::raw_style) {
    // there is no array wildcard or slice parent, emit this string without
    // quotes write current string in parser to generator
    g.write_raw(p);
    return true;
  }
  // case (START_ARRAY, Nil) if style == FlattenStyle
  else if (json_token::START_ARRAY == token && path_is_empty(path_size) &&
           style == write_style::flatten_style) {
    // flatten this array into the parent
    bool dirty = false;
    while (json_token::END_ARRAY != p.next_token()) {
      // JSON validation check
      if (json_token::ERROR == p.get_current_token()) { return false; }

      dirty |= evaluate_path(p, g, true, style, nullptr, 0);
    }
    return dirty;
  }
  // case (_, Nil)
  else if (path_is_empty(path_size)) {
    // general case: just copy the child tree verbatim
    return g.copy_current_structure(p);
  }
  // case (START_OBJECT, Key :: xs)
  else if (json_token::START_OBJECT == token &&
           path_match_element(path_ptr, path_size, path_instruction_type::key)) {
    bool dirty = false;
    while (json_token::END_OBJECT != p.next_token()) {
      // JSON validation check
      if (json_token::ERROR == p.get_current_token()) { return false; }

      if (dirty) {
        // once a match has been found we can skip other fields
        if (!p.try_skip_children()) {
          // JSON validation check
          return false;
        }
      } else {
        dirty = evaluate_path(p, g, true, style, path_ptr + 1, path_size - 1);
      }
    }
    return dirty;
  }
  // case (START_ARRAY, Subscript :: Wildcard :: Subscript :: Wildcard :: xs)
  else if (json_token::START_ARRAY == token &&
           path_match_elements(path_ptr,
                               path_size,
                               path_instruction_type::subscript,
                               path_instruction_type::wildcard,
                               path_instruction_type::subscript,
                               path_instruction_type::wildcard)) {
    // special handling for the non-structure preserving double wildcard
    // behavior in Hive
    bool dirty = false;
    g.write_start_array();
    while (p.next_token() != json_token::END_ARRAY) {
      // JSON validation check
      if (json_token::ERROR == p.get_current_token()) { return false; }

      dirty |= evaluate_path(p, g, true, write_style::flatten_style, path_ptr + 4, path_size - 4);
    }
    g.write_end_array();
    return dirty;
  }
  // case (START_ARRAY, Subscript :: Wildcard :: xs) if style != QuotedStyle
  else if (json_token::START_ARRAY == token &&
           path_match_elements(path_ptr,
                               path_size,
                               path_instruction_type::subscript,
                               path_instruction_type::wildcard) &&
           style != write_style::quoted_style) {
    // retain Flatten, otherwise use Quoted... cannot use Raw within an array
    write_style next_style;
    switch (style) {
      case write_style::raw_style: next_style = write_style::quoted_style; break;
      case write_style::flatten_style: next_style = write_style::flatten_style;
    }

    // temporarily buffer child matches, the emitted json will need to be
    // modified slightly if there is only a single element written

    int dirty    = 0;
    auto child_g = g.new_child_generator();

    // child generator write a fake start array
    child_g.write_start_array_fake();

    while (p.next_token() != json_token::END_ARRAY) {
      // JSON validation check
      if (json_token::ERROR == p.get_current_token()) { return false; }

      // track the number of array elements and only emit an outer array if
      // we've written more than one element, this matches Hive's behavior
      dirty += (evaluate_path(p, child_g, false, next_style, path_ptr + 2, path_size - 2) ? 1 : 0);
    }

    // child generator write a fake end array
    child_g.write_end_array_fake();

    char* child_g_start = child_g.get_output_start_position();
    size_t child_g_len  = child_g.get_output_len() - 2;  // exclude [ ]

    if (dirty > 1) {
      // add outer array tokens
      g.write_child_raw_value(child_g_start, child_g_len, true);
    } else if (dirty == 1) {
      // remove outer array tokens
      g.write_child_raw_value(child_g_start, child_g_len, false);
    }  // else do not write anything

    return dirty > 0;
  }
  // case (START_ARRAY, Subscript :: Wildcard :: xs)
  else if (json_token::START_ARRAY == token &&
           path_match_elements(path_ptr,
                               path_size,
                               path_instruction_type::subscript,
                               path_instruction_type::wildcard)) {
    bool dirty = false;
    g.write_start_array();
    while (p.next_token() != json_token::END_ARRAY) {
      // JSON validation check
      if (json_token::ERROR == p.get_current_token()) { return false; }

      // wildcards can have multiple matches, continually update the dirty count
      dirty |= evaluate_path(p, g, true, write_style::quoted_style, path_ptr + 2, path_size - 2);
    }
    g.write_end_array();

    return dirty;
  }
  // case (START_ARRAY, Subscript :: Index(idx) :: (xs@Subscript :: Wildcard ::
  // _))
  else if (json_token::START_ARRAY == token &&
           thrust::get<0>(path_match_subscript_index_subscript_wildcard(path_ptr, path_size))) {
    int idx = thrust::get<1>(path_match_subscript_index_subscript_wildcard(path_ptr, path_size));
    p.next_token();
    // JSON validation check
    if (json_token::ERROR == p.get_current_token()) { return false; }

    int i = idx;
    while (i >= 0) {
      if (p.get_current_token() == json_token::END_ARRAY) {
        // terminate, nothing has been written
        return false;
      }
      if (0 == i) {
        bool dirty =
          evaluate_path(p, g, true, write_style::quoted_style, path_ptr + 2, path_size - 2);
        while (p.next_token() != json_token::END_ARRAY) {
          // JSON validation check
          if (json_token::ERROR == p.get_current_token()) { return false; }

          // advance the token stream to the end of the array
          if (!p.try_skip_children()) { return false; }
        }
        return dirty;
      } else {
        // i > 0
        if (!p.try_skip_children()) { return false; }

        p.next_token();
        // JSON validation check
        if (json_token::ERROR == p.get_current_token()) { return false; }
      }
      --i;
    }
    // path parser guarantees idx >= 0
    // will never reach to here
    return false;
  }
  // case (START_ARRAY, Subscript :: Index(idx) :: xs)
  else if (json_token::START_ARRAY == token &&
           thrust::get<0>(path_match_subscript_index(path_ptr, path_size))) {
    int idx = thrust::get<1>(path_match_subscript_index(path_ptr, path_size));
    p.next_token();
    // JSON validation check
    if (json_token::ERROR == p.get_current_token()) { return false; }

    int i = idx;
    while (i >= 0) {
      if (p.get_current_token() == json_token::END_ARRAY) {
        // terminate, nothing has been written
        return false;
      }
      if (0 == i) {
        bool dirty = evaluate_path(p, g, true, style, path_ptr + 2, path_size - 2);
        while (p.next_token() != json_token::END_ARRAY) {
          // JSON validation check
          if (json_token::ERROR == p.get_current_token()) { return false; }

          // advance the token stream to the end of the array
          if (!p.try_skip_children()) { return false; }
        }
        return dirty;
      } else {
        // i > 0
        if (!p.try_skip_children()) { return false; }

        p.next_token();
        // JSON validation check
        if (json_token::ERROR == p.get_current_token()) { return false; }
      }
      --i;
    }
    // path parser guarantees idx >= 0
    // will never reach to here
    return false;
  }
  // case (FIELD_NAME, Named(name) :: xs) if p.getCurrentName == name
  else if (json_token::FIELD_NAME == token &&
           thrust::get<0>(path_match_named(path_ptr, path_size)) &&
           p.match_current_field_name(thrust::get<1>(path_match_named(path_ptr, path_size)))) {
    if (p.next_token() != json_token::VALUE_NULL) {
      // JSON validation check
      if (json_token::ERROR == p.get_current_token()) { return false; }

      return evaluate_path(p, g, true, style, path_ptr + 1, path_size - 1);
    } else {
      return false;
    }
  }
  // case (FIELD_NAME, Wildcard :: xs)
  else if (json_token::FIELD_NAME == token &&
           path_match_element(path_ptr, path_size, path_instruction_type::wildcard)) {
    p.next_token();
    // JSON validation check
    if (json_token::ERROR == p.get_current_token()) { return false; }

    return evaluate_path(p, g, true, style, path_ptr + 1, path_size - 1);
    // case _ =>
  } else {
    if (!p.try_skip_children()) { return false; }
    return false;
  }
}

/**
 * @brief Parse a single json string using the provided command buffer
 *
 * @param j_parser The incoming json string and associated parser
 * @param path_ptr The command buffer to be applied to the string.
 * @param path_size Command buffer size
 * @param output Buffer used to store the results of the query
 * @returns A result code indicating success/fail/empty.
 */
template <int max_json_nesting_depth = curr_max_json_nesting_depth>
__device__ parse_result parse_json_path(json_parser<max_json_nesting_depth>& j_parser,
                                        path_instruction const* path_ptr,
                                        size_t path_size,
                                        json_generator<max_json_nesting_depth>& output)
{
  j_parser.next_token();
  // JSON validation check
  if (json_token::ERROR == j_parser.get_current_token()) { return parse_result::ERROR; }

  auto matched = evaluate_path(j_parser, output, true, write_style::raw_style, path_ptr, path_size);
  return matched ? parse_result::SUCCESS : parse_result::ERROR;
}

/**
 * @brief Parse a single json string using the provided command buffer
 *
 * This function exists primarily as a shim for debugging purposes.
 *
 * @param input The incoming json string
 * @param input_len Size of the incoming json string
 * @param commands The command buffer to be applied to the string. Always ends
 * with a path_operator_type::END
 * @param out_buf Buffer user to store the results of the query (nullptr in the
 * size computation step)
 * @param out_buf_size Size of the output buffer
 * @param options Options controlling behavior
 * @returns A pair containing the result code the output buffer.
 */
template <int max_json_nesting_depth = curr_max_json_nesting_depth>
__device__ thrust::pair<parse_result, json_generator<max_json_nesting_depth>>
get_json_object_single(
  char const* input,
  cudf::size_type input_len,
  path_instruction const* path_commands_ptr,
  int path_commands_size,
  char* out_buf,
  size_t out_buf_size,
  json_parser_options options)  // TODO make this a reference? use a global singleton options?
                                // reduce the copy contructor overhead
{
  json_parser j_parser(options, input, input_len);
  json_generator generator(out_buf, out_buf_size);
  auto const result = parse_json_path(j_parser, path_commands_ptr, path_commands_size, generator);
  return {result, generator};
}

/**
 * @brief Kernel for running the JSONPath query.
 *
 * This kernel operates in a 2-pass way.  On the first pass, it computes
 * output sizes.  On the second pass it fills in the provided output buffers
 * (chars and validity)
 *
 * @param col Device view of the incoming string
 * @param commands JSONPath command buffer
 * @param output_offsets Buffer used to store the string offsets for the results
 * of the query
 * @param out_buf Buffer used to store the results of the query
 * @param out_validity Output validity buffer
 * @param out_valid_count Output count of # of valid bits
 * @param options Options controlling behavior
 */
template <int block_size>
__launch_bounds__(block_size) CUDF_KERNEL
  void get_json_object_kernel(cudf::column_device_view col,
                              path_instruction const* path_commands_ptr,
                              int path_commands_size,
                              cudf::size_type* d_sizes,
                              cudf::detail::input_offsetalator output_offsets,
                              thrust::optional<char*> out_buf,
                              thrust::optional<cudf::bitmask_type*> out_validity,
                              thrust::optional<cudf::size_type*> out_valid_count,
                              json_parser_options options)
{
  auto tid          = cudf::detail::grid_1d::global_thread_id();
  auto const stride = cudf::thread_index_type{blockDim.x} * cudf::thread_index_type{gridDim.x};

  cudf::size_type warp_valid_count{0};

  auto active_threads = __ballot_sync(0xffff'ffffu, tid < col.size());
  while (tid < col.size()) {
    bool is_valid               = false;
    cudf::string_view const str = col.element<cudf::string_view>(tid);
    cudf::size_type output_size = 0;
    if (str.size_bytes() > 0) {
      char* dst = out_buf.has_value() ? out_buf.value() + output_offsets[tid] : nullptr;
      size_t const dst_size =
        out_buf.has_value() ? output_offsets[tid + 1] - output_offsets[tid] : 0;

      // process one single row
      auto [result, out] = get_json_object_single(str.data(),
                                                  str.size_bytes(),
                                                  path_commands_ptr,
                                                  path_commands_size,
                                                  dst,
                                                  dst_size,
                                                  options);
      output_size        = out.get_output_len();
      if (result == parse_result::SUCCESS) { is_valid = true; }
    }

    // filled in only during the precompute step. during the compute step, the
    // offsets are fed back in so we do -not- want to write them out
    if (!out_buf.has_value()) { d_sizes[tid] = output_size; }

    // validity filled in only during the output step
    if (out_validity.has_value()) {
      uint32_t mask = __ballot_sync(active_threads, is_valid);
      // 0th lane of the warp writes the validity
      if (!(tid % cudf::detail::warp_size)) {
        out_validity.value()[cudf::word_index(tid)] = mask;
        warp_valid_count += __popc(mask);
      }
    }

    tid += stride;
    active_threads = __ballot_sync(active_threads, tid < col.size());
  }

  // sum the valid counts across the whole block
  if (out_valid_count) {
    cudf::size_type block_valid_count =
      cudf::detail::single_lane_block_sum_reduce<block_size, 0>(warp_valid_count);
    if (threadIdx.x == 0) { atomicAdd(out_valid_count.value(), block_valid_count); }
  }
}

std::unique_ptr<cudf::column> get_json_object(cudf::strings_column_view const& col,
                                              cudf::string_scalar const& json_path,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr)
{
  if (col.is_empty()) return cudf::make_empty_column(cudf::type_id::STRING);

  // parse the json_path into a command buffer
  auto path_commands_optional = parse_path(json_path);

  // if the json path is empty, return a string column containing all nulls
  if (!path_commands_optional.has_value()) {
    return std::make_unique<cudf::column>(
      cudf::data_type{cudf::type_id::STRING},
      col.size(),
      // no data
      rmm::device_buffer{0, stream, mr},
      cudf::detail::create_null_mask(col.size(), cudf::mask_state::ALL_NULL, stream, mr),
      // null count
      col.size());
  }

  // compute output sizes
  auto sizes = rmm::device_uvector<cudf::size_type>(
    col.size(), stream, rmm::mr::get_current_device_resource());
  auto d_offsets = cudf::detail::offsetalator_factory::make_input_iterator(col.offsets());

  constexpr int block_size = 512;
  cudf::detail::grid_1d const grid{col.size(), block_size};
  auto cdv = cudf::column_device_view::create(col.parent(), stream);

  // create json parser options
  spark_rapids_jni::json_parser_options options;
  options.set_allow_single_quotes(true);
  options.set_allow_unescaped_control_chars(true);
  options.set_max_string_len(true);
  options.set_max_num_len(true);
  options.set_allow_tailing_sub_string(true);

  // preprocess sizes (returned in the offsets buffer)
  get_json_object_kernel<block_size>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      *cdv,
      path_commands_optional.value().data(),
      path_commands_optional.value().size(),
      sizes.data(),
      d_offsets,
      thrust::nullopt,
      thrust::nullopt,
      thrust::nullopt,
      options);

  // convert sizes to offsets
  auto [offsets, output_size] =
    cudf::strings::detail::make_offsets_child_column(sizes.begin(), sizes.end(), stream, mr);
  d_offsets = cudf::detail::offsetalator_factory::make_input_iterator(offsets->view());

  // allocate output string column
  rmm::device_uvector<char> chars(output_size, stream, mr);

  // potential optimization : if we know that all outputs are valid, we could
  // skip creating the validity mask altogether
  rmm::device_buffer validity =
    cudf::detail::create_null_mask(col.size(), cudf::mask_state::UNINITIALIZED, stream, mr);

  // compute results
  rmm::device_scalar<cudf::size_type> d_valid_count{0, stream};

  get_json_object_kernel<block_size>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      *cdv,
      path_commands_optional.value().data(),
      path_commands_optional.value().size(),
      sizes.data(),
      d_offsets,
      chars.data(),
      static_cast<cudf::bitmask_type*>(validity.data()),
      d_valid_count.data(),
      options);

  auto result = make_strings_column(col.size(),
                                    std::move(offsets),
                                    chars.release(),
                                    col.size() - d_valid_count.value(stream),
                                    std::move(validity));
  // unmatched array query may result in unsanitized '[' value in the result
  if (cudf::detail::has_nonempty_nulls(result->view(), stream)) {
    result = cudf::detail::purge_nonempty_nulls(result->view(), stream, mr);
  }
  return result;
}

}  // namespace detail

std::unique_ptr<cudf::column> get_json_object(cudf::strings_column_view const& col,
                                              cudf::string_scalar const& json_path,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr)
{
  // TODO: here do not know if json path is invalid, should handle it in Plugin
  return detail::get_json_object(col, json_path, stream, mr);
}

}  // namespace spark_rapids_jni

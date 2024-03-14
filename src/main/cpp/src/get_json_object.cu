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
#include <cudf/lists/list_device_view.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/optional.h>
#include <thrust/pair.h>
#include <thrust/scan.h>
#include <thrust/tuple.h>

namespace spark_rapids_jni {

namespace detail {
// namespace {

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
  int64_t index{-1};
};

rmm::device_uvector<path_instruction> construct_path_commands(cudf::table_view instructions,
                                                              rmm::cuda_stream_view stream,
                                                              rmm::mr::device_memory_resource* mr)
{
  auto const ins_types   = instructions.column(0);
  auto const ins_names   = instructions.column(1);
  auto const ins_indexes = instructions.column(2);

  auto const s_ins_types = cudf::strings_column_view(ins_types);
  auto const s_ins_names = cudf::strings_column_view(ins_names);

  auto const d_ins_types   = cudf::column_device_view::create(s_ins_types.parent(), stream);
  auto const d_ins_names   = cudf::column_device_view::create(s_ins_names.parent(), stream);
  auto const d_ins_indexes = cudf::column_device_view::create(ins_indexes, stream);

  rmm::device_uvector<path_instruction> path_commands(instructions.num_rows(), stream, mr);

  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(instructions.num_rows()),
                    path_commands.begin(),
                    [d_types   = *d_ins_types,
                     d_names   = *d_ins_names,
                     d_indexes = *d_ins_indexes] __device__(auto idx) {
                      path_instruction instruction(path_instruction_type::named);
                      auto const type_str = d_types.element<cudf::string_view>(idx);
                      if (type_str.data() == "subscript") {
                        instruction.type = path_instruction_type::subscript;
                      } else if (type_str.data() == "wildcard") {
                        instruction.type = path_instruction_type::wildcard;
                      } else if (type_str.data() == "key") {
                        instruction.type = path_instruction_type::key;
                      } else if (type_str.data() == "index") {
                        instruction.type  = path_instruction_type::index;
                        instruction.index = d_indexes.element<int64_t>(idx);
                      } else if (type_str.data() == "named") {
                        instruction.type = path_instruction_type::named;
                        instruction.name = d_names.element<cudf::string_view>(idx);
                      }
                      return instruction;
                    });

  return path_commands;
}

/**
 * TODO: JSON generator
 *
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

  CUDF_HOST_DEVICE json_generator finish_child_generator(json_generator const& child_generator)
  {
    // logically delete child generator
    output_len += child_generator.get_output_len();
  }

  CUDF_HOST_DEVICE void write_start_array()
  {
    // TODO
  }

  CUDF_HOST_DEVICE void write_end_array()
  {
    // TODO
  }

  CUDF_HOST_DEVICE void copy_current_structure(json_parser<max_json_nesting_depth>& parser)
  {
    // TODO
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
    if (output) {
      auto copied = parser.try_copy_raw_text(output + output_len);
      output_len += copied;
    }
  }

  CUDF_HOST_DEVICE inline size_t get_output_len() const { return output_len; }

 private:
  char const* const output;
  size_t output_len;
};

/**
 * @brief Result of calling a parse function.
 *
 * The primary use of this is to distinguish between "success" and
 * "success but no data" return cases.  For example, if you are reading the
 * values of an array you might call a parse function in a while loop. You
 * would want to continue doing this until you either encounter an error
 * (parse_result::ERROR) or you get nothing back (parse_result::EMPTY)
 */
enum class parse_result {
  ERROR,          // failure
  SUCCESS,        // success
  MISSING_FIELD,  // success, but the field is missing
  EMPTY,          // success, but no data
};

/**
 * @brief Parse a single json string using the provided command buffer
 *
 * @param j_state The incoming json string and associated parser
 * @param commands The command buffer to be applied to the string. Always ends
 * with a path_operator_type::END
 * @param output Buffer user to store the results of the query
 * @returns A result code indicating success/fail/empty.
 */
template <int max_json_nesting_depth = curr_max_json_nesting_depth>
__device__ parse_result parse_json_path(json_parser<max_json_nesting_depth>& j_parser,
                                        cudf::device_span<path_instruction const> path_commands,
                                        json_generator<max_json_nesting_depth>& output)
{
  // TODO
  return parse_result::SUCCESS;
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
get_json_object_single(char const* input,
                       cudf::size_type input_len,
                       cudf::device_span<path_instruction const> path_commands,
                       char* out_buf,
                       size_t out_buf_size,
                       json_parser_options options)
{
  json_parser j_parser(options, input, input_len);
  json_generator generator(out_buf, out_buf_size);
  auto const result = parse_json_path(j_parser, path_commands, generator);
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
                              cudf::device_span<path_instruction const> path_commands,
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
      auto [result, out] =
        get_json_object_single(str.data(), str.size_bytes(), path_commands, dst, dst_size, options);
      output_size = out.get_output_len();
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
                                              cudf::table_view const& instructions,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr)
{
  if (col.is_empty()) return cudf::make_empty_column(cudf::type_id::STRING);

  // parse the json_path into a command buffer
  auto path_commands = construct_path_commands(instructions, stream, mr);

  auto options = json_parser_options{};

  // compute output sizes
  auto sizes = rmm::device_uvector<cudf::size_type>(
    col.size(), stream, rmm::mr::get_current_device_resource());
  auto d_offsets = cudf::detail::offsetalator_factory::make_input_iterator(col.offsets());

  constexpr int block_size = 512;
  cudf::detail::grid_1d const grid{col.size(), block_size};
  auto cdv = cudf::column_device_view::create(col.parent(), stream);
  // preprocess sizes (returned in the offsets buffer)
  get_json_object_kernel<block_size>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      *cdv,
      cudf::device_span<path_instruction const>{path_commands.data(), path_commands.size()},
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
      cudf::device_span<path_instruction const>{path_commands.data(), path_commands.size()},
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

// }  // namespace

}  // namespace detail

std::unique_ptr<cudf::column> get_json_object(cudf::strings_column_view const& col,
                                              cudf::table_view const& instructions,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr)
{
  // TODO: main logic
  return detail::get_json_object(col, instructions, stream, mr);
}

}  // namespace spark_rapids_jni

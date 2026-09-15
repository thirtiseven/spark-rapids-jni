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

#include "protobuf/protobuf.hpp"
#include "protobuf/protobuf_host_helpers.hpp"
#include "protobuf/protobuf_kernels.cuh"

#include <cudf_test/base_fixture.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

class ProtobufHelpersTest : public cudf::test::BaseFixture {};

namespace {

namespace protobuf = spark_rapids_jni::protobuf;

protobuf::protobuf_decode_context make_numeric_enum_context(int64_t default_value)
{
  auto const stream = cudf::get_default_stream();

  std::vector<cudf::detail::host_vector<uint8_t>> default_strings;
  default_strings.emplace_back(cudf::detail::make_pinned_vector_async<uint8_t>(0, stream));

  std::vector<cudf::detail::host_vector<int32_t>> enum_valid_values;
  auto& values =
    enum_valid_values.emplace_back(cudf::detail::make_pinned_vector_async<int32_t>(3, stream));
  std::iota(values.begin(), values.end(), 0);

  std::vector<std::vector<cudf::detail::host_vector<uint8_t>>> enum_names(1);
  return {{{.field_number      = 1,
            .parent_idx        = -1,
            .wire_type         = protobuf::proto_wire_type::VARINT,
            .output_type       = cudf::type_id::INT32,
            .encoding          = protobuf::proto_encoding::DEFAULT,
            .has_default_value = true}},
          {default_value},
          {0.0},
          {false},
          std::move(default_strings),
          std::move(enum_valid_values),
          std::move(enum_names),
          true};
}

}  // namespace

TEST_F(ProtobufHelpersTest, NumericEnumDefaultMustFitInt32)
{
  EXPECT_NO_THROW(make_numeric_enum_context(2));
  EXPECT_THROW(make_numeric_enum_context(int64_t{1} << 42), std::invalid_argument);
}

TEST_F(ProtobufHelpersTest, NullMaskFromPaddedValidUsesZeroLogicalRows)
{
  cuda::stream_ref stream = cudf::get_default_stream();

  std::array<bool, 1> h_valid{false};
  rmm::device_uvector<bool> valid(h_valid.size(), stream);
  CUDF_CUDA_TRY(cudaMemcpyAsync(valid.data(),
                                h_valid.data(),
                                h_valid.size() * sizeof(h_valid[0]),
                                cudaMemcpyDefault,
                                stream.get()));

  auto [mask, null_count] = spark_rapids_jni::protobuf::detail::make_null_mask_from_valid(
    valid, 0, stream, cudf::get_current_device_resource_ref());

  EXPECT_EQ(0u, mask.size());
  EXPECT_EQ(nullptr, mask.data());
  EXPECT_EQ(0, null_count);
}

TEST_F(ProtobufHelpersTest, NullMaskFromPaddedValidIgnoresTail)
{
  cuda::stream_ref stream = cudf::get_default_stream();

  std::array<bool, 3> h_valid{true, false, false};
  rmm::device_uvector<bool> valid(h_valid.size(), stream);
  CUDF_CUDA_TRY(cudaMemcpyAsync(valid.data(),
                                h_valid.data(),
                                h_valid.size() * sizeof(h_valid[0]),
                                cudaMemcpyDefault,
                                stream.get()));

  auto [mask, null_count] = spark_rapids_jni::protobuf::detail::make_null_mask_from_valid(
    valid, 2, stream, cudf::get_current_device_resource_ref());

  EXPECT_EQ(cudf::bitmask_allocation_size_bytes(2), mask.size());
  EXPECT_EQ(1, null_count);

  std::vector<cudf::bitmask_type> h_mask(mask.size() / sizeof(cudf::bitmask_type));
  CUDF_CUDA_TRY(
    cudaMemcpyAsync(h_mask.data(), mask.data(), mask.size(), cudaMemcpyDefault, stream.get()));
  stream.sync();

  EXPECT_TRUE(cudf::bit_is_set(h_mask.data(), 0));
  EXPECT_FALSE(cudf::bit_is_set(h_mask.data(), 1));
}

TEST_F(ProtobufHelpersTest, NullMaskFromAllValidRowsIsEmpty)
{
  cuda::stream_ref stream = cudf::get_default_stream();

  std::array<bool, 2> h_valid{true, true};
  rmm::device_uvector<bool> valid(h_valid.size(), stream);
  CUDF_CUDA_TRY(cudaMemcpyAsync(valid.data(),
                                h_valid.data(),
                                h_valid.size() * sizeof(h_valid[0]),
                                cudaMemcpyDefault,
                                stream.get()));

  auto [mask, null_count] = spark_rapids_jni::protobuf::detail::make_null_mask_from_valid(
    valid, h_valid.size(), stream, cudf::get_current_device_resource_ref());

  EXPECT_EQ(0u, mask.size());
  EXPECT_EQ(nullptr, mask.data());
  EXPECT_EQ(0, null_count);
}

TEST_F(ProtobufHelpersTest, UnsignedFragmentLengthsRetainSignedOffsetLimit)
{
  auto const stream     = cudf::get_default_stream();
  auto const mr         = cudf::get_current_device_resource_ref();
  auto const max_length = static_cast<uint32_t>(std::numeric_limits<int32_t>::max());
  auto make_offsets     = [&](uint32_t length, int count) {
    return spark_rapids_jni::protobuf::detail::make_list_offsets_from_counts(
      cuda::make_constant_iterator(length), count, "Merged singular message", stream, mr, mr);
  };

  EXPECT_EQ(make_offsets(0, 1).total_count, 0);
  EXPECT_EQ(make_offsets(max_length, 1).total_count, std::numeric_limits<int32_t>::max());

  std::array<uint32_t, 3> const lengths{0, 1, max_length - 1};
  auto const d_lengths = cudf::detail::make_device_uvector_async(lengths, stream, mr);
  auto const offsets   = spark_rapids_jni::protobuf::detail::make_list_offsets_from_counts(
    d_lengths.begin(), lengths.size(), "Merged singular message", stream, mr, mr);
  EXPECT_EQ(offsets.total_count, std::numeric_limits<int32_t>::max());
  std::vector<int32_t> const expected_offsets{0, 0, 1, std::numeric_limits<int32_t>::max()};
  EXPECT_EQ(cudf::detail::make_std_vector(offsets.offsets, stream), expected_offsets);

  EXPECT_THROW(make_offsets(max_length + 1, 1), cudf::logic_error);
  EXPECT_THROW(make_offsets(max_length, 2), cudf::logic_error);
  EXPECT_THROW(make_offsets(std::numeric_limits<uint32_t>::max(), 1), cudf::logic_error);
}

namespace {

namespace protobuf_detail = spark_rapids_jni::protobuf::detail;

CUDF_KERNEL void resolve_test_locations(protobuf_detail::field_location* output,
                                        protobuf_detail::protobuf_error* error)
{
  using protobuf_detail::field_location;
  cudf::size_type const row_offsets[]{100, 120};
  field_location const children[]{{2, 3}, {4, 0}, field_location::missing()};
  field_location const parents[]{{5, 10}};
  field_location const missing_parents[]{field_location::missing()};
  protobuf_detail::top_level_location_provider top{row_offsets, 100, children, 0, 3};
  output[0]       = top.input_location(0);
  top.field_idx   = 1;
  output[1]       = top.input_location(0);
  top.field_idx   = 2;
  output[2]       = top.input_location(0);
  top.field_idx   = 0;
  top.base_offset = 101;
  output[3]       = top.input_location(0);

  protobuf_detail::nested_location_provider nested{row_offsets, 90, parents, children, 0, 3};
  output[4]               = nested.row_location(0);
  output[5]               = nested.input_location(0);
  nested.parent_locations = missing_parents;
  output[6]               = nested.input_location(0);

  protobuf_detail::field_occurrence const occurrences[]{{0, 2, 3}};
  protobuf_detail::field_occurrence_location_provider repeated{
    {nullptr, 30, row_offsets, 90, 1}, {parents, 1, nullptr}, occurrences};
  output[7]                 = repeated.input_location(0);
  repeated.parent.locations = nullptr;
  output[8]                 = repeated.input_location(0);
  repeated.parent.locations = missing_parents;
  output[9]                 = repeated.input_location(0);

  auto const max_offset = static_cast<uint32_t>(cuda::std::numeric_limits<int32_t>::max());
  output[10]            = protobuf_detail::rebase_location({max_offset, 0}, 0);
  output[11]            = protobuf_detail::rebase_location({max_offset, 0}, 1, error);
}

}  // namespace

TEST_F(ProtobufHelpersTest, LocationProvidersResolveCoordinatesAndPresence)
{
  using protobuf_detail::field_location;
  std::array<field_location, 12> const expected{
    {{2, 3},
     {4, 0},
     field_location::missing(),
     field_location::missing(),
     {7, 3},
     {17, 3},
     field_location::missing(),
     {17, 3},
     {12, 3},
     field_location::missing(),
     {static_cast<uint32_t>(std::numeric_limits<int32_t>::max()), 0},
     field_location::missing()}};
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::get_current_device_resource_ref();
  rmm::device_uvector<field_location> output(expected.size(), stream, mr);
  auto error =
    cudf::detail::make_zeroed_device_uvector_async<protobuf_detail::protobuf_error>(1, stream, mr);
  resolve_test_locations<<<1, 1, 0, stream.get()>>>(output.data(), error.data());
  CUDF_CHECK_CUDA(stream.get());
  auto const actual = cudf::detail::make_std_vector(output, stream);
  for (size_t i = 0; i < expected.size(); ++i) {
    SCOPED_TRACE(i);
    EXPECT_EQ(actual[i].offset, expected[i].offset);
    EXPECT_EQ(actual[i].length, expected[i].length);
    EXPECT_EQ(actual[i].is_present(), expected[i].is_present());
  }
  EXPECT_EQ(cudf::detail::make_std_vector(error, stream)[0],
            protobuf_detail::protobuf_error::OVERFLOW);
}

/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#include "ftos_converter.cuh"

#include <cudf/strings/detail/convert/string_to_float.cuh>
#include <cudf/strings/string_view.hpp>
#include <cudf/types.hpp>

#include <cuda/std/tuple>
#include <cuda/std/utility>

namespace spark_rapids_jni {

/**
 * write style when writing out JSON string
 */
enum class escape_style { UNESCAPED, ESCAPED };

// Changing this limit also requires changing context_stack.
constexpr int MAX_JSON_NESTING_DEPTH = 64;

/**
 * Define the maximum JSON number length. Negative or zero means no
 * limitation.
 *
 * By default, maximum JSON number length is negative one, means no
 * limitation.
 *
 * e.g.: The length of number -123.45e-67 is 7. if maximum JSON number length
 * is 6, then this number is a invalid number.
 */
constexpr int max_num_len = 1000;

enum class json_token : int8_t {
  INIT = 0,
  SUCCESS,
  ERROR,
  START_OBJECT,
  END_OBJECT,
  START_ARRAY,
  END_ARRAY,
  FIELD_NAME,
  VALUE_STRING,
  VALUE_NUMBER_INT,
  VALUE_NUMBER_FLOAT,
  VALUE_TRUE,
  VALUE_FALSE,
  VALUE_NULL
};

// A byte range stores its pointer and byte length without character traversal.
class char_range {
 public:
  __device__ inline char_range(char const* const start, cudf::size_type const len)
    : _data(start), _len(len)
  {
  }

  __device__ inline char_range(cudf::string_view const& input)
    : _data(input.data()), _len(input.size_bytes())
  {
  }

  // Keep explicit null() construction; CUDA miscompiles initializing a member from a static
  // method like this.
  __device__ inline static char_range null() { return char_range(nullptr, 0); }

  __device__ inline char_range(char_range const&)            = default;
  __device__ inline char_range(char_range&&)                 = default;
  __device__ inline char_range& operator=(char_range const&) = default;
  __device__ inline char_range& operator=(char_range&&)      = default;
  __device__ inline ~char_range()                            = default;

  __device__ inline cudf::size_type size() const { return _len; }
  __device__ inline char const* data() const { return _data; }
  __device__ inline bool is_null() const { return _data == nullptr; }
  __device__ inline bool is_empty() const { return _len <= 0; }
  __device__ inline char operator[](cudf::size_type pos) const { return _data[pos]; }

  __device__ inline cudf::string_view slice_sv(cudf::size_type pos, cudf::size_type len) const
  {
    return cudf::string_view(_data + pos, len);
  }

  __device__ inline char_range slice(cudf::size_type pos, cudf::size_type len) const
  {
    return char_range(_data + pos, len);
  }

 protected:
  char const* _data;
  cudf::size_type _len;
};

// Tracks the unread suffix without a separate cursor.
class char_range_reader : public char_range {
 public:
  __device__ inline explicit char_range_reader(char_range range)
    : char_range(cuda::std::move(range))
  {
  }
  __device__ inline void next()
  {
    _data++;
    _len--;
  }

  // Caller must check for an empty range before reading.
  __device__ inline char current_char() const { return _data[0]; }
};

// Spark/Jackson parsing permits single quotes and raw controls in strings, but not leading zeros.
// Whitespace remains limited to space, tab, CR and LF.
//
// JSON format: https://www.json.org/json-en.html
//
// Valid numbers:   0, 102, -0, -102, 0.3, -0.3, 1e-5, 1E+5, 1e0, 1E0, 1.3e5, 1e01
//                  (1e01: leading zero after 'e' is allowed)
// Invalid numbers: 00, -00, infinity, +infinity, -infinity, 1e, 1e+, 1e-, -1., 1.
// Valid strings:   "\'" , "\"" , '\'' , '\"' , '"' , "'"
//                  plus ASCII control chars in [0, 32) inside strings
class json_parser {
 public:
  __device__ inline explicit json_parser(char_range _chars)
    : chars(_chars), curr_pos(0), current_token(json_token::INIT), max_depth_exceeded(false)
  {
  }

 private:
  static __device__ inline bool get_bit_value(int64_t number, int bitIndex)
  {
    long shifted = number >> bitIndex;

    bool bit_value = shifted & 1;

    return bit_value;
  }

  static __device__ inline void set_bit_value(int64_t& number, int bit_index, bool bit_value)
  {
    long mask = 1L << bit_index;

    if (bit_value) {
      number |= mask;
    } else {
      number &= ~mask;
    }
  }

  __device__ inline bool eof(cudf::size_type pos) const { return pos >= chars.size(); }
  __device__ inline bool eof() const { return curr_pos >= chars.size(); }

  static __device__ inline bool is_hex_digit(char c)
  {
    return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F') || (c >= 'a' && c <= 'f');
  }

  static __device__ inline bool is_digit(char c) { return (c >= '0' && c <= '9'); }

  static __device__ inline bool is_whitespace(char c)
  {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
  }

  __device__ inline void skip_whitespaces()
  {
    while (!eof() && is_whitespace(chars[curr_pos])) {
      curr_pos++;
    }
  }

  static __device__ inline bool try_skip(char_range_reader& reader, char expected)
  {
    if (!reader.is_empty() && reader.current_char() == expected) {
      reader.next();
      return true;
    }
    return false;
  }

  __device__ inline bool try_skip(cudf::size_type& pos, char expected) const
  {
    if (!eof(pos) && chars[pos] == expected) {
      pos++;
      return true;
    }
    return false;
  }

  // A failed push marks depth exceeded at the caller.
  __device__ inline bool try_push_context(json_token token)
  {
    if (stack_size < MAX_JSON_NESTING_DEPTH) {
      push_context(token);
      return true;
    } else {
      return false;
    }
  }

  // Context bit: object = 1, array = 0.
  __device__ inline void push_context(json_token token)
  {
    bool v = json_token::START_OBJECT == token ? true : false;
    set_bit_value(context_stack, stack_size, v);
    stack_size++;
  }

  __device__ inline bool is_object_context() const
  {
    return get_bit_value(context_stack, stack_size - 1);
  }

  __device__ inline void pop_curr_context() { stack_size--; }

  __device__ inline bool is_context_stack_empty() const { return stack_size == 0; }

  __device__ inline void set_current_error() { current_token = json_token::ERROR; }

  __device__ inline void parse_first_token_in_value_and_set_current()
  {
    if (eof()) {
      set_current_error();
      return;
    }
    current_token_start_pos = curr_pos;
    char c                  = chars[curr_pos];
    switch (c) {
      case '{':
        if (!try_push_context(json_token::START_OBJECT)) {
          max_depth_exceeded = true;
          set_current_error();
        } else {
          curr_pos++;
          current_token = json_token::START_OBJECT;
        }
        break;
      case '[':
        if (!try_push_context(json_token::START_ARRAY)) {
          max_depth_exceeded = true;
          set_current_error();
        } else {
          curr_pos++;
          current_token = json_token::START_ARRAY;
        }
        break;
      case '"':
        // fall through
      case '\'': parse_string_and_set_current(); break;
      case 't':
        curr_pos++;
        parse_true_and_set_current();
        break;
      case 'f':
        curr_pos++;
        parse_false_and_set_current();
        break;
      case 'n':
        curr_pos++;
        parse_null_and_set_current();
        break;
      default: parse_number_and_set_current(); break;
    }
  }

  __device__ inline void parse_string_and_set_current()
  {
    [[maybe_unused]] auto const [success, matched, end] =
      try_parse_string(char_range_reader{chars.slice(curr_pos, chars.size() - curr_pos)});
    if (success) {
      curr_pos      = static_cast<cudf::size_type>(cuda::std::distance(chars.data(), end));
      current_token = json_token::VALUE_STRING;
    } else {
      set_current_error();
    }
  }

  /**
   * transform int value from [0, 15] to hex char
   */
  static __device__ inline char to_hex_char(unsigned int v)
  {
    if (v < 10)
      return '0' + v;
    else
      return 'A' + (v - 10);
  }

  /**
   * escape control char ( ASCII code value [0, 32) )
   * e.g.: \0  (ASCII code 0) will be escaped to 6 chars: \u0000
   * e.g.: \10 (ASCII code 0) will be escaped to 2 chars: \n
   * @param char to be escaped, c should in range [0, 31)
   * @param[out] escape output
   */
  static __device__ inline int escape_char(unsigned char c, char* output)
  {
    if (nullptr == output) {
      switch (c) {
        case 8:             // \b
        case 9:             // \t
        case 10:            // \n
        case 12:            // \f
        case 13: return 2;  // \r
        default: return 6;  // \u0000
      }
    }
    switch (c) {
      case 8:
        output[0] = '\\';
        output[1] = 'b';
        return 2;
      case 9:
        output[0] = '\\';
        output[1] = 't';
        return 2;
      case 10:
        output[0] = '\\';
        output[1] = 'n';
        return 2;
      case 12:
        output[0] = '\\';
        output[1] = 'f';
        return 2;
      case 13:
        output[0] = '\\';
        output[1] = 'r';
        return 2;
      default:
        output[0] = '\\';
        output[1] = 'u';
        output[2] = '0';
        output[3] = '0';

        // write high digit
        if (c >= 16) {
          output[4] = '1';
        } else {
          output[4] = '0';
        }

        // write low digit
        unsigned int v = c % 16;
        output[5]      = to_hex_char(v);
        return 6;
    }
  }

  static __device__ inline int write_string(char_range_reader& str,
                                            char* copy_destination,
                                            escape_style w_style)
  {
    if (str.is_empty()) { return 0; }
    char const quote_char = str.current_char();
    int output_size_bytes = 0;

    if (escape_style::ESCAPED == w_style) {
      output_size_bytes++;
      if (nullptr != copy_destination) { *copy_destination++ = '"'; }
    }

    str.next();

    while (!str.is_empty()) {
      char const c = str.current_char();
      int const v  = static_cast<int>(c);
      if (c == quote_char) {
        str.next();

        if (escape_style::ESCAPED == w_style) {
          output_size_bytes++;
          if (nullptr != copy_destination) { *copy_destination++ = '"'; }
        }

        return output_size_bytes;
      } else if (v >= 0 && v < 32) {
        if (escape_style::UNESCAPED == w_style) {
          output_size_bytes++;
          if (copy_destination != nullptr) { *copy_destination++ = str.current_char(); }
        } else {
          int const escape_chars = escape_char(str.current_char(), copy_destination);
          if (copy_destination != nullptr) { copy_destination += escape_chars; }
          output_size_bytes += escape_chars;
        }

        str.next();
      } else if ('\\' == c) {
        str.next();
        char_range_reader to_match(char_range::null());  // unused
        bool matched_field_name{false};                  // unused
        if (!try_skip_escape_part(
              str, to_match, copy_destination, w_style, output_size_bytes, matched_field_name)) {
          return output_size_bytes;
        }
      } else {
        // A plain quote inside single quotes needs escaping in ESCAPED style.
        if ('\"' == c && escape_style::ESCAPED == w_style) {
          if (copy_destination != nullptr) { *copy_destination++ = '\\'; }
          output_size_bytes++;
        }

        if (copy_destination != nullptr) { *copy_destination++ = c; }
        str.next();
        output_size_bytes++;
      }
    }

    return output_size_bytes;
  }

  // Parse and optionally match decoded string bytes; return the past-end raw cursor.
  // Utility only: no parser state update; quote_char can be ' or ".
  //
  // UTF-8 layout: 1-byte char has MSB 0; 2-byte: leading 110, then 10; 3-byte: leading 1110,
  // then 10, 10; 4-byte: leading 11110, then 10, 10, 10. JSON structural chars ([ ] { } , :),
  // quote chars (" ') and '\' are all ASCII (MSB 0), so the byte array needs no UTF-8 conversion.
  //
  // When quote is " the grammar is:
  //
  //   STRING
  //     : '"' (ESC | SAFECODEPOINT)* '"'
  //     ;
  //
  //   fragment ESC     : '\\' (["\\/bfnrt] | UNICODE)
  //   fragment UNICODE : 'u' HEX HEX HEX HEX
  //   fragment HEX     : [0-9a-fA-F]
  //
  //   fragment SAFECODEPOINT : see try_skip_safe_code_point
  static __device__ inline cuda::std::tuple<bool, bool, char const*> try_parse_string(
    char_range_reader str, char_range_reader to_match = char_range_reader(char_range::null()))
  {
    if (str.is_empty()) { return cuda::std::make_tuple(false, false, nullptr); }
    char const quote_char   = str.current_char();
    bool matched_field_name = !to_match.is_null();

    // skip left quote char
    // We don't need to actually verify what it is, because we just read it.
    str.next();

    // scan string content
    while (!str.is_empty()) {
      char c = str.current_char();
      int v  = static_cast<int>(c);
      if (c == quote_char) {  // path 1: match closing quote char
        str.next();
        matched_field_name = matched_field_name && (to_match.is_null() || to_match.is_empty());
        return cuda::std::make_tuple(true, matched_field_name, str.data());
      } else if (v >= 0 && v < 32) {  // path 2: unescaped control char
        matched_field_name = matched_field_name && try_match_char(to_match, c);
        str.next();
        continue;
      } else if ('\\' == c) {  // path 3: escape path
        str.next();

        char* copy_dest_nullptr = nullptr;  // unused
        int output_size_bytes   = 0;        // unused
        if (!try_skip_escape_part(str,
                                  to_match,
                                  copy_dest_nullptr,
                                  escape_style::UNESCAPED,
                                  output_size_bytes,
                                  matched_field_name)) {
          return cuda::std::make_tuple(false, false, nullptr);
        }
      } else {  // path 4: safe code point
        if (!try_skip_safe_code_point(str, c)) {
          return cuda::std::make_tuple(false, false, nullptr);
        }
        matched_field_name = matched_field_name && try_match_char(to_match, c);
      }
    }

    return cuda::std::make_tuple(false, false, nullptr);
  }

  static __device__ inline bool try_match_char(char_range_reader& reader, char c)
  {
    if (!reader.is_null()) {
      if (!reader.is_empty() && reader.current_char() == c) {
        reader.next();
        return true;
      } else {
        return false;
      }
    } else {
      return true;
    }
  }

  /**
   * skip the second char in \", \', \\, \/, \b, \f, \n, \r, \t;
   * skip the HEX chars in \u HEX HEX HEX HEX.
   * @return positive escaped ASCII value if success, -1 otherwise
   */
  static __device__ inline bool try_skip_escape_part(char_range_reader& str,
                                                     char_range_reader& to_match,
                                                     char*& copy_dest,
                                                     escape_style w_style,
                                                     int& output_size_bytes,
                                                     bool& matched_field_name)
  {
    // already skipped the first '\'
    // try skip second part
    if (!str.is_empty()) {
      char const c = str.current_char();
      switch (c) {
        // path 1: \", \', \\, \/, \b, \f, \n, \r, \t
        case '\"':
          if (nullptr != copy_dest && escape_style::UNESCAPED == w_style) { *copy_dest++ = c; }
          if (escape_style::ESCAPED == w_style) {
            if (copy_dest != nullptr) {
              *copy_dest++ = '\\';
              *copy_dest++ = '"';
            }
            output_size_bytes++;
          }
          output_size_bytes++;
          str.next();
          matched_field_name = matched_field_name && try_match_char(to_match, c);
          return true;
        case '\'':
          // for both unescaped/escaped writes a single char '
          if (nullptr != copy_dest) { *copy_dest++ = c; }

          output_size_bytes++;
          str.next();
          matched_field_name = matched_field_name && try_match_char(to_match, c);
          return true;
        case '\\':
          if (nullptr != copy_dest && escape_style::UNESCAPED == w_style) { *copy_dest++ = c; }
          if (escape_style::ESCAPED == w_style) {
            if (copy_dest != nullptr) {
              *copy_dest++ = '\\';
              *copy_dest++ = '\\';
            }
            output_size_bytes++;
          }
          output_size_bytes++;
          str.next();
          matched_field_name = matched_field_name && try_match_char(to_match, c);
          return true;
        case '/':
          // for both unescaped/escaped writes a single char /
          if (nullptr != copy_dest) { *copy_dest++ = c; }
          output_size_bytes++;
          str.next();
          matched_field_name = matched_field_name && try_match_char(to_match, c);
          return true;
        case 'b':
          if (nullptr != copy_dest && escape_style::UNESCAPED == w_style) { *copy_dest++ = '\b'; }
          if (escape_style::ESCAPED == w_style) {
            if (copy_dest != nullptr) {
              *copy_dest++ = '\\';
              *copy_dest++ = 'b';
            }
            output_size_bytes++;
          }
          output_size_bytes++;
          str.next();
          matched_field_name = matched_field_name && try_match_char(to_match, '\b');
          return true;
        case 'f':
          if (nullptr != copy_dest && escape_style::UNESCAPED == w_style) { *copy_dest++ = '\f'; }
          if (escape_style::ESCAPED == w_style) {
            if (copy_dest != nullptr) {
              *copy_dest++ = '\\';
              *copy_dest++ = 'f';
            }
            output_size_bytes++;
          }
          output_size_bytes++;
          str.next();
          matched_field_name = matched_field_name && try_match_char(to_match, '\f');
          return true;
        case 'n':
          if (nullptr != copy_dest && escape_style::UNESCAPED == w_style) { *copy_dest++ = '\n'; }
          if (escape_style::ESCAPED == w_style) {
            if (copy_dest != nullptr) {
              *copy_dest++ = '\\';
              *copy_dest++ = 'n';
            }
            output_size_bytes++;
          }
          output_size_bytes++;
          str.next();
          matched_field_name = matched_field_name && try_match_char(to_match, '\n');
          return true;
        case 'r':
          if (nullptr != copy_dest && escape_style::UNESCAPED == w_style) { *copy_dest++ = '\r'; }
          if (escape_style::ESCAPED == w_style) {
            if (copy_dest != nullptr) {
              *copy_dest++ = '\\';
              *copy_dest++ = 'r';
            }
            output_size_bytes++;
          }
          output_size_bytes++;
          str.next();
          matched_field_name = matched_field_name && try_match_char(to_match, '\r');
          return true;
        case 't':
          if (nullptr != copy_dest && escape_style::UNESCAPED == w_style) { *copy_dest++ = '\t'; }
          if (escape_style::ESCAPED == w_style) {
            if (copy_dest != nullptr) {
              *copy_dest++ = '\\';
              *copy_dest++ = 't';
            }
            output_size_bytes++;
          }
          output_size_bytes++;
          str.next();
          matched_field_name = matched_field_name && try_match_char(to_match, '\t');
          return true;
        // path 1 done: \", \', \\, \/, \b, \f, \n, \r, \t
        case 'u':
          // path 2: \u HEX HEX HEX HEX
          str.next();

          // for both unescaped/escaped writes corresponding utf8 bytes, no need
          // to pass in write style
          return try_skip_unicode(str, to_match, copy_dest, output_size_bytes, matched_field_name);
        default:
          // path 3: invalid
          return false;
      }
    } else {
      // eof, no escaped char after char '\'
      return false;
    }
  }

  // Caller already excluded quote/backslash; control bytes take another branch.
  // Implements the ANTLR fragment:
  //   SAFECODEPOINT : ~ ["\\\u0000-\u001F]
  //     1 not " or '     2 not \     3 non control character: Ascii value not in [0, 32)
  static __device__ inline bool try_skip_safe_code_point(char_range_reader& str, char c)
  {
    int v = static_cast<int>(c);
    if (!(v >= 0 && v < 32)) {
      str.next();
      return true;
    } else {
      return false;
    }
  }

  static __device__ inline uint8_t hex_value(char c)
  {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return 0;
  }

  // Count the nonzero bytes in the packed UTF-8 character.
  static __device__ cudf::size_type bytes_in_char_utf8(cudf::char_utf8 character)
  {
    return 1 + static_cast<cudf::size_type>((character & 0x0000'FF00u) > 0) +
           static_cast<cudf::size_type>((character & 0x00FF'0000u) > 0) +
           static_cast<cudf::size_type>((character & 0xFF00'0000u) > 0);
  }

  // Pack the code point into UTF-8 bytes.
  static __device__ cudf::char_utf8 codepoint_to_utf8(uint32_t unchr)
  {
    cudf::char_utf8 utf8 = 0;
    if (unchr < 0x0000'0080) {
      utf8 = unchr;
    } else if (unchr < 0x0000'0800) {
      utf8 = (unchr << 2) & 0x1F00;
      utf8 |= (unchr & 0x3F);
      utf8 |= 0x0000'C080;
    } else if (unchr < 0x0001'0000) {
      utf8 = (unchr << 4) & 0x0F'0000;
      utf8 |= (unchr << 2) & 0x00'3F00;
      utf8 |= (unchr & 0x3F);
      utf8 |= 0x00E0'8080;
    } else if (unchr < 0x0011'0000) {
      utf8 = (unchr << 6) & 0x0700'0000;
      utf8 |= (unchr << 4) & 0x003F'0000;
      utf8 |= (unchr << 2) & 0x0000'3F00;
      utf8 |= (unchr & 0x3F);
      utf8 |= 0xF080'8080u;
    }
    return utf8;
  }

  // Emit the packed UTF-8 bytes in source order.
  static __device__ cudf::size_type from_char_utf8(cudf::char_utf8 character, char* str)
  {
    cudf::size_type const chr_width = bytes_in_char_utf8(character);
    for (cudf::size_type idx = 0; idx < chr_width; ++idx) {
      str[chr_width - idx - 1] = static_cast<char>(character) & 0xFF;
      character                = character >> 8;
    }
    return chr_width;
  }

  // Pair only valid high/low surrogates; otherwise encode the initial code point alone.
  static __device__ bool try_skip_unicode(char_range_reader& str,
                                          char_range_reader& to_match,
                                          char*& copy_dest,
                                          int& output_size_bytes,
                                          bool& matched_field_name)
  {
    cudf::char_utf8 code_point = 0;
    for (size_t i = 0; i < 4; i++) {
      if (str.is_empty()) { return false; }
      char const c = str.current_char();
      str.next();
      if (!is_hex_digit(c)) { return false; }
      code_point = (code_point * 16) + hex_value(c);
    }

    if (code_point >= 0xD800 && code_point <= 0xDBFF) {
      if (str.size() >= 6) {
        if (str[0] == '\\' && str[1] == 'u' && is_hex_digit(str[2]) && is_hex_digit(str[3]) &&
            is_hex_digit(str[4]) && is_hex_digit(str[5])) {
          cudf::char_utf8 low_surrogate = 0;
          for (size_t i = 0; i < 4; i++) {
            char const c  = str[i + 2];
            low_surrogate = (low_surrogate * 16) + hex_value(c);
          }
          if (low_surrogate >= 0xDC00 && low_surrogate <= 0xDFFF) {
            code_point = 0x10000 + ((code_point - 0xD800) << 10) + (low_surrogate - 0xDC00);
            str.next();
            str.next();
            str.next();
            str.next();
            str.next();
            str.next();
          }
        }
      }
    }

    auto utf_char = codepoint_to_utf8(code_point);
    // UTF-8 encodes one code point in at most four bytes.
    char buff[4];
    cudf::size_type const bytes = from_char_utf8(utf_char, buff);
    output_size_bytes += bytes;

    // Unicode escapes retain decoded bytes even in ESCAPED style.
    if (nullptr != copy_dest) {
      for (cudf::size_type i = 0; i < bytes; i++) {
        *copy_dest++ = buff[i];
      }
    }

    if (matched_field_name && !to_match.is_null()) {
      for (cudf::size_type i = 0; i < bytes; i++) {
        if (to_match.is_empty() || to_match.current_char() != buff[i]) {
          matched_field_name = false;
          break;
        }
        to_match.next();
      }
    }

    return true;
  }

  // Number: -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?; Spark rejects leading zeros.
  // ANTLR grammar:
  //   NUMBER : '-'? INT ('.' [0-9]+)? EXP?
  //   INT    : '0' | [1-9] [0-9]*          // forbids leading 0s (e.g. `01`)
  //   EXP    : [Ee] [+\-]? [0-9]+
  // valid number:   0, 0.3, 0e005, 0E005
  // invalid number: 0., 0e, 0E
  __device__ inline void parse_number_and_set_current()
  {
    try_skip(curr_pos, '-');

    // number_digits_length counts digits only: e.g. +1.23e-45 has length 5
    bool is_float            = false;
    int number_digits_length = 0;
    if (try_unsigned_number(is_float, number_digits_length)) {
      if (check_max_num_len(number_digits_length)) {
        current_token = (is_float ? json_token::VALUE_NUMBER_FLOAT : json_token::VALUE_NUMBER_INT);
        number_token_len = curr_pos - current_token_start_pos;
      } else {
        set_current_error();
      }
    } else {
      set_current_error();
    }
  }

  // Only digit bytes count toward the configured maximum.
  static __device__ inline bool check_max_num_len(int number_digits_length)
  {
    return
      // disabled num len check
      max_num_len <= 0 ||
      // enabled num len check
      (max_num_len > 0 && number_digits_length <= max_num_len);
  }

  // An integer part starting with zero cannot have further digits.
  __device__ inline bool try_unsigned_number(bool& is_float, int& number_digits_length)
  {
    if (!eof()) {
      char const c = chars[curr_pos];
      if (c >= '1' && c <= '9') {
        curr_pos++;
        number_digits_length++;
        number_digits_length += skip_zero_or_more_digits();
        return parse_number_from_fraction(is_float, number_digits_length);
      } else if (c == '0') {
        curr_pos++;
        number_digits_length++;

        if (!eof()) {
          char const next_char_after_zero = chars[curr_pos];
          if (next_char_after_zero >= '0' && next_char_after_zero <= '9') {
            // e.g.: 01 is invalid
            return false;
          }
        }

        return parse_number_from_fraction(is_float, number_digits_length);
      } else {
        return false;
      }
    } else {
      return false;
    }
  }

  __device__ inline bool parse_number_from_fraction(bool& is_float, int& number_digits_length)
  {
    if (try_skip(curr_pos, '.')) {
      is_float = true;
      if (!try_skip_one_or_more_digits(number_digits_length)) { return false; }
    }

    if (!eof() && (chars[curr_pos] == 'e' || chars[curr_pos] == 'E')) {
      curr_pos++;
      is_float = true;
      return try_parse_exp(number_digits_length);
    }

    return true;
  }

  __device__ inline int skip_zero_or_more_digits()
  {
    int digits = 0;
    while (!eof()) {
      if (is_digit(chars[curr_pos])) {
        digits++;
        curr_pos++;
      } else {
        break;
      }
    }
    return digits;
  }

  __device__ inline bool try_skip_one_or_more_digits(int& number_digits_length)
  {
    if (!eof() && is_digit(chars[curr_pos])) {
      curr_pos++;
      number_digits_length++;
      number_digits_length += skip_zero_or_more_digits();
      return true;
    } else {
      return false;
    }
  }

  __device__ inline bool try_parse_exp(int& number_digits_length)
  {
    if (!eof() && (chars[curr_pos] == '+' || chars[curr_pos] == '-')) { curr_pos++; }

    return try_skip_one_or_more_digits(number_digits_length);
  }

  // =========== Parse number end ===========

  /**
   * parse true
   */
  __device__ inline void parse_true_and_set_current()
  {
    // already parsed 't'
    if (try_skip(curr_pos, 'r') && try_skip(curr_pos, 'u') && try_skip(curr_pos, 'e')) {
      current_token = json_token::VALUE_TRUE;
    } else {
      set_current_error();
    }
  }

  /**
   * parse false
   */
  __device__ inline void parse_false_and_set_current()
  {
    // already parsed 'f'
    if (try_skip(curr_pos, 'a') && try_skip(curr_pos, 'l') && try_skip(curr_pos, 's') &&
        try_skip(curr_pos, 'e')) {
      current_token = json_token::VALUE_FALSE;
    } else {
      set_current_error();
    }
  }

  /**
   * parse null
   */
  __device__ inline void parse_null_and_set_current()
  {
    // already parsed 'n'
    if (try_skip(curr_pos, 'u') && try_skip(curr_pos, 'l') && try_skip(curr_pos, 'l')) {
      current_token = json_token::VALUE_NULL;
    } else {
      set_current_error();
    }
  }

  __device__ inline void parse_field_name_and_set_current(
    bool& matched_field_name, char_range to_match_field_name = char_range::null())
  {
    current_token_start_pos = curr_pos;
    auto const [success, matched, end] =
      try_parse_string(char_range_reader{chars.slice(curr_pos, chars.size() - curr_pos)},
                       char_range_reader{cuda::std::move(to_match_field_name)});
    if (success) {
      matched_field_name = matched;
      curr_pos           = static_cast<cudf::size_type>(cuda::std::distance(chars.data(), end));
      current_token      = json_token::FIELD_NAME;
    } else {
      set_current_error();
    }
  }

  __device__ inline void parse_next_token_and_set_current(
    bool& has_comma_before_token,
    bool& has_colon_before_token,
    bool& matched_field_name,
    char_range to_match_field_name = char_range::null())
  {
    skip_whitespaces();
    if (!eof()) {
      char const c = chars[curr_pos];
      if (is_context_stack_empty()) {
        // stack is empty

        if (current_token == json_token::INIT) {
          // main root entry point
          parse_first_token_in_value_and_set_current();
        } else {
          // previous token is not INIT, means already get a token; stack is
          // empty; Successfully parsed. Note: ignore the tailing sub-string
          current_token = json_token::SUCCESS;
        }
      } else {
        // stack is non-empty

        if (is_object_context()) {
          // in JSON object context
          if (current_token == json_token::START_OBJECT) {
            // previous token is '{'
            if (c == '}') {
              // empty object
              // close curr object context
              current_token_start_pos = curr_pos;
              curr_pos++;
              pop_curr_context();
              current_token = json_token::END_OBJECT;
            } else {
              // parse key in key:value pair
              parse_field_name_and_set_current(matched_field_name, to_match_field_name);
            }
          } else if (current_token == json_token::FIELD_NAME) {
            if (c == ':') {
              has_colon_before_token = true;
              // skip ':' and parse value in key:value pair
              curr_pos++;
              skip_whitespaces();
              parse_first_token_in_value_and_set_current();
            } else {
              set_current_error();
            }
          } else {
            // expect next key:value pair or '}'
            if (c == '}') {
              // end of object
              current_token_start_pos = curr_pos;
              curr_pos++;
              pop_curr_context();
              current_token = json_token::END_OBJECT;
            } else if (c == ',') {
              has_comma_before_token = true;
              // parse next key:value pair
              curr_pos++;
              skip_whitespaces();
              parse_field_name_and_set_current(matched_field_name, to_match_field_name);
            } else {
              set_current_error();
            }
          }
        } else {
          // in Json array context
          if (current_token == json_token::START_ARRAY) {
            // previous token is '['
            if (c == ']') {
              // curr: ']', empty array
              current_token_start_pos = curr_pos;
              curr_pos++;
              pop_curr_context();
              current_token = json_token::END_ARRAY;
            } else {
              // non-empty array, parse the first value in the array
              parse_first_token_in_value_and_set_current();
            }
          } else {
            if (c == ',') {
              has_comma_before_token = true;
              // skip ',' and parse the next value
              curr_pos++;
              skip_whitespaces();
              parse_first_token_in_value_and_set_current();
            } else if (c == ']') {
              // end of array
              current_token_start_pos = curr_pos;
              curr_pos++;
              pop_curr_context();
              current_token = json_token::END_ARRAY;
            } else {
              set_current_error();
            }
          }
        }
      }
    } else {
      // eof
      if (is_context_stack_empty() && current_token != json_token::INIT) {
        // reach eof; stack is empty; previous token is not INIT
        current_token = json_token::SUCCESS;
      } else {
        // eof, and meet the following cases:
        //   - has unclosed JSON array/object;
        //   - the whole JSON is empty
        set_current_error();
      }
    }
  }

 public:
  __device__ json_token next_token()
  {
    bool has_comma_before_token;  // no-initialization because of do not care here
    bool has_colon_before_token;  // no-initialization because of do not care here
    bool matched_field_name;      // no-initialization because of do not care here
    parse_next_token_and_set_current(
      has_comma_before_token, has_colon_before_token, matched_field_name);
    return current_token;
  }

  __device__ bool parse_next_token_with_matching(cudf::string_view to_match_field_name)
  {
    bool has_comma_before_token;  // no-initialization because of do not care here
    bool has_colon_before_token;  // no-initialization because of do not care here
    bool matched_field_name;
    parse_next_token_and_set_current(has_comma_before_token,
                                     has_colon_before_token,
                                     matched_field_name,
                                     char_range{to_match_field_name});
    return matched_field_name;
  }

  __device__ json_token get_current_token() const { return current_token; }

  // Raw token range includes quotes; reading it does not advance the parser.
  __device__ inline char_range current_range() const
  {
    return chars.slice(current_token_start_pos, curr_pos - current_token_start_pos);
  }

  // Skip to the matching close token, reporting malformed descendants.
  __device__ bool try_skip_children()
  {
    if (current_token == json_token::ERROR || current_token == json_token::INIT ||
        current_token == json_token::SUCCESS) {
      return false;
    }

    if (current_token != json_token::START_OBJECT && current_token != json_token::START_ARRAY) {
      return true;
    }

    json_token t;
    int open = 1;
    do {
      t = next_token();
      if (t == json_token::START_OBJECT || t == json_token::START_ARRAY) {
        ++open;
      } else if (t == json_token::END_OBJECT || t == json_token::END_ARRAY) {
        if (--open == 0) { return true; }
      } else if (t == json_token::ERROR) {
        return false;
      }
    } while (t != json_token::SUCCESS);
    return false;
  }

  __device__ cudf::size_type compute_unescaped_len() const { return write_unescaped_text(nullptr); }

  /**
   * unescape current token text, then write to destination
   * e.g.: '\\r' is a string with 2 chars '\' 'r', writes 1 char '\r'
   * e.g.: "\u4e2d\u56FD" are code points for Chinese chars "中国",
   *   writes 6 utf8 bytes: -28  -72 -83 -27 -101 -67
   * For number, write verbatim without normalization
   */
  __device__ cudf::size_type write_unescaped_text(char* destination) const
  {
    switch (current_token) {
      case json_token::VALUE_STRING: {
        // can not copy from JSON directly due to escaped chars
        // rewind the pos; parse again with copy
        char_range_reader reader(current_range());
        return write_string(reader, destination, escape_style::UNESCAPED);
      }
      case json_token::VALUE_NUMBER_INT:
        if (number_token_len == 2 && chars[current_token_start_pos] == '-' &&
            chars[current_token_start_pos + 1] == '0') {
          if (nullptr != destination) *destination++ = '0';
          return 1;
        }
        if (nullptr != destination) {
          for (cudf::size_type i = 0; i < number_token_len; ++i) {
            *destination++ = chars[current_token_start_pos + i];
          }
        }
        return number_token_len;
      case json_token::VALUE_NUMBER_FLOAT: {
        // number normalization:
        // 0.03E-2 => 0.3E-5, 200.000 => 200.0, 351.980 => 351.98,
        // 12345678900000000000.0 => 1.23456789E19, 1E308 => 1.0E308
        // 0.0000000000003 => 3.0E-13; 0.003 => 0.003; 0.0003 => 3.0E-4
        // 1.0E309 => "Infinity", -1E309 => "-Infinity"
        double d_value =
          cudf::strings::detail::stod(chars.slice_sv(current_token_start_pos, number_token_len));
        return spark_rapids_jni::ftos_converter::double_normalization(d_value, destination);
      }
      case json_token::VALUE_TRUE:
        if (nullptr != destination) {
          *destination++ = 't';
          *destination++ = 'r';
          *destination++ = 'u';
          *destination++ = 'e';
        }
        return 4;
      case json_token::VALUE_FALSE:
        if (nullptr != destination) {
          *destination++ = 'f';
          *destination++ = 'a';
          *destination++ = 'l';
          *destination++ = 's';
          *destination++ = 'e';
        }
        return 5;
      case json_token::VALUE_NULL:
        if (nullptr != destination) {
          *destination++ = 'n';
          *destination++ = 'u';
          *destination++ = 'l';
          *destination++ = 'l';
        }
        return 4;
      case json_token::FIELD_NAME: {
        // can not copy from JSON directly due to escaped chars
        // rewind the pos; parse again with copy
        char_range_reader reader(current_range());
        return write_string(reader, destination, escape_style::UNESCAPED);
      }
      case json_token::START_ARRAY:
        if (nullptr != destination) { *destination++ = '['; }
        return 1;
      case json_token::END_ARRAY:
        if (nullptr != destination) { *destination++ = ']'; }
        return 1;
      case json_token::START_OBJECT:
        if (nullptr != destination) { *destination++ = '{'; }
        return 1;
      case json_token::END_OBJECT:
        if (nullptr != destination) { *destination++ = '}'; }
        return 1;
      // for the following tokens, return false
      case json_token::SUCCESS:
      case json_token::ERROR:
      case json_token::INIT: return 0;
    }
    return 0;
  }

  __device__ cudf::size_type compute_escaped_len() const { return write_escaped_text(nullptr); }
  /**
   * escape current token text, then write to destination
   * e.g.: '"' is a string with 1 char '"', writes out 4 chars '"' '\' '\"' '"'
   * e.g.: "\u4e2d\u56FD" are code points for Chinese chars "中国",
   *   writes 8 utf8 bytes: '"' -28  -72 -83 -27 -101 -67 '"'
   * For number, write verbatim without normalization
   */
  __device__ cudf::size_type write_escaped_text(char* destination) const
  {
    switch (current_token) {
      case json_token::VALUE_STRING: {
        // can not copy from JSON directly due to escaped chars
        char_range_reader reader(current_range());
        return write_string(reader, destination, escape_style::ESCAPED);
      }
      case json_token::VALUE_NUMBER_INT: {
        if (number_token_len == 2 && chars[current_token_start_pos] == '-' &&
            chars[current_token_start_pos + 1] == '0') {
          if (nullptr != destination) *destination++ = '0';
          return 1;
        }
        if (nullptr != destination) {
          for (cudf::size_type i = 0; i < number_token_len; ++i) {
            *destination++ = chars[current_token_start_pos + i];
          }
        }
        return number_token_len;
      }
      case json_token::VALUE_NUMBER_FLOAT: {
        // number normalization:
        double d_value =
          cudf::strings::detail::stod(chars.slice_sv(current_token_start_pos, number_token_len));
        return spark_rapids_jni::ftos_converter::double_normalization(d_value, destination);
      }
      case json_token::VALUE_TRUE:
        if (nullptr != destination) {
          *destination++ = 't';
          *destination++ = 'r';
          *destination++ = 'u';
          *destination++ = 'e';
        }
        return 4;
      case json_token::VALUE_FALSE:
        if (nullptr != destination) {
          *destination++ = 'f';
          *destination++ = 'a';
          *destination++ = 'l';
          *destination++ = 's';
          *destination++ = 'e';
        }
        return 5;
      case json_token::VALUE_NULL:
        if (nullptr != destination) {
          *destination++ = 'n';
          *destination++ = 'u';
          *destination++ = 'l';
          *destination++ = 'l';
        }
        return 4;
      case json_token::FIELD_NAME: {
        // can not copy from JSON directly due to escaped chars
        char_range_reader reader(current_range());
        return write_string(reader, destination, escape_style::ESCAPED);
      }
      case json_token::START_ARRAY:
        if (nullptr != destination) { *destination++ = '['; }
        return 1;
      case json_token::END_ARRAY:
        if (nullptr != destination) { *destination++ = ']'; }
        return 1;
      case json_token::START_OBJECT:
        if (nullptr != destination) { *destination++ = '{'; }
        return 1;
      case json_token::END_OBJECT:
        if (nullptr != destination) { *destination++ = '}'; }
        return 1;
      // for the following tokens, return false
      case json_token::SUCCESS:
      case json_token::ERROR:
      case json_token::INIT: return 0;
    }
    return 0;
  }

  /**
   * copy current structure to destination.
   * return false if meets JSON format error,
   * reurn true otherwise.
   * @param[out] copy_to
   */
  __device__ cuda::std::pair<bool, size_t> copy_current_structure(char* copy_to)
  {
    switch (current_token) {
      case json_token::INIT:
      case json_token::ERROR:
      case json_token::SUCCESS:
      case json_token::FIELD_NAME:
      case json_token::END_ARRAY:
      case json_token::END_OBJECT: return {false, 0};
      case json_token::VALUE_NUMBER_INT:
      case json_token::VALUE_NUMBER_FLOAT:
      case json_token::VALUE_STRING:
      case json_token::VALUE_TRUE:
      case json_token::VALUE_FALSE:
      case json_token::VALUE_NULL:
        // copy terminal token
        if (nullptr != copy_to) {
          size_t copy_len = write_escaped_text(copy_to);
          return {true, copy_len};
        } else {
          size_t copy_len = compute_escaped_len();
          return {true, copy_len};
        }
      case json_token::START_ARRAY:
      case json_token::START_OBJECT:
        // stack size increased by 1 when meet start object/array
        // copy until meet matched end object/array
        size_t sum_copy_len   = 0;
        int backup_stack_size = stack_size;

        // copy start object/array
        if (nullptr != copy_to) {
          int len = write_escaped_text(copy_to);
          sum_copy_len += len;
          copy_to += len;
        } else {
          sum_copy_len += compute_escaped_len();
        }

        while (true) {
          bool has_comma_before_token = false;
          bool has_colon_before_token = false;

          // parse and get has_comma_before_token, has_colon_before_token
          bool matched_field_name;  // unused
          parse_next_token_and_set_current(
            has_comma_before_token, has_colon_before_token, matched_field_name);

          // check the JSON format
          if (current_token == json_token::ERROR) { return {false, 0}; }

          // write out the token
          if (nullptr != copy_to) {
            if (has_comma_before_token) {
              sum_copy_len++;
              *copy_to++ = ',';
            }
            if (has_colon_before_token) {
              sum_copy_len++;
              *copy_to++ = ':';
            }
            int len = write_escaped_text(copy_to);
            sum_copy_len += len;
            copy_to += len;
          } else {
            if (has_comma_before_token) { sum_copy_len++; }
            if (has_colon_before_token) { sum_copy_len++; }
            sum_copy_len += compute_escaped_len();
          }

          if (backup_stack_size - 1 == stack_size) {
            // indicate meet the matched end object/array
            return {true, sum_copy_len};
          }
        }
        return {false, 0};
    }

    // never happen
    return {false, 0};
  }

  __device__ inline bool max_nesting_depth_exceeded() const { return max_depth_exceeded; }

 private:
  char_range const chars;
  cudf::size_type curr_pos;

  // A context bit is one for object and zero for array (at most 64 levels).
  int64_t context_stack;
  int stack_size = 0;

  cudf::size_type current_token_start_pos;
  cudf::size_type number_token_len;

  json_token current_token;

  bool max_depth_exceeded;
};

}  // namespace spark_rapids_jni

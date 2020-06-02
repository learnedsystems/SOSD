#pragma once

#include <dtl/dtl.hpp>

namespace dtl {


template<typename T>
struct env {};

template<>
struct env<std::string> {

  static std::string
  get(const std::string name, const std::string default_value = "") {
    std::string value = default_value;
    if (const char* env = std::getenv(name.c_str())) {
      value = std::string(env);
    }
    return value;
  }

};

template<>
struct env<$i32> {

  static $i32
  get(const std::string name, const $i32 default_value = 0) {
    $i32 value = default_value;
    if (const char* env = std::getenv(name.c_str())) {
      value = std::stoi(env);
    }
    return value;
  }

};

template<>
struct env<$i64> {

  static $i64
  get(const std::string name, const $i64 default_value = 0) {
    $i64 value = default_value;
    if (const char* env = std::getenv(name.c_str())) {
      value = std::stoll(env);
    }
    return value;
  }

};

template<>
struct env<$u64> {

  static $u64
  get(const std::string name, const $u64 default_value = 0) {
    $u64 value = default_value;
    if (const char* env = std::getenv(name.c_str())) {
      value = std::stoull(env);
    }
    return value;
  }

};

template<>
struct env<$f64> {

  static $f64
  get(const std::string name, const $f64 default_value = 0.0) {
    $f64 value = default_value;
    if (const char* env = std::getenv(name.c_str())) {
      value = std::stod(env);
    }
    return value;
  }

};

} // namespace dtl

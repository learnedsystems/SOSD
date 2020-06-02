#pragma once

#ifndef _DTL_STORAGE_INCLUDED
#error "Never use <dtl/storage/types.hpp> directly; include <dtl/storage.hpp> instead."
#endif

#include <dtl/dtl.hpp>
#include <string>

namespace dtl {


enum class null { value };

/// the supported runtime types
enum class rtt {
  i8,  u8,
  i16, u16,
  i32, u32,
  i64, u64,
  f32,
  f64,
  str,
};


/// map runtime types to native types
template<rtt T> struct map {};
template<> struct map<rtt::i8>  { using type = $i8; };
template<> struct map<rtt::u8>  { using type = $u8; };
template<> struct map<rtt::i16> { using type = $i16; };
template<> struct map<rtt::u16> { using type = $u16; };
template<> struct map<rtt::i32> { using type = $i32; };
template<> struct map<rtt::u32> { using type = $u32; };
template<> struct map<rtt::i64> { using type = $i64; };
template<> struct map<rtt::u64> { using type = $u64; };
template<> struct map<rtt::f32> { using type = $f32; };
template<> struct map<rtt::f64> { using type = $f64; };
template<> struct map<rtt::str> { using type = std::string; };


template<rtt T> struct parse {};

#define DTL_GENERATE(T, FN)                         \
template<> struct parse<rtt::T>  {                  \
  inline map<rtt::T>::type                          \
  operator()(const std::string& str) const {        \
    return static_cast<map<rtt::T>::type>(FN(str)); \
  }                                                 \
};
DTL_GENERATE(u8,  std::stol)
DTL_GENERATE(i8,  std::stoul)
DTL_GENERATE(u16, std::stol)
DTL_GENERATE(i16, std::stoul)
DTL_GENERATE(u32, std::stol)
DTL_GENERATE(i32, std::stoul)
DTL_GENERATE(u64, std::stoll)
DTL_GENERATE(i64, std::stoull)
DTL_GENERATE(f32, std::stof)
DTL_GENERATE(f64, std::stod)
#undef DTL_GENERATE

// identity. returns a copy of the given string
template<> struct parse<rtt::str>  {
  inline map<rtt::str>::type
  operator()(const std::string& str) const {
    return str;
  }
};


} // namespace dtl
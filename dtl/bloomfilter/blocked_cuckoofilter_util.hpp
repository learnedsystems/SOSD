#pragma once

#include <dtl/dtl.hpp>

namespace dtl {
namespace cuckoofilter {


namespace {


// TODO remove macros
// inspired from http://www-graphics.stanford.edu/~seander/bithacks.html#ZeroInWord
// adapted from https://github.com/efficient/cuckoofilter
#define haszero2_u32(x) (((x)-0x55555555u) & (~(x)) & 0xAAAAAAAAu)
#define hasvalue2_u32(x, n) (haszero2_u32((x) ^ ((n) * 0x55555555u)))

#define haszero2_u64(x) (((x)-0x5555555555555555ull) & (~(x)) & 0xAAAAAAAAAAAAAAAAull)
#define hasvalue2_u64(x, n) (haszero2_u64((x) ^ ((n) * 0x5555555555555555ull)))

#define haszero3_u32(x) (((x)-0b01001001001001001001001001001001u) & (~(x)) & 0b00100100100100100100100100100100u)
#define hasvalue3_u32(x, n) (haszero3_u32((x) ^ ((n) * 0b01001001001001001001001001001001u)))

#define haszero3_u64(x) (((x)-0b1001001001001001001001001001001001001001001001001001001001001001ull) & (~(x)) & 0b0100100100100100100100100100100100100100100100100100100100100100ull)
#define hasvalue3_u64(x, n) (haszero3_u64((x) ^ ((n) * 0b1001001001001001001001001001001001001001001001001001001001001001ull)))

#define haszero4_u32(x) (((x)-0x11111111u) & (~(x)) & 0x88888888u)
#define hasvalue4_u32(x, n) (haszero4_u32((x) ^ ((n) * 0x11111111u)))

#define haszero4_u64(x) (((x)-0x1111111111111111ull) & (~(x)) & 0x8888888888888888ull)
#define hasvalue4_u64(x, n) (haszero4_u64((x) ^ ((n) * 0x1111111111111111ull)))

#define haszero5_u32(x) (((x)-0b01000010000100001000010000100001u) & (~(x)) & 0b00100001000010000100001000010000u)
#define hasvalue5_u32(x, n) (haszero5_u32((x) ^ ((n) * 0b01000010000100001000010000100001u)))

#define haszero5_u64(x) (((x)-0b0001000010000100001000010000100001000010000100001000010000100001ull) & (~(x)) & 0b0000100001000010000100001000010000100001000010000100001000010000ull)
#define hasvalue5_u64(x, n) (haszero5_u64((x) ^ ((n) * 0b0001000010000100001000010000100001000010000100001000010000100001ull)))

#define haszero6_u32(x) (((x)-0b01000001000001000001000001000001u) & (~(x)) & 0b00100000100000100000100000100000u)
#define hasvalue6_u32(x, n) (haszero6_u32((x) ^ ((n) * 0b01000001000001000001000001000001)))

#define haszero6_u64(x) (((x)-0b0001000001000001000001000001000001000001000001000001000001000001ull) & (~(x)) & 0b0000100000100000100000100000100000100000100000100000100000100000ull)
#define hasvalue6_u64(x, n) (haszero6_u64((x) ^ ((n) * 0b0001000001000001000001000001000001000001000001000001000001000001ull)))

#define haszero7_u32(x) (((x)-0b00010000001000000100000010000001u) & (~(x)) & 0b00001000000100000010000001000000u)
#define hasvalue7_u32(x, n) (haszero7_u32((x) ^ ((n) * 0b00010000001000000100000010000001u)))

#define haszero7_u64(x) (((x)-0b1000000100000010000001000000100000010000001000000100000010000001ull) & (~(x)) & 0b0100000010000001000000100000010000001000000100000010000001000000ull)
#define hasvalue7_u64(x, n) (haszero7_u64((x) ^ ((n) * 0b1000000100000010000001000000100000010000001000000100000010000001ull)))

#define haszero8_u32(x) (((x)-0x01010101u) & (~(x)) & 0x80808080u)
#define hasvalue8_u32(x, n) (haszero8_u32((x) ^ ((n) * 0x01010101u)))

#define haszero8_u64(x) (((x)-0x0101010101010101ull) & (~(x)) & 0x8080808080808080ull)
#define hasvalue8_u64(x, n) (haszero8_u64((x) ^ ((n) * 0x0101010101010101ull)))

#define haszero10_u32(x) (((x)-0b00000000000100000000010000000001u) & (~(x)) & 0b00100000000010000000001000000000u)
#define hasvalue10_u32(x, n) (haszero10_u32((x) ^ ((n) * 0b00000000000100000000010000000001u)))

#define haszero10_u64(x) (((x)-0b0000000000000100000000010000000001000000000100000000010000000001ull) & (~(x)) & 0b0000100000000010000000001000000000100000000010000000001000000000ull)
#define hasvalue10_u64(x, n) (haszero10_u64((x) ^ ((n) * 0b0000000000000100000000010000000001000000000100000000010000000001ull)))

#define haszero12_u32(x) (((x)-0x00001001u) & (~(x)) & 0x00800800u)
#define hasvalue12_u32(x, n) (haszero12_u32((x) ^ ((n) * 0x00001001u)))

#define haszero12_u64(x) (((x)-0x001001001001001ull) & (~(x)) & 0x800800800800800ull)
#define hasvalue12_u64(x, n) (haszero12_u64((x) ^ ((n) * 0x001001001001001ull)))

#define haszero15_u32(x) (((x)-0b00000000000000001000000000000001u) & (~(x)) & 0b100000000000000100000000000000u)
#define hasvalue15_u32(x, n) (haszero15_u32((x) ^ ((n) * 0b00000000000000001000000000000001u)))

#define haszero15_u64(x) (((x)-0b000000000000001000000000000001000000000000001000000000000001ull) & (~(x)) & 0b100000000000000100000000000000100000000000000100000000000000ull)
#define hasvalue15_u64(x, n) (haszero15_u64((x) ^ ((n) * 0b000000000000001000000000000001000000000000001000000000000001ull)))

#define haszero16_u32(x) \
  (((x)-0x0001000100010001u) & (~(x)) & 0x8000800080008000u)
#define hasvalue16_u32(x, n) (haszero16_u32((x) ^ ((n) * 0x0001000100010001u)))

#define haszero16_u64(x) \
  (((x)-0x0001000100010001ull) & (~(x)) & 0x8000800080008000ull)
#define hasvalue16_u64(x, n) (haszero16_u64((x) ^ ((n) * 0x0001000100010001ull)))


template<typename T, uint32_t bits_per_value>
struct packed_value { };

#define __GENERATE(T,B)                                                     \
template<>                                                                  \
struct packed_value<uint##T##_t, B> {                                       \
  __forceinline__ static bool                                               \
  contains(const uint##T##_t packed_value, const uint32_t search_value) {   \
    return hasvalue##B##_u##T(packed_value, search_value);                  \
  }                                                                         \
};
__GENERATE(32,2)
__GENERATE(64,2)
__GENERATE(32,3)
__GENERATE(64,3)
__GENERATE(32,4)
__GENERATE(64,4)
__GENERATE(32,5)
__GENERATE(64,5)
__GENERATE(32,6)
__GENERATE(64,6)
__GENERATE(32,7)
__GENERATE(64,7)
__GENERATE(32,8)
__GENERATE(64,8)
__GENERATE(32,10)
__GENERATE(64,10)
__GENERATE(32,12)
__GENERATE(64,12)
__GENERATE(32,16)
__GENERATE(64,16)
#undef __GENERATE

} // anonymous namespace


} // namespace cuckoofilter
} // namespace dtl

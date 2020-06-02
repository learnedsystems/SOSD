#pragma once

#include <bitset>
#include <string>
#include <sstream>

#include <dtl/dtl.hpp>
#include <dtl/math.hpp>
#include <dtl/color.hpp>

// --- helper ---
static std::string
to_string_u64(const __m512i values, const __mmask8 m, const __mmask8 p = 0) {
  dtl::color_modifier active(dtl::color::light_green);
  dtl::color_modifier protect(dtl::color::light_blue);
  dtl::color_modifier inactive(dtl::color::gray);
  dtl::color_modifier reset(dtl::color::gray);
  auto a = reinterpret_cast<const uint64_t*>(&values);
  auto mb = std::bitset<8>(m);
  auto mp = std::bitset<8>(p);
  std::stringstream str;
  str << "[ ";
  str << (mp[0] ? protect : (mb[0] ? active : inactive)) << a[0] << reset;
  for (std::size_t i = 1; i < 8; i++) {
    str << ", " << (mp[i] ? protect : (mb[i] ? active : inactive)) << a[i] << reset;
  }
  str << " ]";
  return str.str();
}

static std::string
to_string_u32(const __m256i values, const __mmask8 m, const __mmask8 p = 0) {
  dtl::color_modifier active(dtl::color::light_green);
  dtl::color_modifier protect(dtl::color::light_blue);
  dtl::color_modifier inactive(dtl::color::gray);
  dtl::color_modifier reset(dtl::color::gray);
  auto a = reinterpret_cast<const uint32_t*>(&values);
  auto mb = std::bitset<8>(m);
  auto mp = std::bitset<8>(p);
  std::stringstream str;
  str << "[ ";
  str << (mp[0] ? protect : (mb[0] ? active : inactive)) << a[0] << reset;
  for (std::size_t i = 1; i < 8; i++) {
    str << ", " << (mp[i] ? protect : (mb[i] ? active : inactive)) << a[i] << reset;
  }
  str << " ]";
  return str.str();
}

static std::string
to_string_u32(const __m512i values, const __mmask16 m, const __mmask16 p = 0) {
  dtl::color_modifier active(dtl::color::light_green);
  dtl::color_modifier protect(dtl::color::light_blue);
  dtl::color_modifier inactive(dtl::color::gray);
  dtl::color_modifier reset(dtl::color::gray);
  auto a = reinterpret_cast<const uint32_t*>(&values);
  auto mb = std::bitset<16>(m);
  auto mp = std::bitset<16>(p);
  std::stringstream str;
  str << "[ ";
  str << (mp[0] ? protect : (mb[0] ? active : inactive)) << a[0] << reset;
  for (std::size_t i = 1; i < 16; i++) {
    str << ", " << (mp[i] ? protect : (mb[i] ? active : inactive)) << a[i] << reset;
  }
  str << " ]";
  return str.str();
}

static std::string
to_string_u32_b16(const __m512i values, const __mmask16 m, const __mmask16 p = 0) {
  dtl::color_modifier active(dtl::color::light_green);
  dtl::color_modifier protect(dtl::color::light_blue);
  dtl::color_modifier inactive(dtl::color::gray);
  dtl::color_modifier reset(dtl::color::gray);
  auto a = reinterpret_cast<const uint32_t*>(&values);
  auto mb = std::bitset<16>(m);
  auto mp = std::bitset<16>(p);
  std::stringstream str;
  str << "[ ";
  str << (mp[0] ? protect : (mb[0] ? active : inactive)) << std::bitset<16>(a[0]) << reset;
  for (std::size_t i = 1; i < 16; i++) {
    str << ", " << (mp[i] ? protect : (mb[i] ? active : inactive)) << std::bitset<16>(a[i]) << reset;
  }
  str << " ]";
  return str.str();
}

static std::string
to_string_m8(const __mmask8 m) {
  dtl::color_modifier active(dtl::color::light_green);
  dtl::color_modifier inactive(dtl::color::gray);
  dtl::color_modifier reset(dtl::color::gray);
  auto b = std::bitset<8>(m);
  std::stringstream str;
  str << "[ ";
  str << (b[0] ? active : inactive) << b[0] << reset;
  for (std::size_t i = 1; i < 8; i++) {
    str << ", " << (b[i] ? active : inactive) << b[i] << reset;
  }
  str << " ]";
  return str.str();
}

static std::string
to_string_m16(const __mmask16 m) {
  dtl::color_modifier active(dtl::color::light_green);
  dtl::color_modifier inactive(dtl::color::gray);
  dtl::color_modifier reset(dtl::color::gray);
  auto b = std::bitset<16>(m);
  std::stringstream str;
  str << "[ ";
  str << (b[0] ? active : inactive) << b[0] << reset;
  for (std::size_t i = 1; i < 16; i++) {
    str << ", " << (b[i] ? active : inactive) << b[i] << reset;
  }
  str << " ]";
  return str.str();
}

static std::string
to_csv_m8(const __mmask8 m) {
  auto b = std::bitset<8>(m);
  std::stringstream str;
  str << "\\Mask{" << b[0] << "}";
  for (std::size_t i = 1; i < 8; i++) {
    str << "{" << b[i] << "}";
  }
  return str.str();
}

static std::string
to_csv_m8(const __mmask8 m, const __mmask8 p) {
  auto mb = std::bitset<8>(m);
  auto pb = std::bitset<8>(p);
  std::stringstream str;
  str << "\\Mask{" << (pb[0] ? 2 : mb[0]) << "}";
  for (std::size_t i = 1; i < 8; i++) {
    str << "{" << (pb[i] ? 2 : mb[i]) << "}";
  }
  return str.str();
}




/**
 * Calculates the number of bits set in each 32-bit element (but only the low 16-bits) in the AVX 512 register
 * using a in-register lookup table. This is pretty much the naive way and requires 4 permutes.
 * Author: Tim Gubner (CWI)
 */

// compile-time population count
static inline constexpr uint32_t
population_count(uint32_t v)
{
  uint32_t c = 0;

  for (; v; v >>= 1) {
    c += v & 1;
  }

  return c;
}

/** Generate m512i using a lambda function */
template<typename GEN>
static inline __m512i
_m512_generate_epi32(GEN&& gen)
{
  return _mm512_set_epi32(gen(15), gen(14), gen(13), gen(12), gen(11), gen(10), gen(9), gen(8),
                          gen(7), gen(6), gen(5), gen(4), gen(3), gen(2), gen(1), gen(0));
}

static const auto popcount_up_to_16_epi32_table = _m512_generate_epi32([] (int idx) { return population_count(idx); });

// 5.4 cycles
inline static __m512i
popcount_up_to_16_epi32(__m512i a, __m512i pop_table)
{
  const auto lo4bits = _mm512_set1_epi32(0x0000000F);

  /* get the first 4 bits and use them as index into a in-register table */
  auto idx = _mm512_and_epi32(a, lo4bits);
  auto pop = _mm512_permutexvar_epi32(idx, pop_table);

  for (int i=0; i<3; i++) {
    /* get the next 4 bits */
    idx = _mm512_srli_epi32(a, (i+1)*4);
    idx = _mm512_and_epi32(idx, lo4bits);
    pop = _mm512_add_epi32(pop, _mm512_permutexvar_epi32(idx, pop_table));
  }

#ifdef POPCOUNT_DEBUG
  /* check results */
	int k = 0;
	_m512_map_epi32(a, pop, [&] (int a, int b) {
		int c = population_count(a);
		if (b != c) {
			printf("k=%d: a=%d pop(a)=%d b=%d\n", k, a, c, b);
			assert(false);
		}
		k++;
	});
#endif

  return pop;
}


static inline __m512i
_mm512_popcount_up_to_16_epi32(__m512i a) {
  return popcount_up_to_16_epi32(a, popcount_up_to_16_epi32_table);
}
// ---------------------------------------------------------------------------------------------------------------------
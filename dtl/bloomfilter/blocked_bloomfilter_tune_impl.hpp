#pragma once

#include <chrono>
#include <map>
#include <random>

#include <dtl/dtl.hpp>
#include <dtl/thread.hpp>
#include <dtl/bloomfilter/block_addressing_logic.hpp>
#include <dtl/bloomfilter/blocked_bloomfilter.hpp>
#include <dtl/bloomfilter/blocked_bloomfilter_tune.hpp>

#include "immintrin.h"

namespace dtl {


namespace internal {

  struct blocked_bloomfilter_tune_mock : blocked_bloomfilter_tune {

    u32 unroll_factor;

    blocked_bloomfilter_tune_mock(u32 unroll_factor) : unroll_factor(unroll_factor) {};

    $u32
    get_unroll_factor(u32 k,
                      u32 word_size,
                      u32 word_cnt_per_block,
                      u32 sector_cnt,
                      dtl::block_addressing addr_mode) const override {
      return unroll_factor;
    }
  };

} // namespace dtl

//===----------------------------------------------------------------------===//
template<typename word_t>
struct blocked_bloomfilter_tune_impl : blocked_bloomfilter_tune {

  // TODO memoization in a global file / tool to calibrate

  static constexpr u32 max_k = 16;
  static constexpr u32 max_unroll_factor = 4;


  std::map<config, tuning_params>
  m;


  void
  set_unroll_factor(u32 k,
                    u32 word_size,
                    u32 word_cnt_per_block,
                    u32 sector_cnt,
                    dtl::block_addressing addr_mode,
                    u32 unroll_factor) override {

    u1 is_sectorized = sector_cnt >= word_cnt_per_block;
    u32 sector_cnt_actual = is_sectorized ? word_cnt_per_block : 1;
    config c {k, word_size, word_cnt_per_block, sector_cnt_actual, addr_mode};
    tuning_params tp { unroll_factor };
    m.insert(std::pair<config, tuning_params>(c, tp));
  }


  $u32
  get_unroll_factor(u32 k,
                    u32 word_size,
                    u32 word_cnt_per_block,
                    u32 sector_cnt,
                    dtl::block_addressing addr_mode) const override {
    u1 is_sectorized = sector_cnt >= word_cnt_per_block;
    u32 sector_cnt_actual = is_sectorized ? word_cnt_per_block : 1;
    config c {k, word_size, word_cnt_per_block, sector_cnt_actual, addr_mode};
    if (m.count(c)) {
      auto p = m.find(c)->second;
      return p.unroll_factor;
    }
    return 1; // default
  }


  $u32
  tune_unroll_factor(u32 k,
                     u32 word_size,
                     u32 word_cnt_per_block,
                     u32 sector_cnt,
                     dtl::block_addressing addr_mode) override {
    assert(sizeof(word_t) == word_size);

    u1 is_sectorized = sector_cnt >= word_cnt_per_block;
    u32 sector_cnt_actual = is_sectorized ? word_cnt_per_block : 1;
    config c {k, word_size, word_cnt_per_block, sector_cnt_actual, addr_mode};

    tuning_params tp = calibrate(c);

    m.insert(std::pair<config, tuning_params>(c, tp));
    return tp.unroll_factor;
  }


  void
  tune_unroll_factor() {
    for ($u32 word_cnt_per_block = 1; word_cnt_per_block <= 16; word_cnt_per_block *= 2) {
      for (auto addr_mode : {dtl::block_addressing::POWER_OF_TWO, dtl::block_addressing::MAGIC}) {
        for ($u32 k = 1; k <= 16; k++) {
          tune_unroll_factor(k, sizeof(word_t), word_cnt_per_block, 1, addr_mode);
          if (word_cnt_per_block > 1) {
            tune_unroll_factor(k, sizeof(word_t), word_cnt_per_block, word_cnt_per_block, addr_mode);
          }
        }
      }
    }
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  tuning_params
  calibrate(const config& c) {
    using key_t = $u32;
//    if (early_out) {
//      std::cerr << "WARNING: Using 'early out' in combination with SIMD unrolling may cause performance degradations!" << std::endl;
//    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis;

    static constexpr u32 data_size = 4u*1024*8;
    std::vector<key_t> random_data;
    random_data.reserve(data_size);
    for (std::size_t i = 0; i < data_size; i++) {
      random_data.push_back(dis(gen));
    }

    try {
      std::cerr << "w = " << std::setw(2) << c.word_cnt_per_block << ", "
                << "s = " << std::setw(2) << c.sector_cnt << ", "
                << "addr = " << std::setw(5) << (c.addr_mode == block_addressing::POWER_OF_TWO ? "pow2" : "magic") << ", "
                << "k = " <<  std::setw(2) << c.k << ": " << std::flush;

      $f64 cycles_per_lookup_min = std::numeric_limits<$f64>::max();
      $u32 u_min = 1;

      std::size_t match_count = 0;
      uint32_t match_pos[dtl::BATCH_SIZE];

      // baselines
      $f64 cycles_per_lookup_scalar = 0.0;
      $f64 cycles_per_lookup_unroll_by_one = 0.0;

      for ($u32 u = 0; u <= max_unroll_factor; u = (u == 0) ? 1 : u * 2) {
        std::cerr << std::setw(2) << "u(" << std::to_string(u) + ") = "<< std::flush;

        u64 desired_filter_size_bits = 4ull * 1024 * 8;
        const std::size_t m = desired_filter_size_bits
            + (128 * static_cast<u32>(c.addr_mode)); // enforce MAGIC addressing

        // Instantiate bloom filter logic.
        internal::blocked_bloomfilter_tune_mock tune_mock { u };
        blocked_bloomfilter<word_t> bbf(m , c.k, c.word_cnt_per_block, c.sector_cnt, tune_mock);

        // Allocate memory.
        dtl::mem::allocator_config alloc_config = dtl::mem::allocator_config::local();
        if (dtl::mem::hbm_available()) {
          // Use HBM if available
          const auto cpu_id = dtl::this_thread::get_cpu_affinity().find_first();
          const auto node_id = dtl::mem::get_node_of_cpu(cpu_id);
          const auto hbm_node_id = dtl::mem::get_nearest_hbm_node(node_id);
          alloc_config = dtl::mem::allocator_config::on_node(hbm_node_id);
        }

        dtl::mem::numa_allocator<word_t> alloc(alloc_config);
        std::vector<word_t, dtl::mem::numa_allocator<word_t>>
            filter_data(bbf.size(), 0, alloc); // TODO how to pass different allocator (for KNL/HBM)

        // Note: No need to insert elements, as the BBF is branch-free.

        // Run the micro benchmark.
        $u64 rep_cntr = 0;
        auto start = std::chrono::high_resolution_clock::now();
        auto tsc_start = _rdtsc();
        while (true) {
          dtl::batch_wise(random_data.begin(), random_data.end(),
                          [&](const auto batch_begin, const auto batch_end) {
                            match_count += bbf.batch_contains(&filter_data[0], &batch_begin[0], batch_end - batch_begin, match_pos, 0);
                          });

          rep_cntr++;
          std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - start;
          if (diff.count() > 0.25) break; // Run micro benchmark for at least 250ms.
        }
        auto tsc_end = _rdtsc();

        auto cycles_per_lookup = (tsc_end - tsc_start) / (data_size * rep_cntr * 1.0);

        if (u == 0) cycles_per_lookup_scalar = cycles_per_lookup;
        if (u == 1) cycles_per_lookup_unroll_by_one = cycles_per_lookup;
        std::cerr << std::setprecision(3) << std::setw(4) << std::right << cycles_per_lookup << ", ";
        if (cycles_per_lookup < cycles_per_lookup_min) {
          cycles_per_lookup_min = cycles_per_lookup;
          u_min = u;
        }
      }
      std::cerr << " picked u = " << u_min << " (" << cycles_per_lookup_min << " [cycles/lookup])"
                << ", speedup over u(0) = " << std::setprecision(3) << std::setw(4) << std::right << (cycles_per_lookup_scalar / cycles_per_lookup_min)
                << ", speedup over u(1) = " << std::setprecision(3) << std::setw(4) << std::right << (cycles_per_lookup_unroll_by_one / cycles_per_lookup_min)
                << " (chksum: " << match_count << ")" << std::endl;

      return tuning_params { u_min };

    } catch (...) {
      std::cerr<< " -> Failed to calibrate for k = " << c.k << "." << std::endl;
    }
    return tuning_params { 0 }; // defaults to scalar code
  }
  //===----------------------------------------------------------------------===//

};

} // namespace dtl
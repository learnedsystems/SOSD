cmake_minimum_required(VERSION 3.5)
project(dtl)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++14 -march=native")

include_directories(SYSTEM ../)

add_subdirectory(simd)
#add_subdirectory(storage)

set(SOURCE_FILES ${SOURCE_FILES}
        main.cpp
        adept.hpp
        barrier.hpp
        batchwise.hpp
        bits.hpp
        bloomfilter/dynamic/block_addressing_logic.hpp
        bloomfilter/dynamic/blocked_bloomfilter.hpp
        bloomfilter/dynamic/blocked_bloomfilter_logic.hpp
        bloomfilter/dynamic/blocked_bloomfilter_block_logic.hpp
        bloomfilter/dynamic/hash.hpp
        bloomfilter/bloomfilter_logic.hpp
        bloomfilter/blocked_bloomfilter.hpp
        bloomfilter/old/bloomfilter_runtime.hpp
        bloomfilter/old/bloomfilter_runtime_sectorized.hpp
        bloomfilter/old/bloomfilter_runtime_types.hpp
        bloomfilter/old/bloomfilter_logic.hpp
        bloomfilter/vector_helper.hpp
        bloomfilter/block_addressing_logic.hpp
        bloomfilter/old/bloomfilter_block_logic.hpp
        bloomfilter/old/bloomfilter_h1.hpp
        bloomfilter/old/bloomfilter_h1_logic.hpp
        bloomfilter/old/bloomfilter_h1_vec.hpp
        bloomfilter/old/bloomfilter_h1_mod.hpp
        bloomfilter/old/bloomfilter_h1_mod_vec.hpp
        bloomfilter/old/bloomfilter_h2.hpp
        bloomfilter/old/bloomfilter_h2_logic.hpp
#        bloomfilter_logic/bloomfilter_h2_mb_logic.hpp
        bloomfilter/old/bloomfilter_h2_vec.hpp
        bloomfilter/old/bloomfilter_h2_mod.hpp
        bloomfilter/old/bloomfilter_h2_mod_vec.hpp
        bloomfilter/old/bloomfilter_h3.hpp
        bloomfilter/old/bloomfilter_h3_mod.hpp
        bloomfilter/old/bloomfilter_h3_mod_vec.hpp
        bloomfilter/old/bloomfilter_h3_vec.hpp
        bloomfilter/blocked_bloomfilter.hpp
        bloomfilter/blocked_bloomfilter_logic.hpp
        bloomfilter/blocked_bloomfilter_block_logic.hpp
#        bloomfilter_logic/old/blocked_bloomfilter_block_logic_u32_w1.hpp
#        bloomfilter_logic/old/blocked_bloomfilter_block_logic_u32_w1.cpp
#        bloomfilter_logic/old/blocked_bloomfilter_block_logic_u32_w2.hpp
#        bloomfilter_logic/old/blocked_bloomfilter_block_logic_u32_w2.cpp
#        bloomfilter_logic/old/blocked_bloomfilter_block_logic_u32_w4.hpp
#        bloomfilter_logic/old/blocked_bloomfilter_block_logic_u32_w4.cpp
#        bloomfilter_logic/old/blocked_bloomfilter_block_logic_u32_w8.hpp
#        bloomfilter_logic/old/blocked_bloomfilter_block_logic_u32_w8.cpp
#        bloomfilter_logic/old/blocked_bloomfilter_block_logic_u64_w1.hpp
#        bloomfilter_logic/old/blocked_bloomfilter_block_logic_u64_w1.cpp
#        bloomfilter_logic/old/blocked_bloomfilter_block_logic_u64_w2.hpp
#        bloomfilter_logic/old/blocked_bloomfilter_block_logic_u64_w2.cpp
#        bloomfilter_logic/old/blocked_bloomfilter_block_logic_u64_w4.hpp
#        bloomfilter_logic/old/blocked_bloomfilter_block_logic_u64_w4.cpp
#        bloomfilter_logic/old/blocked_bloomfilter_block_logic_u64_w8.hpp
#        bloomfilter_logic/old/blocked_bloomfilter_block_logic_u64_w8.cpp
        bloomfilter/blocked_bloomfilter_block_logic_sgew.hpp
        bloomfilter/blocked_bloomfilter_block_logic_sltw.hpp
        bloomfilter/blocked_bloomfilter_tune.hpp
        bloomfilter/blocked_bloomfilter_tune_impl.hpp
        bloomfilter/old/bloomfilter_hn.hpp
        bloomfilter/old/bloomfilter_hn_vec.hpp
        bloomfilter/hash_family.hpp
        bloomfilter/blocked_cuckoofilter.hpp
        bloomfilter/blocked_cuckoofilter_logic.hpp
        bloomfilter/blocked_cuckoofilter_block_logic.hpp
#        bloomfilter/blocked_cuckoofilter_concurrent_multiword_table.hpp
        bloomfilter/blocked_cuckoofilter_multiword_table.hpp
        bloomfilter/blocked_cuckoofilter_simd.hpp
        bloomfilter/blocked_cuckoofilter_util.hpp
        bloomfilter/blocked_cuckoofilter_word_table.hpp
        bloomfilter/math.hpp
        bitmask.hpp
        bitpack.hpp
        bitset.hpp
        bitset_util.hpp
        color.hpp
        dict.hpp
        div.hpp
        dtl.hpp
        env.hpp
        hash.hpp
        index.hpp
        like.hpp
        lockstripe.hpp
        math.hpp
        mem.hpp
        polyfill.hpp
        psma.hpp
        simd.hpp
        sma.hpp
        storage.hpp
        #old/vec.hpp
        thread.hpp
        tree.hpp
        tree_mask.hpp
        zone_mask.hpp
#        compression/lz4.h
#        compression/lz4.c
#        compression/lz4hc.h
#        compression/lz4hc.c
        # third party
        thirdparty/libdivide/libdivide.h
        )

add_library(dtl ${SOURCE_FILES})
target_include_directories(dtl PUBLIC ..)

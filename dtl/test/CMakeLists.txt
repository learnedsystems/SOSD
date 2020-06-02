cmake_minimum_required(VERSION 3.5)
project(dtl_tester)

set(CMAKE_VERBOSE_MAKEFILE OFF)
#set(DEFAULT_ARCH -march=knl)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++14")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++14 -march=knl")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++14 -mavx512f -mavx512cd")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++14 -fuse-ld=gold -fsplit-stack")

# build optimized executables for the following architectures
set(ARCHS
        core-avx2
        knl
        corei7
        )


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(LIBS
        gtest
        gtest_main
        pthread
        numa
        )

# check for NUMA support
#find_package(NUMA)
#if (NUMA_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_NUMA")
    set(LIBS ${LIBS} numa)
#endif (NUMA_FOUND)

# check for CUDA support
find_package(CUDA)
if (CUDA_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_CUDA")
    # TODO add CUDA libs
endif (CUDA_FOUND)


# replacement for 'add_executable'
function(add_multiarch_executable executable_name files)
    foreach(arch ${ARCHS})
        add_executable(${executable_name}_${arch} ${files})
        set_target_properties(${executable_name}_${arch} PROPERTIES COMPILE_FLAGS "-march=${arch}")
        target_link_libraries(${executable_name}_${arch} ${LIBS})
    endforeach(arch ${ARCHS})
endfunction(add_multiarch_executable)

# note for ubuntu users: gtest lib is no longer shipped as dep package (see http://askubuntu.com/questions/145887/why-no-library-files-installed-for-google-test)
add_subdirectory(lib/googletest)
include_directories(lib/googletest/include lib/googletest)

add_subdirectory(../ dtl)
include_directories(SYTEM ../../)


set(TEST_FILES
        bitmask_test.cpp
        div_test.cpp
        cuckoofilter_test.cpp
#        bitset_test.cpp
#        bitpack_test.cpp

#        bloomfilter_test.cpp
#        mem_test.cpp
#        vec_test.cpp
#        like_test.cpp
#        enumerate_groups_test.cpp # avx-512 only
        )
add_multiarch_executable(dtl_tester "${TEST_FILES}")


set(VECTOR_TEST_FILES
        vec_test.cpp
        )
add_multiarch_executable(vector_tester "${VECTOR_TEST_FILES}")


set(STORAGE_TEST_FILES
        bitmask_test.cpp
        bitmask_experiments.cpp
#        io_test.cpp
        )
add_multiarch_executable(storage_tester "${STORAGE_TEST_FILES}")


set(BLOOMFILTER_EXPERIMENTS_FILES
        bloomfilter_experiments.cpp
        )
add_multiarch_executable(bloomfilter_experiments "${BLOOMFILTER_EXPERIMENTS_FILES}")


set(BLOOMFILTER_RTS_EXPERIMENTS_FILES
        bloomfilter_rts_experiments.cpp
        )
add_multiarch_executable(bloomfilter_rts_experiments "${BLOOMFILTER_RTS_EXPERIMENTS_FILES}")


set(BLOOMFILTER_TEST_FILES
        bloomfilter_test.cpp
#        bloomfilter_blocked_cacheline_test.cpp
#        bloomfilter_blocked_cacheline_magic_test.cpp
        )
add_multiarch_executable(bloomfilter_tester "${BLOOMFILTER_TEST_FILES}")


#set(COMPRESSION_TEST_FILES
#        bitpack_test.cpp
#        )
#add_multiarch_executable(bitpack_tester "${COMPRESSION_TEST_FILES}")


#set(THREAD_TEST_FILES
#        thread_test.cpp
#        )
#add_multiarch_executable(thread_tester "${THREAD_TEST_FILES}")


set(BITSET_TEST_FILES
        bitset_test.cpp
        )
add_multiarch_executable(bitset_tester "${BITSET_TEST_FILES}")


set(MEM_TEST_FILES
        mem_test.cpp
        )
add_multiarch_executable(mem_tester "${MEM_TEST_FILES}")

#set(LUT_TEST_FILES
#        lut_test.cpp
#        )
#add_multiarch_executable(lut_tester "${LUT_TEST_FILES}")

#set(BF_TOOL_FILES
#        bloomfilter_util.hpp
#        bloomfilter_tool.cpp
#        )
#add_executable(bf_tool "${BF_TOOL_FILES}")

configure_file(probe_performance.csv probe_performance.csv COPYONLY)
configure_file(false_positive_rate.csv false_positive_rate.csv COPYONLY)
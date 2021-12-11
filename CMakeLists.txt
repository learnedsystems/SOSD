cmake_minimum_required(VERSION 3.10)
project(SOSD)

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -Wall -Wfatal-errors -march=native")

# Enable OpenMP if available
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-fopenmp HAS_OPENMP)
if (HAS_OPENMP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

set(CMAKE_CXX_STANDARD 17)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_subdirectory(dtl)

if (${APPLE})
    include_directories(/usr/local/include/) # required by Mac OS to find boost
endif ()

set(SOURCE_FILES util.h)
file(GLOB_RECURSE RMI_SOURCES "competitors/rmi/*.cpp")
file(GLOB_RECURSE BENCH_SOURCES "benchmarks/*.cc")

if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10.0)
    # FIXME cannot link with g++-9
    add_definitions(-DDISABLE_FST=1)
    list(REMOVE_ITEM BENCH_SOURCES "${CMAKE_SOURCE_DIR}/benchmarks/benchmark_fst.cc")
endif ()

if (${APPLE})
    # Certain headers (e.g., byteswap) not supported under Mac OS.
    list(REMOVE_ITEM BENCH_SOURCES "${CMAKE_SOURCE_DIR}/benchmarks/benchmark_wormhole.cc")
endif ()

add_executable(generate generate.cc ${SOURCE_FILES})
target_link_libraries(generate Threads::Threads)

add_subdirectory(robin-map)

add_custom_target(
        libfast64
        COMMAND cargo build --release
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/fast64/
)

if (${LINUX})
    add_custom_target(
            libwh
            COMMAND make libwh.so
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/wormhole/
    )
endif ()

add_executable(benchmark benchmark.cc ${SOURCE_FILES} ${RMI_SOURCES} ${BENCH_SOURCES})
target_include_directories(benchmark
        PRIVATE "competitors/CHT/include"
        PRIVATE "competitors/FST/include"
        PRIVATE "competitors/PGM-index/include"
        PRIVATE "competitors/rs/include"
        PRIVATE "competitors/stx-btree-0.9/include"
        PRIVATE "competitors/ts/include"
        PRIVATE "fast64")
add_dependencies(benchmark libfast64)

if (${LINUX})
    target_include_directories(benchmark PRIVATE "wormhole")
    target_link_libraries(benchmark PRIVATE ${CMAKE_SOURCE_DIR}/wormhole/libwh.so)
    add_dependencies(benchmark libwh)
endif ()

target_link_libraries(benchmark
        PRIVATE ${CMAKE_SOURCE_DIR}/fast64/target/release/libfast64.a
        PRIVATE tsl::robin_map
        PRIVATE Threads::Threads dtl
        PRIVATE dl)

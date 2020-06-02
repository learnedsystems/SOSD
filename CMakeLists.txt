cmake_minimum_required(VERSION 3.10)
project(SOSD)

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()

if (${LINUX})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -Wall -Wfatal-errors -march=native -fopenmp")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -Wall -Wfatal-errors -march=native")
endif ()

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
if (${APPLE})
    # FIXME cannot link.
    list(REMOVE_ITEM BENCH_SOURCES "${CMAKE_SOURCE_DIR}/benchmarks/benchmark_fst.cc")
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
        PRIVATE "competitors/stx-btree-0.9/include"
        PRIVATE "competitors/FST/include"
        private "competitors/PGM-index/include"
        PRIVATE "fast64")
target_link_libraries(benchmark
        PRIVATE Threads::Threads dtl
        PRIVATE dl
        PRIVATE ${CMAKE_SOURCE_DIR}/fast64/target/release/libfast64.a
        PRIVATE tsl::robin_map)
add_dependencies(benchmark libfast64)

if (${LINUX})
    target_include_directories(benchmark PRIVATE "wormhole")
    target_link_libraries(benchmark PRIVATE ${CMAKE_SOURCE_DIR}/wormhole/libwh.so)
    add_dependencies(benchmark libwh)
endif ()

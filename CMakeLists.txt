cmake_minimum_required(VERSION 3.10)
project(SOSD)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "7.5.0")
    message(FATAL_ERROR "SOSD requires GCC >= 7.5.0")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0.1")
    message(FATAL_ERROR "SOSD requires Clang >= 9.0.1")
  endif()
else()
  message(FATAL_ERROR "SOSD only compiles with GCC or Clang")
endif()



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -Wall -g -march=native")
set(CMAKE_CXX_STANDARD 14)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(SOURCE_FILES util.h)
file(GLOB_RECURSE RMI_SOURCES "competitors/rmi/*.cpp")

add_custom_command(
    OUTPUT competitors/rmi/all_rmis.h
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/competitors/rmi/gen.sh
    DEPENDS scripts/datasets_under_test.txt
)

add_executable(generate generate.cc ${SOURCE_FILES})
target_link_libraries(generate Threads::Threads)

add_executable(benchmark benchmark.cc competitors/rmi/all_rmis.h ${SOURCE_FILES} ${RMI_SOURCES})
target_include_directories(benchmark PRIVATE "competitors/stx-btree-0.9/include")
target_link_libraries(benchmark Threads::Threads)

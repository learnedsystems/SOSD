#pragma once
#include "benchmark.h"
#include <string>

template<template<typename> typename Searcher>
void benchmark_32_rmi(sosd::Benchmark<uint32_t, Searcher>& benchmark, bool pareto, const std::string& filename);

template<template<typename> typename Searcher>
void benchmark_64_rmi(sosd::Benchmark<uint64_t, Searcher>& benchmark, bool pareto, const std::string& filename);

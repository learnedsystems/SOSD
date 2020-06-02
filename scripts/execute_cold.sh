#! /usr/bin/env bash

echo "Executing cold-cache benchmark and saving results..."

BENCHMARK=build/benchmark
if [ ! -f $BENCHMARK ]; then
    echo "benchmark binary does not exist"
    exit
fi

function do_benchmark() {

    RESULTS=./results/$1_results_cold.txt
    if [ -f $RESULTS ]; then
        echo "Already have results for $1"
    else
        echo "Executing workload $1"
        $BENCHMARK -r 1 ./data/$1 ./data/$1_equality_lookups_20K --cold-cache --pareto | tee $RESULTS
    fi
}

mkdir -p ./results

#do_benchmark osm_cellids_200M_uint64
#do_benchmark books_200M_uint64

do_benchmark fb_200M_uint64
do_benchmark wiki_ts_200M_uint64

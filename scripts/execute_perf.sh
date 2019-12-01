#! /usr/bin/env bash

echo "Executing perf benchmark and saving results..."

BENCHMARK=build/benchmark
if [ ! -f $BENCHMARK ]; then
    echo "benchmark binary does not exist"
    exit
fi

function do_benchmark() {

    RESULTS=./results_perf/$1_perf_results.txt
    if [ -f $RESULTS ]; then
        echo "Already have results for $1"
    else
        echo "Executing workload $1"
        $BENCHMARK --perf ./data/$1 ./data/$1_equality_lookups_1M | tee $RESULTS
    fi
}

mkdir -p ./results_perf

for dataset in $(cat scripts/datasets_under_test.txt); do
    do_benchmark "$dataset"
done

chmod -R 755 results_perf



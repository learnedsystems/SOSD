#! /usr/bin/env bash

echo "Executing perf benchmark and saving results..."

BENCHMARK=build/benchmark
if [ ! -f $BENCHMARK ]; then
    echo "benchmark binary does not exist"
    exit
fi

function do_benchmark() {

    RESULTS=./results/$1_results_perf.txt
    if [ -f $RESULTS ]; then
        echo "Already have results for $1"
    else
        echo "Executing workload $1"
        $BENCHMARK --perf --pareto ./data/$1 ./data/$1_equality_lookups_1M | tee $RESULTS
    fi
}

mkdir -p ./results

for dataset in $(cat scripts/datasets_under_test_searches.txt); do
    do_benchmark "$dataset"
done

chmod -R 755 results_perf



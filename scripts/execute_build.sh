#! /usr/bin/env bash

echo "Executing build time benchmark and saving results..."

BENCHMARK=build/benchmark
if [ ! -f $BENCHMARK ]; then
    echo "benchmark binary does not exist"
    exit
fi

function do_benchmark() {

    RESULTS=./results/$1_build_results.txt
    if [ -f $RESULTS ]; then
        echo "Already have results for $1"
    else
        echo "Executing workload $1"
        $BENCHMARK ./data/$1 ./data/$1_equality_lookups_10M --build --pareto | tee $RESULTS
    fi
}

mkdir -p ./results_build

for dataset in $(cat scripts/datasets_under_test_build.txt); do
    do_benchmark "$dataset"
done

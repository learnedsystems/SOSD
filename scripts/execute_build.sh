#! /usr/bin/env bash

echo "Executing build time benchmark and saving results..."

BENCHMARK=build/benchmark
if [ ! -f $BENCHMARK ]; then
    echo "benchmark binary does not exist"
    exit
fi

function do_benchmark() {

    RESULTS=./results_build/$1_results.txt
    if [ -f $RESULTS ]; then
        echo "Already have results for $1"
    else
        echo "Executing workload $1"
        $BENCHMARK ./data/$1 ./data/$1_equality_lookups_10M --build | tee ./results_build/$1_results.txt
    fi
}

mkdir -p ./results_build

for dataset in $(cat scripts/datasets_under_test.txt); do
    do_benchmark "$dataset"
done

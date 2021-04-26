#! /usr/bin/env bash

echo "Executing benchmark and saving results..."

while getopts "c" arg; do
    case $arg in
        c) do_csv=true;;
    esac
done

BENCHMARK=build/benchmark
if [ ! -f $BENCHMARK ]; then
    echo "benchmark binary does not exist"
    exit
fi

function do_benchmark() {

    RESULTS=./results/$1_results.txt
    if [ -f $RESULTS ]; then
        echo "Already have results for $1"
    else
        echo "Executing workload $1"
        $BENCHMARK -r 1 ./data/$1 ./data/$1_equality_lookups_10M --pareto | tee ./results/$1_results.txt
    fi
}

function do_benchmark_csv() {

    RESULTS=./results/$1_results_table.csv
    if [ -f $RESULTS ]; then
	# Previously existing file could be from incomplete run
        echo "Removing results CSV for $1"
	rm $RESULTS
    fi
    echo "Executing workload $1 and printing to CSV"
    $BENCHMARK -r 5 ./data/$1 ./data/$1_equality_lookups_10M --pareto --csv
}

mkdir -p ./results

for dataset in $(cat scripts/datasets_under_test.txt); do
    if [ "$do_csv" = true ]; then
        do_benchmark_csv "$dataset"
    else
        do_benchmark "$dataset"
    fi
done

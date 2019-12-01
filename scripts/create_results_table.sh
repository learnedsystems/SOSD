#! /usr/bin/env bash

FOLDER=results

function main() {
  PrintTableHeader
  ParseFile books_200M_uint32_results.txt
  ParseFile fb_200M_uint32_results.txt
  ParseFile lognormal_200M_uint32_results.txt
  ParseFile normal_200M_uint32_results.txt
  ParseFile uniform_dense_200M_uint32_results.txt
  ParseFile uniform_sparse_200M_uint32_results.txt
  ParseFile books_200M_uint64_results.txt
  ParseFile fb_200M_uint64_results.txt
  ParseFile lognormal_200M_uint64_results.txt
  ParseFile normal_200M_uint64_results.txt
  ParseFile osm_cellids_200M_uint64_results.txt
  ParseFile uniform_dense_200M_uint64_results.txt
  ParseFile uniform_sparse_200M_uint64_results.txt
  ParseFile wiki_ts_200M_uint64_results.txt
}

function PrintTableHeader() {
  echo "|               | ART       | B-tree    | BS        | FAST      | IS        | RBS       | RMI       | RS        | TIP       |"
  echo "| ------------- | ---------:| ---------:| ---------:| ---------:| ---------:| ---------:| ---------:| ---------:| ---------:|"
}

function ParseFile() {
  file_name=$1

  cat $FOLDER/$file_name | grep -v Repeating | grep -v read | grep -v "data contains duplicates" | \
  awk -v file_name=$file_name -F'[, ]' '
  # Simple insertion sort (awk does not have a sort, only gawk)
  function my_sort(arr, new_arr) {
    new_arr[0] = arr[0];
    new_len = 1;
    for(i=1; i<repeat_count; i++) {
      for(j=0; j<new_len; j++) {
        if(arr[i] < new_arr[j]) {
          tmp = arr[i];
          arr[i] = new_arr[j];
          new_arr[j] = tmp;
        }
      }
      new_arr[new_len] = arr[i];
      new_len++;
    }
  }

  BEGIN {
    art = "n/a"
    btree = "n/a"
    bs = "n/a"
    fast = "n/a"
    is = "n/a"
    rbs = "n/a"
    rmi = "n/a"
    rs = "n/a"
    tip = "n/a"

    data_set = "unknown"
    if(file_name == "books_200M_uint32_results.txt") data_set = "amzn32"
    if(file_name == "fb_200M_uint32_results.txt") data_set = "face32"
    if(file_name == "lognormal_200M_uint32_results.txt") data_set = "logn32"
    if(file_name == "normal_200M_uint32_results.txt") data_set = "norm32"
    if(file_name == "uniform_dense_200M_uint32_results.txt") data_set = "uden32"
    if(file_name == "uniform_sparse_200M_uint32_results.txt") data_set = "uspr32"

    if(file_name == "books_200M_uint64_results.txt") data_set = "amzn64"
    if(file_name == "fb_200M_uint64_results.txt") data_set = "face64"
    if(file_name == "lognormal_200M_uint64_results.txt") data_set = "logn64"
    if(file_name == "normal_200M_uint64_results.txt") data_set = "norm64"
    if(file_name == "osm_cellids_200M_uint64_results.txt") data_set = "osmc64"
    if(file_name == "uniform_dense_200M_uint64_results.txt") data_set = "uden64"
    if(file_name == "uniform_sparse_200M_uint64_results.txt") data_set = "uspr64"
    if(file_name == "wiki_ts_200M_uint64_results.txt") data_set = "wiki64"
  }

  /RESULT/ {
    # Adjust repeat count
    repeat_count = NF - 3

    # Gather all measurements in
    for(i=0; i<repeat_count; i++) {
      j = i+3
      arr[i] = $j;
    }

    # Sort, get median and round
    my_sort(arr, new_arr)
    result = new_arr[int(repeat_count/2)]
    if(result < 100) {
      result = int(result*10)/10.0
    } else {
      result = int(result)
    }

    # Remember in variable for respective data structure
    if($2 == "ART") art = result
    if($2 == "stx::btree_multimap") btree = result
    if($2 == "BinarySearch") bs = result
    if($2 == "FAST") fast = result
    if($2 == "InterpolationSearch") is = result
    if($2 == "RadixBinarySearch18") rbs = result
    if($2 == "rmi") rmi = result
    if($2 == "RadixSpline") rs = result
    if($2 == "TIP") tip = result
  }

  END {
    printf("| %-13s | %9s | %9s | %9s | %9s | %9s | %9s | %9s | %9s | %9s |\n", data_set, art, btree, bs, fast, is, rbs, rmi, rs, tip);
  }
  '
}

main
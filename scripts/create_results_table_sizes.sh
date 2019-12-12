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
    if(file_name == "books_200M_uint32_results.txt") { data_set = "amzn32"; bit_count = 32; }
    if(file_name == "fb_200M_uint32_results.txt") { data_set = "face32"; bit_count = 32; }
    if(file_name == "lognormal_200M_uint32_results.txt") { data_set = "logn32"; bit_count = 32; }
    if(file_name == "normal_200M_uint32_results.txt") { data_set = "norm32"; bit_count = 32; }
    if(file_name == "uniform_dense_200M_uint32_results.txt") { data_set = "uden32"; bit_count = 32; }
    if(file_name == "uniform_sparse_200M_uint32_results.txt") { data_set = "uspr32"; bit_count = 32; }

    if(file_name == "books_200M_uint64_results.txt") { data_set = "amzn64"; bit_count = 64; }
    if(file_name == "fb_200M_uint64_results.txt") { data_set = "face64"; bit_count = 64; }
    if(file_name == "lognormal_200M_uint64_results.txt") { data_set = "logn64"; bit_count = 64; }
    if(file_name == "normal_200M_uint64_results.txt") { data_set = "norm64"; bit_count = 64; }
    if(file_name == "osm_cellids_200M_uint64_results.txt") { data_set = "osmc64"; bit_count = 64; }
    if(file_name == "uniform_dense_200M_uint64_results.txt") { data_set = "uden64"; bit_count = 64; }
    if(file_name == "uniform_sparse_200M_uint64_results.txt") { data_set = "uspr64"; bit_count = 64; }
    if(file_name == "wiki_ts_200M_uint64_results.txt") { data_set = "wiki64"; bit_count = 64; }

    if(data_set == "unknown") {
      print "Unknown data set, please extend the script."
      exit
    }
  }

  /RESULT/ {
    # Calculate size overhead
    ds_size = $NF
    byte_per_tuple = (bit_count + 64) / 8
    data_size = byte_per_tuple * 200 * 1000 * 1000
    overhead = (ds_size - data_size) / data_size
    if(overhead * 100 >= 10 || overhead * 100 <= -10) {
      result = sprintf("%.3f%%", overhead * 100)
    } else {
      result = sprintf("%.3f%%", overhead * 100)
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

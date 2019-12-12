#! /usr/bin/env bash

FOLDER=results

function CreateTable() {
  ParseFile books_200M_uint32_results.txt 1
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

function ParseFile() {
  file_name=$1
  is_first_file=$2

  cat $FOLDER/$file_name | grep -v Repeating | grep -v read | grep -v "data contains duplicates" | \
  awk -v file_name=$file_name -v is_first_file=$is_first_file -F'[, ]' '
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
    idx_names[0] = "RMI"
    idx_names[1] = "RS"
    idx_names[2] = "ART"
    idx_names[3] = "FAST"
    idx_names[4] = "RBS"
    idx_names[5] = "B-tree"
    idx_names[6] = "BS"
    idx_names[7] = "TIP"
    idx_names[8] = "IS"

    idx_name_mapping["ART"] = "ART"
    idx_name_mapping["stx::btree_multimap"] = "B-tree"
    idx_name_mapping["BinarySearch"] = "BS"
    idx_name_mapping["FAST"] = "FAST"
    idx_name_mapping["InterpolationSearch"] = "IS"
    idx_name_mapping["RadixBinarySearch18"] = "RBS"
    idx_name_mapping["rmi"] = "RMI"
    idx_name_mapping["RadixSpline"] = "RS"
    idx_name_mapping["TIP"] = "TIP"

    if(is_first_file) {
      printf("| %-13s |", "")
      for(i=0; i<length(idx_names); i++) {
        printf("%10s |", idx_names[i])
      }
      print ""
      print "| ------------- | ---------:| ---------:| ---------:| ---------:| ---------:| ---------:| ---------:| ---------:| ---------:|"
    }

    # Translate file name into nice date set name to be displayed in the table
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

    if(data_set == "unknown") {
      print "Unknown data set, please extend the script."
      exit
    }
  }

  /RESULT/ {
    if($2 == "Oracle") next

    # Check if this index exists
    if(length(idx_name_mapping[$2]) == 0) {
      print "Sorry index does not exist!"
      exit
    }

    # Adjust repeat count
    repeat_count = NF - 3

    # Gather all measurements
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
    idx_result[idx_name_mapping[$2]] = result
  }

  END {
    printf("| %-13s |", data_set)
    for(i=0; i<length(idx_names); i++) {
      res = idx_result[idx_names[i]]
      printf("%10s |", length(res) == 0 ? "n/a" : res)
    }
    print ""
  }
  '
}

function AddAverageLine() {
  CreateTable | awk -F '|' '
    BEGIN {
      line = 0
    }

    # index_name := holds the name of the index in column i
    # index_sum := sums up the lookup times for index in column i
    # index_count := counts how often the lookup time was not "n/a"
    {
      line++

      # Remember the index names
      if(line == 1) {
        for(i=3; i<NF; i++) {
          index_names[i] = $i
        }
        next
      }

      # Skip header line of table
      if(line == 2) {
        next
      }

      # Parse regular result line
      for(i=3; i<NF; i++) {
        if($i ~ /n\/a/) {
          continue
        }

        index_sum[i] += $i
        index_count[i] ++
      }
    }

    END {
      print "| ------------- | ---------:| ---------:| ---------:| ---------:| ---------:| ---------:| ---------:| ---------:| ---------:|"
      printf "| avg           |"
      for(i in index_names) {
        res = (index_sum[i] / index_count[i])
        printf("%10i |", res)
      }
      print ""
    }
    '
}

CreateTable
AddAverageLine

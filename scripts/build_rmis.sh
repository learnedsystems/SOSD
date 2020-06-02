#! /usr/bin/env bash
git submodule update --init --recursive

mkdir -p rmi_data

function build_rmi() {
    DATA_NAME=$1
    HEADER1_NAME=${DATA_NAME}_rmi.h
    HEADER2_NAME=${DATA_NAME}_rmi_data.h
    C_NAME=${DATA_NAME}_rmi.cpp
    HEADER_PATH=competitors/rmi/$HEADER1_NAME

    shift 1
    if [ ! -f $HEADER_PATH ]; then
        echo "Building RMI for $DATA_NAME ($INCR)"
        RUST_BACKTRACE=1 RUST_LOG=trace RMI/target/release/rmi data/$DATA_NAME ${DATA_NAME}_rmi $@ -d rmi_data/ -e
        mv $HEADER1_NAME competitors/rmi/
        mv $HEADER2_NAME competitors/rmi/
        mv $C_NAME competitors/rmi/
    fi
}

function build_rmi_set() {
    DATA_NAME=$1
    HEADER_PATH=competitors/rmi/${DATA_NAME}_0.h
    JSON_PATH=scripts/rmi_specs/${DATA_NAME}.json

    shift 1
    if [ ! -f $HEADER_PATH ]; then
        echo "Building RMI set for $DATA_NAME"
        RMI/target/release/rmi data/$DATA_NAME --param-grid ${JSON_PATH} -d rmi_data/
        mv ${DATA_NAME}_* competitors/rmi/
    fi
}


cd RMI && cargo build --release && cd ..

build_rmi_set fb_200M_uint64
build_rmi_set wiki_ts_200M_uint64

build_rmi_set osm_cellids_200M_uint64
build_rmi_set osm_cellids_400M_uint64
build_rmi_set osm_cellids_600M_uint64
build_rmi_set osm_cellids_800M_uint64

build_rmi_set books_200M_uint64
build_rmi_set books_400M_uint64
build_rmi_set books_600M_uint64
build_rmi_set books_800M_uint64

build_rmi_set books_200M_uint32


scripts/rmi_specs/gen.sh


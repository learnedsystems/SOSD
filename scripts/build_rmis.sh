#! /usr/bin/env bash
git submodule update --init --recursive


function build_rmi() {
    DATA_NAME=$1
    HEADER1_NAME=${DATA_NAME}_rmi.h
    HEADER2_NAME=${DATA_NAME}_rmi_data.h
    C_NAME=${DATA_NAME}_rmi.cpp
    HEADER_PATH=competitors/rmi/$HEADER1_NAME

    shift 1
    if [ ! -f $HEADER_PATH ]; then
        echo "Building RMI for $DATA_NAME"
        RUST_BACKTRACE=1 RUST_LOG=trace RMI/target/release/rmi data/$DATA_NAME ${DATA_NAME}_rmi $@
        mv $HEADER1_NAME competitors/rmi/
        mv $HEADER2_NAME competitors/rmi/
        mv $C_NAME competitors/rmi/
    fi
}

cd RMI && cargo build --release && cd ..
build_rmi normal_200M_uint32         radix,linear_spline      262144
build_rmi normal_200M_uint64         radix,linear_spline      262144
build_rmi lognormal_200M_uint32      radix,linear_spline      2097152
build_rmi lognormal_200M_uint64      histogram,cubic,linear   300     -e
build_rmi osm_cellids_200M_uint64    bradix,linear            2097152 -e
build_rmi wiki_ts_200M_uint64        bradix,linear            2097152 -e
build_rmi books_200M_uint32          bradix,linear            16384   -e
build_rmi books_200M_uint64          bradix,linear            32768   -e
build_rmi fb_200M_uint64             linear,linear            1000000 -e
build_rmi fb_200M_uint32             linear,linear            1000000 -e
build_rmi uniform_dense_200M_uint32  linear                   1
build_rmi uniform_dense_200M_uint64  linear                   1
build_rmi uniform_sparse_200M_uint64 linear,linear            500000
build_rmi uniform_sparse_200M_uint32 linear,linear            500000


#!/usr/bin/env bash
# prof.sh OUTDIR CORPUS CASE T : perf record cycles (+ page-faults) pinned to idle cores of one L3 domain.
out=$1; corpus=$2; case=$3; t=$4
root=$(cd "$(dirname "$0")/../../../../../../.." && pwd)
cpus=$(python3 $root/benchmarks/scripts/idle_cpus.py pick $t --seconds 2) || exit 1
tag=$case-${t}t
export BENCH_RUNS=${BENCH_RUNS:-60} BENCH_CASE=$case
bin="$root/target/release/contract --threads $t --corpus $corpus"
taskset -c $cpus perf record -q -F 4999 -o $out/$tag.data -- $bin > $out/$tag.csv 2> $out/$tag.err
taskset -c $cpus perf record -q -e page-faults -c 1 -o $out/$tag.pf.data -- $bin > /dev/null 2>> $out/$tag.err
perf script -i $out/$tag.data -F tid,time,ip,sym 2>/dev/null | gzip > $out/$tag.cycles.gz
perf script -i $out/$tag.pf.data -F tid,time,ip,sym 2>/dev/null | gzip > $out/$tag.pf.gz
rm -f $out/$tag.data $out/$tag.pf.data
echo "$tag cpus=$cpus" >> $out/cpus.txt

#!/usr/bin/env bash
# Per case: pick one idle CPU, run new then base on that same CPU back to back.
set -u
T=/home/shinaoka/tensor4all; N=$T/tprims-rs-smoke; B=$T/tprims-rs-smoke-base
C=$N/benchmarks/benchmarks/tprims/corpus
R=$N/benchmarks/benchmarks/tprims/contract/results/2026-10-02-phase1-smoke/hadamard-rerun-8f430b6
export PINNED_RETRIES=1 PINNED_WAIT=1
runone() { # side root bin corpus case cpu
  local CORP=$7; local side=$1 root=$2 bin=$3 f=$4 c=$5 cpu=$6 tmp
  tmp=$(mktemp)
  if BENCH_CASE=$c $N/benchmarks/scripts/pinned.sh $cpu -- $root/target/release/$bin --threads 1 --corpus $CORP/$f.json > $tmp 2>>$R/pinned.err; then
    grep -v -E '^(#|case,|CHECK)' $tmp >> $R/$side-$f.csv
    grep -E '^(#|CHECK)' $tmp | sed "s/^/[$c cpu=$cpu] /" >> $R/$side-$f.log
    rm -f $tmp; return 0
  fi
  rm -f $tmp; return 1
}
for f in hadamard; do
  for s in new base; do echo "case,variant,threads,median_ns,samples" > $R/$s-$f.csv; : > $R/$s-$f.log; done
  bbin=contract; [[ $f == *gemm* ]] && bbin=blas
  mapfile -t cases < <($N/target/release/contract --list --corpus $C/$f.json)
  for c in "${cases[@]}"; do
    ok=0
    for attempt in $(seq 1 60); do
      cpu=$(python3 $N/benchmarks/scripts/idle_cpus.py pick 1 --seconds 2 2>/dev/null) || { sleep 5; continue; }
      ln=$(wc -l < $R/new-$f.csv); lb=$(wc -l < $R/base-$f.csv)
      if runone new $N contract $f $c $cpu $C && runone base $B $bbin $f $c $cpu $B/benchmarks/benchmarks/tprims/corpus; then ok=1; echo "$f,$c,$cpu,attempt$attempt" >> $R/cpus.csv; break; fi
      head -n $ln $R/new-$f.csv > $R/.t && mv $R/.t $R/new-$f.csv
      head -n $lb $R/base-$f.csv > $R/.t && mv $R/.t $R/base-$f.csv
      sed -i "/^\[$c cpu=/d" $R/new-$f.log $R/base-$f.log
    done
    [[ $ok == 1 ]] || echo "$f,$c,SKIPPED" >> $R/skipped.csv
  done
done
echo DONE > $R/done.flag

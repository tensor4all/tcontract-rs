@CLAUDE.md

## Critical Rules

- **NEVER run benchmarks in parallel.** Run them sequentially (1 thread first, then 4 threads, etc.); concurrent runs interfere and produce misleading results.
- **Record the exact tcontract-rs commit for every published result table**, next to the table, together with the CPU, core set and profile.
- **Campaign results go to [tcontract-benchmark](https://github.com/tensor4all/tcontract-benchmark), not here.**
  That repository owns the revision pin, the run manifests, the published
  reports and the staleness index; it builds `tcbench` from a pinned checkout of
  this one. `benchmarks/benchmarks/tcbench/results/` holds the raw runs cited as
  evidence by the PRs that produced them and is frozen. An ad-hoc run recorded
  here, or a table copied out of a campaign report without its commit, is not
  evidence.
- **Always pin CPU cores with `taskset` on Linux, including 1T**, inside one L3/CCD domain (`lscpu -e` maps cores to L3). Check `ps -eo pid,psr,%cpu,comm --sort=-%cpu | head` for busy cores first and avoid their CCD. On macOS, state that pinning was unavailable.

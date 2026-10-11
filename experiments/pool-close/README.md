# Rayon pool close: exit handler vs join

**Question:** When a pool owned by a C execution handle is closed, what marks the point after which no worker code runs: the Rayon exit handler, or joining the worker threads?

Found by the audit in [#1](https://github.com/tensor4all/tcontract-rs/issues/1). This is a correctness probe, not a timing experiment.

## Setup

| Item | Value |
| --- | --- |
| Date | 2026-09-29 |
| Machine | Apple M5 Max, macOS 26.5.1, rustc 1.96.0, release profile |
| Rayon | rayon 1.11.0, rayon-core 1.13.0 (`Cargo.lock`) |

```sh
cargo run --release --locked
```

`src/main.rs` builds a one-worker pool with `spawn_handler` (keeping the `JoinHandle`) and an `exit_handler`, installs a worker thread-local whose destructor signals that it started and then blocks until the host releases it, and drops the pool. Every wait is bounded (2 s), so a hang fails instead of stalling.

## Result

```text
exit handler signaled; TLS destructor is still running
join still waiting while TLS destructor blocks: OK
join returned only after TLS destructor finished: OK
```

The exit handler runs inside the worker's main loop (`rayon-core/src/registry.rs`, `main_loop`), before thread-local teardown, so it cannot back a "no worker code is running" guarantee. Joining the handle kept from `spawn_handler` can. The probe also checks that `ThreadPool::current_thread_index` is `None` on the host thread and `Some` on a worker, which is the self-join guard.

## Decision taken from this result

Recorded in the [architecture](../../docs/architecture.md#execution-context) and [decision log](../../docs/decision-log.md): close of an owned pool joins its workers; close from one of its own workers returns an error; closing a borrowed context only detaches.

# Sharing a host-owned Rayon pool

## Decision

For tenferro's CPU integration, add `Pool::shared(Arc<rayon::ThreadPool>)` rather than a raw-pool getter or a second worker team. The host retains its existing raw Arc for tlinalg/strided/FFT and one persistent tcontract wrapper for contractions. Clones share that wrapper, preserving its arena, counters and SPMD mutex; plans still own no resources.

A per-operation `Pool::borrow` would discard workspace reuse. Keeping that borrowed wrapper alongside its owning Arc would require self-reference. Shared stdlib ownership avoids both without a new executor or registry. Independent pre-review approved the shared constructor and rejected the getter alternative.

## Contract and verification

The underlying pool, dispatch policies and worker-context refusal remain unchanged. Serialization is wrapper-local, not a guarantee for arbitrary independent wrappers or unrelated raw-pool work. Hosts must serialize potentially conflicting numerical resource loans.

The focused check verifies supplied worker IDs, selected-pool execution, width-one caller execution, stable arena identity under wrapper Arc clones, `Send + Sync`, existing same-pool broadcast refusal and Arc ownership retention. Weak-pointer expiry is not evidence of worker shutdown completion. No numerical or performance improvement is claimed; the full CPU integration performance protocol remains separate.

Local gates passed: workspace debug and CI-profile tests, standalone exec tests, release exec and contract tests, all-target clippy, warning-denied docs, Rust 1.89 workspace/all-target build, README example and benchmark-info smoke, prebuilt/installed C consumers, script checks and aarch64-Apple cross-check. The aarch64-Linux check could not run because that standard-library target is not installed; no Linux cross-test claim is made. Independent post-review found no demonstrated blocker or correction within the ownership/dispatch/test/documentation scope. Actual numerical arena-buffer reuse and cross-library integration still belong to downstream validation.

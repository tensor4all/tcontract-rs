# t4a-cost

Neutral cost-model types for the tensor4all stack. A provider of a batched
kernel prices its work with three things, and none of them needs the stack the
work runs on: the host's CPU class, the time of one item at a width, and the
fitted row behind that time. They live here, with no dependency at all, so
libraries can share a cost model without one depending on the other
(tensor4all/tcontract-rs#205, consumed by tensor4all/tlinalg-rs#119).

| Item | Role |
| --- | --- |
| `host::CpuClass` | The CPU class (vendor, family, model, logical CPUs from `/proc/cpuinfo`, Linux) a provider keys its offline-fitted rows by; `CpuClass::detect()` reads it once per process. |
| `ItemTime` | A provider's time for one item at a width, without the token's dispatch cost. The lane decision that minimises it is `tcontract-exec`'s, which re-exports this type for its 0.13 API. |
| `ItemFit` | One fitted row of the generic item-time fit (`experiments/faer-item-width/item_fit.py` in tcontract-rs): the six parameters every provider fits the same way, and the one `ItemFit::item_time(muladds, tasks, largest_share, task_ns, slow_below)` rule that turns them into an `ItemTime`. The work, the tasks (their count, the largest's share and a cost per round) and the slow band are the provider's features, supplied per item. |

Nothing here starts a task, owns a pool or allocates scratch: placement is
[`threadteam`](https://crates.io/crates/threadteam)'s, and a fit is data
committed per host class, never a run-time calibration. The runnable example
of a row and its `item_time` is the rustdoc of `ItemFit` (doc-tested with the
crate's tests).

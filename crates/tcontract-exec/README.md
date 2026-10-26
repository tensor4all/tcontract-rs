# tcontract-exec

Execution resources of the tcontract stack: scratch memory and contraction
policy. It starts no task and exposes no scheduler type. Where tasks run, and how
many at once, belongs to [`threadteam`](https://crates.io/crates/threadteam)
(formerly `strided-exec`); this crate adds what is specific to contraction.

| Item | Role |
| --- | --- |
| `Pool<'p>` | A `threadteam::Pool` (borrowed with `Pool::borrow`, owned with `Pool::owned`, or shared by several wrappers with `Pool::shared`) plus the retained scratch arena. One wrapper is one workspace owner: keep it across calls. |
| `Exec<'a>` | The per-call resource bundle: one `threadteam::Exec` token, an optional workspace, and the `BatchPolicy`. Not `Clone`, not `Copy`; operations take `&mut Exec`. |
| `ArenaProvider`, `Workspace`, `WorkspaceReq`, `TeamLease` | Reusable scratch, owned by a `Pool` or lent by a serial caller. |
| `BatchPolicy`, `BatchQuery`, `BatchDecision`, `batch_decision`, `batch_decision_within`, `WorkModel` | The one lane decision (`lanes * item_threads <= budget`) every batch-owning operation shares; its widths are the token's. |
| `Exec::width_for(ns, sections)`, `Exec::width_for_within`, `Exec::pays(ns, sections, width)` | The partition width, and the gate, for work estimated at `serial_ns` that performs `sections` dispatches: the token's rule (`threadteam`'s dispatch cost of the token's kind, times `PAYS`). |

## `Exec`

```rust
use tcontract_exec::{Exec, Pool};

let sp = threadteam::Pool::builder().threads(4).build()?;   // the host owns the pool
let pool = Pool::borrow(&sp);                                  // wrapper: pool + scratch arena
let mut exec = Exec::new(&pool, 4)?;                           // the only budget clamp
exec.token().for_ranges(1000, 64.try_into()?, |_range| {});    // placement is the token's
# Ok::<(), Box<dyn std::error::Error>>(())
```

Constructors: `Exec::serial()` and `Exec::serial_with_workspace(ws)` need no
pool; `Exec::new(&pool, budget)` mints the token from a pool (budget zero is an
error, a budget above the pool width is clamped) and lends the pool's workspace.
Accessors: `budget()`, `workspace()`, `batch_policy()`, `with_batch_policy(p)`,
`batch_decision(&query)`, `width_for(ns, sections)`, `pays(ns, sections, width)`, `limit(max)`, `token_ref()`,
`reborrow(&mut self)`.

Placement is the token's. Take it with `exec.token()` (a reborrow, so one `Exec`
serves a sequence of calls) or `exec.into_token()`, and use its `fork`,
`for_ranges`, `broadcast` and `external`; kernels of strided-rs take the token as
their last argument. `fork` hands out bare child tokens, and
`Exec::from_token(child, workspace, policy)` makes the bundle for a job.

### Naming

`tcontract_exec::Exec` keeps its name, and `Pool` keeps `borrow` and `owned`, so
cpueinsum, tlinalg and tenferro change their imports as little as possible.
What changes for them: operations take `&mut Exec` (a token cannot be copied),
`Exec::rayon(&pool).with_budget(n)` becomes `Exec::new(&pool, n)?`,
`install`/`for_each_partition`/`broadcast`/`rayon_pool`/`PoolStats` are gone (use
the token), and a pool is a `threadteam::Pool`, never a Rayon type.

## What is checked

- `cargo deny check bans` rejects a direct `rayon` dependency anywhere in the
  workspace (vendor wrappers listed in `deny.toml`).
- `scripts/check-no-public-rayon.sh` (`cargo public-api`) fails when `rayon::`
  appears in this crate's public API.
- Tests read the `threadteam` ledger: `max_live(root) <= budget`, pool entries
  per call, and `global_pool_unstarted()` in a binary of its own.

//! Enforced thread counts for benchmark binaries.

use tcontract_exec::{ArenaProvider, Exec, Pool};

/// Thread environment variables that must agree with `--threads`.
pub const THREAD_ENV_VARS: [&str; 3] = [
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
];

/// The first variable in [`THREAD_ENV_VARS`] that is set to a value other
/// than `requested`, with that value.
pub fn conflicting_env(
    requested: usize,
    get: impl Fn(&str) -> Option<String>,
) -> Option<(String, String)> {
    THREAD_ENV_VARS.iter().find_map(|name| {
        let value = get(name)?;
        (value.trim() != requested.to_string()).then(|| (name.to_string(), value))
    })
}

/// Name prefixes of the environment variables the libraries read before
/// tcontract-rs#37 and no longer do. A benchmark that finds one set would silently
/// measure something other than what the caller asked for, so it refuses to
/// run: use the benchmark's own flags, or `tcbench`, whose knobs are `TCBENCH_*`.
pub const REMOVED_ENV_PREFIXES: [&str; 2] = ["TENSORCONTRACT_", "TPRIMS_GEMM_"];

/// The first variable of `vars` with a [`REMOVED_ENV_PREFIXES`] prefix, with its value.
pub fn forbidden_env(vars: impl IntoIterator<Item = (String, String)>) -> Option<(String, String)> {
    vars.into_iter()
        .find(|(name, _)| REMOVED_ENV_PREFIXES.iter().any(|p| name.starts_with(p)))
}

/// Parse `--threads N` from `args` (default 1).
pub fn parse_threads(args: &[String]) -> Result<usize, String> {
    match args.iter().position(|a| a == "--threads") {
        None => Ok(1),
        Some(i) => {
            let v = args.get(i + 1).ok_or("--threads needs a value")?;
            let n: usize = v.parse().map_err(|_| format!("bad --threads {v}"))?;
            if n == 0 {
                Err("--threads must be at least 1".into())
            } else {
                Ok(n)
            }
        }
    }
}

/// A benchmark's thread configuration: a serial `Exec` that lends the
/// caller-owned arena for one thread, a `strided_exec::Pool` borrowed through
/// `Exec` otherwise.
pub struct BenchThreads {
    /// Requested thread count.
    pub requested: usize,
    pool: Option<strided_exec::Pool>,
    /// The plan owns no scratch any more, so the 1T rows own exactly the
    /// steady-state storage they measure and no other row can see it.
    arena: ArenaProvider,
}

impl BenchThreads {
    /// From the process arguments and environment; exits with status 2 on a
    /// bad flag or a conflicting thread environment variable.
    pub fn from_args() -> Self {
        let args: Vec<String> = std::env::args().collect();
        let requested = parse_threads(&args).unwrap_or_else(|e| fail(&e));
        if let Some((name, value)) = conflicting_env(requested, |n| std::env::var(n).ok()) {
            fail(&format!(
                "{name}={value} conflicts with --threads {requested}"
            ));
        }
        if let Some((name, value)) = forbidden_env(std::env::vars()) {
            fail(&format!(
                "{name}={value}: this variable was removed in tcontract-rs#37 and is not read; use the benchmark's flags"
            ));
        }
        // tcontract paths run on this pool and start no ambient one.
        let pool = (requested > 1).then(|| {
            strided_exec::Pool::builder()
                .threads(requested)
                .build()
                .unwrap_or_else(|e| fail(&format!("pool: {e}")))
        });
        Self {
            requested,
            pool,
            arena: ArenaProvider::new(),
        }
    }

    /// Run `f` with the configured context and, for multi-thread runs, the
    /// borrowed pool (for its size).
    pub fn with_exec<R>(&self, f: impl FnOnce(&mut Exec<'_>, Option<&Pool<'_>>) -> R) -> R {
        match &self.pool {
            None => f(&mut Exec::serial_with_workspace(&self.arena), None),
            Some(sp) => {
                let pool = Pool::borrow(sp);
                let mut exec =
                    Exec::new(&pool, pool.size()).unwrap_or_else(|e| fail(&format!("budget: {e}")));
                f(&mut exec, Some(&pool))
            }
        }
    }

    /// Print `# threads: ...` and assert the effective width equals the
    /// request.
    pub fn verify(&self) {
        self.with_exec(|exec, pool| {
            let size = pool.map_or(1, Pool::size);
            println!(
                "# threads: requested={} pool={} budget={}",
                self.requested,
                size,
                exec.budget()
            );
            assert_eq!(exec.budget(), self.requested, "effective budget");
            assert_eq!(size, self.requested, "pool size");
        });
    }
}

fn fail(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(2)
}

#[cfg(test)]
mod tests;

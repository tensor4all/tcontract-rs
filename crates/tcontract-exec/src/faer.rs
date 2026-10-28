//! The parallelism faer gets from a placement token (feature `faer`).

/// Runs `f` with the parallelism faer should use for `token`.
///
/// A token that carries a team (minted by a `threadteam::Team`, or a child of
/// one) and has more than one slot hands faer its own team as `Par::Team`:
/// faer's regions run on the reserved members as team sections, at the
/// dispatch cost the floors were derived from. Any other token runs faer with
/// `Par::Seq` on the calling thread and never enters a pool: only a team gives
/// faer internal parallelism. Batch lanes still fork through `threadteam`
/// before this point, so batch-level parallelism stays.
///
/// # Examples
///
/// ```
/// use std::num::NonZeroUsize;
/// let pool = threadteam::Pool::builder()
///     .threads(2)
///     .acquire_timeout(std::time::Duration::from_secs(10))
///     .build()
///     .unwrap();
/// // A token without a team runs faer sequentially.
/// let token = threadteam::Exec::new(&pool, 2).unwrap();
/// assert!(tcontract_exec::with_faer_par(token, |par| matches!(par, faer::Par::Seq)));
/// // A team token of two slots hands faer the team.
/// let team = pool.into_team(2).unwrap();
/// let token = team.exec(2);
/// assert!(tcontract_exec::with_faer_par(token, |par| matches!(par, faer::Par::Team(_))));
/// // One slot is sequential even on a team.
/// let one = team.exec(2).limit(NonZeroUsize::MIN);
/// assert!(tcontract_exec::with_faer_par(one, |par| matches!(par, faer::Par::Seq)));
/// ```
pub fn with_faer_par<R: Send>(
    mut token: threadteam::Exec<'_>,
    f: impl FnOnce(faer::Par<'_>) -> R + Send,
) -> R {
    if token.budget().get() > 1 && token.carries_team() {
        // A carried team is handed over without acquisition, so `team` never
        // refuses it.
        return token
            .team(|team| f(faer::Par::Team(team)))
            .expect("a carried team is never refused");
    }
    f(faer::Par::Seq)
}

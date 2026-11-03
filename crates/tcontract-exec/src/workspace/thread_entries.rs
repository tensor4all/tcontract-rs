use super::*;

fn thread_slots(provider: &ArenaProvider) -> usize {
    lock(&provider.state).threads.len()
}

fn trace_records(owner: u64) -> usize {
    lock(&TRACE)
        .iter()
        .filter(|(key, _, _)| *key == owner)
        .count()
}

#[test]
fn one_entry_per_thread_and_none_carried_over() {
    let req = WorkspaceReq::default();
    let provider = ArenaProvider::default();
    assert_eq!(thread_slots(&provider), 0);
    provider.with_worker(&req, &mut |_, _, _| {});
    assert_eq!(thread_slots(&provider), 1);
    provider.with_worker(&req, &mut |_, _, _| {});
    assert_eq!(thread_slots(&provider), 1);

    // The map belongs to the provider, so a new one never inherits an
    // entry: nothing outlives the owner that created it.
    let fresh = ArenaProvider::default();
    assert_eq!(thread_slots(&fresh), 0);
    fresh.with_worker(&req, &mut |_, _, _| {});
    assert_eq!(thread_slots(&fresh), 1);
    assert_eq!(thread_slots(&provider), 1);
}

#[test]
fn two_threads_get_two_entries() {
    let provider = ArenaProvider::default();
    let barrier = std::sync::Barrier::new(2);
    std::thread::scope(|scope| {
        for _ in 0..2 {
            scope.spawn(|| {
                provider.with_worker(&WorkspaceReq::default(), &mut |_, _, _| {});
                barrier.wait();
            });
        }
    });
    assert_eq!(thread_slots(&provider), 2);
}

#[test]
fn dropping_a_traced_provider_drops_its_trace_records() {
    let provider = ArenaProvider::traced();
    let key = provider.key;
    provider.with_worker(
        &WorkspaceReq {
            a_bytes: 4096,
            ..Default::default()
        },
        &mut |_, _, _| {},
    );
    assert!(trace_records(key) > 0);
    drop(provider);
    assert_eq!(
        trace_records(key),
        0,
        "a dropped traced provider must not keep records in the process-global log"
    );
}

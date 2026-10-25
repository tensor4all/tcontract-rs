//! Serial and pooled use of `Exec` never starts Rayon's process-global pool.
//! One test, alone in its binary: the check starts the global pool itself and
//! passes only when nothing earlier did.
use tcontract_exec::{Exec, Pool};

#[test]
fn serial_and_pooled_exec_leave_the_global_pool_unstarted() {
    let mut serial = Exec::serial();
    serial.token().fork(4, |_, _| {});
    serial
        .token()
        .for_ranges(100, 1.try_into().unwrap(), |_| {});
    serial.token().broadcast(|_| {}).unwrap();
    assert_eq!(serial.token().external(|w| w.get()), 1);

    let sp = strided_exec::Pool::builder().threads(3).build().unwrap();
    let pool = Pool::borrow(&sp);
    let mut exec = Exec::new(&pool, 3).unwrap();
    exec.token().fork(4, |_, _| {});
    exec.token().for_ranges(100, 1.try_into().unwrap(), |_| {});
    exec.token().broadcast(|_| {}).unwrap();
    assert_eq!(exec.token().external(|w| w.get()), 3);

    assert!(
        strided_exec::ledger::global_pool_unstarted(),
        "something started the process-global Rayon pool"
    );
}

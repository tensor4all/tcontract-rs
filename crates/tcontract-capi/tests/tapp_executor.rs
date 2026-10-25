//! Executors under the TAPP product entry points: plans are reusable across
//! executors, width comes from the executor alone, small work stays on the
//! caller.
mod raw;

use raw::*;
use tcontract_capi::*;

struct Problem {
    m: usize,
    n: usize,
    k: usize,
    a: Vec<f64>,
    b: Vec<f64>,
    plan: isize,
}

impl Problem {
    fn new(m: usize, n: usize, k: usize) -> Self {
        let (ea, eb, ed) = (
            [m as i64, k as i64],
            [k as i64, n as i64],
            [m as i64, n as i64],
        );
        let (sa, sb, sd) = (cm(&ea), cm(&eb), cm(&ed));
        let (la, lb, ld) = ([I, K], [K, J], [I, J]);
        let (rc, plan) = unsafe {
            plan_f64(
                (&ea, &sa, &la),
                (&eb, &sb, &lb),
                (&ed, &sd, &ld),
                (&ed, &sd, &ld),
            )
        };
        assert_eq!(rc, TAPP_SUCCESS);
        Self {
            m,
            n,
            k,
            a: seq(m * k, 1),
            b: seq(k * n, 2),
            plan,
        }
    }

    fn run(&self, exec: isize) -> Vec<f64> {
        let mut d = vec![f64::NAN; self.m * self.n];
        let rc = unsafe {
            exec_f64(
                self.plan,
                exec,
                1.0,
                self.a.as_ptr(),
                self.b.as_ptr(),
                0.0,
                std::ptr::null(),
                d.as_mut_ptr(),
            )
        };
        assert_eq!(rc, TAPP_SUCCESS);
        d
    }

    fn want(&self) -> Vec<f64> {
        naive(self.m, self.n, self.k, &self.a, &self.b)
    }
}

impl Drop for Problem {
    fn drop(&mut self) {
        unsafe { TAPP_destroy_tensor_product(self.plan) };
    }
}

#[test]
fn one_plan_runs_on_default_serial_and_four_thread_executors() {
    let p = Problem::new(200, 190, 180);
    let want = p.want();
    unsafe {
        let serial = serial_exec();
        let four = rayon_exec(4);
        let one = rayon_exec(1);
        let two = rayon_exec(2);
        for exec in [0, serial, four, one, two] {
            assert_close(&p.run(exec), &want, 1e-12);
        }
        // A different pair of buffers through the same plan and executor.
        let q = Problem {
            a: seq(200 * 180, 7),
            b: seq(180 * 190, 8),
            plan: 0,
            ..Problem::new(200, 190, 180)
        };
        let mut d = vec![0.0; 200 * 190];
        assert_eq!(
            exec_f64(
                p.plan,
                four,
                1.0,
                q.a.as_ptr(),
                q.b.as_ptr(),
                0.0,
                std::ptr::null(),
                d.as_mut_ptr()
            ),
            TAPP_SUCCESS
        );
        assert_close(&d, &q.want(), 1e-12);
        for e in [serial, four, one, two] {
            assert_eq!(TAPP_destroy_executor(e), TAPP_SUCCESS);
        }
    }
}

/// Where the work runs is `exec_placement.rs`'s subject (it reads the
/// placement ledger); here the product is right at every width, and a one-thread
/// or default executor has no pool at all.
#[test]
fn large_and_small_work_are_right_on_the_executors_pool() {
    unsafe {
        let four = rayon_exec(4);
        assert_eq!(pool_width(four), 4);
        let big = Problem::new(256, 256, 256);
        let want = big.want();
        assert_close(&big.run(four), &want, 1e-12);

        let small = Problem::new(4, 4, 4);
        for _ in 0..10 {
            assert_close(&small.run(four), &small.want(), 1e-12);
        }

        // Explicit 1T and the default executor never have a pool to enter.
        let one = rayon_exec(1);
        assert_eq!((pool_width(one), pool_width(0)), (0, 0));
        assert_close(&big.run(one), &want, 1e-12);
        assert_close(&big.run(0), &want, 1e-12);
        assert_eq!(TAPP_destroy_executor(one), TAPP_SUCCESS);
        assert_eq!(TAPP_destroy_executor(four), TAPP_SUCCESS);
    }
}

#[test]
fn width_is_the_executors_not_the_environment() {
    unsafe {
        let big = Problem::new(256, 256, 256);
        let want = big.want();
        // The environment asks for one thread; the 4T executor still decides.
        std::env::set_var("TENSORCONTRACT_THREADS", "1");
        std::env::set_var("RAYON_NUM_THREADS", "1");
        let four = rayon_exec(4);
        assert_close(&big.run(four), &want, 1e-12);
        // The environment asks for many; a budget of one keeps the call serial.
        std::env::set_var("TENSORCONTRACT_THREADS", "16");
        assert_eq!(tcontract_tapp_executor_set_budget(four, 1), TAPP_SUCCESS);
        assert_close(&big.run(four), &want, 1e-12);
        std::env::remove_var("TENSORCONTRACT_THREADS");
        std::env::remove_var("RAYON_NUM_THREADS");
        assert_eq!(TAPP_destroy_executor(four), TAPP_SUCCESS);
    }
}

#[test]
fn plans_sharing_an_executor_share_its_pool() {
    unsafe {
        let four = rayon_exec(4);
        let p1 = Problem::new(256, 256, 256);
        let p2 = Problem::new(240, 250, 260);
        assert_close(&p1.run(four), &p1.want(), 1e-12);
        assert_close(&p2.run(four), &p2.want(), 1e-12);
        assert_eq!(tcontract_tapp_executor_set_budget(four, 4), TAPP_SUCCESS);
        let (mut pool, mut budget) = (0usize, 0usize);
        assert_eq!(
            tcontract_tapp_executor_get_threads(four, &mut pool, &mut budget),
            TAPP_SUCCESS
        );
        assert_eq!((pool, budget), (4, 4));
        assert_eq!(TAPP_destroy_executor(four), TAPP_SUCCESS);
    }
}

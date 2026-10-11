//! C's write-only D contract includes overwrite and separate-C accumulation.
mod raw;

use core::mem::MaybeUninit;
use raw::*;
use tcontract_capi::*;

#[test]
fn row_major_matvec_writes_fresh_d_with_and_without_separate_c() {
    // Faer's row-major matvec Replace path forms destination references; the
    // C plan must therefore prepare and use its packed fresh-output alternative.
    let (ea, eb, ed) = ([4, 2], [2, 1], [4, 1]);
    let (sa, sb, sd) = ([2, 1], [1, 1], [1, 1]);
    let (la, lb, ld) = ([I, K], [K, J], [I, J]);
    // SAFETY: descriptors and live inputs below cover every declared element;
    // D is writable, and successful execution initializes all four elements.
    unsafe {
        let (rc, plan) = plan_f64(
            (&ea, &sa, &la),
            (&eb, &sb, &lb),
            (&ed, &sd, &ld),
            (&ed, &sd, &ld),
        );
        assert_eq!(rc, TAPP_SUCCESS);
        let exec = serial_exec(); // Explicit 1T executor.
        let a = [1., 2., 3., 4., 5., 6., 7., 8.];
        let b = [2., 3.];
        let c = [10., 20., 30., 40.];
        for (beta, cp, expected) in [
            (0., core::ptr::null(), [16., 36., 56., 76.]),
            (0.5, c.as_ptr(), [21., 46., 71., 96.]),
        ] {
            let mut d = [MaybeUninit::<f64>::uninit(); 4];
            assert_eq!(
                exec_f64(
                    plan,
                    exec,
                    2.,
                    a.as_ptr(),
                    b.as_ptr(),
                    beta,
                    cp,
                    d.as_mut_ptr().cast()
                ),
                TAPP_SUCCESS
            );
            assert_eq!(d.map(|x| x.assume_init()), expected);
        }
        assert_eq!(TAPP_destroy_executor(exec), TAPP_SUCCESS);
        assert_eq!(TAPP_destroy_tensor_product(plan), TAPP_SUCCESS);
    }
}

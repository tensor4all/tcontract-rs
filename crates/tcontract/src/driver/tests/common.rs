//! Shared operand construction for the family tests.
//!
//! Included by more than one test binary, so each binary sees a subset of it.
//! One small, non-trivial GEMM per registered family, run against the
//! reference oracle. The sizes are chosen just past a register tile so an
//! edge tile and second `MR`/`NR` block are always exercised, and the
//! reduction is cut into two `KC` blocks so the accumulate path runs too.
#![allow(dead_code)]
//! edge tile and second `MR`/`NR` block are always exercised, and the
//! reduction is cut into two `KC` blocks so the accumulate path runs too.
use super::compat::{contract_reference, ElementOp, Layout, Operand, Plan, RefOperand};
use crate::api::Scalar;
use crate::driver::ResolvedCall;
use tcontract_kernel::ResolvedGemm;
use tcontract_kernel::{Blocking, CpuFeatures, Element, Families, KernelChoice, Real, Registry};

/// One case's operand and call options.
#[derive(Clone, Copy, Debug)]
pub struct Opts {
    /// `alpha` real part.
    pub alpha_re: f64,
    /// `alpha` imaginary part, ignored by real element types.
    pub alpha_im: f64,
    /// `beta` real part.
    pub beta_re: f64,
    /// `beta` imaginary part, ignored by real element types.
    pub beta_im: f64,
    /// `C` and `D` are one buffer, so `C` aliases `D` exactly.
    pub alias: bool,
    /// Conjugate the `A` operand.
    pub conj_a: bool,
    /// Conjugate the `B` operand.
    pub conj_b: bool,
    /// Conjugate the `C` operand.
    pub conj_c: bool,
    /// Conjugate the `D` operand.
    pub conj_d: bool,
    /// Store `D` row-major, which makes the engine exchange the two operands.
    pub row_major_d: bool,
    /// Threads the plan asks for. One unless a test says otherwise.
    pub width: usize,
}

impl Default for Opts {
    fn default() -> Self {
        Self {
            alpha_re: 1.3,
            alpha_im: 0.2,
            beta_re: -0.4,
            beta_im: 0.1,
            alias: false,
            conj_a: false,
            conj_b: false,
            conj_c: false,
            conj_d: false,
            row_major_d: false,
            width: 1,
        }
    }
}

impl Opts {
    /// Real scalars, everything else default: the direct-family cases.
    pub fn real(alpha: f64, beta: f64) -> Self {
        Self {
            alpha_re: alpha,
            alpha_im: 0.0,
            beta_re: beta,
            beta_im: 0.0,
            ..Self::default()
        }
    }

    fn alpha<T: Element>(&self) -> T {
        T::from_parts(
            T::Real::from_f64(self.alpha_re),
            T::Real::from_f64(self.alpha_im),
        )
    }

    fn beta<T: Element>(&self) -> T {
        T::from_parts(
            T::Real::from_f64(self.beta_re),
            T::Real::from_f64(self.beta_im),
        )
    }
}

/// Deterministic operand value, distinct per index and non-trivial in both
/// parts, so a conjugation or a stride error cannot cancel out.
pub fn value<T: Element>(i: usize) -> T {
    let r = T::Real::from_f64((i as f64 + 1.0) * 0.13);
    let im = T::Real::from_f64((i as f64 - 2.0) * 0.07);
    T::from_parts(r, if T::IS_COMPLEX { im } else { T::Real::ZERO })
}

/// Every CPU-available family for `T`, after registering the providers.
pub fn all_families<T>() -> Vec<&'static str>
where
    T: Element + Families,
{
    Registry::families::<T>(CpuFeatures::detect(), false)
        .into_iter()
        .map(|f| f.id)
        .collect()
}

/// Run `id` against the reference oracle and assert the relative error is
/// within `tol`. Returns the call's operand-dependent decisions, so a test can
/// pin them from the same run it measured.
pub fn check_family_vs_oracle<T>(id: &'static str, opts: Opts, tol: f64) -> ResolvedCall
where
    T: Scalar,
{
    // Probe the family's tile before sizing the operands: the interesting
    // shapes are one row past `MR` and several columns past `NR`.
    let probe = Plan::new(
        Operand::new(&Layout::col_major(&[2, 2]), &[0, 2]),
        Operand::new(&Layout::col_major(&[2, 2]), &[2, 1]),
        None,
        Operand::new(&Layout::col_major(&[2, 2]), &[0, 1]),
    )
    .unwrap()
    .with_kernel(KernelChoice::Id(id.into()))
    .unwrap();
    let rg = probe.resolved::<T>().unwrap();
    assert_eq!(rg.family().id, id);

    let (m, k, n) = (rg.mr + 1, 3, rg.nr.max(rg.mr) + 1);
    let la = Layout::col_major(&[m as i64, k as i64]);
    let lb = Layout::col_major(&[k as i64, n as i64]);
    let ld = if opts.row_major_d {
        Layout::row_major(&[m as i64, n as i64])
    } else {
        Layout::col_major(&[m as i64, n as i64])
    };
    let lc = if opts.row_major_d {
        Layout::col_major(&[m as i64, n as i64])
    } else {
        Layout::row_major(&[m as i64, n as i64])
    };
    let (ia, ib, idd) = (vec![0i64, 2], vec![2i64, 1], vec![0i64, 1]);
    let op = |conj| {
        if conj {
            ElementOp::Conjugate
        } else {
            ElementOp::Identity
        }
    };
    let a: Vec<T> = (0..la.storage_len() as usize).map(value).collect();
    let b: Vec<T> = (0..lb.storage_len() as usize).map(value).collect();
    // `start` is `D` before the call and, when `alias`, `C` as well.
    let start: Vec<T> = (0..ld.storage_len() as usize)
        .map(|i| value(i + 31))
        .collect();
    let separate_c: Vec<T> = (0..lc.storage_len() as usize)
        .map(|i| value(i + 17))
        .collect();
    let mut got = start.clone();
    // `C` is always described, whether or not it is read: `beta = 0` skips it,
    // and with `alias` it *is* `D`. A plan without a `C` operand has no C
    // scatter for a nonzero beta to follow.
    let c_layout = if opts.alias { &ld } else { &lc };
    let plan = Plan::new(
        Operand {
            layout: &la,
            idx: &ia,
            op: op(opts.conj_a),
        },
        Operand {
            layout: &lb,
            idx: &ib,
            op: op(opts.conj_b),
        },
        Some(Operand {
            layout: c_layout,
            idx: &idd,
            op: op(opts.conj_c),
        }),
        Operand {
            layout: &ld,
            idx: &idd,
            op: op(opts.conj_d),
        },
    )
    .unwrap()
    .with_kernel(KernelChoice::Id(id.into()))
    .unwrap()
    .with_threads(opts.width)
    .with_blocking(Blocking {
        mc: rg.mr,
        kc: 2,
        nc: rg.nr,
    });
    let rg = plan.resolved::<T>().unwrap();
    let (alpha, beta) = (opts.alpha::<T>(), opts.beta::<T>());
    let c_for_oracle: &[T] = if opts.alias { &start } else { &separate_c };
    let c_ptr = if opts.alias {
        got.as_ptr()
    } else {
        separate_c.as_ptr()
    };
    // Pin the decisions the driver will take, on the real pointers.
    let calls = plan.decisions::<T>(&rg, c_ptr, got.as_mut_ptr(), beta);

    // SAFETY: every buffer is at least as long as its layout's storage, all
    // offsets come from the plan, and `D` is borrowed exclusively here.
    unsafe { plan.run_raw::<T>(alpha, a.as_ptr(), b.as_ptr(), beta, c_ptr, got.as_mut_ptr()) };

    let mut want = start.clone();
    contract_reference::<T>(
        alpha,
        &RefOperand {
            data: &a,
            layout: &la,
            idx: &ia,
            op: op(opts.conj_a),
        },
        &RefOperand {
            data: &b,
            layout: &lb,
            idx: &ib,
            op: op(opts.conj_b),
        },
        beta,
        Some(&RefOperand {
            data: c_for_oracle,
            layout: c_layout,
            idx: &idd,
            op: op(opts.conj_c),
        }),
        &mut want,
        &ld,
        &idd,
        op(opts.conj_d),
    )
    .unwrap();

    let diff: f64 = got
        .iter()
        .zip(&want)
        .map(|(x, y)| x.sub(*y).norm().powi(2))
        .sum();
    let scale: f64 = want.iter().map(|x| x.norm().powi(2)).sum();
    let err = (diff / scale.max(1.0)).sqrt();
    assert!(
        err < tol,
        "family {id}, opts {opts:?}: relative error {err:e} exceeds {tol:e}"
    );
    calls
}

/// Problem shape for the width-sweep tests.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    /// Rows of `D`.
    pub m: usize,
    /// Columns of `D`.
    pub n: usize,
    /// Contracted length.
    pub k: usize,
}

/// A pool of eight workers shared by every width-sweep run. Each run takes a
/// budget of it, so the pool is built once instead of once per call.
fn sweep_pool() -> &'static tcontract_exec::Pool<'static> {
    static TP: std::sync::OnceLock<strided_exec::Pool> = std::sync::OnceLock::new();
    static POOL: std::sync::OnceLock<tcontract_exec::Pool<'static>> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let tp = TP.get_or_init(|| strided_exec::Pool::builder().threads(8).build().unwrap());
        tcontract_exec::Pool::borrow(tp)
    })
}

/// One `ij,jk->ik` contraction at `width`, through an `Exec` of the
/// shared pool and the resolved driver, with `D` overwritten (`beta = 0`).
pub fn run_with_width<T>(
    id: &'static str,
    width: usize,
    shape: Shape,
    align_c_lines: bool,
) -> Vec<T>
where
    T: Scalar,
{
    let Shape { m, n, k } = shape;
    let la = Layout::col_major(&[m as i64, k as i64]);
    let lb = Layout::col_major(&[k as i64, n as i64]);
    let ld = Layout::col_major(&[m as i64, n as i64]);
    let (ia, ib, idd) = (vec![0i64, 2], vec![2i64, 1], vec![0i64, 1]);
    let a: Vec<T> = (0..la.storage_len() as usize).map(value).collect();
    let b: Vec<T> = (0..lb.storage_len() as usize).map(value).collect();
    let mut d = vec![<T as Element>::zero(); ld.storage_len() as usize];
    let choice = KernelChoice::Id(id.into());
    let plan = Plan::new(
        Operand::new(&la, &ia),
        Operand::new(&lb, &ib),
        None,
        Operand::new(&ld, &idd),
    )
    .unwrap()
    .with_kernel(choice.clone())
    .unwrap()
    .with_threads(width);
    let rg = if align_c_lines {
        ResolvedGemm::<T::Real>::resolve_with::<T>(
            &choice,
            width,
            tcontract_kernel::PartitionPolicy::default(),
            tcontract_kernel::PartitionOpts { align_c_lines },
        )
        .unwrap()
    } else {
        plan.resolved::<T>().unwrap()
    };
    let mut exec = tcontract_exec::Exec::new(sweep_pool(), width).unwrap();
    // SAFETY: every buffer is sized by its layout, `beta` is zero so `C` is
    // never read, and `D` is exclusively borrowed here.
    unsafe {
        super::compat::execute_resolved(
            &plan,
            &rg,
            &mut exec,
            None,
            <T as Element>::one(),
            a.as_ptr(),
            b.as_ptr(),
            <T as Element>::zero(),
            std::ptr::null(),
            d.as_mut_ptr(),
        )
    }
    .expect("the driver must serve this plan at this width");
    d
}

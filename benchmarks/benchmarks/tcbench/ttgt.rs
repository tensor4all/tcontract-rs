//! TTGT baseline: Transpose-Transpose-GEMM-Transpose.
//!
//! The classical way to do a tensor contraction: permute `A` into a dense
//! `M x K` matrix, `B` into `K x N`, call GEMM, and permute the `M x N` result
//! into the output tensor's layout.
//!
//! This implementation reads the same validated [`Problem`] roles the planner
//! lowers, so TTGT and the contraction library see the identical M/N/K
//! decomposition. The only thing that differs is the execution strategy. That
//! makes the comparison a clean measurement of "materialise a transposed copy,
//! then call a vendor GEMM" versus "fuse the transposition into packing".
//!
//! The permutation itself writes its output contiguously and gathers its
//! input, which is the sensible naive strategy. A production TTGT would use a
//! blocked/vectorised transpose such as HPTT; treat these numbers as an upper
//! bound on TTGT's transposition cost, and TBLIS as the strong baseline.

#![allow(dead_code)] // only reachable with the `blas` feature

use tcontract::api::{OperandId, Problem, RoleAxis, Scalar};

use crate::blas::GemmScalar;

/// The M/N/K/H offset tables of one problem: element `(i, p)` of `A` in batch
/// `h` lives at `h_a[h] + a_m[i] + a_k[p]`, and likewise for the others.
#[derive(Clone, Debug)]
pub struct TtgtPlan {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub batch: usize,
    a_m: Vec<isize>,
    a_k: Vec<isize>,
    b_k: Vec<isize>,
    b_n: Vec<isize>,
    d_m: Vec<isize>,
    d_n: Vec<isize>,
    c_m: Vec<isize>,
    c_n: Vec<isize>,
    h_a: Vec<isize>,
    h_b: Vec<isize>,
    h_c: Vec<isize>,
    h_d: Vec<isize>,
}

/// Mixed-radix offsets over `axes`, the first axis fastest.
fn offsets(axes: &[RoleAxis], operand: OperandId) -> Vec<isize> {
    let mut out = vec![0isize];
    for ax in axes {
        let s = ax.stride(operand);
        let prev = out.clone();
        out.clear();
        for j in 0..ax.extent() as isize {
            out.extend(prev.iter().map(|&o| o + j * s));
        }
    }
    out
}

impl TtgtPlan {
    pub fn new(problem: &Problem) -> Self {
        let r = problem.roles();
        let (m, n, k, h) = (r.m(), r.n(), r.k(), r.h());
        let o = offsets;
        use OperandId::{A, B, C, D};
        let len = |axes: &[RoleAxis]| axes.iter().map(|x| x.extent()).product::<usize>();
        TtgtPlan {
            m: len(m),
            n: len(n),
            k: len(k),
            batch: len(h),
            a_m: o(m, A),
            a_k: o(k, A),
            b_k: o(k, B),
            b_n: o(n, B),
            d_m: o(m, D),
            d_n: o(n, D),
            c_m: o(m, C),
            c_n: o(n, C),
            h_a: o(h, A),
            h_b: o(h, B),
            h_c: o(h, C),
            h_d: o(h, D),
        }
    }
}

/// Scratch buffers, reused across repetitions so the benchmark measures the
/// algorithm rather than the allocator.
pub struct TtgtScratch<T> {
    pub am: Vec<T>,
    pub bm: Vec<T>,
    pub cm: Vec<T>,
}

impl<T: Scalar> TtgtScratch<T> {
    pub fn new(plan: &TtgtPlan) -> Self {
        let zero = <T as tcontract_kernel::Element>::zero();
        TtgtScratch {
            am: vec![zero; plan.m * plan.k],
            bm: vec![zero; plan.k * plan.n],
            cm: vec![zero; plan.m * plan.n],
        }
    }
}

/// `D = alpha * A * B + beta * C` via TTGT. Batch (Hadamard) indices are
/// handled by looping over them, as a TTGT implementation would.
///
/// Slices must be large enough for every offset the problem generates.
#[allow(clippy::too_many_arguments)]
pub fn ttgt<T>(
    plan: &TtgtPlan,
    alpha: T,
    a: &[T],
    b: &[T],
    beta: T,
    c: &[T],
    d: &mut [T],
    scratch: &mut TtgtScratch<T>,
) where
    T: Scalar + GemmScalar,
{
    use tcontract_kernel::Element;
    let (m, n, k) = (plan.m, plan.n, plan.k);
    // No work, no read: an empty output extent or an empty batch returns here,
    // before anything is indexed, even for a nonzero beta.
    if m == 0 || n == 0 || plan.batch == 0 {
        return;
    }
    // With work to do, C is read only when `beta` is nonzero; a caller that
    // overwrites passes no C at all, which is what the benchmark does. Checked
    // here so a later nonzero beta cannot index an empty slice.
    debug_assert!(
        beta == <T as Element>::zero() || !c.is_empty(),
        "a nonzero beta reads C, so the caller must supply it"
    );

    for h in 0..plan.batch {
        let (oa, ob, oc, od) = (plan.h_a[h], plan.h_b[h], plan.h_c[h], plan.h_d[h]);

        // A -> column-major M x K
        for (p, &kp) in plan.a_k.iter().enumerate() {
            let dst = &mut scratch.am[p * m..(p + 1) * m];
            for (i, &mi) in plan.a_m.iter().enumerate() {
                dst[i] = a[(oa + mi + kp) as usize];
            }
        }
        // B -> column-major K x N
        for (j, &nj) in plan.b_n.iter().enumerate() {
            let dst = &mut scratch.bm[j * k..(j + 1) * k];
            for (p, &kp) in plan.b_k.iter().enumerate() {
                dst[p] = b[(ob + kp + nj) as usize];
            }
        }

        if k > 0 {
            // SAFETY: buffers are sized m*k, k*n, m*n by `TtgtScratch::new`.
            unsafe {
                T::gemm(
                    m,
                    n,
                    k,
                    scratch.am.as_ptr(),
                    scratch.bm.as_ptr(),
                    scratch.cm.as_mut_ptr(),
                );
            }
        } else {
            scratch
                .cm
                .iter_mut()
                .for_each(|x| *x = <T as Element>::zero());
        }

        // M x N -> D
        let beta_zero = beta == <T as Element>::zero();
        for (j, (&dn, &cn)) in plan.d_n.iter().zip(&plan.c_n).enumerate() {
            for (i, (&dm, &cm)) in plan.d_m.iter().zip(&plan.c_m).enumerate() {
                let mut v = Element::mul(alpha, scratch.cm[j * m + i]);
                if !beta_zero {
                    v = Element::add(v, Element::mul(beta, c[(oc + cm + cn) as usize]));
                }
                d[(od + dm + dn) as usize] = v;
            }
        }
    }
}

#[cfg(all(test, feature = "blas"))]
mod tests {
    //! The two empty-output routes and the overwrite route the benchmark takes.
    //!
    //! The regression these pin: a `debug_assert` used to reject an empty `C`
    //! before the no-op return, so a valid call with a zero output extent (or an
    //! empty batch) panicked in debug builds even though it reads no `C`.

    use super::*;
    use tcontract::api::{CSpec, DType, Labels, LayoutSpec, Op, OperandSpec};

    fn operand(dims: &[usize], strides: &[isize]) -> OperandSpec {
        OperandSpec::new(LayoutSpec::new(dims, strides, 0).expect("valid layout"))
    }

    /// `D[a,b] = sum_c A[a,c] B[c,b]` with `a = 0`: m = 0, nothing to read or write.
    #[test]
    fn a_zero_output_extent_returns_before_c_is_consulted() {
        let problem = Problem::from_labels(
            DType::F64,
            operand(&[0, 1], &[1, 1]),
            operand(&[1, 1], &[1, 1]),
            CSpec::Output(Op::Identity),
            operand(&[0, 1], &[1, 1]),
            &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
        )
        .expect("a zero extent is a valid problem");
        let plan = TtgtPlan::new(&problem);
        assert_eq!(plan.m, 0);
        let mut scratch = TtgtScratch::<f64>::new(&plan);
        let mut d: Vec<f64> = Vec::new();
        ttgt(&plan, 1.0, &[], &[1.0], 1.0, &[], &mut d, &mut scratch);
        assert!(d.is_empty());
    }

    /// `D[a,b,h] = sum_c A[a,c,h] B[c,b,h]` with `h = 0`: batch = 0, same route.
    #[test]
    fn an_empty_batch_returns_before_c_is_consulted() {
        let problem = Problem::from_labels(
            DType::F64,
            operand(&[1, 1, 0], &[1, 1, 1]),
            operand(&[1, 1, 0], &[1, 1, 1]),
            CSpec::Output(Op::Identity),
            operand(&[1, 1, 0], &[1, 1, 1]),
            &Labels::new(&[0, 2, 3], &[2, 1, 3], &[0, 1, 3]),
        )
        .expect("a zero batch extent is a valid problem");
        let plan = TtgtPlan::new(&problem);
        assert_eq!((plan.m, plan.n, plan.k, plan.batch), (1, 1, 1, 0));
        let mut scratch = TtgtScratch::<f64>::new(&plan);
        let mut d: Vec<f64> = Vec::new();
        ttgt(&plan, 1.0, &[1.0], &[1.0], 1.0, &[], &mut d, &mut scratch);
        assert!(d.is_empty());
    }

    /// The route the timed arm takes: overwrite (`beta = 0`), so no `C` is passed
    /// and none is read -- and the result is still the product.
    #[test]
    fn an_overwrite_with_no_c_computes_the_product() {
        let problem = Problem::from_labels(
            DType::F64,
            operand(&[1, 1], &[1, 1]),
            operand(&[1, 1], &[1, 1]),
            CSpec::Absent,
            operand(&[1, 1], &[1, 1]),
            &Labels::new(&[0, 2], &[2, 1], &[0, 1]),
        )
        .expect("1x1x1 is a valid problem");
        let plan = TtgtPlan::new(&problem);
        let mut scratch = TtgtScratch::<f64>::new(&plan);
        let mut d = vec![0.0f64];
        ttgt(&plan, 1.0, &[2.0], &[3.0], 0.0, &[], &mut d, &mut scratch);
        assert_eq!(d, vec![6.0]);
    }
}

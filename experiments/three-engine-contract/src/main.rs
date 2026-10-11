//! tcontract vs tensorprimitives-rs (`tensorcontract`) vs tenferro on the
//! per-shape corpus of tensor4all/tcontract-rs#61, single-threaded.
//!
//! Every case is a *program*: a list of input tensors and a list of pairwise
//! steps, each step contracting two slots into a new slot with a local einsum
//! equation. A single pairwise case has one step; `ij,jk,kl->il` has two in the
//! fixed order `((ij,jk),kl)`; the MPS chain has two per site. All three engines
//! run the same steps on the same data, so only the per-step machinery differs.
//!
//! Arms (all at one thread):
//!
//! | arm | boundary | what runs per step |
//! | --- | --- | --- |
//! | `tc_exec` | exec | prebuilt `tensorcontract::Plan::run` into a preallocated output |
//! | `tc_call` | call | `tensorcontract::contract` (plans) into a freshly allocated output |
//! | `tp_exec` | exec | prebuilt `tcontract::Plan::execute_into` (planner's choice) |
//! | `tp_packed_exec` | exec | as `tp_exec`, packed driver forced |
//! | `tp_call` | call | `Problem` + `Plan::new` + fresh output, then execute |
//! | `tf_exec` | exec | prepared `ConcreteEinsumPlan::execute_into`, one session for the whole timing loop |
//! | `tf_call` | call | ordinary `einsum` (plans, allocates), one session for the whole timing loop |
//! | `tf_call_spc` | call | ordinary `einsum`, a new `with_backend_session` per step |
//! | `tf_eager` | call | `EagerRuntime` session einsum on constants, one eager session per program |
//!
//! The backend (`CpuBackend::with_threads(1)`) is constructed once per arm and
//! never inside the timed region. Outputs are checked against a naive reference
//! before any timing.

use std::collections::HashMap;
use std::hint::black_box;
use std::time::{Duration, Instant};

use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

// ---------------------------------------------------------------------------
// Element abstraction
// ---------------------------------------------------------------------------

trait Elem:
    tensorcontract::Element
    + tcontract::api::Scalar
    + cpueinsum::Scalar
    + tenferro_tensor::TensorScalar
    + Copy
    + Default
    + std::ops::Add<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::fmt::Debug
    + Send
    + Sync
    + 'static
where
    <Self as tensorcontract::Element>::Real: tensorcontract::KernelSet,
{
    const NAME: &'static str;
    fn rand(rng: &mut ChaCha8Rng) -> Self;
    fn unit() -> Self;
    fn abs(self) -> f64;
    fn diff(self, o: Self) -> Self;
}

impl Elem for f64 {
    const NAME: &'static str = "f64";
    fn rand(rng: &mut ChaCha8Rng) -> Self {
        rng.gen_range(-1.0..1.0)
    }
    fn unit() -> Self {
        1.0
    }
    fn abs(self) -> f64 {
        f64::abs(self)
    }
    fn diff(self, o: Self) -> Self {
        self - o
    }
}

impl Elem for Complex64 {
    const NAME: &'static str = "c64";
    fn rand(rng: &mut ChaCha8Rng) -> Self {
        Complex64::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0))
    }
    fn unit() -> Self {
        Complex64::new(1.0, 0.0)
    }
    fn abs(self) -> f64 {
        self.norm()
    }
    fn diff(self, o: Self) -> Self {
        self - o
    }
}

// ---------------------------------------------------------------------------
// Programs
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Step {
    lhs: usize,
    rhs: usize,
    a: Vec<char>,
    b: Vec<char>,
    d: Vec<char>,
}

impl Step {
    fn eq(&self) -> String {
        let s = |v: &[char]| v.iter().collect::<String>();
        format!("{},{}->{}", s(&self.a), s(&self.b), s(&self.d))
    }
}

#[derive(Clone, Debug)]
struct Program {
    name: String,
    dtype: &'static str,
    inputs: Vec<Vec<usize>>,
    steps: Vec<Step>,
    /// Shape of every slot (inputs first, then one per step).
    shapes: Vec<Vec<usize>>,
    macs: u128,
}

fn parse(eq: &str) -> (Vec<char>, Vec<char>, Vec<char>) {
    let (lhs, d) = eq.split_once("->").unwrap();
    let (a, b) = lhs.split_once(',').unwrap();
    (a.chars().collect(), b.chars().collect(), d.chars().collect())
}

/// Build a program from input shapes and `(lhs_slot, rhs_slot, equation)` steps.
fn program(
    name: String,
    dtype: &'static str,
    inputs: Vec<Vec<usize>>,
    steps: Vec<(usize, usize, &str)>,
) -> Program {
    let mut shapes = inputs.clone();
    let mut out = Vec::new();
    let mut macs = 0u128;
    for (lhs, rhs, eq) in steps {
        let (a, b, d) = parse(eq);
        let mut ext: HashMap<char, usize> = HashMap::new();
        for (l, &e) in a.iter().zip(&shapes[lhs]) {
            ext.insert(*l, e);
        }
        for (l, &e) in b.iter().zip(&shapes[rhs]) {
            if let Some(&prev) = ext.get(l) {
                assert_eq!(prev, e, "{name}: extent of {l}");
            }
            ext.insert(*l, e);
        }
        let mut all: Vec<char> = a.iter().chain(&b).copied().collect();
        all.sort();
        all.dedup();
        macs += all.iter().map(|l| ext[l] as u128).product::<u128>();
        shapes.push(d.iter().map(|l| ext[l]).collect());
        out.push(Step { lhs, rhs, a, b, d });
    }
    Program {
        name,
        dtype,
        inputs,
        steps: out,
        shapes,
        macs,
    }
}

fn corpus() -> Vec<Program> {
    let mut v = Vec::new();
    // MPS chain: <phi|psi> (bilinear, no conjugation) of two random open-chain
    // MPS with uniform bond dimension chi and physical dimension 2, swept left to
    // right from a random chi x chi left environment. Per site:
    //   E[a,b] A[a,s,c] -> T[b,s,c];  T[b,s,c] B[b,s,d] -> E'[c,d].
    for chi in [4usize, 8, 16, 32, 64] {
        let l = 32;
        let mut inputs = vec![vec![chi, chi]];
        for _ in 0..l {
            inputs.push(vec![chi, 2, chi]); // ket site
            inputs.push(vec![chi, 2, chi]); // bra site
        }
        let n_in = inputs.len();
        let mut steps = Vec::new();
        let mut env = 0usize;
        for site in 0..l {
            let t = n_in + steps.len();
            steps.push((env, 1 + 2 * site, "ab,asc->bsc"));
            env = n_in + steps.len();
            steps.push((t, 2 + 2 * site, "bsc,bsd->cd"));
        }
        v.push(program(
            format!("mps_chain_L{l}_chi{chi}"),
            "c64",
            inputs,
            steps,
        ));
    }
    for batch in [16usize, 64, 256] {
        for n in [2usize, 4, 8, 16] {
            v.push(program(
                format!("ikb_knb_inb_n{n}_b{batch}"),
                "f64",
                vec![vec![n, n, batch], vec![n, n, batch]],
                vec![(0, 1, "ikb,knb->inb")],
            ));
        }
    }
    v.push(program(
        "ijk_jkl_il_8x16x8".into(),
        "f64",
        vec![vec![8, 16, 8], vec![16, 8, 8]],
        vec![(0, 1, "ijk,jkl->il")],
    ));
    v.push(program(
        "ij_jk_ik_c64_n32".into(),
        "c64",
        vec![vec![32, 32], vec![32, 32]],
        vec![(0, 1, "ij,jk->ik")],
    ));
    v.push(program(
        "ij_jk_ik_f64_n64".into(),
        "f64",
        vec![vec![64, 64], vec![64, 64]],
        vec![(0, 1, "ij,jk->ik")],
    ));
    v.push(program(
        "ij_jk_kl_il_n64".into(),
        "f64",
        vec![vec![64, 64], vec![64, 64], vec![64, 64]],
        vec![(0, 1, "ij,jk->ik"), (3, 2, "ik,kl->il")],
    ));
    v
}

fn col_major_strides(dims: &[usize]) -> Vec<isize> {
    let mut s = Vec::with_capacity(dims.len());
    let mut acc = 1isize;
    for &d in dims {
        s.push(acc);
        acc *= d as isize;
    }
    s
}

fn numel(dims: &[usize]) -> usize {
    dims.iter().product()
}

fn label_ids(v: &[char]) -> Vec<i64> {
    v.iter().map(|&c| c as i64).collect()
}

// ---------------------------------------------------------------------------
// Naive reference
// ---------------------------------------------------------------------------

fn naive_step<T: Elem>(st: &Step, a: &[T], da: &[usize], b: &[T], db: &[usize]) -> Vec<T>
where
    <T as tensorcontract::Element>::Real: tensorcontract::KernelSet,
{
    let mut labels: Vec<char> = st.a.iter().chain(&st.b).copied().collect();
    labels.sort();
    labels.dedup();
    let mut ext = HashMap::new();
    for (l, &e) in st.a.iter().zip(da) {
        ext.insert(*l, e);
    }
    for (l, &e) in st.b.iter().zip(db) {
        ext.insert(*l, e);
    }
    let dd: Vec<usize> = st.d.iter().map(|l| ext[l]).collect();
    let (sa, sb, sd) = (
        col_major_strides(da),
        col_major_strides(db),
        col_major_strides(&dd),
    );
    let pos = |l: char| labels.iter().position(|&x| x == l).unwrap();
    let ia: Vec<usize> = st.a.iter().map(|&l| pos(l)).collect();
    let ib: Vec<usize> = st.b.iter().map(|&l| pos(l)).collect();
    let id: Vec<usize> = st.d.iter().map(|&l| pos(l)).collect();
    let extents: Vec<usize> = labels.iter().map(|l| ext[l]).collect();
    let mut out = vec![T::default(); numel(&dd)];
    let mut idx = vec![0usize; labels.len()];
    loop {
        let oa: isize = ia.iter().zip(&sa).map(|(&p, &s)| idx[p] as isize * s).sum();
        let ob: isize = ib.iter().zip(&sb).map(|(&p, &s)| idx[p] as isize * s).sum();
        let od: isize = id.iter().zip(&sd).map(|(&p, &s)| idx[p] as isize * s).sum();
        out[od as usize] = out[od as usize] + a[oa as usize] * b[ob as usize];
        let mut k = 0;
        loop {
            if k == idx.len() {
                return out;
            }
            idx[k] += 1;
            if idx[k] < extents[k] {
                break;
            }
            idx[k] = 0;
            k += 1;
        }
    }
}

fn naive_program<T: Elem>(p: &Program, inputs: &[Vec<T>]) -> Vec<T>
where
    <T as tensorcontract::Element>::Real: tensorcontract::KernelSet,
{
    let mut slots: Vec<Vec<T>> = inputs.to_vec();
    for st in &p.steps {
        let r = naive_step(
            st,
            &slots[st.lhs],
            &p.shapes[st.lhs],
            &slots[st.rhs],
            &p.shapes[st.rhs],
        );
        slots.push(r);
    }
    slots.pop().unwrap()
}

fn rel_err<T: Elem>(x: &[T], r: &[T]) -> f64
where
    <T as tensorcontract::Element>::Real: tensorcontract::KernelSet,
{
    assert_eq!(x.len(), r.len());
    let scale = r.iter().map(|v| v.abs()).fold(0.0, f64::max).max(f64::MIN_POSITIVE);
    x.iter()
        .zip(r)
        .map(|(a, b)| a.diff(*b).abs())
        .fold(0.0, f64::max)
        / scale
}

// ---------------------------------------------------------------------------
// Engines. Each arm returns a closure `run()` that executes the whole program
// once and returns the final output (copied out only for the check).
// ---------------------------------------------------------------------------

type Runner<'a, T> = Box<dyn FnMut(bool) -> Option<Vec<T>> + 'a>;

/// tensorcontract with prebuilt plans and preallocated slots.
fn tc_exec<'a, T: Elem>(p: &'a Program, inputs: &'a [Vec<T>]) -> Runner<'a, T>
where
    <T as tensorcontract::Element>::Real: tensorcontract::KernelSet,
{
    use tensorcontract::{Layout, Operand, Plan, TensorView, TensorViewMut};
    let layouts: Vec<Layout> = p.shapes.iter().map(|d| Layout::col_major(&dims_i64(d))).collect();
    let idx: Vec<(Vec<i64>, Vec<i64>, Vec<i64>)> = p
        .steps
        .iter()
        .map(|s| (label_ids(&s.a), label_ids(&s.b), label_ids(&s.d)))
        .collect();
    let n_in = p.inputs.len();
    let plans: Vec<Plan> = p
        .steps
        .iter()
        .enumerate()
        .map(|(k, s)| {
            let (ia, ib, id) = &idx[k];
            Plan::new(
                Operand::new(&layouts[s.lhs], ia),
                Operand::new(&layouts[s.rhs], ib),
                None,
                Operand::new(&layouts[n_in + k], id),
            )
            .expect("tc plan")
        })
        .collect();
    assert!(plans.iter().all(|pl| pl.threads() == 1), "tensorcontract threads != 1");
    let mut slots: Vec<Vec<T>> = p.shapes[n_in..]
        .iter()
        .map(|d| vec![T::default(); numel(d)])
        .collect();
    Box::new(move |want| {
        for (k, s) in p.steps.iter().enumerate() {
            let (ia, ib, id) = &idx[k];
            let (done, rest) = slots.split_at_mut(k);
            let get = |i: usize| -> &[T] {
                if i < n_in {
                    &inputs[i]
                } else {
                    &done[i - n_in]
                }
            };
            plans[k]
                .run(
                    T::unit(),
                    TensorView::new(get(s.lhs), &layouts[s.lhs], ia),
                    TensorView::new(get(s.rhs), &layouts[s.rhs], ib),
                    T::default(),
                    None,
                    TensorViewMut::new(&mut rest[0], &layouts[n_in + k], id),
                )
                .expect("tc run");
        }
        black_box(&slots);
        want.then(|| slots.last().unwrap().clone())
    })
}

/// tensorcontract planning per call (`contract`), fresh outputs.
fn tc_call<'a, T: Elem>(p: &'a Program, inputs: &'a [Vec<T>]) -> Runner<'a, T>
where
    <T as tensorcontract::Element>::Real: tensorcontract::KernelSet,
{
    use tensorcontract::{contract, Layout, TensorView, TensorViewMut};
    let n_in = p.inputs.len();
    Box::new(move |want| {
        let mut slots: Vec<Vec<T>> = Vec::with_capacity(p.steps.len());
        for (k, s) in p.steps.iter().enumerate() {
            let (la, lb, ld) = (
                Layout::col_major(&dims_i64(&p.shapes[s.lhs])),
                Layout::col_major(&dims_i64(&p.shapes[s.rhs])),
                Layout::col_major(&dims_i64(&p.shapes[n_in + k])),
            );
            let (ia, ib, id) = (label_ids(&s.a), label_ids(&s.b), label_ids(&s.d));
            let mut out = vec![T::default(); numel(&p.shapes[n_in + k])];
            {
                let get = |i: usize| -> &[T] {
                    if i < n_in {
                        &inputs[i]
                    } else {
                        &slots[i - n_in]
                    }
                };
                contract(
                    T::unit(),
                    TensorView::new(get(s.lhs), &la, &ia),
                    TensorView::new(get(s.rhs), &lb, &ib),
                    T::default(),
                    None,
                    TensorViewMut::new(&mut out, &ld, &id),
                )
                .expect("tc contract");
            }
            slots.push(out);
        }
        black_box(&slots);
        want.then(|| slots.pop().unwrap())
    })
}

fn tp_problem<T: Elem>(p: &Program, k: usize) -> tcontract::api::Problem
where
    <T as tensorcontract::Element>::Real: tensorcontract::KernelSet,
{
    use tcontract::api::{CSpec, Labels, LayoutSpec, OperandSpec, Problem};
    let s = &p.steps[k];
    let n_in = p.inputs.len();
    let spec = |d: &[usize]| {
        OperandSpec::new(LayoutSpec::new(d, &col_major_strides(d), 0).expect("layout"))
    };
    Problem::from_labels(
        <T as tcontract::api::Scalar>::STORAGE,
        spec(&p.shapes[s.lhs]),
        spec(&p.shapes[s.rhs]),
        CSpec::Absent,
        spec(&p.shapes[n_in + k]),
        &Labels::new(&label_ids(&s.a), &label_ids(&s.b), &label_ids(&s.d)),
    )
    .expect("tp problem")
}

fn tp_view<'b, T: Elem>(data: &'b [T], d: &'b [usize], s: &'b [isize]) -> strided_view::StridedView<'b, T>
where
    <T as tensorcontract::Element>::Real: tensorcontract::KernelSet,
{
    strided_view::StridedView::new(data, d, s, 0).expect("view")
}

/// tcontract with prebuilt plans (planner's choice, or packed forced).
fn tp_exec<'a, T: Elem>(p: &'a Program, inputs: &'a [Vec<T>], packed: bool) -> Runner<'a, T>
where
    <T as tensorcontract::Element>::Real: tensorcontract::KernelSet,
{
    use tcontract::{Plan, PlanConfig};
    let n_in = p.inputs.len();
    let cfg = if packed {
        PlanConfig::packed()
    } else {
        PlanConfig::default()
    };
    let plans: Vec<Plan<T>> = (0..p.steps.len())
        .map(|k| Plan::<T>::new(&tp_problem::<T>(p, k), &cfg).expect("tp plan"))
        .collect();
    let strides: Vec<Vec<isize>> = p.shapes.iter().map(|d| col_major_strides(d)).collect();
    let mut slots: Vec<Vec<T>> = p.shapes[n_in..]
        .iter()
        .map(|d| vec![T::default(); numel(d)])
        .collect();
    let exec = tcontract_exec::Exec::serial();
    Box::new(move |want| {
        for (k, s) in p.steps.iter().enumerate() {
            let (done, rest) = slots.split_at_mut(k);
            let get = |i: usize| -> &[T] {
                if i < n_in {
                    &inputs[i]
                } else {
                    &done[i - n_in]
                }
            };
            let av = tp_view(get(s.lhs), &p.shapes[s.lhs], &strides[s.lhs]);
            let bv = tp_view(get(s.rhs), &p.shapes[s.rhs], &strides[s.rhs]);
            let mut dv = strided_view::StridedViewMut::new(
                &mut rest[0],
                &p.shapes[n_in + k],
                &strides[n_in + k],
                0,
            )
            .expect("view");
            plans[k]
                .execute_into(&exec, T::unit(), &av, &bv, &mut dv)
                .expect("tp exec");
        }
        black_box(&slots);
        want.then(|| slots.last().unwrap().clone())
    })
}

/// tcontract planning per call, fresh outputs.
fn tp_call<'a, T: Elem>(p: &'a Program, inputs: &'a [Vec<T>]) -> Runner<'a, T>
where
    <T as tensorcontract::Element>::Real: tensorcontract::KernelSet,
{
    use tcontract::{Plan, PlanConfig};
    let n_in = p.inputs.len();
    let exec = tcontract_exec::Exec::serial();
    let cfg = PlanConfig::default();
    Box::new(move |want| {
        let mut slots: Vec<Vec<T>> = Vec::with_capacity(p.steps.len());
        for (k, s) in p.steps.iter().enumerate() {
            let plan = Plan::<T>::new(&tp_problem::<T>(p, k), &cfg).expect("tp plan");
            let (sa, sb, sd) = (
                col_major_strides(&p.shapes[s.lhs]),
                col_major_strides(&p.shapes[s.rhs]),
                col_major_strides(&p.shapes[n_in + k]),
            );
            let mut out = vec![T::default(); numel(&p.shapes[n_in + k])];
            {
                let get = |i: usize| -> &[T] {
                    if i < n_in {
                        &inputs[i]
                    } else {
                        &slots[i - n_in]
                    }
                };
                let av = tp_view(get(s.lhs), &p.shapes[s.lhs], &sa);
                let bv = tp_view(get(s.rhs), &p.shapes[s.rhs], &sb);
                let mut dv =
                    strided_view::StridedViewMut::new(&mut out, &p.shapes[n_in + k], &sd, 0)
                        .expect("view");
                plan.execute_into(&exec, T::unit(), &av, &bv, &mut dv)
                    .expect("tp exec");
            }
            slots.push(out);
        }
        black_box(&slots);
        want.then(|| slots.pop().unwrap())
    })
}

/// The program as a cpueinsum spec: global labels and the SSA step order.
fn ce_spec(p: &Program) -> cpueinsum::EinsumSpec {
    let (inputs, output) = global_labels(p);
    let i64s = |v: &[u32]| -> Vec<i64> { v.iter().map(|&x| x as i64).collect() };
    let inputs: Vec<Vec<i64>> = inputs.iter().map(|v| i64s(v)).collect();
    let refs: Vec<&[i64]> = inputs.iter().map(|v| v.as_slice()).collect();
    let path: Vec<[usize; 2]> = p.steps.iter().map(|s| [s.lhs, s.rhs]).collect();
    cpueinsum::EinsumSpec::new(&refs, &i64s(&output), &path).expect("ce spec")
}

/// cpueinsum with a prebuilt `EinsumPlan` and a kept `Scratch`; the input and
/// output views are built per call (once per program, not per step).
fn ce_exec<'a, T: Elem>(p: &'a Program, inputs: &'a [Vec<T>]) -> Runner<'a, T>
where
    <T as tensorcontract::Element>::Real: tensorcontract::KernelSet,
{
    use cpueinsum::{EinsumPlan, Layout, Scratch};
    let n_in = p.inputs.len();
    let spec = ce_spec(p);
    let strides: Vec<Vec<isize>> = p.shapes.iter().map(|d| col_major_strides(d)).collect();
    let out_shape = p.shapes.last().unwrap();
    let plan = {
        let layouts: Vec<Layout<'_>> = (0..n_in)
            .map(|i| Layout::new(&p.shapes[i], &strides[i]).expect("layout"))
            .collect();
        let out = Layout::new(out_shape, strides.last().unwrap()).expect("layout");
        EinsumPlan::<T>::new(&spec, &layouts, out).expect("ce plan")
    };
    let mut scratch = Scratch::with_len(plan.scratch_len());
    let mut out = vec![T::default(); numel(out_shape)];
    let exec = tcontract_exec::Exec::serial();
    Box::new(move |want| {
        let views: Vec<_> = (0..n_in).map(|i| tp_view(&inputs[i], &p.shapes[i], &strides[i])).collect();
        let mut dv = strided_view::StridedViewMut::new(&mut out, out_shape, strides.last().unwrap(), 0)
            .expect("view");
        plan.execute_into(&exec, &views, &mut dv, &mut scratch).expect("ce exec");
        black_box(&out);
        want.then(|| out.clone())
    })
}

/// cpueinsum planning per call: `einsum_into` (plan + fresh scratch) into a
/// fresh output; the spec is built per call too.
fn ce_call<'a, T: Elem>(p: &'a Program, inputs: &'a [Vec<T>]) -> Runner<'a, T>
where
    <T as tensorcontract::Element>::Real: tensorcontract::KernelSet,
{
    let n_in = p.inputs.len();
    let (gin, gout) = global_labels(p);
    let gin: Vec<Vec<i64>> = gin.iter().map(|v| v.iter().map(|&x| x as i64).collect()).collect();
    let gout: Vec<i64> = gout.iter().map(|&x| x as i64).collect();
    let path: Vec<[usize; 2]> = p.steps.iter().map(|s| [s.lhs, s.rhs]).collect();
    let strides: Vec<Vec<isize>> = p.shapes.iter().map(|d| col_major_strides(d)).collect();
    let exec = tcontract_exec::Exec::serial();
    Box::new(move |want| {
        let refs: Vec<&[i64]> = gin.iter().map(|v| v.as_slice()).collect();
        let spec = cpueinsum::EinsumSpec::new(&refs, &gout, &path).expect("ce spec");
        let views: Vec<_> = (0..n_in).map(|i| tp_view(&inputs[i], &p.shapes[i], &strides[i])).collect();
        let out_shape = p.shapes.last().unwrap();
        let mut out = vec![T::default(); numel(out_shape)];
        {
            let mut dv =
                strided_view::StridedViewMut::new(&mut out, out_shape, strides.last().unwrap(), 0).expect("view");
            cpueinsum::einsum_into(&exec, &spec, &views, &mut dv).expect("ce call");
        }
        black_box(&out);
        want.then_some(out)
    })
}

fn tf_backend() -> tenferro_cpu::CpuBackend {
    let b = tenferro_cpu::CpuBackend::with_threads(1).expect("backend");
    assert_eq!(b.num_threads(), 1, "tenferro threads != 1");
    b
}

fn tf_inputs<T: Elem>(p: &Program, inputs: &[Vec<T>]) -> Vec<tenferro_tensor::Tensor>
where
    <T as tensorcontract::Element>::Real: tensorcontract::KernelSet,
{
    inputs
        .iter()
        .zip(&p.inputs)
        .map(|(v, d)| tenferro_tensor::Tensor::from_vec_col_major(d.clone(), v.clone()).expect("tensor"))
        .collect()
}

#[derive(Clone, Copy, PartialEq)]
enum TfMode {
    Exec,
    Call,
    CallSessionPerStep,
    /// Ordinary `einsum_subscripts` (integer labels built outside timing): no string parsing.
    CallSubscripts,
    /// `ConcreteEinsumPlan::prepare_subscripts` + `execute` per step: planning + allocation.
    PrepareExecute,
    /// Prebuilt plan, `execute` (allocating) per step: allocation only.
    PlanExecute,
}

/// tenferro concrete einsum. The timed closure receives the session from the
/// caller so that `Exec` and `Call` share one session across the timing loop.
fn tf_run<T: Elem>(
    p: &Program,
    mode: TfMode,
    inputs: &[Vec<T>],
    target: Duration,
    samples: usize,
) -> (Vec<f64>, Vec<T>)
where
    <T as tensorcontract::Element>::Real: tensorcontract::KernelSet,
{
    use tenferro_einsum::{ConcreteEinsumPlan, TensorEinsumExt};
    use tenferro_tensor::{BackendSessionHost, Tensor, TensorWrite};
    let mut backend = tf_backend();
    let ins = tf_inputs(p, inputs);
    let n_in = p.inputs.len();
    let eqs: Vec<String> = p.steps.iter().map(|s| s.eq()).collect();
    let subs: Vec<tenferro_einsum::EinsumSubscripts> = p
        .steps
        .iter()
        .map(|s| {
            let u = |v: &[char]| v.iter().map(|&c| c as u32).collect::<Vec<u32>>();
            tenferro_einsum::EinsumSubscripts::new(&[&u(&s.a), &u(&s.b)], &u(&s.d))
        })
        .collect();
    // Prepared plans and preallocated outputs for Exec.
    let mut outs: Vec<Tensor> = p.shapes[n_in..]
        .iter()
        .map(|d| Tensor::from_vec_col_major(d.clone(), vec![T::default(); numel(d)]).expect("out"))
        .collect();
    let plans: Vec<ConcreteEinsumPlan> = {
        // Plans need example tensors of the right shape; use the real inputs
        // and the preallocated outputs as stand-ins for intermediates.
        p.steps
            .iter()
            .enumerate()
            .map(|(k, s)| {
                let get = |i: usize| -> &Tensor {
                    if i < n_in {
                        &ins[i]
                    } else {
                        &outs[i - n_in]
                    }
                };
                ConcreteEinsumPlan::prepare([get(s.lhs), get(s.rhs)], eqs[k].as_str()).expect("tf plan")
            })
            .collect()
    };

    let once_exec = |session: &mut dyn tenferro_tensor::BackendSession, outs: &mut Vec<Tensor>| {
        for (k, s) in p.steps.iter().enumerate() {
            let (done, rest) = outs.split_at_mut(k);
            let get = |i: usize| -> &Tensor {
                if i < n_in {
                    &ins[i]
                } else {
                    &done[i - n_in]
                }
            };
            plans[k]
                .execute_into([get(s.lhs), get(s.rhs)], session, TensorWrite::from_tensor(&mut rest[0]))
                .expect("tf exec");
        }
    };
    let once_call = |session: &mut dyn tenferro_tensor::BackendSession| -> Tensor {
        let mut slots: Vec<Tensor> = Vec::with_capacity(p.steps.len());
        for (k, s) in p.steps.iter().enumerate() {
            let r = {
                let get = |i: usize| -> &Tensor {
                    if i < n_in {
                        &ins[i]
                    } else {
                        &slots[i - n_in]
                    }
                };
                [get(s.lhs), get(s.rhs)].einsum(eqs[k].as_str(), session).expect("tf einsum")
            };
            slots.push(r);
        }
        slots.pop().unwrap()
    };

    match mode {
        TfMode::Exec => {
            let (times, ()) = backend
                .with_backend_session(|session| {
                    let t = time_it(target, samples, || once_exec(session, &mut outs));
                    (t, ())
                })
                .expect("session");
            let last = outs.last().unwrap().as_slice::<T>().expect("slice").to_vec();
            (times, last)
        }
        TfMode::Call => {
            let (times, last) = backend
                .with_backend_session(|session| {
                    let t = time_it(target, samples, || {
                        black_box(once_call(session));
                    });
                    let last = once_call(session).as_slice::<T>().expect("slice").to_vec();
                    (t, last)
                })
                .expect("session");
            (times, last)
        }
        TfMode::CallSubscripts | TfMode::PrepareExecute | TfMode::PlanExecute => {
            let once = |session: &mut dyn tenferro_tensor::BackendSession| -> Tensor {
                let mut slots: Vec<Tensor> = Vec::with_capacity(p.steps.len());
                for (k, s) in p.steps.iter().enumerate() {
                    let r = {
                        let get = |i: usize| -> &Tensor {
                            if i < n_in {
                                &ins[i]
                            } else {
                                &slots[i - n_in]
                            }
                        };
                        let pair = [get(s.lhs), get(s.rhs)];
                        match mode {
                            TfMode::CallSubscripts => pair.einsum_subscripts(&subs[k], session),
                            TfMode::PrepareExecute => ConcreteEinsumPlan::prepare_subscripts(pair, &subs[k])
                                .and_then(|pl| pl.execute(pair, session)),
                            _ => plans[k].execute(pair, session),
                        }
                        .expect("tf step")
                    };
                    slots.push(r);
                }
                slots.pop().unwrap()
            };
            backend
                .with_backend_session(|session| {
                    let t = time_it(target, samples, || {
                        black_box(once(session));
                    });
                    (t, once(session).as_slice::<T>().expect("slice").to_vec())
                })
                .expect("session")
        }
        TfMode::CallSessionPerStep => {
            let run = |backend: &mut tenferro_cpu::CpuBackend| -> Tensor {
                let mut slots: Vec<Tensor> = Vec::with_capacity(p.steps.len());
                for (k, s) in p.steps.iter().enumerate() {
                    let r = {
                        let get = |i: usize| -> &Tensor {
                            if i < n_in {
                                &ins[i]
                            } else {
                                &slots[i - n_in]
                            }
                        };
                        let pair = [get(s.lhs), get(s.rhs)];
                        backend
                            .with_backend_session(|session| pair.einsum(eqs[k].as_str(), session))
                            .expect("session")
                            .expect("tf einsum")
                    };
                    slots.push(r);
                }
                slots.pop().unwrap()
            };
            let times = time_it(target, samples, || {
                black_box(run(&mut backend));
            });
            let last = run(&mut backend).as_slice::<T>().expect("slice").to_vec();
            (times, last)
        }
    }
}

/// tenferro eager (PyTorch-style) on constants, one eager session per program.
fn tf_eager<T: Elem>(p: &Program, inputs: &[Vec<T>], target: Duration, samples: usize) -> (Vec<f64>, Vec<T>)
where
    <T as tensorcontract::Element>::Real: tensorcontract::KernelSet,
{
    tf_eager_with(p, inputs, target, samples, tf_backend())
}

fn tf_eager_with<T: Elem>(
    p: &Program,
    inputs: &[Vec<T>],
    target: Duration,
    samples: usize,
    backend: tenferro_cpu::CpuBackend,
) -> (Vec<f64>, Vec<T>)
where
    <T as tensorcontract::Element>::Real: tensorcontract::KernelSet,
{
    use tenferro_ad::{EagerRuntime, EagerTensor};
    use tenferro_einsum::EagerSessionEinsumExt;
    let runtime = EagerRuntime::with_cpu_backend(backend).expect("runtime");
    let ins: Vec<EagerTensor> = tf_inputs(p, inputs)
        .into_iter()
        .map(|t| runtime.constant_from(t).expect("constant"))
        .collect();
    let n_in = p.inputs.len();
    let eqs: Vec<String> = p.steps.iter().map(|s| s.eq()).collect();
    let run = || -> EagerTensor {
        runtime
            .with_eager_session(|sess| {
                let mut slots: Vec<EagerTensor> = Vec::with_capacity(p.steps.len());
                for (k, s) in p.steps.iter().enumerate() {
                    let r = {
                        let get = |i: usize| -> &EagerTensor {
                            if i < n_in {
                                &ins[i]
                            } else {
                                &slots[i - n_in]
                            }
                        };
                        sess.einsum(&[get(s.lhs), get(s.rhs)], eqs[k].as_str()).expect("eager einsum")
                    };
                    slots.push(r);
                }
                slots.pop().unwrap()
            })
            .expect("eager session")
    };
    let times = time_it(target, samples, || {
        black_box(run());
    });
    let out = run();
    let last = out.value().expect("value").as_slice::<T>().expect("slice").to_vec();
    (times, last)
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

/// Calibrate an inner repeat count so one sample lasts about `target`, warm up
/// for one sample, then return `samples` per-call times in nanoseconds.
fn time_it(target: Duration, samples: usize, mut f: impl FnMut()) -> Vec<f64> {
    let mut reps = 1u64;
    loop {
        let t0 = Instant::now();
        for _ in 0..reps {
            f();
        }
        let el = t0.elapsed();
        if el >= target / 4 || reps >= 1 << 30 {
            let per = el.as_secs_f64() / reps as f64;
            reps = ((target.as_secs_f64() / per).ceil() as u64).max(1);
            break;
        }
        reps *= 4;
    }
    // warm-up sample
    for _ in 0..reps {
        f();
    }
    (0..samples)
        .map(|_| {
            let t0 = Instant::now();
            for _ in 0..reps {
                f();
            }
            t0.elapsed().as_secs_f64() * 1e9 / reps as f64
        })
        .collect()
}

fn time_runner<T: Elem>(mut r: Runner<'_, T>, target: Duration, samples: usize) -> (Vec<f64>, Vec<T>)
where
    <T as tensorcontract::Element>::Real: tensorcontract::KernelSet,
{
    let out = r(true).unwrap();
    let t = time_it(target, samples, || {
        r(false);
    });
    (t, out)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

const ARMS: [&str; 17] = [
    "tc_exec",
    "tc_call",
    "tp_exec",
    "tp_packed_exec",
    "tp_call",
    "ce_exec",
    "ce_call",
    "tf_exec",
    "tf_call",
    "tf_call_spc",
    "tf_call_subs",
    "tf_prep_exec",
    "tf_plan_exec",
    "tf_eager",
    "tf_eager_scoped",
    "tf_traced",
    "tf_traced_nary",
];

fn run_case<T: Elem>(p: &Program, arms: &[String], target: Duration, samples: usize, seed: u64)
where
    <T as tensorcontract::Element>::Real: tensorcontract::KernelSet,
{
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // Scale MPS inputs so the chain neither overflows nor underflows.
    let scale = if p.name.starts_with("mps") {
        1.0 / (2.0 * p.shapes[0][0] as f64).sqrt()
    } else {
        1.0
    };
    let inputs: Vec<Vec<T>> = p
        .inputs
        .iter()
        .map(|d| {
            (0..numel(d))
                .map(|_| {
                    let x = T::rand(&mut rng);
                    // multiply by a real scale via repeated construction
                    let mut y = T::default();
                    let _ = &mut y;
                    scale_elem(x, scale)
                })
                .collect()
        })
        .collect();
    let reference = naive_program(p, &inputs);
    for arm in arms {
        let (times, out) = match arm.as_str() {
            "tc_exec" => time_runner(tc_exec(p, &inputs), target, samples),
            "tc_call" => time_runner(tc_call(p, &inputs), target, samples),
            "tp_exec" => time_runner(tp_exec(p, &inputs, false), target, samples),
            "tp_packed_exec" => time_runner(tp_exec(p, &inputs, true), target, samples),
            "tp_call" => time_runner(tp_call(p, &inputs), target, samples),
            "ce_exec" => time_runner(ce_exec(p, &inputs), target, samples),
            "ce_call" => time_runner(ce_call(p, &inputs), target, samples),
            "tf_exec" => tf_run(p, TfMode::Exec, &inputs, target, samples),
            "tf_call" => tf_run(p, TfMode::Call, &inputs, target, samples),
            "tf_call_spc" => tf_run(p, TfMode::CallSessionPerStep, &inputs, target, samples),
            "tf_call_subs" => tf_run(p, TfMode::CallSubscripts, &inputs, target, samples),
            "tf_prep_exec" => tf_run(p, TfMode::PrepareExecute, &inputs, target, samples),
            "tf_plan_exec" => tf_run(p, TfMode::PlanExecute, &inputs, target, samples),
            "tf_eager" => tf_eager(p, &inputs, target, samples),
            "tf_eager_scoped" => tf_eager_scoped(p, &inputs, target, samples),
            "tf_traced" => tf_traced(p, TraceMode::Steps, &inputs, target, samples),
            "tf_traced_nary" => tf_traced(p, TraceMode::Nary, &inputs, target, samples),
            other => panic!("unknown arm {other}"),
        };
        let err = rel_err(&out, &reference);
        assert!(err < 1e-10, "{} {arm}: rel err {err:e}", p.name);
        let med = median(times.clone());
        let min = times.iter().copied().fold(f64::INFINITY, f64::min);
        println!(
            "{},{},{},{},{},{:.1},{:.1},{:.2e}",
            p.name,
            p.dtype,
            p.steps.len(),
            p.macs,
            arm,
            med,
            min,
            err
        );
    }
}

fn scale_elem<T: Elem>(x: T, s: f64) -> T
where
    <T as tensorcontract::Element>::Real: tensorcontract::KernelSet,
{
    // T has no generic real scaling in our bound set; go through the type name.
    if T::NAME == "f64" {
        let v: f64 = unsafe { std::mem::transmute_copy(&x) };
        unsafe { std::mem::transmute_copy(&(v * s)) }
    } else {
        let v: Complex64 = unsafe { std::mem::transmute_copy(&x) };
        unsafe { std::mem::transmute_copy(&(v * s)) }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let filter = get("--case");
    let arms: Vec<String> = get("--arms")
        .map(|s| s.split(',').map(String::from).collect())
        .unwrap_or_else(|| ARMS.iter().map(|s| s.to_string()).collect());
    let target = Duration::from_millis(get("--sample-ms").map(|s| s.parse().unwrap()).unwrap_or(20));
    let samples: usize = get("--samples").map(|s| s.parse().unwrap()).unwrap_or(11);
    let seed: u64 = get("--seed").map(|s| s.parse().unwrap()).unwrap_or(61);
    if args.iter().any(|a| a == "--report") {
        // Which strategy the tcontract planner picks for each distinct step.
        for p in corpus() {
            let mut seen = std::collections::BTreeMap::new();
            for k in 0..p.steps.len() {
                let r = match p.dtype {
                    "f64" => tp_report::<f64>(&p, k),
                    _ => tp_report::<Complex64>(&p, k),
                };
                seen.entry(p.steps[k].eq()).or_insert(r);
            }
            for (eq, r) in seen {
                println!("{} {} {}", p.name, eq, r);
            }
        }
        return;
    }
    if args.iter().any(|a| a == "--list") {
        for p in corpus() {
            println!("{} {} steps={} macs={}", p.name, p.dtype, p.steps.len(), p.macs);
        }
        return;
    }
    println!("# three-engine-contract threads=1 sample_ms={} samples={samples} seed={seed}", target.as_millis());
    println!("case,dtype,steps,macs,arm,median_ns,min_ns,rel_err");
    for p in corpus() {
        if let Some(f) = &filter {
            if !p.name.contains(f.as_str()) {
                continue;
            }
        }
        match p.dtype {
            "f64" => run_case::<f64>(&p, &arms, target, samples, seed),
            "c64" => run_case::<Complex64>(&p, &arms, target, samples, seed),
            _ => unreachable!(),
        }
    }
}

fn dims_i64(d: &[usize]) -> Vec<i64> {
    d.iter().map(|&x| x as i64).collect()
}

/// The whole program as one N-ary einsum: globally unique integer labels
/// (shared indices unified across steps) and the JAX-style positional path that
/// reproduces the pairwise step order exactly.
fn nary_of(p: &Program) -> (tenferro_einsum::EinsumSubscripts, Vec<(usize, usize)>) {
    let (inputs, output) = global_labels(p);
    let refs: Vec<&[u32]> = inputs.iter().map(|v| v.as_slice()).collect();
    let subs = tenferro_einsum::EinsumSubscripts::new(&refs, &output);

    let mut list: Vec<usize> = (0..p.inputs.len()).collect();
    let mut path = Vec::new();
    for (k, st) in p.steps.iter().enumerate() {
        let i = list.iter().position(|&s| s == st.lhs).unwrap();
        let j = list.iter().position(|&s| s == st.rhs).unwrap();
        path.push((i, j));
        let (hi, lo) = (i.max(j), i.min(j));
        list.remove(hi);
        list.remove(lo);
        list.push(p.inputs.len() + k);
    }
    (subs, path)
}

/// Globally unique integer labels for the inputs and the final output, shared
/// indices unified across steps.
fn global_labels(p: &Program) -> (Vec<Vec<u32>>, Vec<u32>) {
    // Union-find over axis ids; inputs get fresh ids, step outputs inherit.
    let mut parent: Vec<usize> = Vec::new();
    fn find(parent: &mut Vec<usize>, x: usize) -> usize {
        let mut r = x;
        while parent[r] != r {
            r = parent[r];
        }
        let mut y = x;
        while parent[y] != r {
            let n = parent[y];
            parent[y] = r;
            y = n;
        }
        r
    }
    let mut ids: Vec<Vec<usize>> = Vec::new();
    for d in &p.inputs {
        ids.push(
            d.iter()
                .map(|_| {
                    parent.push(parent.len());
                    parent.len() - 1
                })
                .collect(),
        );
    }
    for st in &p.steps {
        let mut by_char: HashMap<char, usize> = HashMap::new();
        for (c, &id) in st.a.iter().zip(&ids[st.lhs]) {
            by_char.insert(*c, id);
        }
        for (c, &id) in st.b.iter().zip(&ids[st.rhs]) {
            if let Some(&prev) = by_char.get(c) {
                let (ra, rb) = (find(&mut parent, prev), find(&mut parent, id));
                parent[ra] = rb;
            } else {
                by_char.insert(*c, id);
            }
        }
        ids.push(st.d.iter().map(|c| by_char[c]).collect());
    }
    let mut resolve = |v: &[usize]| -> Vec<u32> { v.iter().map(|&x| find(&mut parent, x) as u32).collect() };
    let inputs: Vec<Vec<u32>> = (0..p.inputs.len()).map(|i| resolve(&ids[i])).collect();
    let output = resolve(ids.last().unwrap());
    (inputs, output)
}

#[derive(Clone, Copy, PartialEq)]
enum TraceMode {
    /// One traced einsum op per pairwise step.
    Steps,
    /// One N-ary traced einsum with an explicit path equal to the step order.
    Nary,
}

/// tenferro traced: trace, compile and prepare once (untimed), then time
/// `run_prepared` inside one `with_execution_scope`.
fn tf_traced<T: Elem>(p: &Program, mode: TraceMode, inputs: &[Vec<T>], target: Duration, samples: usize) -> (Vec<f64>, Vec<T>)
where
    <T as tensorcontract::Element>::Real: tensorcontract::KernelSet,
{
    use tenferro_einsum::{EinsumOptimize, TraceContextEinsumExt};
    use tenferro_runtime::program::ProgramInputSpec;
    use tenferro_runtime::{GraphCompiler, Runtime, TraceContext};
    let backend = tf_backend();
    let ins = tf_inputs(p, inputs);
    let mut trace = TraceContext::new();
    let mut slots = Vec::new();
    for (t, d) in ins.iter().zip(&p.inputs) {
        let shape: Vec<_> = d.iter().map(|&x| x.into()).collect();
        slots.push(trace.input(ProgramInputSpec::new(t.dtype(), shape)).expect("trace input"));
    }
    let out = match mode {
        TraceMode::Steps => {
            for st in &p.steps {
                let v = trace
                    .einsum(&[slots[st.lhs], slots[st.rhs]], st.eq().as_str())
                    .expect("trace einsum");
                slots.push(v);
            }
            *slots.last().unwrap()
        }
        TraceMode::Nary => {
            let (subs, path) = nary_of(p);
            trace
                .einsum_subscripts_with(&slots, &subs, EinsumOptimize::Path(path))
                .expect("trace nary einsum")
        }
    };
    let graph = trace.finish(&[out]).expect("finish");
    let program = GraphCompiler::new().compile_traced_graph(&graph).expect("compile");
    let mut builder = Runtime::builder();
    builder
        .register_engine(tenferro_cpu::runtime_engine_registration(&backend).expect("reg"))
        .expect("register");
    builder
        .install_extension_module(
            tenferro_einsum::extension_module::<tenferro_cpu::CpuBackend>(tenferro_cpu::runtime_engine_id().expect("id"))
                .expect("module"),
        )
        .expect("install");
    let runtime = builder.build().expect("runtime");
    let refs: Vec<&tenferro_tensor::Tensor> = ins.iter().collect();
    let prepared = runtime.prepare_compiled(&program, &refs).expect("prepare");
    backend
        .with_execution_scope(|| {
            let times = time_it(target, samples, || {
                black_box(runtime.run_prepared(&prepared, &refs).expect("replay"));
            });
            let mut o = runtime.run_prepared(&prepared, &refs).expect("replay");
            (times, o.remove(0).as_slice::<T>().expect("slice").to_vec())
        })
        .expect("scope")
}

/// `tf_eager`, but inside one `with_execution_scope` (the documented
/// steady-state entry for sequential eager work).
fn tf_eager_scoped<T: Elem>(p: &Program, inputs: &[Vec<T>], target: Duration, samples: usize) -> (Vec<f64>, Vec<T>)
where
    <T as tensorcontract::Element>::Real: tensorcontract::KernelSet,
{
    let backend = tf_backend();
    let b2 = backend.clone();
    backend
        .with_execution_scope(|| tf_eager_with(p, inputs, target, samples, b2))
        .expect("scope")
}

fn tp_report<T: Elem>(p: &Program, k: usize) -> String
where
    <T as tensorcontract::Element>::Real: tensorcontract::KernelSet,
{
    let plan = tcontract::Plan::<T>::new(&tp_problem::<T>(p, k), &tcontract::PlanConfig::default())
        .expect("tp plan");
    let r = plan.report();
    format!(
        "algorithm={} beta_zero={:?} family={:?}",
        r.algorithm.name(),
        r.beta_zero.map(|a| a.name()),
        r.packed.as_ref().map(|x| x.family_id)
    )
}

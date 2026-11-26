//! `Plan::retained_bytes` of a two-qubit gate applied to an n-qubit complex64
//! state, at the positions of the tenferro-rs#2742 reproducer (tcontract-rs#221).
//!
//! `cargo run --release -p tcontract --example retained_gate -- [qubits [plans]]`
//!
//! The state is `2^n` elements (so the tensors dominate); what this prints is
//! what each plan keeps *besides* them. Only plans are built; nothing executes.
use num_complex::Complex32;
use tcontract::api::{CSpec, Labels, LayoutSpec, OperandSpec, Problem, Scalar};
use tcontract::{Plan, PlanConfig};

fn col_major(dims: &[usize]) -> Vec<isize> {
    dims.iter()
        .scan(1isize, |acc, &d| {
            let s = *acc;
            *acc *= d as isize;
            Some(s)
        })
        .collect()
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .map_or(26, |s| s.parse().expect("qubit count"));
    let pairs = [
        (0, 13),
        (3, 17),
        (6, 21),
        (9, 25),
        (3, 12),
        (7, 15),
        (11, 18),
        (15, 21),
    ];
    let spec = |d: &[usize]| OperandSpec::new(LayoutSpec::new(d, &col_major(d), 0).unwrap());
    let state_dims = vec![2usize; n];
    let gate_dims = [2usize; 4];
    let state_labels: Vec<i64> = (0..n as i64).collect();
    let mut total = 0usize;
    let mut plans = Vec::new();
    let plans_wanted: usize = std::env::args()
        .nth(2)
        .map_or(pairs.len(), |s| s.parse().expect("plan count"));
    for &(p, q) in pairs.iter().take(plans_wanted) {
        let out: Vec<i64> = state_labels
            .iter()
            .map(|&l| match l {
                l if l == p => 1000,
                l if l == q => 1001,
                l => l,
            })
            .collect();
        let labels = Labels::new(&state_labels, &[1000, 1001, p, q], &out);
        let problem = Problem::from_labels(
            <Complex32 as Scalar>::STORAGE,
            spec(&state_dims),
            spec(&gate_dims),
            CSpec::Absent,
            spec(&state_dims),
            &labels,
        )
        .unwrap();
        let plan = Plan::<Complex32>::new(&problem, &PlanConfig::default()).unwrap();
        println!(
            "gate ({p:2}, {q:2}): {:?}, retained_bytes {}",
            plan.report().algorithm,
            plan.retained_bytes()
        );
        total += plan.retained_bytes();
        plans.push(plan);
    }
    println!(
        "{n} qubits, {} plans: retained_bytes total {total} ({:.3} MiB)",
        plans.len(),
        total as f64 / (1 << 20) as f64
    );
}

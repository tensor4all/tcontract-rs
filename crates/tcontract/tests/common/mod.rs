//! Reference contraction and view helpers for the tcontract tests.
#![allow(dead_code)]

use tcontract::api::{DotGeneral, Scalar};
use tcontract_kernel::{Element, Real};

/// A dense tensor with explicit strides over its own storage.
#[derive(Clone, Debug)]
pub struct T<S> {
    pub data: Vec<S>,
    pub dims: Vec<usize>,
    pub strides: Vec<isize>,
    pub offset: isize,
}

pub fn col_major_strides(dims: &[usize]) -> Vec<isize> {
    let mut s = Vec::with_capacity(dims.len());
    let mut acc = 1isize;
    for &d in dims {
        s.push(acc);
        acc *= d.max(1) as isize;
    }
    s
}

pub fn fill<S: Scalar>(len: usize, seed: u64) -> Vec<S> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let re = (s % 1000) as f64 / 1000.0 - 0.5;
            let im = ((s >> 20) % 1000) as f64 / 1000.0 - 0.5;
            <S as Element>::from_parts(Real::from_f64(re), Real::from_f64(im))
        })
        .collect()
}

impl<S: Scalar> T<S> {
    /// Column-major tensor with deterministic data.
    pub fn new(dims: &[usize], seed: u64) -> Self {
        let n: usize = dims.iter().product();
        Self {
            data: fill(n, seed),
            dims: dims.to_vec(),
            strides: col_major_strides(dims),
            offset: 0,
        }
    }

    /// Same logical tensor stored with its axes physically permuted by
    /// `storage_order` (axis `storage_order[0]` fastest).
    pub fn restride(&self, storage_order: &[usize]) -> Self {
        let pd: Vec<usize> = storage_order.iter().map(|&a| self.dims[a]).collect();
        let ps = col_major_strides(&pd);
        let mut strides = vec![0isize; self.dims.len()];
        for (k, &a) in storage_order.iter().enumerate() {
            strides[a] = ps[k];
        }
        let mut out = Self {
            data: vec![<S as Element>::zero(); self.data.len()],
            dims: self.dims.clone(),
            strides,
            offset: 0,
        };
        for_each_index(&self.dims, |idx| {
            let v = self.get(idx);
            out.set(idx, v);
        });
        out
    }

    /// Same logical tensor with every axis reversed in memory (negative strides).
    pub fn reversed(&self) -> Self {
        let dims = self.dims.clone();
        let base = col_major_strides(&dims);
        let strides: Vec<isize> = base.iter().map(|s| -s).collect();
        let offset: isize = dims
            .iter()
            .zip(&base)
            .map(|(&d, &s)| (d.max(1) as isize - 1) * s)
            .sum();
        let mut out = Self {
            data: vec![<S as Element>::zero(); self.data.len()],
            dims: dims.clone(),
            strides,
            offset,
        };
        for_each_index(&dims, |idx| {
            let v = self.get(idx);
            out.set(idx, v);
        });
        out
    }

    pub fn pos(&self, idx: &[usize]) -> usize {
        (self.offset
            + idx
                .iter()
                .zip(&self.strides)
                .map(|(&i, &s)| i as isize * s)
                .sum::<isize>()) as usize
    }
    pub fn get(&self, idx: &[usize]) -> S {
        self.data[self.pos(idx)]
    }
    pub fn set(&mut self, idx: &[usize], v: S) {
        let p = self.pos(idx);
        self.data[p] = v;
    }
    pub fn view(&self) -> strided_view::StridedView<'_, S> {
        strided_view::StridedView::new(&self.data, &self.dims, &self.strides, self.offset).unwrap()
    }
    pub fn view_mut(&mut self) -> strided_view::StridedViewMut<'_, S> {
        strided_view::StridedViewMut::new(&mut self.data, &self.dims, &self.strides, self.offset)
            .unwrap()
    }
}

pub fn for_each_index(dims: &[usize], mut f: impl FnMut(&[usize])) {
    if dims.contains(&0) {
        return;
    }
    let mut idx = vec![0usize; dims.len()];
    loop {
        f(&idx);
        let mut k = 0;
        loop {
            if k == dims.len() {
                return;
            }
            idx[k] += 1;
            if idx[k] < dims[k] {
                break;
            }
            idx[k] = 0;
            k += 1;
        }
    }
}

/// Reference `C = alpha * contract(op(A), op(B)) + beta * C` in the
/// `[lhs_free, rhs_free, batch]` output convention.
#[allow(clippy::too_many_arguments)]
pub fn reference<S: Scalar>(
    cfg: &DotGeneral,
    alpha: S,
    a: &T<S>,
    ca: bool,
    b: &T<S>,
    cb: bool,
    beta: S,
    c: &T<S>,
) -> T<S> {
    let op = |x: S, c: bool| if c { Element::conj(x) } else { x };
    let a_free: Vec<usize> = (0..a.dims.len())
        .filter(|x| !cfg.lhs_contract().contains(x) && !cfg.lhs_batch().contains(x))
        .collect();
    let b_free: Vec<usize> = (0..b.dims.len())
        .filter(|x| !cfg.rhs_contract().contains(x) && !cfg.rhs_batch().contains(x))
        .collect();
    let kdims: Vec<usize> = cfg.lhs_contract().iter().map(|&x| a.dims[x]).collect();
    let mut out = c.clone();
    for_each_index(&c.dims, |o| {
        let (om, rest) = o.split_at(a_free.len());
        let (on, oh) = rest.split_at(b_free.len());
        let mut acc = <S as Element>::zero();
        let mut kidx = vec![0usize; kdims.len()];
        let mut run = |kidx: &[usize]| {
            let mut ai = vec![0usize; a.dims.len()];
            let mut bi = vec![0usize; b.dims.len()];
            for (p, &ax) in a_free.iter().enumerate() {
                ai[ax] = om[p];
            }
            for (p, &ax) in b_free.iter().enumerate() {
                bi[ax] = on[p];
            }
            for (p, (&la, &lb)) in cfg.lhs_batch().iter().zip(cfg.rhs_batch()).enumerate() {
                ai[la] = oh[p];
                bi[lb] = oh[p];
            }
            for (p, (&la, &lb)) in cfg
                .lhs_contract()
                .iter()
                .zip(cfg.rhs_contract())
                .enumerate()
            {
                ai[la] = kidx[p];
                bi[lb] = kidx[p];
            }
            acc = Element::add(acc, Element::mul(op(a.get(&ai), ca), op(b.get(&bi), cb)));
        };
        if kdims.iter().all(|&d| d > 0) {
            if kdims.is_empty() {
                run(&kidx);
            } else {
                for_each_index(&kdims, |k| run(k));
            }
        }
        let _ = &mut kidx;
        let base = if beta == <S as Element>::zero() {
            <S as Element>::zero()
        } else {
            Element::mul(beta, c.get(o))
        };
        out.set(o, Element::add(Element::mul(alpha, acc), base));
    });
    out
}

/// max |x - y| / max(1, max |y|) over the logical elements.
pub fn rel_err<S: Scalar>(x: &T<S>, y: &T<S>) -> f64 {
    let mag = |z: S| {
        let (re, im): (f64, f64) = (Real::to_f64(Element::re(z)), Real::to_f64(Element::im(z)));
        re.hypot(im)
    };
    let mut scale = 1.0f64;
    for_each_index(&y.dims, |i| scale = scale.max(mag(y.get(i))));
    let mut err = 0.0f64;
    for_each_index(&y.dims, |i| {
        err = err.max(mag(Element::add(
            x.get(i),
            Element::mul(
                y.get(i),
                <S as Element>::from_parts(Real::from_f64(-1.0), Real::from_f64(0.0)),
            ),
        )))
    });
    err / scale
}

/// The output extents of a validated contraction.
pub fn out_dims(cfg: &DotGeneral, a: &[usize], b: &[usize]) -> Vec<usize> {
    cfg.validate(a, b).unwrap().out_dims.to_vec()
}
pub mod plans;

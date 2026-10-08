//! A small k-d tree for exact nearest-neighbour *values*.
//!
//! Distances are computed with [`crate::nn::sqdist`], identical to R's. Pruning
//! uses a box lower bound built from the same per-dimension operations; since
//! IEEE subtraction, squaring and summation are monotone, the bound never
//! exceeds the computed distance of any point in the box, so pruning with
//! `bound >= best` cannot change the minimum found.

use crate::nn::sqdist;

const LEAF: usize = 16;

struct Node {
    lo: usize, // range into `idx`
    hi: usize,
    left: u32,  // child node ids (u32::MAX for leaves)
    right: u32,
    bmin: Vec<f64>,
    bmax: Vec<f64>,
}

pub struct KdTree<'a> {
    x: &'a [f64],
    p: usize,
    idx: Vec<u32>,
    nodes: Vec<Node>,
}

impl<'a> KdTree<'a> {
    pub fn build(x: &'a [f64], p: usize, points: Vec<u32>) -> Self {
        let mut t = KdTree { x, p, idx: points, nodes: Vec::new() };
        if !t.idx.is_empty() {
            let n = t.idx.len();
            t.build_rec(0, n);
        }
        t
    }

    fn coord(&self, i: u32, d: usize) -> f64 {
        self.x[i as usize * self.p + d]
    }

    fn build_rec(&mut self, lo: usize, hi: usize) -> u32 {
        let p = self.p;
        let mut bmin = vec![f64::INFINITY; p];
        let mut bmax = vec![f64::NEG_INFINITY; p];
        for &i in &self.idx[lo..hi] {
            for d in 0..p {
                let v = self.coord(i, d);
                if v < bmin[d] {
                    bmin[d] = v;
                }
                if v > bmax[d] {
                    bmax[d] = v;
                }
            }
        }
        let id = self.nodes.len() as u32;
        self.nodes.push(Node { lo, hi, left: u32::MAX, right: u32::MAX, bmin: bmin.clone(), bmax: bmax.clone() });
        if hi - lo > LEAF {
            let dim = (0..p).max_by(|&a, &b| (bmax[a] - bmin[a]).partial_cmp(&(bmax[b] - bmin[b])).unwrap()).unwrap();
            let mid = (lo + hi) / 2;
            let x = self.x;
            self.idx[lo..hi].select_nth_unstable_by(mid - lo, |&a, &b| {
                x[a as usize * p + dim].partial_cmp(&x[b as usize * p + dim]).unwrap()
            });
            let l = self.build_rec(lo, mid);
            let r = self.build_rec(mid, hi);
            self.nodes[id as usize].left = l;
            self.nodes[id as usize].right = r;
        }
        id
    }

    #[inline]
    fn box_bound(&self, node: &Node, q: &[f64]) -> f64 {
        let mut s = 0.0f64;
        for d in 0..self.p {
            let v = q[d];
            let diff = if v < node.bmin[d] {
                node.bmin[d] - v
            } else if v > node.bmax[d] {
                v - node.bmax[d]
            } else {
                0.0
            };
            s += diff * diff;
        }
        s
    }

    /// Lower `best` (a squared distance) to the nearest point in the tree.
    pub fn search(&self, q: &[f64], best: &mut f64) {
        if self.nodes.is_empty() {
            return;
        }
        self.search_rec(0, q, best);
    }

    fn search_rec(&self, id: u32, q: &[f64], best: &mut f64) {
        let node = &self.nodes[id as usize];
        if self.box_bound(node, q) >= *best {
            return;
        }
        if node.left == u32::MAX {
            let p = self.p;
            for &i in &self.idx[node.lo..node.hi] {
                let d = sqdist(&self.x[i as usize * p..(i as usize + 1) * p], q);
                if d < *best {
                    *best = d;
                }
            }
            return;
        }
        let (l, r) = (node.left, node.right);
        let bl = self.box_bound(&self.nodes[l as usize], q);
        let br = self.box_bound(&self.nodes[r as usize], q);
        if bl <= br {
            self.search_rec(l, q, best);
            self.search_rec(r, q, best);
        } else {
            self.search_rec(r, q, best);
            self.search_rec(l, q, best);
        }
    }
}

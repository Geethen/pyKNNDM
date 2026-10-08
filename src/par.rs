//! Minimal data-parallel helper on `std::thread::scope` (no dependencies).

use std::sync::OnceLock;

pub fn threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("KNNDM_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))
            .max(1)
    })
}

/// Fill `out[i] = f(i)` in parallel, splitting into contiguous blocks.
pub fn par_fill<T: Send, F: Fn(usize) -> T + Sync>(out: &mut [T], f: F) {
    let n = out.len();
    let nt = threads().min(n.max(1));
    if nt <= 1 || n < 64 {
        for (i, o) in out.iter_mut().enumerate() {
            *o = f(i);
        }
        return;
    }
    // more blocks than threads for load balancing
    let nblocks = (nt * 8).min(n);
    let bsize = n.div_ceil(nblocks);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let blocks: Vec<std::sync::Mutex<Option<&mut [T]>>> =
        out.chunks_mut(bsize).map(|c| std::sync::Mutex::new(Some(c))).collect();
    std::thread::scope(|s| {
        for _ in 0..nt {
            s.spawn(|| loop {
                let b = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if b >= blocks.len() {
                    break;
                }
                let chunk = blocks[b].lock().unwrap().take().unwrap();
                let base = b * bsize;
                for (t, o) in chunk.iter_mut().enumerate() {
                    *o = f(base + t);
                }
            });
        }
    });
}

pub fn par_map<T: Send + Default + Clone, F: Fn(usize) -> T + Sync>(n: usize, f: F) -> Vec<T> {
    let mut out = vec![T::default(); n];
    par_fill(&mut out, f);
    out
}

//! Log-bucketed histogram for approximate percentiles of `u64` samples.
//!
//! Values below 16 get a bucket each. Above that, every power-of-two range is
//! split into 16 equal slices, so a bucket is at most 1/16 as wide as its lower
//! bound and reporting its midpoint keeps the relative error under ~3.2%.
//! The bucket table is fixed at 976 counters no matter how many samples arrive.
//! `min`, `max`, `count` and `mean` are tracked separately and are exact.

const EXACT: u64 = 16;
const SUB_BITS: u32 = 4;
const SUBS: usize = 1 << SUB_BITS;
const BUCKETS: usize = EXACT as usize + (64 - SUB_BITS as usize) * SUBS;

#[derive(Clone, Debug)]
pub struct Histogram {
    buckets: Vec<u64>,
    count: u64,
    sum: u128,
    min: u64,
    max: u64,
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

fn bucket_index(v: u64) -> usize {
    if v < EXACT {
        return v as usize;
    }
    let octave = 63 - v.leading_zeros();
    let shift = octave - SUB_BITS;
    let sub = ((v >> shift) as usize) & (SUBS - 1);
    EXACT as usize + (shift as usize) * SUBS + sub
}

/// Inclusive value range covered by a bucket.
fn bucket_range(idx: usize) -> (u64, u64) {
    if idx < EXACT as usize {
        return (idx as u64, idx as u64);
    }
    let j = idx - EXACT as usize;
    let shift = (j / SUBS) as u32;
    let sub = (j % SUBS) as u64;
    let lo = (EXACT + sub) << shift;
    (lo, lo + ((1u64 << shift) - 1))
}

impl Histogram {
    pub fn new() -> Self {
        Histogram {
            buckets: vec![0; BUCKETS],
            count: 0,
            sum: 0,
            min: u64::MAX,
            max: 0,
        }
    }

    pub fn record(&mut self, v: u64) {
        self.buckets[bucket_index(v)] += 1;
        self.count += 1;
        self.sum += v as u128;
        self.min = self.min.min(v);
        self.max = self.max.max(v);
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn min(&self) -> Option<u64> {
        (self.count > 0).then_some(self.min)
    }

    pub fn max(&self) -> Option<u64> {
        (self.count > 0).then_some(self.max)
    }

    pub fn mean(&self) -> Option<f64> {
        (self.count > 0).then(|| self.sum as f64 / self.count as f64)
    }

    /// Approximate value at quantile `q` (0.0 to 1.0, clamped), using the
    /// nearest-rank definition. `None` when nothing has been recorded.
    pub fn quantile(&self, q: f64) -> Option<u64> {
        if self.count == 0 {
            return None;
        }
        let q = if q.is_nan() { 0.0 } else { q.clamp(0.0, 1.0) };
        let rank = ((q * self.count as f64).ceil() as u64).clamp(1, self.count);
        let mut seen = 0u64;
        for (idx, &n) in self.buckets.iter().enumerate() {
            seen += n;
            if seen >= rank {
                let (lo, hi) = bucket_range(idx);
                // The extremes are known exactly, so never report past them.
                return Some((lo + (hi - lo) / 2).clamp(self.min, self.max));
            }
        }
        Some(self.max)
    }

    pub fn merge(&mut self, other: &Histogram) {
        for (a, b) in self.buckets.iter_mut().zip(&other.buckets) {
            *a += b;
        }
        self.count += other.count;
        self.sum += other.sum;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_histogram_has_no_answers() {
        let h = Histogram::new();
        assert_eq!(h.count(), 0);
        assert_eq!(h.min(), None);
        assert_eq!(h.max(), None);
        assert_eq!(h.mean(), None);
        assert_eq!(h.quantile(0.5), None);
    }

    #[test]
    fn small_values_are_exact() {
        let mut h = Histogram::new();
        for v in 0..16 {
            h.record(v);
        }
        assert_eq!(h.quantile(0.0), Some(0));
        assert_eq!(h.quantile(0.5), Some(7));
        assert_eq!(h.quantile(1.0), Some(15));
    }

    #[test]
    fn buckets_tile_the_range_without_gaps() {
        let mut expected_lo = 0u64;
        for idx in 0..BUCKETS {
            let (lo, hi) = bucket_range(idx);
            assert_eq!(lo, expected_lo, "bucket {idx}");
            assert!(hi >= lo);
            assert_eq!(bucket_index(lo), idx);
            assert_eq!(bucket_index(hi), idx);
            expected_lo = hi.wrapping_add(1);
        }
        assert_eq!(expected_lo, 0, "last bucket must end at u64::MAX");
    }

    #[test]
    fn extremes_do_not_overflow() {
        let mut h = Histogram::new();
        h.record(0);
        h.record(u64::MAX);
        assert_eq!(h.min(), Some(0));
        assert_eq!(h.max(), Some(u64::MAX));
        assert_eq!(h.quantile(1.0), Some(u64::MAX));
    }

    #[test]
    fn quantile_error_stays_under_bound() {
        let n = 100_000u64;
        let mut h = Histogram::new();
        for v in 1..=n {
            h.record(v * 7);
        }
        for &q in &[0.01, 0.1, 0.5, 0.9, 0.99, 0.999] {
            let exact = ((q * n as f64).ceil() as u64) * 7;
            let got = h.quantile(q).unwrap();
            let err = (got as f64 - exact as f64).abs() / exact as f64;
            assert!(err < 0.032, "q={q} exact={exact} got={got} err={err}");
        }
    }

    #[test]
    fn min_max_mean_are_exact() {
        let mut h = Histogram::new();
        for v in [25u64, 791, 472, 100_000] {
            h.record(v);
        }
        assert_eq!(h.min(), Some(25));
        assert_eq!(h.max(), Some(100_000));
        assert_eq!(h.mean(), Some(101_288.0 / 4.0));
    }

    #[test]
    fn quantile_is_clamped_to_observed_range() {
        let mut h = Histogram::new();
        h.record(1000);
        assert_eq!(h.quantile(0.5), Some(1000));
        assert_eq!(h.quantile(-3.0), Some(1000));
        assert_eq!(h.quantile(f64::NAN), Some(1000));
    }

    #[test]
    fn merge_matches_recording_everything_in_one() {
        let mut a = Histogram::new();
        let mut b = Histogram::new();
        let mut all = Histogram::new();
        for v in 0..5000u64 {
            let v = v * 13 + 1;
            if v % 2 == 0 {
                a.record(v);
            } else {
                b.record(v);
            }
            all.record(v);
        }
        a.merge(&b);
        assert_eq!(a.count(), all.count());
        assert_eq!(a.min(), all.min());
        assert_eq!(a.max(), all.max());
        assert_eq!(a.mean(), all.mean());
        for &q in &[0.1, 0.5, 0.9, 0.99] {
            assert_eq!(a.quantile(q), all.quantile(q));
        }
    }
}

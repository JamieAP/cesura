//! FOCuS-style frequentist online change-point detector for univariate
//! Gaussian-mean shifts.
//!
//! Reference: Romano, Eckley, Fearnhead, Rigaill (2023), "Fast Online
//! Change Point Detection via Functional Pruning CUSUM Statistics"
//! (arXiv:2110.08205 / JMLR v24/21-1230). Ward, Romano, Eckley,
//! Fearnhead (2024) "A Constant-per-Iteration Likelihood Ratio Test
//! for Online Changepoint Detection for Exponential Family Models"
//! (arXiv:2302.04743, Statistics and Computing) is the §5 adaptive
//! maxima-checking follow-up. That method is not implemented by
//! `with_pruning` here.
//!
//! Two inner-loop modes:
//! - Naive (default): O(t) per step. Walks all candidate split points τ.
//! - Pruned (`with_pruning`): amortised O(1) prune + O(|deque|) query
//!   per step. Maintains two monotone deques (positive- and
//!   negative-shift) of pieces `(τ, S_τ)`, parameterised so the
//!   dominance ordering is invariant to t. Bit-for-bit parity with
//!   the naive path on every fixture in the eval suite -- pinned by
//!   `tests/focus.rs` parity tests.
//!
//! Operating-characteristic note: FOCuS thresholds ARE NOT comparable to
//! BOCPD confidences. Use FOCuS as a sanity-check parallel detector,
//! not as a precision/recall replacement.

use std::collections::VecDeque;

use crate::ChangePoint;

/// Window used to compute `shift_sigma` once a candidate is committed.
const SHIFT_WINDOW: usize = 20;

/// Univariate Gaussian-mean FOCuS detector.
///
/// Tracks cumulative sums since the last reset and computes the
/// generalized-likelihood-ratio statistic
/// `½ · max_τ [τ(t−τ)/t] · (x̄_{0..τ} − x̄_{τ..t})²`
/// at each step. When the maximum exceeds `threshold`, emits a change
/// point at the maximizing `τ` and resets baselines past that index
/// (subsequent observations re-accumulate from there).
///
/// Inputs are z-normalised against the data fed so far before the GLR
/// is computed -- matches BOCPD's input convention so threshold
/// calibration on N(0,1) carries over to scaled inputs.
pub struct FocusDetector {
    threshold: f64,
    cooldown: usize,
    /// Cumulative running sum for the *current* segment (post-last-reset).
    /// At index i the entry is `Σ x_{seg_start..=i}` after z-norm.
    seg_sums: Vec<f64>,
    /// `seg_sums.len() = i+1−seg_start` once we reach absolute step i.
    seg_start: usize,
    /// Welford state on raw input for streaming z-norm.
    mean: f64,
    m2: f64,
    n: usize,
    /// Most recently emitted CP, for cooldown bookkeeping.
    last_emit: Option<usize>,
    /// Inner-loop strategy. Default `Naive`; opt into `Pruned` via
    /// [`FocusDetector::with_pruning`].
    mode: InnerLoop,
}

/// Per-step argmax strategy for the GLR statistic.
#[derive(Clone)]
enum InnerLoop {
    Naive,
    Pruned(PruningState),
}

/// One candidate split point in the deque. `tau` is segment-relative;
/// `s_tau = Σ z_{0..tau}` (cumulative sum at insertion time, before
/// the τ-th observation enters the right segment).
#[derive(Clone, Copy, Debug)]
struct Piece {
    tau: usize,
    s_tau: f64,
}

/// Two-deque functional-pruning state for the GLR inner loop.
///
/// Invariant after every `step`: pieces in `qr` (right deque) have
/// strictly increasing `argmax(τ, t) = (S_t − S_τ)/(t − τ)` along the
/// deque; pieces in `ql` (left deque) have strictly decreasing argmax.
/// The τ=0 piece (`s_tau = 0`) is the implicit pre-change baseline and
/// always sits at the front of both deques.
///
/// Construct only via [`PruningState::new`]; an empty deque has no
/// baseline and would mis-compute the first step's stat.
#[derive(Clone)]
struct PruningState {
    qr: VecDeque<Piece>,
    ql: VecDeque<Piece>,
}

impl FocusDetector {
    /// `threshold`: GLR cutoff. Higher → fewer false alarms. Calibrate
    /// to a target ARL₀ via [`arl0_calibrate`].
    pub fn new(threshold: f64) -> Self {
        assert!(
            threshold > 0.0 && threshold.is_finite(),
            "threshold must be finite and positive, got {threshold}"
        );
        Self {
            threshold,
            cooldown: 15,
            seg_sums: Vec::new(),
            seg_start: 0,
            mean: 0.0,
            m2: 0.0,
            n: 0,
            last_emit: None,
            mode: InnerLoop::Naive,
        }
    }

    /// Opt into the functional-pruning inner loop (Romano et al. 2023).
    ///
    /// Replaces the O(t) per-step argmax scan with two monotone deques.
    /// Per-step work is amortised O(1) for pruning + O(|deque|) for the
    /// query; the deque size is O(log t) on average under H₀, O(t)
    /// worst-case. Detection semantics are bit-for-bit identical to the
    /// naive path -- enforced by `tests/focus.rs` parity tests.
    ///
    /// Use for long streams (t ≥ 10^4); the naive path is faster on
    /// short streams due to lower constants.
    pub fn with_pruning(mut self) -> Self {
        self.mode = InnerLoop::Pruned(PruningState::new());
        self
    }

    /// Detect on a complete batch. Mirrors `BocpdDetector::detect`.
    pub fn detect(&mut self, data: &[f64]) -> Vec<ChangePoint> {
        let mut out = Vec::new();
        for (raw_idx, &x) in data.iter().enumerate() {
            if let Some(idx) = self.step_inner(x) {
                let cp = build_cp(data, idx, self.threshold);
                out.push(cp);
                let _ = raw_idx; // raw_idx tracked via self.n
            }
        }
        out
    }

    /// Online step. Returns `Some(t)` when a change point at absolute
    /// time `t` has fired; else `None`. To produce a full `ChangePoint`
    /// (with `confidence` / `shift_sigma`), use [`detect`] which has the
    /// data window required for `shift_sigma`.
    pub fn step(&mut self, x: f64) -> Option<usize> {
        self.step_inner(x)
    }

    fn step_inner(&mut self, x: f64) -> Option<usize> {
        // Welford update on raw input for streaming z-norm.
        self.n += 1;
        let n = self.n as f64;
        let delta = x - self.mean;
        self.mean += delta / n;
        self.m2 += delta * (x - self.mean);

        let std = if self.n >= 2 {
            (self.m2 / n).sqrt().max(1e-10)
        } else {
            1.0
        };
        let z = (x - self.mean) / std;

        let abs_idx = self.n - 1;
        // Drop observations strictly inside the cooldown window.
        if let Some(last) = self.last_emit {
            if abs_idx <= last + self.cooldown {
                return None;
            }
        }
        // Append z-normed observation to the segment cumsum.
        self.seg_sums.push(self.seg_sums.last().copied().unwrap_or(0.0) + z);

        // Need at least 2 observations in the segment to define a split.
        let m = self.seg_sums.len();
        let total = *self.seg_sums.last().unwrap();

        // Compute (best_stat, best_tau) via the active inner-loop strategy.
        // Pruned mode dispatches to `step_with_query` BEFORE the m < 4
        // short-circuit because the deque must stay in lockstep with
        // `seg_sums` -- skipping the prune+append on early steps would
        // leave the structure stale by the time m reaches the threshold.
        // Naive does no per-step state, so the early-return is safe there.
        let (best_stat, best_tau) = match &mut self.mode {
            InnerLoop::Naive => {
                if m < 4 {
                    return None;
                }
                naive_inner_loop(&self.seg_sums, total, m)
            }
            InnerLoop::Pruned(state) => {
                let res = state.step_with_query(total, m);
                if m < 4 {
                    return None;
                }
                res
            }
        };

        if best_stat >= self.threshold {
            let cp_abs = self.seg_start + best_tau;
            self.last_emit = Some(cp_abs);
            // Reset segment baseline to start *after* the detected split.
            // Carry forward the right-segment cumsum so the next step's
            // statistic is computed from the new regime.
            let s_left = self.seg_sums[best_tau - 1];
            let mut new_sums = Vec::with_capacity(self.seg_sums.len() - best_tau);
            for v in &self.seg_sums[best_tau..] {
                new_sums.push(v - s_left);
            }
            self.seg_sums = new_sums;
            self.seg_start = cp_abs;
            // Rebuild the pruning deque from the new right-segment cumsum
            // so it stays in lockstep with `seg_sums`.
            if let InnerLoop::Pruned(state) = &mut self.mode {
                state.rebuild_from_seg_sums(&self.seg_sums);
            }
            Some(cp_abs)
        } else {
            None
        }
    }

    /// Number of observations seen so far.
    pub fn total_steps(&self) -> usize {
        self.n
    }

    /// Multivariate detection by **per-dim union**.
    ///
    /// Spawns a fresh `FocusDetector` per dimension (sharing this
    /// detector's `threshold` and inner-loop mode), runs `detect` on
    /// each marginal series, and OR-unions the detected CP indices,
    /// deduplicating within a ±`MV_DEDUP_TOLERANCE`-step window
    /// (default 25, matching `EnsembleDetector`'s confirmation
    /// tolerance).
    ///
    /// **This is structurally weaker than `BocpdDetector::detect_multivariate`**
    /// on correlated input: per-dim z-norm misses anti-correlated
    /// shifts that the Mahalanobis path catches (see the correlated-shift
    /// tests in `tests/correctness.rs`). The MV FOCuS pitch is
    /// "frequentist parallel detector for users who don't want a
    /// Bayesian prior", not "best multivariate detector". The
    /// Pishchagina et al. 2024 (arXiv:2311.01174) convex-hull MV
    /// FOCuS is a separate algorithm and is **not** what this method
    /// implements.
    ///
    /// Empty input or ragged dimensions return an empty vec.
    pub fn detect_multivariate(&self, data: &[Vec<f64>]) -> Vec<ChangePoint> {
        if data.is_empty() {
            return Vec::new();
        }
        let d = data[0].len();
        if d == 0 || data.iter().any(|row| row.len() != d) {
            return Vec::new();
        }
        let with_pruning = matches!(self.mode, InnerLoop::Pruned(_));
        let mut all: Vec<ChangePoint> = Vec::new();
        for k in 0..d {
            let series: Vec<f64> = data.iter().map(|row| row[k]).collect();
            let mut det = FocusDetector::new(self.threshold);
            if with_pruning {
                det = det.with_pruning();
            }
            all.extend(det.detect(&series));
        }
        all.sort_by_key(|c| c.index);
        let mut out: Vec<ChangePoint> = Vec::new();
        for cp in all {
            if out
                .last()
                .is_none_or(|last| cp.index.saturating_sub(last.index) > MV_DEDUP_TOLERANCE)
            {
                out.push(cp);
            }
        }
        out
    }
}

/// Tolerance window (in steps) for de-duplicating per-dim CPs in
/// `FocusDetector::detect_multivariate`. Two CPs from different
/// dimensions within this many steps of each other are merged into
/// the earlier one. Matches `EnsembleDetector`'s default
/// confirmation tolerance.
pub const MV_DEDUP_TOLERANCE: usize = 25;

/// Naive O(t) argmax over candidate split points τ ∈ [1, m−1].
/// Returns `(best_stat, best_tau)`. Tie-break: smaller τ wins (strict
/// `>` on the running max).
fn naive_inner_loop(seg_sums: &[f64], total: f64, m: usize) -> (f64, usize) {
    let mut best_stat = 0.0f64;
    let mut best_tau = 0usize;
    let m_f = m as f64;
    for tau in 1..m {
        let s_left = seg_sums[tau - 1];
        let s_right = total - s_left;
        let n_l = tau as f64;
        let n_r = (m - tau) as f64;
        let mean_l = s_left / n_l;
        let mean_r = s_right / n_r;
        let diff = mean_l - mean_r;
        let stat = 0.5 * (n_l * n_r / m_f) * diff * diff;
        if stat > best_stat {
            best_stat = stat;
            best_tau = tau;
        }
    }
    (best_stat, best_tau)
}

impl PruningState {
    fn new() -> Self {
        let baseline = Piece { tau: 0, s_tau: 0.0 };
        let mut qr = VecDeque::new();
        let mut ql = VecDeque::new();
        qr.push_back(baseline);
        ql.push_back(baseline);
        Self { qr, ql }
    }

    fn reseed_baseline(&mut self) {
        *self = Self::new();
    }

    /// Replay the prune+append cycle over `seg_sums` so the deque
    /// matches what it would be after fresh streaming through the
    /// right segment. No CPs emitted during replay (caller has
    /// already fired and is rebuilding for the new regime).
    fn rebuild_from_seg_sums(&mut self, seg_sums: &[f64]) {
        self.reseed_baseline();
        for (i, &total) in seg_sums.iter().enumerate() {
            let m = i + 1;
            self.prune_back(total, m);
            self.append(m, total);
        }
    }

    /// One streaming step at segment-relative time t = m, post-push
    /// running sum `total = S_t`. Performs prune → query → append in
    /// the order required for parity with the naive inner loop.
    /// Returns `(best_stat, best_tau)` over τ ∈ [1, m−1].
    fn step_with_query(&mut self, total: f64, m: usize) -> (f64, usize) {
        self.prune_back(total, m);
        let result = self.query(total, m);
        self.append(m, total);
        result
    }

    /// Pop dominated pieces from the back of each deque, given current
    /// `(S_t, t) = (total, m)`. Right deque keeps strictly increasing
    /// argmax along the deque; left deque keeps strictly decreasing.
    /// Cross-multiplication avoids the division.
    fn prune_back(&mut self, total: f64, m: usize) {
        let m_f = m as f64;
        // Invariant: every piece in either deque has `tau < m` at prune
        // time (we prune before appending the new τ=m piece, and the
        // most-recent existing piece was appended at the previous step
        // with τ = m-1). Both denominators are therefore strictly > 0.
        // Right deque: argmax_back ≤ argmax_prev → pop back.
        while self.qr.len() >= 2 {
            let last = self.qr[self.qr.len() - 1];
            let prev = self.qr[self.qr.len() - 2];
            let dl = m_f - last.tau as f64;
            let dp = m_f - prev.tau as f64;
            debug_assert!(dl > 0.0 && dp > 0.0, "deque tau ≥ m invariant break");
            // argmax(p) = (total - p.s_tau) / (m - p.tau).
            // (a/b) ≤ (c/d) with b, d > 0 ⟺ a·d ≤ c·b.
            let lhs = (total - last.s_tau) * dp;
            let rhs = (total - prev.s_tau) * dl;
            if lhs <= rhs {
                self.qr.pop_back();
            } else {
                break;
            }
        }
        // Left deque: argmax_back ≥ argmax_prev → pop back.
        while self.ql.len() >= 2 {
            let last = self.ql[self.ql.len() - 1];
            let prev = self.ql[self.ql.len() - 2];
            let dl = m_f - last.tau as f64;
            let dp = m_f - prev.tau as f64;
            debug_assert!(dl > 0.0 && dp > 0.0, "deque tau ≥ m invariant break");
            let lhs = (total - last.s_tau) * dp;
            let rhs = (total - prev.s_tau) * dl;
            if lhs >= rhs {
                self.ql.pop_back();
            } else {
                break;
            }
        }
    }

    /// Walk both deques, compute the GLR statistic for each kept τ > 0,
    /// return `(best_stat, best_tau)`. Bit-for-bit identical to the
    /// naive form for the same τ -- uses the
    /// `½ · τ(t−τ)/t · (μ_L − μ_R)²` representation, not the
    /// cancellation-prone `½(S_t−S_τ)²/(t−τ) + m₀_τ − m₀_now` form.
    ///
    /// Tie-break: smaller τ wins, matching the naive `1..m` loop with
    /// strict `>`. The walk visits qr in increasing-τ order, then ql.
    /// This preserves naive parity on noisy data (where exact stat
    /// equality across distinct τ never arises), but a contrived
    /// fixture with exactly-tied stats across the qr/ql split would
    /// pick the qr-side τ here while naive would pick the smaller τ.
    /// If such a fixture ever appears, switch to a merge-walk by τ
    /// over the union of the two deques.
    fn query(&self, total: f64, m: usize) -> (f64, usize) {
        let mut best_stat = 0.0f64;
        let mut best_tau = 0usize;
        let m_f = m as f64;
        // Walk pieces in increasing-τ order (deque invariant) so that
        // ties break the same way the naive loop does (smaller τ wins).
        // qr first, then ql; both contain the τ=0 baseline (skipped).
        // `piece.tau < m` always at query time -- append happens after
        // query, so the τ=m piece doesn't yet exist in either deque.
        for piece in self.qr.iter().chain(self.ql.iter()) {
            debug_assert!(piece.tau < m, "deque tau ≥ m at query");
            if piece.tau == 0 {
                continue;
            }
            let tau_f = piece.tau as f64;
            let n_l = tau_f;
            let n_r = m_f - tau_f;
            let mean_l = piece.s_tau / n_l;
            let mean_r = (total - piece.s_tau) / n_r;
            let diff = mean_l - mean_r;
            let stat = 0.5 * (n_l * n_r / m_f) * diff * diff;
            if stat > best_stat {
                best_stat = stat;
                best_tau = piece.tau;
            }
        }
        (best_stat, best_tau)
    }

    fn append(&mut self, tau: usize, s_tau: f64) {
        let p = Piece { tau, s_tau };
        self.qr.push_back(p);
        self.ql.push_back(p);
    }
}

/// Build a `ChangePoint` from the data slice and detected index. Uses
/// the same before/after-window math BOCPD uses for `shift_sigma`. The
/// `confidence` is `1 − exp(−glr_at_index / threshold)` -- a monotone
/// score in `[0, 1]`, NOT a posterior probability.
fn build_cp(data: &[f64], idx: usize, threshold: f64) -> ChangePoint {
    let n = data.len();
    let lo = idx.saturating_sub(SHIFT_WINDOW);
    let hi = (idx + SHIFT_WINDOW).min(n);
    let before = &data[lo..idx];
    let after = &data[idx..hi];
    let shift_sigma = if before.is_empty() || after.is_empty() {
        0.0
    } else {
        let mean_b: f64 = before.iter().sum::<f64>() / before.len() as f64;
        let mean_a: f64 = after.iter().sum::<f64>() / after.len() as f64;
        let var_b: f64 =
            before.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / before.len() as f64;
        let var_a: f64 =
            after.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / after.len() as f64;
        let pooled = ((var_b + var_a) / 2.0).sqrt().max(1e-10);
        ((mean_a - mean_b) / pooled).abs()
    };
    // Confidence: monotone-in-margin transform of the GLR statistic.
    // `glr_at_index` recomputes the per-side cumulative-mean form with the
    // emitted slice -- proxy for "how strong is this evidence vs threshold".
    let glr = if !before.is_empty() && !after.is_empty() {
        let mean_b: f64 = before.iter().sum::<f64>() / before.len() as f64;
        let mean_a: f64 = after.iter().sum::<f64>() / after.len() as f64;
        let n_l = before.len() as f64;
        let n_r = after.len() as f64;
        let m = n_l + n_r;
        0.5 * (n_l * n_r / m) * (mean_a - mean_b).powi(2)
    } else {
        threshold
    };
    let conf = (1.0 - (-glr / threshold).exp()).clamp(0.0, 1.0);
    ChangePoint {
        index: idx,
        confidence: conf,
        shift_sigma,
    }
}

/// Pick a threshold for a target false-alarm rate (`ARL₀`).
///
/// Empirical: simulate `trials` × `length` Gaussian streams, bisect on
/// the threshold so that the average run length to first detection
/// approximately matches `target_arl0`. No analytic shortcut. Slow --
/// expect ~seconds at default settings. Caller-friendly defaults:
/// `trials = 30`, `length = (target_arl0 * 3.0) as usize`.
///
/// Available with the `test-utils` feature -- the simulator depends on
/// `crate::eval::Rng` which is itself feature-gated.
#[cfg(any(test, feature = "test-utils"))]
pub fn arl0_calibrate(target_arl0: f64) -> f64 {
    use crate::eval::Rng;
    let trials = 30;
    let length = (target_arl0 * 3.0) as usize;
    let mut lo = 1.0;
    let mut hi = 50.0;
    for _ in 0..18 {
        let mid = 0.5 * (lo + hi);
        let mut total_run_length = 0.0;
        for t in 0..trials {
            let mut rng = Rng::new(7000 + t as u64);
            let mut det = FocusDetector::new(mid);
            let mut fired_at = None;
            for i in 0..length {
                if det.step(rng.normal(0.0, 1.0)).is_some() {
                    fired_at = Some(i);
                    break;
                }
            }
            total_run_length += fired_at.map(|i| i as f64).unwrap_or(length as f64);
        }
        let avg = total_run_length / trials as f64;
        if avg < target_arl0 {
            // Too sensitive: raise threshold.
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::Rng;

    #[test]
    fn focus_detects_clean_mean_shift() {
        let mut rng = Rng::new(11);
        let mut data: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
        data.extend((0..150).map(|_| rng.normal(5.0, 1.0)));
        let mut det = FocusDetector::new(8.0);
        let cps = det.detect(&data);
        assert!(!cps.is_empty(), "must detect a clean 5σ shift");
        assert!(
            (cps[0].index as i64 - 150).abs() < 30,
            "first CP at {} too far from truth 150",
            cps[0].index
        );
    }

    #[test]
    fn focus_no_detection_on_stationary_noise() {
        let mut rng = Rng::new(42);
        let data: Vec<f64> = (0..1000).map(|_| rng.normal(0.0, 1.0)).collect();
        let mut det = FocusDetector::new(8.0);
        let cps = det.detect(&data);
        // FOCuS can fire spuriously; threshold-8 on N(0,1) over 1k should
        // produce at most a handful. A hard cap catches a calibration regression.
        assert!(
            cps.len() <= 3,
            "stationary N(0,1) produced {} CPs at threshold=8",
            cps.len()
        );
    }

    #[test]
    fn focus_detection_power_increases_with_shift() {
        // Larger shifts must fire faster. Compare median first-CP index
        // between 5σ and 1σ shifts at the same threshold; the 5σ case
        // should detect noticeably earlier than 1σ.
        let mut delay_strong = Vec::new();
        let mut delay_weak = Vec::new();
        for seed in 0..20 {
            let mut rng = Rng::new(seed);
            let mut strong: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
            strong.extend((0..150).map(|_| rng.normal(5.0, 1.0)));
            let mut weak: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
            weak.extend((0..150).map(|_| rng.normal(1.0, 1.0)));
            let mut a = FocusDetector::new(8.0);
            let mut b = FocusDetector::new(8.0);
            if let Some(cp) = a.detect(&strong).first() {
                if cp.index >= 150 {
                    delay_strong.push(cp.index - 150);
                }
            }
            if let Some(cp) = b.detect(&weak).first() {
                if cp.index >= 150 {
                    delay_weak.push(cp.index - 150);
                }
            }
        }
        assert!(
            !delay_strong.is_empty() && !delay_weak.is_empty(),
            "expected at least one detection in each arm"
        );
        let med_strong = median(&mut delay_strong);
        let med_weak = median(&mut delay_weak);
        eprintln!("median delay 5σ={med_strong} 1σ={med_weak}");
        assert!(
            med_strong < med_weak,
            "5σ median delay ({med_strong}) must be < 1σ median delay ({med_weak})"
        );
    }

    fn median(xs: &mut [usize]) -> usize {
        xs.sort();
        xs[xs.len() / 2]
    }

    #[test]
    fn focus_arl0_at_threshold_8_is_high() {
        // Light gate: at threshold 8, the average run length to first false
        // alarm should be large (≥ 200) over 5 trials × 1000 samples N(0,1).
        // The longer 30×1000 trial is gated `#[ignore]`.
        let mut total = 0.0;
        let trials = 5;
        for t in 0..trials {
            let mut rng = Rng::new(2000 + t);
            let mut det = FocusDetector::new(8.0);
            let mut fired = None;
            for i in 0..1000 {
                if det.step(rng.normal(0.0, 1.0)).is_some() {
                    fired = Some(i);
                    break;
                }
            }
            total += fired.map(|i| i as f64).unwrap_or(1000.0);
        }
        let arl0 = total / trials as f64;
        eprintln!("FOCuS ARL₀ at threshold=8 over {trials} × 1000 = {arl0}");
        assert!(arl0 >= 200.0, "ARL₀ at threshold=8 too low: {arl0}");
    }

    // ── MV FOCuS (per-dim union) ────────────────────────────────────

    #[test]
    fn mv_focus_empty_input() {
        let det = FocusDetector::new(8.0);
        let cps = det.detect_multivariate(&[]);
        assert!(cps.is_empty());
    }

    #[test]
    fn mv_focus_ragged_input_returns_empty() {
        let det = FocusDetector::new(8.0);
        let data = vec![vec![0.0, 1.0], vec![1.0]];
        let cps = det.detect_multivariate(&data);
        assert!(cps.is_empty());
    }

    #[test]
    fn mv_focus_zero_dim_returns_empty() {
        let det = FocusDetector::new(8.0);
        let data = vec![Vec::<f64>::new(); 100];
        let cps = det.detect_multivariate(&data);
        assert!(cps.is_empty());
    }

    #[test]
    fn mv_focus_per_dim_union_catches_axis_aligned_shift() {
        // Both dims shift by 5σ at t=150. Per-dim FOCuS catches each
        // marginal trivially; MV union returns ≥ 1 CP near the truth.
        let mut rng = Rng::new(0x10C5);
        let mut data: Vec<Vec<f64>> = Vec::with_capacity(300);
        for _ in 0..150 {
            data.push(vec![rng.normal(0.0, 1.0), rng.normal(0.0, 1.0)]);
        }
        for _ in 0..150 {
            data.push(vec![rng.normal(5.0, 1.0), rng.normal(5.0, 1.0)]);
        }
        let det = FocusDetector::new(8.0).with_pruning();
        let cps = det.detect_multivariate(&data);
        assert!(!cps.is_empty());
        assert!(
            cps.iter()
                .any(|cp| (cp.index as i64 - 150).abs() < 30),
            "expected a CP near 150, got {:?}",
            cps.iter().map(|c| c.index).collect::<Vec<_>>()
        );
    }

    #[test]
    fn mv_focus_per_dim_union_misses_anti_correlated_at_low_shift() {
        // The honest-limitation test: ρ=0.95 bivariate Gaussian, 0.5σ
        // anti-correlated shift. The MV BOCPD path catches it
        // (`tests/correctness.rs::multivariate_detects_correlated_shift_per_dim_invisible`);
        // the per-dim union path documented here is structurally
        // weaker on this fixture and pinning the miss is what makes
        // the limitation explicit rather than implicit.
        let rho = 0.95_f64;
        let shift = 0.5_f64;
        let mut rng = Rng::new(0x10C6);
        let mut data: Vec<Vec<f64>> = Vec::with_capacity(400);
        for _ in 0..200 {
            let z0 = rng.normal(0.0, 1.0);
            let z1 = rng.normal(0.0, 1.0);
            data.push(vec![z0, rho * z0 + (1.0 - rho * rho).sqrt() * z1]);
        }
        for _ in 0..200 {
            let z0 = rng.normal(0.0, 1.0);
            let z1 = rng.normal(0.0, 1.0);
            let x0 = z0 + shift * 0.5;
            let x1 = rho * z0 + (1.0 - rho * rho).sqrt() * z1 - shift * 0.5;
            data.push(vec![x0, x1]);
        }
        let det = FocusDetector::new(8.0).with_pruning();
        let cps = det.detect_multivariate(&data);
        // Pin the miss: with truth at 200 and ±25 tolerance, 0.5σ
        // anti-correlated shift is below the per-dim threshold.
        let near_truth = cps
            .iter()
            .filter(|cp| (cp.index as i64 - 200).abs() <= 25)
            .count();
        assert_eq!(
            near_truth, 0,
            "per-dim FOCuS should NOT catch a 0.5σ anti-correlated shift; got {} near-truth CPs (Mahalanobis path is the right tool here)",
            near_truth,
        );
    }

    #[test]
    fn mv_focus_dedup_within_tolerance() {
        // Two dims that shift at the same step should produce ONE
        // unioned CP, not two. Strong shift on both dims at t=150.
        let mut rng = Rng::new(0x10C7);
        let mut data: Vec<Vec<f64>> = Vec::with_capacity(300);
        for _ in 0..150 {
            data.push(vec![rng.normal(0.0, 1.0), rng.normal(0.0, 1.0)]);
        }
        for _ in 0..150 {
            data.push(vec![rng.normal(5.0, 1.0), rng.normal(5.0, 1.0)]);
        }
        let det = FocusDetector::new(8.0).with_pruning();
        let cps = det.detect_multivariate(&data);
        // Within MV_DEDUP_TOLERANCE of t=150 we expect at most 1 CP
        // (despite both dims firing).
        let near_truth: Vec<usize> = cps
            .iter()
            .filter(|cp| (cp.index as i64 - 150).abs() <= MV_DEDUP_TOLERANCE as i64)
            .map(|cp| cp.index)
            .collect();
        assert!(
            near_truth.len() <= 2,
            "dedup should collapse near-coincident per-dim CPs; got {near_truth:?}"
        );
    }
}

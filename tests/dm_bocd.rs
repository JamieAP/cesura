//! Operating-characteristics tests for `DmBocdDetector`.
//!
//! Mirrors the patterns in `tests/correctness.rs` and `tests/properties.rs`
//! used for `BocpdDetector::detect_multivariate`. These pin the
//! Path A_min behaviour: detects mean shifts on whitened MV data,
//! does not over-fire on stationary streams, comparable hit rates to
//! the existing NIW-BOCPD path on the fixtures both detectors are
//! supposed to win on.

use cesura::eval::Rng;
use cesura::{BocpdDetector, ConformalCpWrapper, DmBocdDetector};

/// Two-regime, anti-correlated 1σ shift on d=2 with ρ=0.95.
/// This is the fixture the joint-whitening MV path is supposed to
/// catch (covariance-aware), and Dm-BOCD-mean-only should also catch
/// it because the shift survives whitening.
#[test]
fn detects_anticorrelated_2d_shift() {
    let regime_len = 200usize;
    let n_pre = regime_len;
    let n_post = regime_len;
    let rho = 0.95_f64;
    let shift = 1.0_f64;
    let mut rng = Rng::new(0xDA0F_C0DE);

    let mut data: Vec<Vec<f64>> = Vec::with_capacity(n_pre + n_post);
    for _ in 0..n_pre {
        let z0 = rng.normal(0.0, 1.0);
        let z1 = rng.normal(0.0, 1.0);
        data.push(vec![z0, rho * z0 + (1.0 - rho * rho).sqrt() * z1]);
    }
    for _ in 0..n_post {
        let z0 = rng.normal(0.0, 1.0);
        let z1 = rng.normal(0.0, 1.0);
        let off = shift * 0.5;
        let x0 = z0 + off;
        let x1 = rho * z0 + (1.0 - rho * rho).sqrt() * z1 - off;
        data.push(vec![x0, x1]);
    }

    let det = DmBocdDetector::new(2, 200.0, n_pre + n_post + 50);
    let cps = det.detect_multivariate(&data);
    assert!(
        !cps.is_empty(),
        "expected ≥ 1 CP on anti-correlated 1σ shift, got 0"
    );
    let post: Vec<&cesura::ChangePoint> =
        cps.iter().filter(|c| c.index >= n_pre - 30 && c.index <= n_pre + 60).collect();
    assert!(
        !post.is_empty(),
        "expected at least one CP within 60 steps of the true CP at {n_pre}, got indices {:?}",
        cps.iter().map(|c| c.index).collect::<Vec<_>>()
    );
}

/// Stationary correlated 2D Gaussian -- must not over-fire.
#[test]
fn no_oversfire_on_stationary_2d() {
    let n = 500;
    let rho = 0.7_f64;
    let mut rng = Rng::new(0x5EED_5EED);
    let data: Vec<Vec<f64>> = (0..n)
        .map(|_| {
            let z0 = rng.normal(0.0, 1.0);
            let z1 = rng.normal(0.0, 1.0);
            vec![z0, rho * z0 + (1.0 - rho * rho).sqrt() * z1]
        })
        .collect();

    let det = DmBocdDetector::new(2, 500.0, n + 50);
    let cps = det.detect_multivariate(&data);
    // Allow up to 2 false alarms over 500 steps at λ=500. λ=500 implies
    // ~1 CP / 500 steps; 2 is the loose ceiling that still catches
    // catastrophic over-firing.
    assert!(
        cps.len() <= 2,
        "stationary 2D over-fired: {} CPs at indices {:?}",
        cps.len(),
        cps.iter().map(|c| c.index).collect::<Vec<_>>()
    );
}

/// d=1 stationary parity smoke: Dm-BOCD mean-only on d=1 should not
/// over-fire any worse than `BocpdDetector::detect`. Counts ≤ baseline + 2
/// across a long stationary stream.
#[test]
fn d1_stationary_does_not_overshoot_nig_baseline() {
    let n = 800;
    let mut rng = Rng::new(0x00C0_FFEE);
    let data: Vec<f64> = (0..n).map(|_| rng.normal(0.0, 1.0)).collect();
    let mv: Vec<Vec<f64>> = data.iter().map(|&x| vec![x]).collect();

    let nig = BocpdDetector::new(500.0, n + 50);
    let nig_cps = nig.detect(&data).len();

    let dm = DmBocdDetector::new(1, 500.0, n + 50);
    let dm_cps = dm.detect_multivariate(&mv).len();

    assert!(
        dm_cps <= nig_cps + 2,
        "Dm-BOCD d=1 over-fired vs NIG baseline: dm={dm_cps} nig={nig_cps}"
    );
}

/// Detects a clean d=1 mean shift. Loose bound -- within 60 steps of the
/// true CP. This is the "does the math even fire?" smoke.
#[test]
fn d1_clean_mean_shift_fires() {
    let mut rng = Rng::new(0xBEEF);
    let mut data: Vec<f64> = (0..150).map(|_| rng.normal(0.0, 1.0)).collect();
    data.extend((0..150).map(|_| rng.normal(4.0, 1.0)));
    let mv: Vec<Vec<f64>> = data.iter().map(|&x| vec![x]).collect();

    let det = DmBocdDetector::new(1, 200.0, 350);
    let cps = det.detect_multivariate(&mv);
    let near: Vec<usize> = cps.iter().map(|c| c.index).filter(|&i| (i as i64 - 150).abs() <= 60).collect();
    assert!(!near.is_empty(), "expected ≥ 1 CP within 60 steps of 150; got {:?}", cps.iter().map(|c| c.index).collect::<Vec<_>>());
}

/// Builder smoke: with_omega + with_mass_cutoff + with_prior chain
/// without panicking, and the resulting detector still fires on a
/// mean shift.
#[test]
fn builder_chain_smoke() {
    let prior_mu = vec![0.0, 0.0];
    let prior_sigma_inv = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

    let det = DmBocdDetector::new(2, 200.0, 350)
        .with_omega(0.2)
        .with_mass_cutoff(1e-3)
        .with_prior(prior_mu, prior_sigma_inv);

    let mut rng = Rng::new(0xFADE);
    let mut data: Vec<Vec<f64>> = (0..150)
        .map(|_| vec![rng.normal(0.0, 1.0), rng.normal(0.0, 1.0)])
        .collect();
    data.extend((0..150).map(|_| vec![rng.normal(3.0, 1.0), rng.normal(3.0, 1.0)]));

    let cps = det.detect_multivariate(&data);
    assert!(!cps.is_empty(), "builder-chained detector failed to fire");
}

/// Conformal compat: `DmBocdDetector` impls `MvScoredDetect`, so the
/// rolling-quantile timing-interval wrapper composes over it the same
/// way it does over `BocpdDetector`. Pin: emission-count parity between
/// raw `detect_multivariate` and `ConformalCpWrapper::detect_multivariate`.
#[test]
fn mv_scored_detect_impl_composes_with_conformal_wrapper() {
    let mut rng = Rng::new(0x60CC_AB1E);
    let mut data: Vec<Vec<f64>> = (0..200)
        .map(|_| vec![rng.normal(0.0, 1.0), rng.normal(0.0, 1.0)])
        .collect();
    data.extend((0..200).map(|_| vec![rng.normal(3.0, 1.0), rng.normal(-3.0, 1.0)]));
    data.extend((0..200).map(|_| vec![rng.normal(0.0, 1.0), rng.normal(0.0, 1.0)]));

    let det = DmBocdDetector::new(2, 200.0, 700);
    let raw = det.detect_multivariate(&data);
    let scored = det.detect_multivariate_with_score(&data);
    assert_eq!(
        raw.len(),
        scored.len(),
        "with_score must emit one (cp, score) per cp"
    );
    assert!(scored.iter().all(|(_, s)| s.is_finite() && *s >= 0.0));

    let mut wrapped = ConformalCpWrapper::new(DmBocdDetector::new(2, 200.0, 700))
        .with_calibration_capacity(50)
        .with_coverage(0.9);
    let conformal = wrapped.detect_multivariate(&data);
    assert_eq!(
        conformal.len(),
        raw.len(),
        "wrapper must preserve raw emission count"
    );
    let raw_idx: Vec<usize> = raw.iter().map(|c| c.index).collect();
    let conf_idx: Vec<usize> = conformal.iter().map(|c| c.cp.index).collect();
    assert_eq!(raw_idx, conf_idx, "wrapper must preserve emission indices");
}

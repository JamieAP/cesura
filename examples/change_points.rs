use cesura::BocpdDetector;

fn main() {
    let samples: Vec<f64> = (0..400)
        .map(|i| if i < 200 { 0.0 } else { 5.0 })
        .collect();
    let detector = BocpdDetector::new(200.0, 250);
    for point in detector.detect(&samples) {
        println!("index={} confidence={:.3}", point.index, point.confidence);
    }
}

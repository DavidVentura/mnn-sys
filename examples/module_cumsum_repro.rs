use mnn_sys::{InferenceConfig, ModuleEngine, NamedInput, TensorData};
use std::env;
use std::path::Path;

fn input_data(t: usize, seed: f32) -> Vec<f32> {
    let mut data = Vec::with_capacity(9 * t);
    for i in 0..(9 * t) {
        let x = (i as f32 + 1.0) * seed;
        data.push((x.sin() * 0.7) + (x.cos() * 0.3));
    }
    data
}

fn run(engine: &ModuleEngine, t: usize, seed: f32) -> Vec<f32> {
    let data = input_data(t, seed);
    let shape = [1_usize, 9, t];
    let outputs = engine
        .run_named_dynamic(
            &[NamedInput {
                name: "x",
                data: TensorData::F32(&data),
                shape: &shape,
            }],
            &["y"],
        )
        .expect("run module");
    outputs.into_iter().next().expect("output").data
}

fn mae(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum::<f32>()
        / a.len().min(b.len()) as f32
}

fn load(path: &Path) -> ModuleEngine {
    ModuleEngine::from_file(
        path,
        &["x"],
        &["y"],
        Some(InferenceConfig::new().with_threads(4)),
    )
    .expect("load module")
}

fn main() {
    let model = env::args().nth(1).expect("usage: module_cumsum_repro <model.mnn>");
    let model = Path::new(&model);

    let shared = load(model);
    let a0 = run(&shared, 8, 0.17);
    let b0 = run(&shared, 5, 0.31);
    let a1 = run(&shared, 8, 0.17);

    let fresh_a0 = run(&load(model), 8, 0.17);
    let fresh_b0 = run(&load(model), 5, 0.31);
    let fresh_a1 = run(&load(model), 8, 0.17);

    println!("shared_a0_vs_fresh_a0_mae={:.9}", mae(&a0, &fresh_a0));
    println!("shared_b0_vs_fresh_b0_mae={:.9}", mae(&b0, &fresh_b0));
    println!("shared_a1_vs_fresh_a1_mae={:.9}", mae(&a1, &fresh_a1));
    println!("shared_a0_vs_shared_a1_mae={:.9}", mae(&a0, &a1));
}

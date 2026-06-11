//! Raw inference timing for the PP-OCR det + rec MNN graphs, isolated from
//! the translator-rs pipeline (no image decode, preprocessing, or CTC).
//! Defaults match the app's session config: precision Low, memory Low, det
//! at 4 intra-threads, rec at 1 thread x N parallel sessions.
//!
//! Usage: bench_ocr <det.mnn> <rec.mnn> [iters]
//!          [--precision low|normal|high] [--memory low|normal|high]
//!          [--section det|rec|par|batch|all] [--det-threads N]

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use mnn_sys::{Backend, InferenceConfig, InferenceEngine, MemoryMode, PrecisionMode};

const REC_HEIGHT: usize = 48;

#[derive(Clone)]
struct Options {
    precision: PrecisionMode,
    memory: MemoryMode,
    backend: Backend,
    rec_gpu: bool,
    section: String,
    det_threads: Vec<i32>,
    par_sessions: usize,
    pin: Vec<usize>,
    iters: usize,
}

fn pin_current_thread(cpu: usize) {
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    unsafe {
        libc::CPU_SET(cpu, &mut set);
        let rc = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
        assert_eq!(rc, 0, "pin to cpu{cpu} failed");
    }
}

fn pseudo_random_input(len: usize) -> Vec<f32> {
    let mut state = 0x2545F4914F6CDD1Du64;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

fn load_engine(path: &Path, threads: i32, opts: &Options) -> InferenceEngine {
    // For GPU backends MNN reinterprets numThread as a mode bitmask, where
    // 4 = TUNING_WIDE (the default); 1 means TUNING_NONE and silently falls
    // back to CPU-like behavior, so keep CPU thread counts away from it.
    let threads = if opts.backend == Backend::CPU { threads } else { threads.max(4) };
    let config = InferenceConfig::new()
        .with_threads(threads)
        .with_precision(opts.precision)
        .with_memory(opts.memory)
        .with_backend(opts.backend);
    InferenceEngine::from_file(path, Some(config))
        .unwrap_or_else(|e| panic!("load {}: {e}", path.display()))
}

fn time_runs(engine: &InferenceEngine, input: &[f32], shape: &[usize], iters: usize) -> Vec<f32> {
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        engine.run_dynamic_raw(input, shape).expect("inference");
        samples.push(t.elapsed().as_secs_f32() * 1000.0);
    }
    samples
}

fn stats(samples: &mut [f32]) -> (f32, f32, f32) {
    samples.sort_by(f32::total_cmp);
    let min = samples[0];
    let median = samples[samples.len() / 2];
    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
    (min, median, mean)
}

fn bench_det(det_path: &Path, opts: &Options) {
    // 960x2144 is what the app's still cap feeds for a 1440x3216 screenshot;
    // the other shapes show how cost scales with area down to the live-path
    // 650kpx budget.
    let shapes: &[(usize, usize)] = &[(960, 2144), (736, 1632), (544, 1216)];
    println!("det: {}", det_path.display());
    println!(
        "{:>12} {:>8} {:>24} {:>12}",
        "shape", "threads", "ms min/median/mean", "Mpx/s"
    );
    for &(w, h) in shapes {
        let input = pseudo_random_input(3 * w * h);
        let shape = [1usize, 3, h, w];
        for &threads in &opts.det_threads {
            let engine = load_engine(det_path, threads, opts);
            let _ = engine.run_dynamic_raw(&input, &shape).expect("warmup");
            let mut samples = time_runs(&engine, &input, &shape, opts.iters);
            let (min, median, mean) = stats(&mut samples);
            let mpx_s = (w * h) as f32 / (median / 1000.0) / 1e6;
            println!(
                "{:>12} {:>8} {:>8.1}/{:>6.1}/{:>6.1} {:>12.2}",
                format!("{w}x{h}"),
                threads,
                min,
                median,
                mean,
                mpx_s
            );
        }
    }
}

fn bench_rec_widths(rec_path: &Path, opts: &Options) {
    println!("\nrec (1 session, 1 thread): {}", rec_path.display());
    println!(
        "{:>6} {:>24} {:>12}",
        "width", "ms min/median/mean", "ms/100px"
    );
    let engine = load_engine(rec_path, 1, opts);
    for &w in &[64usize, 192, 320, 576, 832, 960] {
        let input = pseudo_random_input(3 * REC_HEIGHT * w);
        let shape = [1usize, 3, REC_HEIGHT, w];
        let _ = engine.run_dynamic_raw(&input, &shape).expect("warmup");
        let mut samples = time_runs(&engine, &input, &shape, opts.iters);
        let (min, median, mean) = stats(&mut samples);
        println!(
            "{:>6} {:>8.1}/{:>6.1}/{:>6.1} {:>12.2}",
            w,
            min,
            median,
            mean,
            median / (w as f32 / 100.0)
        );
    }
}

/// Same total pixels submitted as N batch-1 runs, but as one [N,3,48,W]
/// call — measures whether batching amortizes per-run overhead or improves
/// kernel utilization.
fn bench_rec_batch(rec_path: &Path, opts: &Options) {
    let engine = load_engine(rec_path, 1, opts);
    for &w in &[320usize, 544] {
        println!("\nrec batch (1 session, 1 thread, width {w}):");
        println!(
            "{:>6} {:>24} {:>12}",
            "batch", "ms min/median/mean", "ms/strip"
        );
        for &n in &[1usize, 2, 4, 8, 16] {
            let input = pseudo_random_input(n * 3 * REC_HEIGHT * w);
            let shape = [n, 3, REC_HEIGHT, w];
            let _ = engine.run_dynamic_raw(&input, &shape).expect("warmup");
            let mut samples = time_runs(&engine, &input, &shape, opts.iters);
            let (min, median, mean) = stats(&mut samples);
            println!(
                "{:>6} {:>8.1}/{:>6.1}/{:>6.1} {:>12.2}",
                n,
                min,
                median,
                mean,
                median / n as f32
            );
        }
    }
}

/// Concurrent det (2 threads, `--backend`) + rec (4 CPU sessions plus one
/// GPU session when `--rec-gpu` is set, sharing one strip queue): tells
/// whether stages contend for the same resource and what a hybrid
/// GPU-det + CPU/GPU-rec pipeline would sustain.
fn bench_overlap(det_path: &Path, rec_path: &Path, opts: &Options) {
    let (w, h) = (960usize, 2144usize);
    let strip_w = 544usize;
    let total = 71usize;
    println!(
        "\noverlap: det {w}x{h} ({:?}) concurrent with rec {total} strips ({} CPU sessions{})",
        opts.backend,
        opts.par_sessions,
        if opts.rec_gpu { " + 1 GPU session" } else { "" }
    );
    let det = load_engine(det_path, if opts.backend == Backend::CPU { 2 } else { 4 }, opts);
    let det_input = pseudo_random_input(3 * w * h);
    let det_shape = [1usize, 3, h, w];
    let _ = det.run_dynamic_raw(&det_input, &det_shape).expect("warmup");
    let cpu_opts = Options { backend: Backend::CPU, ..opts.clone() };
    let gpu_opts = Options { backend: Backend::OpenCL, ..opts.clone() };
    let mut rec_sessions: Vec<InferenceEngine> = (0..opts.par_sessions)
        .map(|_| load_engine(rec_path, 1, &cpu_opts))
        .collect();
    if opts.rec_gpu {
        rec_sessions.push(load_engine(rec_path, 4, &gpu_opts));
    }
    let rec_input = pseudo_random_input(3 * REC_HEIGHT * strip_w);
    let rec_shape = [1usize, 3, REC_HEIGHT, strip_w];
    for engine in &rec_sessions {
        let _ = engine.run_dynamic_raw(&rec_input, &rec_shape).expect("warmup");
    }

    let queue = AtomicUsize::new(0);
    let t0 = Instant::now();
    let (det_ms, rec_ms) = std::thread::scope(|scope| {
        let det_handle = scope.spawn(|| {
            det.run_dynamic_raw(&det_input, &det_shape).expect("inference");
            t0.elapsed().as_secs_f32() * 1000.0
        });
        let rec_handles: Vec<_> = rec_sessions
            .iter()
            .map(|engine| {
                let rec_input = &rec_input;
                let rec_shape = &rec_shape;
                let queue = &queue;
                scope.spawn(move || {
                    while queue.fetch_add(1, Ordering::Relaxed) < total {
                        engine.run_dynamic_raw(rec_input, rec_shape).expect("inference");
                    }
                    t0.elapsed().as_secs_f32() * 1000.0
                })
            })
            .collect();
        let det_ms = det_handle.join().expect("det thread");
        let rec_ms = rec_handles
            .into_iter()
            .map(|jh| jh.join().expect("rec thread"))
            .fold(0.0f32, f32::max);
        (det_ms, rec_ms)
    });
    println!(
        "  det done at {det_ms:.1}ms  rec done at {rec_ms:.1}ms  wall={:.1}ms",
        det_ms.max(rec_ms)
    );
}

/// Det parallelism in MNN caps at 2 threads for this graph, so a 4-core
/// cluster idles half its capacity. This measures the workaround: split the
/// page into two horizontal tiles (+32px overlap each, /32-aligned) and run
/// them concurrently on two 2-thread sessions.
fn bench_det_tiled(det_path: &Path, opts: &Options) {
    let (w, h) = (960usize, 2144usize);
    let tile_h = (h / 2 + 32).div_ceil(32) * 32;
    println!("\ndet tiled: {w}x{h} as 2 concurrent {w}x{tile_h} tiles, 2 threads each");
    let whole = load_engine(det_path, 2, opts);
    let input = pseudo_random_input(3 * w * h);
    let shape = [1usize, 3, h, w];
    let _ = whole.run_dynamic_raw(&input, &shape).expect("warmup");
    let mut samples = time_runs(&whole, &input, &shape, opts.iters);
    let (_, whole_ms, _) = stats(&mut samples);

    let tiles: Vec<InferenceEngine> = (0..2).map(|_| load_engine(det_path, 2, opts)).collect();
    let tile_input = pseudo_random_input(3 * w * tile_h);
    let tile_shape = [1usize, 3, tile_h, w];
    for engine in &tiles {
        let _ = engine.run_dynamic_raw(&tile_input, &tile_shape).expect("warmup");
    }
    let mut tiled = Vec::with_capacity(opts.iters);
    for _ in 0..opts.iters {
        let t = Instant::now();
        std::thread::scope(|scope| {
            for engine in &tiles {
                let tile_input = &tile_input;
                let tile_shape = &tile_shape;
                scope.spawn(move || {
                    engine.run_dynamic_raw(tile_input, tile_shape).expect("inference");
                });
            }
        });
        tiled.push(t.elapsed().as_secs_f32() * 1000.0);
    }
    let (_, tiled_ms, _) = stats(&mut tiled);
    println!(
        "  whole={whole_ms:.1}ms  tiled={tiled_ms:.1}ms  speedup={:.2}x (tile overhead {:.0}% extra pixels)",
        whole_ms / tiled_ms,
        (2.0 * tile_h as f32 / h as f32 - 1.0) * 100.0
    );
}

/// Mimics the pipeline's strip workload: `total` strips of width `strip_w`
/// pulled from a shared queue by one thread per session — the same shape of
/// parallelism recognize_text_in_boxes_image uses via rayon work-stealing.
fn bench_rec_parallel(rec_path: &Path, strip_w: usize, total: usize, opts: &Options) {
    let n = opts.par_sessions;
    println!(
        "\nrec parallel: {total} strips of {strip_w}x{REC_HEIGHT} across {n} sessions, pin={:?}{}",
        opts.pin,
        if opts.rec_gpu { " + 1 GPU session" } else { "" }
    );
    let mut sessions: Vec<InferenceEngine> =
        (0..n).map(|_| load_engine(rec_path, 1, opts)).collect();
    if opts.rec_gpu {
        let gpu_opts = Options { backend: Backend::OpenCL, ..opts.clone() };
        sessions.push(load_engine(rec_path, 4, &gpu_opts));
    }
    let input = pseudo_random_input(3 * REC_HEIGHT * strip_w);
    let shape = [1usize, 3, REC_HEIGHT, strip_w];
    for engine in &sessions {
        let _ = engine.run_dynamic_raw(&input, &shape).expect("warmup");
    }

    let serial_t = Instant::now();
    for i in 0..total {
        let engine = &sessions[i % n];
        engine.run_dynamic_raw(&input, &shape).expect("inference");
    }
    let serial_ms = serial_t.elapsed().as_secs_f32() * 1000.0;

    let queue = AtomicUsize::new(0);
    let parallel_t = Instant::now();
    std::thread::scope(|scope| {
        for (idx, engine) in sessions.iter().enumerate() {
            let input = &input;
            let shape = &shape;
            let queue = &queue;
            let pin = opts.pin.get(idx).copied();
            scope.spawn(move || {
                if let Some(cpu) = pin {
                    pin_current_thread(cpu);
                }
                while queue.fetch_add(1, Ordering::Relaxed) < total {
                    engine.run_dynamic_raw(input, shape).expect("inference");
                }
            });
        }
    });
    let parallel_ms = parallel_t.elapsed().as_secs_f32() * 1000.0;

    println!(
        "  serial={serial_ms:.1}ms ({:.1}ms/strip)  parallel={parallel_ms:.1}ms ({:.1}ms/strip)  speedup={:.2}x",
        serial_ms / total as f32,
        parallel_ms / total as f32,
        serial_ms / parallel_ms
    );
}

fn print_profile(label: &str, rows: &[(String, String, f64, f64, String)]) {
    let total: f64 = rows.iter().map(|r| r.2).sum();
    println!("\nprofile {label}: {:.1}ms total across {} ops", total, rows.len());

    let mut by_type: std::collections::HashMap<&str, (f64, usize)> = std::collections::HashMap::new();
    for (ty, _, ms, _, _) in rows {
        let e = by_type.entry(ty).or_default();
        e.0 += ms;
        e.1 += 1;
    }
    let mut types: Vec<_> = by_type.into_iter().collect();
    types.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
    println!("{:>16} {:>8} {:>9} {:>6}", "op type", "ms", "share", "count");
    for (ty, (ms, count)) in &types {
        println!("{:>16} {:>8.1} {:>8.1}% {:>6}", ty, ms, ms / total * 100.0, count);
    }

    println!("top ops:");
    println!(
        "{:>14} {:>8} {:>7} {:>9} {:>18}  name",
        "type", "ms", "share", "GFLOP/s", "out shape"
    );
    for (ty, name, ms, mflops, shape) in rows.iter().take(25) {
        println!(
            "{:>14} {:>8.2} {:>6.1}% {:>9.1} {:>18}  {}",
            ty,
            ms,
            ms / total * 100.0,
            mflops / ms,
            shape,
            name
        );
    }
}

fn bench_profile(det_path: &Path, rec_path: &Path, opts: &Options) {
    let (w, h) = (960usize, 2144usize);
    let det = load_engine(det_path, opts.det_threads[0], opts);
    let det_input = pseudo_random_input(3 * w * h);
    let rows = det
        .run_profile(&det_input, &[1, 3, h, w], opts.iters)
        .expect("det profile");
    print_profile(&format!("det {w}x{h} ({} threads)", opts.det_threads[0]), &rows);

    let strip_w = 544usize;
    let rec = load_engine(rec_path, 1, opts);
    let rec_input = pseudo_random_input(3 * REC_HEIGHT * strip_w);
    let rows = rec
        .run_profile(&rec_input, &[1, 3, REC_HEIGHT, strip_w], opts.iters)
        .expect("rec profile");
    print_profile(&format!("rec {strip_w}x{REC_HEIGHT} (1 thread)"), &rows);
}

fn parse_options(args: &[String]) -> Options {
    let mut opts = Options {
        precision: PrecisionMode::Low,
        memory: MemoryMode::Low,
        backend: Backend::CPU,
        rec_gpu: false,
        section: "all".to_string(),
        det_threads: vec![1, 2, 4],
        par_sessions: 4,
        pin: Vec::new(),
        iters: 5,
    };
    if let Some(iters) = args.get(3).and_then(|s| s.parse().ok()) {
        opts.iters = iters;
    }
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--precision" => {
                i += 1;
                opts.precision = match args[i].as_str() {
                    "low" => PrecisionMode::Low,
                    "high" => PrecisionMode::High,
                    _ => PrecisionMode::Normal,
                };
            }
            "--memory" => {
                i += 1;
                opts.memory = match args[i].as_str() {
                    "low" => MemoryMode::Low,
                    "high" => MemoryMode::High,
                    _ => MemoryMode::Normal,
                };
            }
            "--section" => {
                i += 1;
                opts.section = args[i].clone();
            }
            "--det-threads" => {
                i += 1;
                opts.det_threads = args[i]
                    .split(',')
                    .map(|s| s.parse().expect("thread count"))
                    .collect();
            }
            "--backend" => {
                i += 1;
                opts.backend = match args[i].as_str() {
                    "opencl" => Backend::OpenCL,
                    "vulkan" => Backend::Vulkan,
                    _ => Backend::CPU,
                };
            }
            "--rec-gpu" => {
                opts.rec_gpu = true;
            }
            "--par-sessions" => {
                i += 1;
                opts.par_sessions = args[i].parse().expect("session count");
            }
            "--pin" => {
                i += 1;
                opts.pin = args[i]
                    .split(',')
                    .map(|s| s.parse().expect("cpu id"))
                    .collect();
            }
            _ => {}
        }
        i += 1;
    }
    opts
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: bench_ocr <det.mnn> <rec.mnn> [iters] [--precision low|normal|high] \
             [--memory low|normal|high] [--section det|rec|par|batch|all] [--det-threads 1,2,4]"
        );
        std::process::exit(2);
    }
    let det_path = Path::new(&args[1]);
    let rec_path = Path::new(&args[2]);
    let opts = parse_options(&args);

    println!(
        "mnn {} precision={:?} memory={:?} iters={}",
        mnn_sys::get_version(),
        opts.precision,
        opts.memory,
        opts.iters
    );
    if matches!(opts.section.as_str(), "det" | "all") {
        bench_det(det_path, &opts);
    }
    if opts.section == "dettile" {
        bench_det_tiled(det_path, &opts);
    }
    if opts.section == "overlap" {
        bench_overlap(det_path, rec_path, &opts);
    }
    if opts.section == "profile" {
        bench_profile(det_path, rec_path, &opts);
    }
    if matches!(opts.section.as_str(), "rec" | "all") {
        bench_rec_widths(rec_path, &opts);
    }
    if matches!(opts.section.as_str(), "batch" | "all") {
        bench_rec_batch(rec_path, &opts);
    }
    if matches!(opts.section.as_str(), "par" | "all") {
        // 71 strips at ~544 wide matches what the reference screenshot produces.
        bench_rec_parallel(rec_path, 544, 71, &opts);
    }
}

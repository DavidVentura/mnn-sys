use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let debug = env::var("DEBUG").unwrap();

    let coreml_enabled = env::var("CARGO_FEATURE_COREML").is_ok();
    let metal_enabled = env::var("CARGO_FEATURE_METAL").is_ok();
    let cuda_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();
    let opencl_enabled = env::var("CARGO_FEATURE_OPENCL").is_ok();
    let opengl_enabled = env::var("CARGO_FEATURE_OPENGL").is_ok();
    let vulkan_enabled = env::var("CARGO_FEATURE_VULKAN").is_ok();

    let mnn_source_dir = mnn_source_dir(&manifest_dir);
    println!(
        "cargo:rerun-if-changed={}",
        mnn_source_dir.join("CMakeLists.txt").display()
    );

    let dst = build_mnn_with_cmake(
        &mnn_source_dir,
        &arch,
        &os,
        &debug,
        coreml_enabled,
        metal_enabled,
        cuda_enabled,
        opencl_enabled,
        opengl_enabled,
        vulkan_enabled,
    );

    let mnn_include_dirs = vec![dst.join("include"), mnn_source_dir.join("include")];
    let mnn_lib_dirs = vec![dst.clone(), dst.join("lib")];

    build_wrapper(&manifest_dir, &mnn_include_dirs, &os, vulkan_enabled);

    link_libraries(
        &mnn_lib_dirs,
        &os,
        coreml_enabled,
        metal_enabled,
        cuda_enabled,
        opencl_enabled,
        opengl_enabled,
        vulkan_enabled,
    );

    bind_gen(&manifest_dir, &mnn_include_dirs, &os, &arch);
}

/// Return the path to the vendored MNN source tree. The submodule under
/// `3rd_party/MNN` is the only supported source.
fn mnn_source_dir(manifest_dir: &Path) -> PathBuf {
    let path = manifest_dir.join("3rd_party/MNN");
    if !path.join("CMakeLists.txt").exists() {
        panic!(
            "MNN submodule not initialized at {}. Run:\n\
             \tgit submodule update --init --depth=1 -- 3rd_party/MNN",
            path.display()
        );
    }
    path
}

fn build_mnn_with_cmake(
    mnn_source_dir: &Path,
    arch: &str,
    os: &str,
    debug: &str,
    coreml_enabled: bool,
    metal_enabled: bool,
    cuda_enabled: bool,
    opencl_enabled: bool,
    opengl_enabled: bool,
    vulkan_enabled: bool,
) -> PathBuf {
    let mut config = cmake::Config::new(mnn_source_dir);

    config
        .define("MNN_BUILD_SHARED_LIBS", "OFF")
        .define("MNN_BUILD_TOOLS", "OFF")
        .define("MNN_BUILD_DEMO", "OFF")
        .define("MNN_BUILD_TEST", "OFF")
        .define("MNN_BUILD_BENCHMARK", "OFF")
        .define("MNN_BUILD_QUANTOOLS", "OFF")
        .define("MNN_BUILD_CONVERTER", "OFF")
        .define("MNN_PORTABLE_BUILD", "ON")
        .define("MNN_SEP_BUILD", "OFF");

    if os == "windows" {
        config.generator("NMake Makefiles");
        config.define("CMAKE_BUILD_TYPE", "Release");
        if env::var("CARGO_CFG_TARGET_FEATURE").map_or(false, |f| f.contains("crt-static")) {
            config.define("MNN_WIN_RUNTIME_MT", "ON");
            config.define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreaded");
            config.define("CMAKE_C_FLAGS_RELEASE", "/MT /O2 /Ob2 /DNDEBUG");
            config.define("CMAKE_CXX_FLAGS_RELEASE", "/MT /O2 /Ob2 /DNDEBUG");
            config.define("CMAKE_C_FLAGS", "/MT");
            config.define("CMAKE_CXX_FLAGS", "/MT");
        }
    } else if debug == "true" {
        config.define("CMAKE_BUILD_TYPE", "Debug");
    } else {
        config.define("CMAKE_BUILD_TYPE", "Release");
    }

    if os == "android" {
        let ndk = env::var("ANDROID_NDK_ROOT")
            .or_else(|_| env::var("ANDROID_NDK_HOME"))
            .or_else(|_| env::var("ANDROID_NDK"))
            .or_else(|_| env::var("NDK_HOME"))
            .expect(
                "Android NDK not found. Set one of: ANDROID_NDK_ROOT, ANDROID_NDK_HOME, ANDROID_NDK, NDK_HOME",
            );

        config
            .define(
                "CMAKE_TOOLCHAIN_FILE",
                PathBuf::from(&ndk).join("build/cmake/android.toolchain.cmake"),
            )
            .define("ANDROID_STL", "c++_static")
            .define("ANDROID_NATIVE_API_LEVEL", "android-21")
            .define("ANDROID_TOOLCHAIN", "clang")
            .define("MNN_BUILD_FOR_ANDROID_COMMAND", "ON")
            .define("MNN_USE_SSE", "OFF");

        match arch {
            "arm" => {
                config.define("ANDROID_ABI", "armeabi-v7a");
            }
            "aarch64" => {
                config.define("ANDROID_ABI", "arm64-v8a");
            }
            "x86" => {
                config.define("ANDROID_ABI", "x86");
            }
            "x86_64" => {
                config.define("ANDROID_ABI", "x86_64");
            }
            _ => {}
        }
    }

    if os == "ios" {
        let rust_target = env::var("TARGET").unwrap_or_default();
        let is_simulator = rust_target.contains("-sim") || arch == "x86_64";

        config
            .define("CMAKE_SYSTEM_NAME", "iOS")
            .define("MNN_BUILD_FOR_IOS", "ON")
            .define("CMAKE_OSX_DEPLOYMENT_TARGET", "13.0");

        if arch == "aarch64" {
            config.define("CMAKE_OSX_ARCHITECTURES", "arm64");
        } else if arch == "x86_64" {
            config.define("CMAKE_OSX_ARCHITECTURES", "x86_64");
        }

        if is_simulator {
            config.define("CMAKE_OSX_SYSROOT", "iphonesimulator");
        } else {
            config.define("CMAKE_OSX_SYSROOT", "iphoneos");
        }

        // MNN's CMakeLists.txt only sets CMAKE_SYSTEM_PROCESSOR from
        // CMAKE_OSX_ARCHITECTURES when CMAKE_SYSTEM_NAME == "Darwin",
        // but for iOS it's "iOS". Without this, ARM assembly sources
        // (NEON, AArch64) are not compiled, causing undefined symbols.
        if arch == "aarch64" {
            config.define("CMAKE_SYSTEM_PROCESSOR", "arm64");
            config.define("ARCHS", "arm64");
        } else if arch == "x86_64" {
            config.define("CMAKE_SYSTEM_PROCESSOR", "x86_64");
        }
    }

    if arch == "x86_64" && os != "android" && os != "ios" {
        config.define("MNN_USE_SSE", "ON");
    } else {
        config.define("MNN_USE_SSE", "OFF");
        config.define("MNN_USE_AVX", "OFF");
        config.define("MNN_USE_AVX2", "OFF");
        config.define("MNN_USE_AVX512", "OFF");
    }

    if coreml_enabled && matches!(os, "macos" | "ios") {
        config.define("MNN_COREML", "ON");
    }
    if metal_enabled && matches!(os, "macos" | "ios") {
        config.define("MNN_METAL", "ON");
    }
    if cuda_enabled && matches!(os, "linux" | "windows") {
        config.define("MNN_CUDA", "ON");
    }
    if opencl_enabled {
        config.define("MNN_OPENCL", "ON");
    }
    if opengl_enabled && matches!(os, "android" | "linux") {
        config.define("MNN_OPENGL", "ON");
    }
    if vulkan_enabled {
        config.define("MNN_VULKAN", "ON");
    }
    config.build()
}

fn build_wrapper(
    manifest_dir: &Path,
    mnn_include_dirs: &[PathBuf],
    os: &str,
    vulkan_enabled: bool,
) {
    let wrapper_file = manifest_dir.join("cpp/src/mnn_wrapper.cpp");

    println!("cargo:rerun-if-changed=cpp/src/mnn_wrapper.cpp");
    println!("cargo:rerun-if-changed=cpp/include/mnn_wrapper.h");

    let mut build = cc::Build::new();

    build
        .cpp(true)
        .file(&wrapper_file)
        .include(manifest_dir.join("cpp/include"));

    for inc in mnn_include_dirs {
        build.include(inc);
    }

    build.define("OCR_RS_FORCE_VULKAN_LINK", Some("0"));

    if vulkan_enabled {
        let mnn_root = manifest_dir.join("3rd_party/MNN");
        let mnn_source_root = mnn_root.join("source");
        let vulkan_root = mnn_source_root.join("backend/vulkan");
        let vulkan_include_dirs = [
            vulkan_root.clone(),
            vulkan_root.join("component"),
            vulkan_root.join("runtime"),
            vulkan_root.join("schema/current"),
            vulkan_root.join("image/backend"),
            vulkan_root.join("image/execution"),
            vulkan_root.join("image/shaders"),
            vulkan_root.join("image/compiler"),
            vulkan_root.join("buffer/backend"),
            vulkan_root.join("buffer/execution"),
            vulkan_root.join("buffer/shaders"),
            vulkan_root.join("buffer/compiler"),
            vulkan_root.join("buffer/render"),
            vulkan_root.join("buffer/render/compiler"),
            vulkan_root.join("buffer/render/glsl"),
            vulkan_root.join("vulkan"),
        ];
        let flatbuffers_include_dir = mnn_root.join("3rd_party/flatbuffers/include");
        if mnn_source_root.exists() {
            build.include(&mnn_source_root);
        }
        for dir in &vulkan_include_dirs {
            if dir.exists() {
                build.include(dir);
            }
        }
        if flatbuffers_include_dir.exists() {
            build.include(&flatbuffers_include_dir);
        }
    }

    if os == "windows" {
        build.flag("/std:c++14").flag("/EHsc").flag("/W3");
    } else {
        build.flag("-std=c++14").flag("-fvisibility=hidden");
    }

    build.compile("mnn_wrapper");
}

fn link_libraries(
    lib_dirs: &[PathBuf],
    os: &str,
    coreml_enabled: bool,
    metal_enabled: bool,
    cuda_enabled: bool,
    opencl_enabled: bool,
    opengl_enabled: bool,
    vulkan_enabled: bool,
) {
    for dir in lib_dirs {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }

    // whole-archive: keep MNN backend runtime creators registered via static init.
    println!("cargo:rustc-link-arg=-Wl,--whole-archive");
    println!("cargo:rustc-link-lib=static=MNN");
    println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");

    match os {
        "macos" | "ios" => {
            println!("cargo:rustc-link-lib=c++");
        }
        "linux" => {
            println!("cargo:rustc-link-lib=stdc++");
            println!("cargo:rustc-link-lib=m");
            println!("cargo:rustc-link-lib=pthread");
        }
        "android" => {
            println!("cargo:rustc-link-lib=c++_static");
            println!("cargo:rustc-link-lib=log");
        }
        _ => {}
    }

    if coreml_enabled && matches!(os, "macos" | "ios") {
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    }
    if metal_enabled && matches!(os, "macos" | "ios") {
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    }
    if cuda_enabled && matches!(os, "linux" | "windows") {
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cudnn");
    }
    if opencl_enabled {
        if os == "macos" {
            println!("cargo:rustc-link-lib=framework=OpenCL");
        } else {
            println!("cargo:rustc-link-lib=OpenCL");
        }
    }
    if opengl_enabled && matches!(os, "android" | "linux") {
        if os == "android" {
            println!("cargo:rustc-link-lib=GLESv3");
            println!("cargo:rustc-link-lib=EGL");
        } else {
            println!("cargo:rustc-link-lib=GL");
        }
    }
    if vulkan_enabled {
        println!("cargo:rustc-link-lib=vulkan");
    }
}

fn bind_gen(manifest_dir: &Path, mnn_include_dirs: &[PathBuf], os: &str, arch: &str) {
    let header_path = manifest_dir.join("cpp/include/mnn_wrapper.h");

    let mut builder = bindgen::Builder::default()
        .header(header_path.to_string_lossy())
        .allowlist_function("mnnr_.*")
        .allowlist_type("MNN.*")
        .allowlist_type("MNNR.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .layout_tests(false);

    for inc in mnn_include_dirs {
        builder = builder.clang_arg(format!("-I{}", inc.display()));
    }

    if os == "linux" {
        builder = add_linux_system_include_args(builder);
    }

    if os == "android" {
        let ndk = env::var("ANDROID_NDK_ROOT")
            .or_else(|_| env::var("ANDROID_NDK_HOME"))
            .or_else(|_| env::var("ANDROID_NDK"))
            .or_else(|_| env::var("NDK_HOME"))
            .unwrap_or_default();

        let api_level = "21";
        let target = match arch {
            "aarch64" => "aarch64-linux-android",
            "arm" => "armv7-linux-androideabi",
            "x86_64" => "x86_64-linux-android",
            "x86" => "i686-linux-android",
            _ => "aarch64-linux-android",
        };
        builder = builder.clang_arg(format!("--target={}{}", target, api_level));

        if !ndk.is_empty() {
            let host_tag = if cfg!(target_os = "macos") {
                "darwin-x86_64"
            } else {
                "linux-x86_64"
            };
            let sysroot = PathBuf::from(&ndk)
                .join("toolchains/llvm/prebuilt")
                .join(host_tag)
                .join("sysroot");
            if sysroot.exists() {
                builder = builder.clang_arg(format!("--sysroot={}", sysroot.display()));
            }
        }
    }

    if os == "ios" {
        let rust_target = env::var("TARGET").unwrap_or_default();
        let clang_target = if rust_target == "aarch64-apple-ios-sim" {
            "arm64-apple-ios13.0-simulator".to_string()
        } else if rust_target == "aarch64-apple-ios" {
            "arm64-apple-ios13.0".to_string()
        } else if rust_target == "x86_64-apple-ios" {
            "x86_64-apple-ios13.0-simulator".to_string()
        } else {
            rust_target
        };
        builder = builder.clang_arg(format!("--target={}", clang_target));
    }

    let bindings = builder.generate().expect("Unable to generate bindings");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    fs::write(out_path.join("mnn_bindings.rs"), bindings.to_string())
        .expect("Couldn't write bindings!");
}

fn add_linux_system_include_args(mut builder: bindgen::Builder) -> bindgen::Builder {
    let mut include_dirs = Vec::new();
    let mut seen = HashSet::new();

    let compiler = cc::Build::new().get_compiler();
    let compiler_path = compiler.path();

    if let Some(include_dir) = command_path_output(compiler_path, &["-print-file-name=include"]) {
        push_unique_path(&mut include_dirs, &mut seen, PathBuf::from(include_dir));
    }

    let sysroot = command_path_output(compiler_path, &["-print-sysroot"])
        .filter(|value| !value.is_empty() && value != "/");

    let target_include = command_path_output(compiler_path, &["-dumpmachine"])
        .map(PathBuf::from)
        .or_else(|| env::var("TARGET").ok().map(PathBuf::from));

    if let Some(sysroot) = sysroot.as_ref() {
        let sysroot_path = PathBuf::from(sysroot);
        push_unique_path(
            &mut include_dirs,
            &mut seen,
            sysroot_path.join("usr/local/include"),
        );
        if let Some(target) = target_include.as_ref() {
            push_unique_path(
                &mut include_dirs,
                &mut seen,
                sysroot_path.join("usr/include").join(target),
            );
        }
        push_unique_path(
            &mut include_dirs,
            &mut seen,
            sysroot_path.join("usr/include"),
        );
    }

    push_unique_path(
        &mut include_dirs,
        &mut seen,
        PathBuf::from("/usr/local/include"),
    );
    if let Some(target) = target_include.as_ref() {
        push_unique_path(
            &mut include_dirs,
            &mut seen,
            PathBuf::from("/usr/include").join(target),
        );
    }
    push_unique_path(&mut include_dirs, &mut seen, PathBuf::from("/usr/include"));

    for dir in include_dirs {
        builder = builder.clang_arg(format!("-isystem{}", dir.display()));
    }

    builder
}

fn command_path_output(program: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }

    let value = String::from_utf8(output.stdout).ok()?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn push_unique_path(paths: &mut Vec<PathBuf>, seen: &mut HashSet<PathBuf>, path: PathBuf) {
    if path.exists() && seen.insert(path.clone()) {
        paths.push(path);
    }
}

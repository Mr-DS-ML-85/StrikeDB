fn main() {
    // Link CUDA driver and NVRTC libraries.
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=nvrtc");
    // Add CUDA library path.
    if let Ok(cuda_home) = std::env::var("CUDA_HOME") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_home);
    } else {
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    }
}

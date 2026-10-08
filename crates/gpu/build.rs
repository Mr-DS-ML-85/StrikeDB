fn main() {
    // Nothing to link: CUDA and NVRTC are loaded at runtime with dlopen (see
    // the `late_bound!` block in src/lib.rs), so the server builds without the
    // CUDA toolkit and starts on machines without an NVIDIA driver.
    println!("cargo:rerun-if-changed=build.rs");
}

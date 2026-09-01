fn main() {
    let target = std::env::var("TARGET").expect("Cargo provides TARGET to build scripts");
    println!("cargo:rustc-env=GLR_BUILD_TARGET={target}");
}

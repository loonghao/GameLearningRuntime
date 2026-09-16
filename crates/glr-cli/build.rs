use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

fn files(root: &Path, current: &Path, output: &mut Vec<String>) {
    for entry in fs::read_dir(current).expect("dashboard directory is readable") {
        let entry = entry.expect("dashboard file is readable");
        assert!(
            !entry.file_type().unwrap().is_symlink(),
            "dashboard inputs cannot be links"
        );
        if entry.file_type().unwrap().is_dir() {
            files(root, &entry.path(), output);
        } else {
            output.push(
                entry
                    .path()
                    .strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .replace('\\', "/"),
            );
        }
    }
}

fn main() {
    let target = std::env::var("TARGET").expect("Cargo provides TARGET to build scripts");
    println!("cargo:rustc-env=GLR_BUILD_TARGET={target}");
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let ui = manifest
        .join("../../dashboard-ui")
        .canonicalize()
        .expect("dashboard-ui source is required");
    let dist = ui.join("dist");
    println!("cargo:rerun-if-changed={}", dist.display());
    let mut inputs = [
        ".npmrc",
        "package.json",
        "package-lock.json",
        "tsconfig.json",
        "vite.config.ts",
        "index.html",
        "components.json",
    ]
    .map(String::from)
    .to_vec();
    for dir in ["src", "scripts"] {
        println!("cargo:rerun-if-changed={}", ui.join(dir).display());
        files(&ui, &ui.join(dir), &mut inputs);
    }
    inputs.sort();
    let mut hash = Sha256::new();
    for name in inputs {
        println!("cargo:rerun-if-changed={}", ui.join(&name).display());
        let source = fs::read_to_string(ui.join(&name))
            .expect("UTF-8 dashboard source")
            .replace("\r\n", "\n");
        hash.update(format!("{name}\0{source}\0"));
    }
    let expected = format!("{:x}", hash.finalize());
    let actual=fs::read_to_string(dist.join("source.sha256")).expect("Dashboard assets missing. Run `vx just dashboard-build` before Cargo; CI must download the dashboard-ui artifact first.");
    assert_eq!(
        actual.trim(),
        expected,
        "Dashboard assets are stale. Run `vx just dashboard-build` before Cargo."
    );
    assert!(
        dist.join("index.html").is_file(),
        "dashboard entrypoint missing"
    );
    let mut assets = Vec::new();
    files(&dist, &dist, &mut assets);
    assets.sort();
    let mut generated = format!(
        "pub const SOURCE_HASH: &str = {expected:?};\npub fn asset(path: &str) -> Option<(&'static str, &'static [u8])> {{\nmatch path {{\n"
    );
    for name in assets {
        if name == "source.sha256" {
            continue;
        }
        let mime = match Path::new(&name).extension().and_then(|e| e.to_str()) {
            Some("html") => "text/html; charset=utf-8",
            Some("js") => "text/javascript; charset=utf-8",
            Some("css") => "text/css; charset=utf-8",
            Some("svg") => "image/svg+xml",
            Some("png") => "image/png",
            Some("ico") => "image/x-icon",
            Some("woff2") => "font/woff2",
            _ => panic!("unexpected dashboard asset: {name}"),
        };
        let path = dist.join(&name);
        assert!(
            fs::metadata(&path).unwrap().len() <= 8 * 1024 * 1024,
            "dashboard asset exceeds 8 MiB"
        );
        let route = format!("/{name}");
        if name == "index.html" {
            generated.push_str("\"/\" | ");
        }
        generated.push_str(&format!(
            "{route:?} => Some(({mime:?}, include_bytes!({path:?}))),\n"
        ));
    }
    generated.push_str("_ => None,\n}\n}\n");
    let output = PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("dashboard_assets.rs");
    fs::write(output, generated).expect("write embedded dashboard asset table");
}

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    pyo3_build_config::add_extension_module_link_args();
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let source = manifest.join("../../../src/core/qgraph");
    let output = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("qgraph");
    fs::create_dir_all(&output).unwrap();
    for name in ["mod.rs", "vertex.rs", "kernels.rs", "build_kernels.rs"] {
        let path = source.join(name);
        println!("cargo:rerun-if-changed={}", path.display());
        let mut bytes = fs::read(&path).unwrap();
        if name == "vertex.rs" {
            let helper = manifest.join("src/vertex_support.rs");
            println!("cargo:rerun-if-changed={}", helper.display());
            bytes.extend_from_slice(b"\n");
            bytes.extend(fs::read(helper).unwrap());
        }
        fs::write(output.join(name), bytes).unwrap();
    }
    let declaration = format!(
        "#[allow(dead_code)] pub mod core {{ pub mod error {{ #[derive(Debug, thiserror::Error)] pub enum WarmError {{ #[error(\"Invalid parameter: {{details}}\")] InvalidParam {{ details: String }} }} }} #[path = {:?}] pub mod qgraph; }}",
        output.join("mod.rs").to_str().unwrap()
    );
    fs::write(output.parent().unwrap().join("core.rs"), declaration).unwrap();
}

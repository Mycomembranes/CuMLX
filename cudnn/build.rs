// build.rs — compile the oneDNN C++ backend (onednn_backend.cpp) and link it
// into libcudnn.dylib alongside the prebuilt oneDNN CPU library (libdnnl).
//
// The vendored oneDNN tree (lib + headers) lives under cudnn/vendor/onednn,
// produced by the recipe in README.md. If a different oneDNN is desired, set
// ONEDNN_DIR to a prefix containing lib/libdnnl.dylib + include/oneapi/dnnl/.

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let onednn = env::var("ONEDNN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| manifest.join("vendor/onednn"));

    let inc = onednn.join("include");
    let lib = onednn.join("lib");

    // Compile the C++ backend against the oneDNN headers.
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .include(&inc)
        .file("src/onednn_backend.cpp")
        .flag_if_supported("-Wno-unused-parameter")
        .compile("onednn_backend");

    // Link the prebuilt oneDNN CPU library + set an rpath so libdnnl.3.dylib
    // resolves at load time (its install_name is @rpath/libdnnl.3.dylib).
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=dylib=dnnl");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());

    println!("cargo:rerun-if-changed=src/onednn_backend.cpp");
    println!("cargo:rerun-if-env-changed=ONEDNN_DIR");
}

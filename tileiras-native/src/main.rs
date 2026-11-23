// SPDX-License-Identifier: Apache-2.0
//! Offline, pinned TileIR load planning. No foreign entry, mapping or GPU I/O.
use std::{env, fs::OpenOptions, io::Read, path::PathBuf};

const PIN: &str = "297d5448eb73dd0c6ea07a28678415af59c25d289a30508b1049460b6d717f1a";
const MAX_INPUT: u64 = 256 * 1024 * 1024;
const DEFAULT_ELF: &str = ".config/cuda-oxide/cuda-redist-linux-sbsa-13.2.51/extract/cuda_tileiras-linux-sbsa-13.2.51-archive/bin/tileiras";

fn help() {
    println!(
        r#"tileiras-native: native-port foundation (NOT a runnable NVIDIA compiler)

Usage:
  tileiras-native plan [--elf FILE] [--json] [--materialize-check]
                       [--page-size 4096|16384]
  tileiras-native rebase-audit [--elf FILE]
  tileiras-native --version

plan                  Verify the pinned Linux/AArch64 ELF and describe its load plan.
rebase-audit          Classify every data word the loader's rebase rule slides by
                      independent evidence (section typing, FDE starts, ADRP+ADD /
                      ADR targets, pointer-bearing objects, C strings) and count
                      the uncorroborated ("ambiguous") ones. JSON; executes nothing.
--materialize-check   Copy into ordinary non-executable host memory and create a
                      separate TLS template; never run constructors or ELF code.
--elf FILE            Default: cached NVIDIA linux-sbsa TileIR 13.2.51 binary.
--page-size BYTES     Planning granularity; default 16384. Does not map memory.
--json                Emit machine-readable readiness and remaining blockers.

Run/compile modes are intentionally unavailable. This command does not test
NVIDIA tileiras --version, emit a cubin, download dependencies, or touch a GPU."#
    );
}

fn quote(value: &str) -> String {
    let mut out = String::from("\"");
    for c in value.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn read_snapshot(path: &PathBuf) -> Result<Vec<u8>, String> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        #[cfg(target_os = "macos")]
        options.custom_flags(0x0004); // O_NONBLOCK: a FIFO must not hang admission.
        #[cfg(target_os = "linux")]
        options.custom_flags(0x0800);
    }
    let mut file = options
        .open(path)
        .map_err(|e| format!("open {}: {e}", path.display()))?;
    let before = file.metadata().map_err(|e| e.to_string())?;
    if !before.is_file() || before.len() == 0 || before.len() > MAX_INPUT {
        return Err("input must be a nonempty regular ELF file <=256 MiB".into());
    }
    let mut bytes = Vec::with_capacity(before.len() as usize);
    (&mut file)
        .take(MAX_INPUT + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    let after = file.metadata().map_err(|e| e.to_string())?;
    if bytes.len() as u64 != before.len()
        || after.len() != before.len()
        || after.modified().ok() != before.modified().ok()
    {
        return Err("input changed while taking snapshot".into());
    }
    Ok(bytes)
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        None | Some("-h" | "--help") => {
            help();
            return Ok(());
        }
        Some("--version") => {
            println!("tileiras-native planner 0.1.0; NVIDIA compiler execution NOT implemented in this planner (the isolated worker is tileiras-native-helper)");
            return Ok(());
        }
        Some("plan") => (),
        Some("rebase-audit") => {
            let mut input = None;
            while let Some(option) = args.next() {
                match option.as_str() {
                    "--elf" if input.is_none() => {
                        input = Some(PathBuf::from(args.next().ok_or("--elf needs a file")?))
                    }
                    "--json" => (),
                    _ => return Err(format!("unknown or duplicate option: {option}")),
                }
            }
            let path = match input {
                Some(path) => path,
                None => PathBuf::from(env::var_os("HOME").ok_or("HOME missing; pass --elf")?)
                    .join(DEFAULT_ELF),
            };
            let bytes = read_snapshot(&path)?;
            let sha = tileiras_native::sha256::hex(&bytes);
            if sha != PIN {
                return Err(format!(
                    "unadmitted TileIR input SHA256 {sha}; expected {PIN}"
                ));
            }
            println!(
                "{}",
                tileiras_native::rebase_audit::audit(&bytes)?.to_json(&sha)
            );
            return Ok(());
        }
        Some(_) => return Err(
            "only non-executing 'plan' is available; native TileIR run/compile is not implemented"
                .into(),
        ),
    }
    let mut input = None;
    let mut json = false;
    let mut materialize = false;
    let mut page_size = 16384;
    while let Some(option) = args.next() {
        match option.as_str() {
            "--elf" if input.is_none() => {
                input = Some(PathBuf::from(args.next().ok_or("--elf needs a file")?))
            }
            "--json" => json = true,
            "--materialize-check" => materialize = true,
            "--page-size" => {
                page_size = args
                    .next()
                    .ok_or("--page-size needs a number")?
                    .parse::<u64>()
                    .map_err(|_| "invalid page size")?;
                if ![4096, 16384].contains(&page_size) {
                    return Err("page size must be 4096 or 16384".into());
                }
            }
            "-h" | "--help" => {
                help();
                return Ok(());
            }
            _ => return Err(format!("unknown or duplicate option: {option}")),
        }
    }
    let path = match input {
        Some(path) => path,
        None => {
            PathBuf::from(env::var_os("HOME").ok_or("HOME missing; pass --elf")?).join(DEFAULT_ELF)
        }
    };
    let bytes = read_snapshot(&path)?;
    let plan = tileiras_native::plan(&bytes, page_size).map_err(|e| e.to_string())?;
    if plan.input_sha256 != PIN {
        return Err(format!(
            "unadmitted TileIR input SHA256 {}; expected {PIN}",
            plan.input_sha256
        ));
    }
    let mut copied_bytes = 0;
    let mut tls_copy_bytes = 0;
    if materialize {
        copied_bytes = plan.materialize(&bytes).map_err(|e| e.to_string())?.len();
        if let Some(tls) = &plan.tls {
            tls_copy_bytes = tls.instantiate().map_err(|e| e.to_string())?.len();
        }
    }
    let tls_bytes = plan.tls.as_ref().map_or(0, |tls| tls.memory_size);
    let relro_json = format!("[{}]", plan.relro.iter().map(|r| format!(
        "{{\"virtual_address\":{},\"size\":{},\"touched_page_start\":{},\"touched_page_end\":{},\"whole_page_start\":{},\"whole_page_end\":{},\"exact_page_protection\":{}}}",
        r.virtual_address, r.size, r.page_start, r.page_end, r.whole_page_start, r.whole_page_end, r.exact_page_protection
    )).collect::<Vec<_>>().join(","));
    if json {
        println!("{{\"relro\":{},\"schema\":\"cuda-oxide.tileiras-native-plan.v1\",\"input\":{},\"input_sha256\":{},\"pinned\":true,\"file_bytes\":{},\"entry\":{},\"load_base\":{},\"load_end\":{},\"planning_page_size\":{},\"load_segments\":{},\"imports\":{},\"relocations\":{},\"constructors\":{},\"tls_bytes_per_thread\":{},\"materialization_checked\":{},\"nonexecutable_image_bytes\":{},\"separate_tls_copy_bytes\":{},\"execution_ready\":false,\"compiler_executed\":false,\"gpu_accessed\":false,\"remaining\":[\"isolated fixed-address worker and collision-safe mapping\",\"Linux entry/auxv and exactly-once constructors\",\"Linux TLS and thread bootstrap\",\"x18/control-flow and ABI-compatible import contracts\",\"unwind/signal/jump handling\",\"real version/help and TileIR compilation tests\"]}}",
            relro_json, quote(&path.to_string_lossy()), quote(&plan.input_sha256), plan.file_size, plan.entry,
            plan.load_base, plan.load_end, page_size, plan.segments.len(), plan.imports.len(),
            plan.relocations.len(), plan.startup.init_array.len(), tls_bytes, materialize,
            copied_bytes, tls_copy_bytes);
    } else {
        println!("Pinned ELF load plan PASS; native compiler execution NOT implemented.");
        println!("{}", path.display());
        println!("SHA256 {}", plan.input_sha256);
        println!("entry={:#x}, segments={}, imports={}, relocations={}, constructors={}, TLS={} bytes/thread",
            plan.entry, plan.segments.len(), plan.imports.len(), plan.relocations.len(),
            plan.startup.init_array.len(), tls_bytes);
        println!(
            "Host copy checked: {materialize}; foreign code executed: false; GPU accessed: false."
        );
        if plan.relro.iter().any(|r| !r.exact_page_protection) {
            println!(
                "RELRO has partial host pages: an explicit protection policy is still required."
            );
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("tileiras-native: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn json_quotes_paths_without_control_injection() {
        assert_eq!(quote("a\n\"\\雪"), "\"a\\u000a\\\"\\\\雪\"");
    }
}

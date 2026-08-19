//! Stdin/stdout runner: one JSON event per input line, one JSON outcome per
//! output line. This is the minimal "统一事件入口" — business embeds the
//! crate as a library for anything richer.
//!
//! Usage:
//!   edge-agent [--config edge-agent.json]
//!   echo '{"kind":"command","payload":"turn on the light"}' | edge-agent

use edge_agent_core::{Config, Event, Kernel};
use std::io::BufRead;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let mut config_path = String::from("edge-agent.json");
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--config" => {
                config_path = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("--config needs a path"))?
            }
            other => anyhow::bail!("unknown argument '{other}'"),
        }
    }

    let cfg = if Path::new(&config_path).exists() {
        Config::load(Path::new(&config_path))?
    } else {
        eprintln!("[main] no {config_path}, using defaults (mock backend)");
        Config::default()
    };

    let mut kernel = Kernel::new(cfg, None)?;
    eprintln!("[main] kernel up, reading events from stdin (one JSON per line)");

    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match serde_json::from_str::<Event>(line) {
            Ok(ev) => {
                kernel.queue.push(ev);
                for outcome in kernel.run_pending() {
                    println!("{}", serde_json::to_string(&outcome)?);
                }
            }
            Err(e) => eprintln!("[main] bad event line rejected: {e}"),
        }
    }
    kernel.shutdown();
    Ok(())
}

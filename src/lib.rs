//! edge-agent-core — minimal deterministic agent runtime kernel for edge
//! devices. See README.md for the architecture; this crate is also usable as
//! a library: business code builds a `Kernel`, registers a `HostBridge` for
//! real hardware capabilities, pushes `Event`s and drains outcomes.

pub mod breaker;
pub mod config;
pub mod context;
pub mod event;
pub mod inference;
pub mod kernel;
pub mod lock;
pub mod plugin;

pub use config::Config;
pub use event::Event;
pub use kernel::{Kernel, TaskOutcome};
pub use plugin::runtime::HostBridge;

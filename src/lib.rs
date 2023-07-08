#![doc = include_str!("../README.md")]

mod distributions;
pub use distributions::{Buffered, Ranged, StandardBuffered};
mod rngs;
pub use rngs::StandardSeedableRng;

#![doc = include_str!("../README.md")]

mod distributions;
pub use distributions::{
    Buffered, Ranged, StandardBuffered, TryDistIter, TryDistMap, TryDistribution, TryRanged,
};
mod rngs;
pub use rngs::StandardSeedableRng;

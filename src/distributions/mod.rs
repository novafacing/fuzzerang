//! Random distributions that are useful for random generation or mutation of fuzzer
//! data

use std::ops::{Range, RangeInclusive};

use anyhow::Result;
use rand::Rng;

mod standard_buffered;
pub use standard_buffered::StandardBuffered;

mod utils;

pub trait Buffered {
    fn try_ensure<R: Rng + ?Sized>(&self, bits: usize, rng: &mut R) -> Result<()>;
    fn ensure<R: Rng + ?Sized>(&self, bits: usize, rng: &mut R);
}

pub trait Ranged<T> {
    fn sample_range<R: Rng + ?Sized>(&self, rng: &mut R, range: Range<T>) -> T;
    fn sample_range_inclusive<R: Rng + ?Sized>(&self, rng: &mut R, range: RangeInclusive<T>) -> T;
}

pub trait BitsNeeded<T> {
    fn bits_needed(&self, value: T) -> usize;
}

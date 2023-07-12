//! Random distributions that are useful for random generation or mutation of fuzzer
//! data

use std::{
    iter::FusedIterator,
    ops::{Range, RangeInclusive},
};

use anyhow::Result;
use rand::Rng;

mod standard_buffered;
pub use standard_buffered::StandardBuffered;

mod utils;
/// Types (distributions) that can be used to create a random instance of `T`.
///
/// It is possible to sample from a distribution through both the
/// `Distribution` and [`Rng`] traits, via `distr.sample(&mut rng)` and
/// `rng.sample(distr)`. They also both offer the [`sample_iter`] method, which
/// produces an iterator that samples from the distribution.
///
/// All implementations are expected to be immutable; this has the significant
/// advantage of not needing to consider thread safety, and for most
/// distributions efficient state-less sampling algorithms are available.
///
/// Implementations are typically expected to be portable with reproducible
/// results when used with a PRNG with fixed seed; see the
/// [portability chapter](https://rust-random.github.io/book/portability.html)
/// of The Rust Rand Book. In some cases this does not apply, e.g. the `usize`
/// type requires different sampling on 32-bit and 64-bit machines.
///
/// [`sample_iter`]: Distribution::sample_iter
pub trait TryDistribution<T> {
    /// Generate a random value of `T`, using `rng` as the source of randomness.
    fn try_sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<T>;

    /// Create an iterator that generates random values of `T`, using `rng` as
    /// the source of randomness.
    ///
    /// Note that this function takes `self` by value. This works since
    /// `Distribution<T>` is impl'd for `&D` where `D: Distribution<T>`,
    /// however borrowing is not automatic hence `distr.sample_iter(...)` may
    /// need to be replaced with `(&distr).sample_iter(...)` to borrow or
    /// `(&*distr).sample_iter(...)` to reborrow an existing reference.
    ///
    /// # Example
    ///
    /// ```
    /// use rand::thread_rng;
    /// use rand::distributions::{Distribution, Alphanumeric, Uniform, Standard};
    ///
    /// let mut rng = thread_rng();
    ///
    /// // Vec of 16 x f32:
    /// let v: Vec<f32> = Standard.sample_iter(&mut rng).take(16).collect();
    ///
    /// // String:
    /// let s: String = Alphanumeric
    ///     .sample_iter(&mut rng)
    ///     .take(7)
    ///     .map(char::from)
    ///     .collect();
    ///
    /// // Dice-rolling:
    /// let die_range = Uniform::new_inclusive(1, 6);
    /// let mut roll_die = die_range.sample_iter(&mut rng);
    /// while roll_die.next().unwrap() != 6 {
    ///     println!("Not a 6; rolling again!");
    /// }
    /// ```
    fn sample_iter<R>(self, rng: R) -> TryDistIter<Self, R, T>
    where
        R: Rng,
        Self: Sized,
    {
        TryDistIter {
            distr: self,
            rng,
            phantom: ::core::marker::PhantomData,
        }
    }

    /// Create a distribution of values of 'S' by mapping the output of `Self`
    /// through the closure `F`
    ///
    /// # Example
    ///
    /// ```
    /// use rand::thread_rng;
    /// use rand::distributions::{Distribution, Uniform};
    ///
    /// let mut rng = thread_rng();
    ///
    /// let die = Uniform::new_inclusive(1, 6);
    /// let even_number = die.map(|num| num % 2 == 0);
    /// while !even_number.sample(&mut rng) {
    ///     println!("Still odd; rolling again!");
    /// }
    /// ```
    fn map<F, S>(self, func: F) -> TryDistMap<Self, F, T, S>
    where
        F: Fn(T) -> S,
        Self: Sized,
    {
        TryDistMap {
            distr: self,
            func,
            phantom: ::core::marker::PhantomData,
        }
    }
}

/// An iterator that generates random values of `T` with distribution `D`,
/// using `R` as the source of randomness.
///
/// This `struct` is created by the [`sample_iter`] method on [`Distribution`].
/// See its documentation for more.
///
/// [`sample_iter`]: Distribution::sample_iter
#[derive(Debug)]
pub struct TryDistIter<D, R, T> {
    distr: D,
    rng: R,
    phantom: ::core::marker::PhantomData<T>,
}

impl<D, R, T> Iterator for TryDistIter<D, R, T>
where
    D: TryDistribution<T>,
    R: Rng,
{
    type Item = T;

    #[inline(always)]
    fn next(&mut self) -> Option<T> {
        // Here, self.rng may be a reference, but we must take &mut anyway.
        // Even if sample could take an R: Rng by value, we would need to do this
        // since Rng is not copyable and we cannot enforce that this is "reborrowable".
        self.distr.try_sample(&mut self.rng).ok()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::max_value(), None)
    }
}

impl<D, R, T> FusedIterator for TryDistIter<D, R, T>
where
    D: TryDistribution<T>,
    R: Rng,
{
}

#[cfg(features = "nightly")]
impl<D, R, T> iter::TrustedLen for TryDistIter<D, R, T>
where
    D: TryDistribution<T>,
    R: Rng,
{
}

/// A distribution of values of type `S` derived from the distribution `D`
/// by mapping its output of type `T` through the closure `F`.
///
/// This `struct` is created by the [`Distribution::map`] method.
/// See its documentation for more.
#[derive(Debug)]
pub struct TryDistMap<D, F, T, S> {
    distr: D,
    func: F,
    phantom: ::core::marker::PhantomData<fn(T) -> S>,
}

impl<D, F, T, S> TryDistribution<S> for TryDistMap<D, F, T, S>
where
    D: TryDistribution<T>,
    F: Fn(T) -> Result<S>,
{
    fn try_sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<S> {
        (self.func)(self.distr.try_sample(rng)?)
    }
}

pub trait Buffered {
    fn try_ensure<R: Rng + ?Sized>(&self, bits: usize, rng: &mut R) -> Result<()>;
    fn ensure<R: Rng + ?Sized>(&self, bits: usize, rng: &mut R);
}

pub trait Ranged<T> {
    fn sample_range<R: Rng + ?Sized>(&self, rng: &mut R, range: Range<T>) -> T;
    fn sample_range_inclusive<R: Rng + ?Sized>(&self, rng: &mut R, range: RangeInclusive<T>) -> T;
}

pub trait TryRanged<T> {
    fn try_sample_range<R: Rng + ?Sized>(&self, rng: &mut R, range: Range<T>) -> Result<T>;
    fn try_sample_range_inclusive<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        range: RangeInclusive<T>,
    ) -> Result<T>;
}

pub trait BitsNeeded<T> {
    fn bits_needed(&self, value: T) -> usize;
}

use super::{Buffered, Ranged};
use anyhow::Result;
use bitvec::{order::Lsb0, vec::BitVec};
use core::ops::{Range, RangeInclusive};
use rand::{prelude::Distribution, Rng};
use std::{cell::RefCell, io::Read, mem::size_of};

/// Similar to the [`rand::distributions::Standard`] distribution in that it generates
/// values in the "expected" way for each type
pub struct StandardBuffered {
    buf: RefCell<BitVec<u8, Lsb0>>,
}

impl StandardBuffered {
    /// Create a new [`StandardBuffered`] rng that buffers data from the sampling RNG and
    /// uses it to generate values using the smallest possible number of bits. For example,
    /// a [`bool`] is 1 bit in size, so only 1 bit will be used to generate it
    pub fn new() -> Self {
        Self {
            buf: RefCell::new(BitVec::new()),
        }
    }
}

impl Buffered for StandardBuffered {
    /// Ensures enough bits are in the buffer to generate an instance of a type.
    /// Returns Ok if enough bits were generated.
    /// Returns Err if not enough bits were generated, and fills the
    /// buffer zero.
    fn try_ensure<R: Rng + ?Sized>(&self, bits: usize, rng: &mut R) -> Result<()> {
        if self.buf.borrow().len() < bits {
            let bits_needed = bits - self.buf.borrow().len();
            let bytes_needed = ((bits_needed + (u8::BITS as usize - 1))
                & (!(u8::BITS as usize - 1)))
                / u8::BITS as usize;
            let mut bits = vec![0u8; bytes_needed];
            rng.try_fill_bytes(&mut bits)?;
            self.buf.borrow_mut().extend(bits);
        }
        Ok(())
    }

    fn ensure<R: Rng + ?Sized>(&self, bits: usize, rng: &mut R) {
        self.try_ensure::<R>(bits, rng)
            .expect("Generator::ensure failed");
    }
}

impl Default for StandardBuffered {
    fn default() -> Self {
        Self::new()
    }
}

impl Distribution<bool> for StandardBuffered {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> bool {
        // Special case: bool only requires 1 bit, even though a bool is a full byte in size
        self.ensure::<R>(1, rng);
        self.buf.borrow_mut().remove(0)
    }
}

impl Distribution<char> for StandardBuffered {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> char {
        self.ensure::<R>(u8::BITS as usize, rng);
        let mut bytes = vec![0u8; 1];
        self.buf
            .borrow_mut()
            .read_exact(&mut bytes)
            .expect("Failed to read into buffer");
        bytes[0] as char
    }
}

macro_rules! impl_distribution_integral {
    ($T:ty) => {
        impl Distribution<$T> for StandardBuffered {
            fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> $T {
                self.ensure::<R>(<$T>::BITS as usize, rng);
                let mut bytes = vec![0u8; size_of::<$T>()];
                self.buf
                    .borrow_mut()
                    .read_exact(&mut bytes)
                    .expect("Failed to read into buffer");
                <$T>::from_le_bytes(bytes.as_slice().try_into().expect("Invalid bytes"))
            }
        }
    };
}

impl_distribution_integral! { u8 }
impl_distribution_integral! { u16 }
impl_distribution_integral! { u32 }
impl_distribution_integral! { u64 }
impl_distribution_integral! { usize }
impl_distribution_integral! { i8 }
impl_distribution_integral! { i16 }
impl_distribution_integral! { i32 }
impl_distribution_integral! { i64 }
impl_distribution_integral! { isize }

macro_rules! impl_ranged_integral {
    ($T:ty, $UT:ty) => {
        impl_ranged_integral! { $T, $UT, $T }
    };
    ($T:ty, $UT:ty, $C:ty) => {
        impl Ranged<$C> for StandardBuffered {
            fn sample_range<R: Rng + ?Sized>(&self, rng: &mut R, range: Range<$C>) -> $C {
                self.sample_range_inclusive(rng, range.start..=(range.end as $T - 1) as $C)
            }

            fn sample_range_inclusive<R: Rng + ?Sized>(
                &self,
                rng: &mut R,
                range: RangeInclusive<$C>,
            ) -> $C {
                let end = *range.end() as $T;
                let start = *range.start() as $T;
                // Get the size of the range
                let range_size = end.wrapping_sub(start).wrapping_add(1) as $UT;

                if range_size == 0 {
                    self.sample(rng)
                } else {
                    // Get the number of bits needed to represent the maximum value in the range
                    let bits_needed = range_size.ilog2() as usize + 1;
                    // Ensure we have enough bits in the buffer to generate a value
                    self.ensure::<R>(bits_needed, rng);
                    // We can use T because we know the range is small enough to fit in T
                    let mut v: $T = 0;
                    // Read bits from the buffer and OR bits into v
                    for i in 0..bits_needed {
                        let bit = self.buf.borrow_mut().remove(0);
                        v |= (bit as $T) << i;
                    }
                    // TODO: Need to do better here than modulus
                    v %= range_size as $T;
                    v += start;
                    v as $C
                }
            }
        }
    };
}

impl_ranged_integral! { u8, u8, char }
impl_ranged_integral! { u8, u8 }
impl_ranged_integral!(u16, u16);
impl_ranged_integral!(u32, u32);
impl_ranged_integral!(u64, u64);
impl_ranged_integral!(usize, usize);
impl_ranged_integral!(i8, u8);
impl_ranged_integral!(i16, u16);
impl_ranged_integral!(i32, u32);
impl_ranged_integral!(i64, u64);
impl_ranged_integral!(isize, usize);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{distributions::Ranged, rngs::StandardSeedableRng};
    use concat_idents::concat_idents;
    use rand::SeedableRng;

    macro_rules! test_sample_impl {
        ($T:ty, $TN:ident) => {
            #[test]
            fn $TN() {
                const SAMPLES: usize = 8;
                const BYTES_NEEDED: usize = size_of::<$T>() * SAMPLES;

                let mut rng = StandardSeedableRng::from_seed(vec![0xff; BYTES_NEEDED]);
                let dist = StandardBuffered::new();
                (0..SAMPLES).for_each(|_| {
                    let s: $T = rng.sample(&dist);
                    assert_eq!(
                        s,
                        <$T>::from_le_bytes([0xff; size_of::<$T>()]),
                        "Expected true"
                    );
                });
            }
        };
    }

    #[test]
    fn test_bool() {
        let mut rng = StandardSeedableRng::from_seed(vec![0xff]);
        let dist = StandardBuffered::new();
        for i in 0..8 {
            let s: bool = rng.sample(&dist);
            assert!(s, "Expected true on iteration {}", i);
        }
    }

    #[test]
    fn test_char() {
        let mut rng = StandardSeedableRng::from_seed(vec![0x41; 8]);
        let dist = StandardBuffered::new();
        for i in 0..8 {
            let s: char = rng.sample(&dist);
            assert_eq!(s, 'A', "Expected character on iteration {}", i);
        }
    }

    test_sample_impl!(u8, test_sample_u8);
    test_sample_impl!(u16, test_sample_u16);
    test_sample_impl!(u32, test_sample_u32);
    test_sample_impl!(u64, test_sample_u64);
    test_sample_impl!(usize, test_sample_usize);
    test_sample_impl!(i8, test_sample_i8);
    test_sample_impl!(i16, test_sample_i16);
    test_sample_impl!(i32, test_sample_i32);
    test_sample_impl!(i64, test_sample_i64);
    test_sample_impl!(isize, test_sample_isize);

    #[test]
    fn test_sample_range_char() {
        const RANGE_MAX: char = 'Z';
        const RANGE_MIN: char = 'A';
        const SAMPLES: usize = 64;
        let bytes_needed: usize =
            ((RANGE_MAX as u8 - RANGE_MIN as u8).ilog2() as usize + 1) * SAMPLES;
        let mut rng = StandardSeedableRng::from_seed(
            (0..255)
                .take(bytes_needed / 2)
                .chain((0..255).rev().take(bytes_needed / 2))
                .collect(),
        );
        let dist = StandardBuffered::new();
        (0..SAMPLES * 2).for_each(|_| {
            let s: char = dist.sample_range(&mut rng, RANGE_MIN..RANGE_MAX);
            assert!(s >= RANGE_MIN, "Unexpected value");
            assert!(s < RANGE_MAX, "Unexpected value");
        });
    }

    macro_rules! test_sample_rangeimpl {
        ($T:ty, $TN:ident) => {
            #[test]
            fn $TN() {
                const RANGE_MAX: $T = 48;
                const RANGE_MIN: $T = 8;
                const SAMPLES: usize = 64;
                let bytes_needed: usize = ((RANGE_MAX - RANGE_MIN).ilog2() as usize + 1) * SAMPLES;
                let mut rng = StandardSeedableRng::from_seed(
                    (0..255)
                        .take(bytes_needed / 2)
                        .chain((0..255).rev().take(bytes_needed / 2))
                        .collect(),
                );
                let dist = StandardBuffered::new();
                (0..SAMPLES * 2).for_each(|_| {
                    let s: $T = dist.sample_range(&mut rng, 0..RANGE_MAX);
                    assert!(s < RANGE_MAX, "Unexpected value");
                });
            }

            concat_idents!(test_name = $TN, _, {
                #[test]
                fn test_name() {
                    const RANGE_MAX: $T = 48;
                    const RANGE_MIN: $T = 8;
                    const SAMPLES: usize = 64;
                    let bytes_needed: usize =
                        ((RANGE_MAX - RANGE_MIN).ilog2() as usize + 1) * SAMPLES;
                    let mut rng = StandardSeedableRng::from_seed(
                        (0..255)
                            .take(bytes_needed / 2)
                            .chain((0..255).rev().take(bytes_needed / 2))
                            .collect(),
                    );
                    let dist = StandardBuffered::new();
                    (0..SAMPLES * 2).for_each(|_| {
                        let s: $T = dist.sample_range_inclusive(&mut rng, 0..=RANGE_MAX);
                        assert!(s <= RANGE_MAX, "Unexpected value");
                    });
                }
            });
        };
    }

    test_sample_rangeimpl!(u8, test_sample_range_u8);
    test_sample_rangeimpl!(u16, test_sample_range_u16);
    test_sample_rangeimpl!(u32, test_sample_range_u32);
    test_sample_rangeimpl!(u64, test_sample_range_u64);
    test_sample_rangeimpl!(usize, test_sample_range_usize);
    test_sample_rangeimpl!(i8, test_sample_range_i8);
    test_sample_rangeimpl!(i16, test_sample_range_i16);
    test_sample_rangeimpl!(i32, test_sample_range_i32);
    test_sample_rangeimpl!(i64, test_sample_range_i64);
    test_sample_rangeimpl!(isize, test_sample_range_isize);
}

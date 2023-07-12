use anyhow::anyhow;
use rand::RngCore;
use rand_core::{
    impls::{next_u32_via_fill, next_u64_via_fill},
    SeedableRng,
};

#[derive(Clone, Debug)]
/// An RNG that generates values directly from a seed (similar to proptest's `PassThrough` RNG).
/// This is mostly only useful in conjunction with the provided distributions like
/// [`StandardBuffered`] because although the [`RandCore`] trait is restrictive and only allows
/// byte-level resolution, we want to do better than this.
pub struct StandardSeedableRng {
    seed: Vec<u8>,
}

impl RngCore for StandardSeedableRng {
    fn next_u32(&mut self) -> u32 {
        next_u32_via_fill(self)
    }

    fn next_u64(&mut self) -> u64 {
        next_u64_via_fill(self)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.try_fill_bytes(dest)
            .expect("Generator::fill_bytes failed");
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        (0..dest.len()).try_for_each(|i| {
            match self.seed.pop().ok_or_else(|| {
                rand_core::Error::new(anyhow!(
                    "Generator::try_fill_bytes unable to fill bytes: seed exhausted on byte {} filling destination of size {}", i, dest.len()
                ))
            }) {
                Ok(b) => {
                    dest[i] = b;
                    Ok(())
                }
                Err(e) => Err(e),
            }
        })?;
        Ok(())
    }
}

impl SeedableRng for StandardSeedableRng {
    type Seed = Vec<u8>;

    fn from_seed(mut seed: Self::Seed) -> Self {
        // Fuzzers and people likely assume that the seed is used like a VecDeque but we
        // need a BitVec which doesn't have an associated Deque type. We can make do by reversing
        // the seed and popping to extract bits, which lets us "take from the front"
        seed.reverse();
        Self { seed }
    }
}

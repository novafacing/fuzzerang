# Fuzzerang

Useful random generators and distributions for use in fuzzers and mutators

Instead of being very random, very fast, or very secure, these generators
and distributions are designed to be useful for fuzzing and mutation by efficiently
utilizing available input data. For example, the default [`Standard`] distribution
in the [`rand`] crate wastes 31 bits of input for every boolean value generated.

In comparison, [`StandardBuffered`] uses the input data more efficiently by consuming
only 1 bit for a boolean, the minimum number of bits to generate a value in a range, and so
on.

# Examples

```rust
use fuzzerang::{StandardSeedableRng, StandardBuffered, Ranged};
use rand::{SeedableRng, distributions::Distribution};

// Use a constant seed of 8 bytes, or 64 bits
let mut rng = StandardSeedableRng::from_seed((0..255).take(8).collect());
let dist = StandardBuffered::new();

// We can generate 10 bools from 8 bytes of input because we're only using 1 bit each
for i in 0..10 {
    let x: bool = dist.sample(&mut rng);
    println!("{}: {}", i, x);
}

// In fact, we are so efficient we can generate some alphabetic characters too, which
// each use 4 bits
for i in 0..10 {
    let x: char = dist.sample_range_inclusive(&mut rng, 'A'..='Z');
    println!("{}: {}", i, x);
}

```
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
use rand::SeedableRng;

let mut rng = StandardSeedableRng::from_seed((0..255).take(8).collect());
let dist = StandardBuffered::new();

for i in 0..10 {
   println!("{}: {}", i, rng.sample(&dist));
}
```
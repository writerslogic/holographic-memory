#![no_main]

use holographic_memory::EntangledHVec;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let dimensions = data.len().saturating_mul(8).clamp(1, 65_536);
    let dense: Vec<f32> = data
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .filter(|value| value.is_finite())
        .collect();
    let vector = EntangledHVec::from_dense(&dense, dimensions);
    let _ = vector.similarity(&vector);
});

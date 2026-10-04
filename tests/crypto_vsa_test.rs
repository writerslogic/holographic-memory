use holographic_memory::core::entangled::EntangledHVec;
use holographic_memory::core::algebra::HolographicAlgebra;

#[test]
fn test_fhe_lite_similarity() {
    let dim = 16384;
    
    // 1. Create two document vectors
    let doc1 = EntangledHVec::new_deterministic(dim, 100);
    let doc2 = EntangledHVec::new_deterministic(dim, 101); // A bit different
    let doc1_sim = EntangledHVec::new_deterministic(dim, 100).permute(1); // Highly similar if we bundle or shift
    
    // 2. Create a "master key" vector (dense, 50% active)
    // To make it 50% dense, we can just bind many vectors or use a custom density
    let mut key = EntangledHVec::new_deterministic(dim, 999);
    for i in 0..100 {
        key = key.bind(&EntangledHVec::new_deterministic(dim, 1000 + i));
    }
    
    let plain_sim_1_2 = doc1.similarity(&doc2);
    
    let enc1 = doc1.bind(&key);
    let enc2 = doc2.bind(&key);
    
    let enc_sim_1_2 = enc1.similarity(&enc2);
    
    println!("Plaintext similarity: {}", plain_sim_1_2);
    println!("Encrypted similarity: {}", enc_sim_1_2);
}

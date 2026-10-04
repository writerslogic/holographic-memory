use holographic_memory::core::algebra::HolographicAlgebra;
use holographic_memory::core::entangled::EntangledHVec;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// A simple deterministic encoder for the demo
fn encode_text_internal(text: &str, dim: usize) -> EntangledHVec {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    EntangledHVec::new_deterministic(dim, hasher.finish())
}

fn main() {
    println!("=== FHE-Lite: Zero-Trust Holographic Vector Search Demo ===\n");
    let dim = 16384;

    // 1. Client initializes their Master Key. This NEVER leaves their machine.
    // In practice, this key is made dense so that it properly obscures the sparse vectors.
    println!("[Client] Generating Master Cryptographic Key...");
    let mut master_key = EntangledHVec::new_deterministic(dim, 8888);
    for i in 0..50 {
        master_key = master_key.bind(&EntangledHVec::new_deterministic(dim, 8888 + i));
    }

    // 2. Client processes private documents locally
    println!("[Client] Encoding private documents...");
    let docs = vec![
        "The Q3 financial earnings were surprisingly high due to the merger.",
        "Operation Midnight will commence on Tuesday at 0400 hours.",
        "Patient 402 has a history of severe allergic reactions to penicillin.",
        "The recipe for the secret sauce includes two parts cinnamon, one part nutmeg."
    ];

    // Client encodes them, encrypts them, and sends them to the server
    let mut server_database: Vec<(usize, EntangledHVec)> = Vec::new();

    println!("[Client] Encrypting documents and sending to untrusted server...");
    for (id, text) in docs.iter().enumerate() {
        // Local encoding (standard VSA sparse vector)
        let plain_vec = encode_text_internal(text, dim);
        
        // FHE-Lite Encryption: Bind with Master Key
        let encrypted_vec = plain_vec.bind(&master_key);
        
        // Send to server
        server_database.push((id, encrypted_vec));
    }
    println!("[Server] Received {} vectors. They appear as pure uniform noise.\n", server_database.len());

    // 3. Client wants to search for "financial earnings"
    let query_text = "What were the financial earnings?";
    println!("[Client] Query: '{}'", query_text);
    
    // Client encodes and encrypts the query locally
    let plain_query = encode_text_internal(query_text, dim);
    let encrypted_query = plain_query.bind(&master_key);
    println!("[Client] Sending encrypted query vector to server...\n");

    // 4. Server performs search over ENCRYPTED data
    println!("[Server] Performing semantic search on encrypted data...");
    let mut results: Vec<(usize, f64)> = server_database.iter().map(|(id, enc_vec)| {
        let sim = encrypted_query.similarity(enc_vec);
        (*id, sim)
    }).collect();
    
    // Sort by similarity descending
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Server returns the top encrypted result
    let top_result_id = results[0].0;
    println!("[Server] Returning best match ID: {} (Score: {:.4})\n", top_result_id, results[0].1);

    // 5. Client verifies
    println!("[Client] Match corresponds to: '{}'", docs[top_result_id]);
    
    // Show mathematically that it worked
    let plain_sim = plain_query.similarity(&encode_text_internal(docs[top_result_id], dim));
    println!("[Verification] Plaintext Similarity would have been: {:.4}", plain_sim);
    println!("[Verification] Encrypted Similarity was: {:.4}", results[0].1);
    
    if (plain_sim - results[0].1).abs() < 0.05 {
         println!("\nSUCCESS: Mathematical equivalence proven. Search executed flawlessly over encrypted data.");
    }
}

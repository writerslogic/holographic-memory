// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! Query-private scoring. The server end answers a query vector it receives from any client,
//! and the client end decodes an answer it receives from any server: both are untrusted word
//! vectors. Neither may panic, and a server must reject a query of the wrong length.

use std::sync::OnceLock;

use holographic_memory::core::private_query::{
    Answer, Client, Hint, Params, PublicMatrix, Query, Server,
};
use holographic_memory::EntangledHVec;
use libfuzzer_sys::fuzz_target;

const DIM: usize = 256;

struct Fixture {
    server: Server,
    client: Client,
}

fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        let params = Params::for_dim(DIM).expect("params");
        let matrix = PublicMatrix::expand(params, [3u8; 32]);
        let rows: Vec<EntangledHVec> = (0..8)
            .map(|i| {
                let dense: Vec<f32> = (0..DIM)
                    .map(|j| if (j + i) % 16 == 0 { 1.0 } else { -1.0 })
                    .collect();
                EntangledHVec::from_dense(&dense, DIM)
            })
            .collect();
        let server = Server::new(&matrix, &rows).expect("server");
        let client = Client::new(matrix);
        Fixture { server, client }
    })
}

fn words(data: &[u8]) -> Vec<u32> {
    data.as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect()
}

fuzz_target!(|data: &[u8]| {
    let fixture = fixture();
    let words = words(data);
    // Server side: arbitrary query words, padded or truncated to the right length half the time.
    let mut query = words.clone();
    if data.first().is_some_and(|b| b & 1 == 1) {
        query.resize(DIM, 0);
    }
    let _ = fixture.server.answer(&Query::from_words(query));
    // Client side: arbitrary answer words against the real hint.
    let code = EntangledHVec::from_dense(&vec![1.0; DIM], DIM);
    if let Ok((_, state)) = fixture.client.query(&code) {
        let mut answer = words;
        if data.first().is_some_and(|b| b & 2 == 2) {
            answer.resize(fixture.server.hint().rows(), 0);
        }
        let _ = state.decode(fixture.server.hint(), &Answer::from_words(answer));
    }
    let _: &Hint = fixture.server.hint();
});

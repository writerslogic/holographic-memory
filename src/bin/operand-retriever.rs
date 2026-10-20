// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: AGPL-3.0-or-later

use anyhow::{ensure, Result};
use holographic_memory::core::operand_retriever::{
    knapsack_prepared, retrieve_prepared, Prepared, Request,
};
use std::io::{self, BufRead, Read, Write};

fn main() -> Result<()> {
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut out = io::BufWriter::new(io::stdout().lock());
    loop {
        let mut line = Vec::new();
        let count = reader
            .by_ref()
            .take(8_388_609)
            .read_until(b'\n', &mut line)?;
        if count == 0 {
            break;
        }
        ensure!(count <= 8_388_608, "request exceeds input allocation limit");
        let request: Request = serde_json::from_slice(&line)?;
        let index_start = std::time::Instant::now();
        let prepared = Prepared::new(&request)?;
        let index_ns = index_start.elapsed().as_nanos();
        let result = retrieve_prepared(&prepared)?;
        let control = knapsack_prepared(&prepared)?;
        let mut operand_ns = Vec::new();
        let mut control_ns = Vec::new();
        for repeat in 0..11 {
            for is_operand in if repeat % 2 == 0 {
                [true, false]
            } else {
                [false, true]
            } {
                let start = std::time::Instant::now();
                if is_operand {
                    std::hint::black_box(retrieve_prepared(&prepared)?);
                    operand_ns.push(start.elapsed().as_nanos());
                } else {
                    std::hint::black_box(knapsack_prepared(&prepared)?);
                    control_ns.push(start.elapsed().as_nanos());
                }
            }
        }
        serde_json::to_writer(
            &mut out,
            &serde_json::json!({"retrieval": result, "control": control, "operand_ns": operand_ns, "control_ns": control_ns, "index_ns": index_ns}),
        )?;
        writeln!(out)?;
        out.flush()?;
    }
    Ok(())
}

use anyhow::Result;
use holographic_memory::core::documents::{LexicalExecution, LexicalOptions, LexicalQuery};
use holographic_memory::HmsCore;
use std::io::{BufRead, BufReader, Write};

fn main() -> Result<()> {
    let index = std::env::args().nth(1).expect("index path");
    let queries = std::env::args().nth(2).expect("queries path");
    let core = HmsCore::new(1024, Some(index), None)?;
    let mut checked = 0;
    for line in BufReader::new(std::fs::File::open(queries)?).lines() {
        let row: serde_json::Value = serde_json::from_str(&line?)?;
        let text = row["query"].as_str().expect("query string");
        let query = LexicalQuery::parse(text)?;
        let reference = core.search_lexical(
            &query,
            &LexicalOptions {
                k: 100,
                count: true,
                execution: LexicalExecution::Exhaustive,
                ..Default::default()
            },
        )?;
        for (k, count) in [(10, false), (100, true), (0, true)] {
            let mut expected = reference.clone();
            expected.hits.truncate(k);
            if !count {
                expected.count = None;
            }
            let actual = core.search_lexical(
                &query,
                &LexicalOptions {
                    k,
                    count,
                    ..Default::default()
                },
            )?;
            anyhow::ensure!(actual == expected, "mismatch {text}, k={k}, count={count}");
            let bytes = serde_json::to_vec(&actual)?;
            let hash = bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| {
                (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
            });
            println!(
                "{}",
                serde_json::json!({"query":text, "k":k, "count":count, "matches":actual.count,"hash":hash})
            );
            std::io::stdout().flush()?;
            checked += 1;
        }
    }
    anyhow::ensure!(checked == 2886, "expected 2886 comparisons, got {checked}");
    eprintln!(
        "{}",
        serde_json::json!({"mode":"exhaustive600", "queries":962, "checked":checked,"mismatches":0,"reference":"one independent exhaustive top100+count per query; prefixes and count reused"})
    );
    Ok(())
}

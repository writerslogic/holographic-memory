// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Story memory: HMS serving its stated user, a writer (scrivener-mcp).
//
// A novelist's world is a graph of characters, relationships, places, and plot
// causality. This example asserts such a world into HMS's experimental phasor
// relational memory and then answers the questions a writer actually asks —
// direct lookups, inverse lookups ("who mentors X?"), and multi-hop reasoning
// ("where does my protagonist's brother's protege work?"). These are the queries
// the sparse-binary store structurally cannot do; the phasor substrate can,
// deterministically and verifiably.
//
// Run: cargo run --features experimental --example story_memory

use holographic_memory::HmsCore;

fn main() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let hms = HmsCore::new(4096, Some(dir.path().to_string_lossy().to_string()), None)?;

    // --- The writer builds the story world, one fact at a time ---------------
    let facts = [
        ("elena", "sister_of", "marco"),
        ("marco", "mentor_of", "dev"),
        ("dev", "works_at", "the_archive"),
        ("the_archive", "located_in", "duskport"),
        ("elena", "rival_of", "vane"),
        ("vane", "works_at", "the_ministry"),
        ("the_ministry", "located_in", "duskport"),
        ("marco", "married_to", "sabine"),
    ];
    for (s, r, o) in facts {
        hms.relate_phase(s, r, o);
    }
    println!("Story world: {} facts asserted.\n", facts.len());

    // --- The questions a writer asks about their own world -------------------

    // 1. Direct: continuity check.
    let sibling = hms
        .phase_retrieve_object("elena", "sister_of")
        .unwrap_or_else(|| "?".into());
    println!("Q: Who is elena's sibling?              A: {sibling}");

    // 2. Inverse (a query the sparse store cannot answer at all).
    let mentor = hms
        .phase_retrieve_subject("mentor_of", "dev")
        .unwrap_or_else(|| "?".into());
    println!("Q: Who mentors dev?                     A: {mentor}");

    let rival_of_elena = hms
        .phase_retrieve_subject("rival_of", "vane")
        .unwrap_or_else(|| "?".into());
    println!("Q: Who is vane's rival?                 A: {rival_of_elena}");

    // 3. Multi-hop reasoning: the payoff of relation algebra.
    let city = hms
        .phase_retrieve_path(
            "elena",
            &["sister_of", "mentor_of", "works_at", "located_in"],
        )
        .unwrap_or_else(|| "?".into());
    println!("Q: What city does elena's brother's     A: {city}");
    println!("   protege work in? (4-hop)");

    // 4. A plot-consistency question the writer might forget: does elena's rival
    //    end up in the same city as her family's circle? (two independent chains)
    let family_city = hms
        .phase_retrieve_path(
            "elena",
            &["sister_of", "mentor_of", "works_at", "located_in"],
        )
        .unwrap_or_else(|| "?".into());
    let rival_city = hms
        .phase_retrieve_path("elena", &["rival_of", "works_at", "located_in"])
        .unwrap_or_else(|| "?".into());
    println!(
        "\nContinuity: family circle is in {family_city}; rival is in {rival_city}. Same city? {}",
        family_city == rival_city
    );

    println!(
        "\nEvery assertion above is an ordered event; the memory is a deterministic,\n\
         tamper-evident fold of that log (see PhaseGraph::verify). A writing tool can\n\
         thus prove what its story bible held at any revision — reasoning memory that is\n\
         also an auditable record."
    );
    Ok(())
}

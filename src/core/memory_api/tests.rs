use std::sync::Arc;

use super::index::Index;
use super::*;
use crate::core::HmsCore;

fn fact(id: &str, text: &str) -> RememberInput {
    RememberInput {
        id: Some(id.into()),
        text: text.into(),
        ..RememberInput::default()
    }
}

#[test]
fn replacing_a_fact_leaves_the_index_equal_to_a_rebuild() {
    let dir = tempfile::tempdir().unwrap();
    let core = Arc::new(HmsCore::new(64, Some(dir.path().display().to_string()), None).unwrap());
    let config = MemoryConfig {
        on_id_conflict: IdConflict::Replace,
        ..MemoryConfig::default()
    };
    let memory = Memory::with_config(core.clone(), config).unwrap();
    memory
        .remember(RememberInput {
            subject: Some("s".into()),
            entities: Some(vec!["e1".into(), "e2".into()]),
            valid_from: Some(10),
            valid_to: Some(20),
            ..fact("a", "alpha one")
        })
        .unwrap();
    memory.remember(fact("b", "beta one")).unwrap();
    memory
        .remember(RememberInput {
            supersedes: Some(vec!["b".into()]),
            ..fact("c", "gamma one")
        })
        .unwrap();
    memory.ensure_fresh();
    let tracked = |memory: &Memory| {
        let live = memory.index.read();
        let tracked = live.revision == Some(core.revision());
        if tracked {
            assert_eq!(live.canonical(), Index::rebuild(&core).canonical());
        }
        tracked
    };

    // Unlinked: updated in place, the old subject, entities and window gone.
    memory
        .remember(RememberInput {
            subject: Some("t".into()),
            entities: Some(vec!["e2".into(), "e3".into()]),
            ..fact("a", "alpha two")
        })
        .unwrap();
    assert!(tracked(&memory));
    // Unlinked before, linked by the new stamp: still in place.
    memory
        .remember(RememberInput {
            supersedes: Some(vec!["c".into()]),
            ..fact("a", "alpha three")
        })
        .unwrap();
    assert!(tracked(&memory));
    // Linked (c supersedes b): only a rebuild reproduces other facts' links to it.
    memory.remember(fact("b", "beta two")).unwrap();
    assert!(!tracked(&memory));
    memory.ensure_fresh();
    assert!(tracked(&memory));
}

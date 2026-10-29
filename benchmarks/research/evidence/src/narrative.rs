// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{ensure, Result};
use narrative_lens::continuity::entity::{
    self, CorrectionAction, EntityCorrection, EntityType, Mention,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AliasCandidates {
    pub surface: String,
    pub canonical_candidates: Vec<String>,
}

pub fn resolve_people(
    mentions: &[String],
    corrections: &[(String, String)],
) -> Result<Vec<AliasCandidates>> {
    ensure!(
        mentions.len() <= 10_000 && corrections.len() <= 10_000,
        "too many entity inputs"
    );
    for surface in mentions {
        crate::contract::validate_id(surface)?;
    }
    for (surface, canonical) in corrections {
        crate::contract::validate_id(surface)?;
        crate::contract::validate_id(canonical)?;
    }
    let mentions: Vec<_> = mentions
        .iter()
        .map(|name| Mention::new(name.clone(), vec![]))
        .collect();
    let corrections: Vec<_> = corrections
        .iter()
        .map(|(surface, canonical)| EntityCorrection {
            surface: surface.clone(),
            action: CorrectionAction::Assign {
                canonical: canonical.clone(),
            },
        })
        .collect();
    let entities =
        entity::resolve_with_corrections_checked(&mentions, EntityType::Character, &corrections)?;
    let mut candidates: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for entity in entities {
        for surface in entity.aliases {
            candidates
                .entry(surface)
                .or_default()
                .insert(entity.canonical.clone());
        }
    }
    Ok(candidates
        .into_iter()
        .map(|(surface, values)| AliasCandidates {
            surface,
            canonical_candidates: values.into_iter().collect(),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_alias_corrections_use_the_real_resolver() -> Result<()> {
        let aliases = resolve_people(
            &["Dr. Rivera".into(), "Maya Rivera".into()],
            &[("Dr. Rivera".into(), "Maya Rivera".into())],
        )?;
        assert!(aliases
            .iter()
            .any(|alias| alias.surface == "Dr. Rivera"
                && alias.canonical_candidates == ["Maya Rivera"]));
        Ok(())
    }

    #[test]
    fn resolver_inputs_are_bounded_before_cloning() -> Result<()> {
        for bytes in [255, 256, 257] {
            let mentions = vec!["x".repeat(bytes)];
            let result = resolve_people(&mentions, &[]);
            if bytes <= 256 {
                result?;
            } else {
                assert!(result.is_err());
            }
        }
        assert!(resolve_people(&["bad\nname".into()], &[]).is_err());
        assert!(resolve_people(&[], &[("alias".into(), "x".repeat(257))]).is_err());
        assert!(resolve_people(&vec!["name".into(); 10_001], &[]).is_err());
        Ok(())
    }
}

// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use serde_json::json;

use super::WarmCore;
use crate::core::models::ModelStages;

#[test]
fn invalid_stage_installation_preserves_the_active_search_pipeline() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let core = WarmCore::new(1024, Some(directory.path().display().to_string()), None)?;
    core.memorize_document(serde_json::from_value(json!({"id":"a", "text":"needle"}))?)?;
    core.set_model_stages(Some(Arc::new(ModelStages::default())))?;
    let expected = serde_json::to_value(core.search_documents("needle", &Default::default())?)?;
    let mut invalid = ModelStages::default();
    invalid.params.k0 = -1.0;
    assert!(core.set_model_stages(Some(Arc::new(invalid))).is_err());
    assert_eq!(
        serde_json::to_value(core.search_documents("needle", &Default::default())?)?,
        expected
    );
    Ok(())
}

#[test]
fn direct_model_search_rejects_invalid_limits_before_preparing_variants() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let core = WarmCore::new(1024, Some(directory.path().display().to_string()), None)?;
    let stages = ModelStages::default();
    for k in [0, 999, 1000, 1001] {
        let options = crate::SearchOptions {
            k: Some(k),
            ..Default::default()
        };
        assert_eq!(
            core.search_documents_with_models("needle", "", &options, &stages)
                .is_ok(),
            (1..=1000).contains(&k)
        );
    }
    let mut stages = ModelStages::default();
    stages.params.rerank_n = usize::MAX;
    assert!(core
        .search_documents_with_models("needle", "", &Default::default(), &stages)
        .is_err());
    Ok(())
}

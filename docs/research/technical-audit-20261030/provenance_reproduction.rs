use holographic_memory::core::provenance::ProvenanceManager;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join(format!("hms-audit-integrity-{}", std::process::id()));
    std::fs::create_dir(&dir)?;
    let manager = ProvenanceManager::new(&dir.join("test.key"), Some(&dir))?;
    manager.create_fact_provenance("f1", b"one", None)?;
    let log = dir.join("provenance_log.jsonl");
    let head = log.with_extension("head");
    assert!(head.is_file());
    let original = std::fs::read(&log)?;
    println!("intact={:?}", manager.verify_log_integrity());
    std::fs::write(&log, b"")?;
    println!("empty_with_signed_head={:?}", manager.verify_log_integrity());
    std::fs::remove_file(&log)?;
    println!("missing_with_signed_head={:?}", manager.verify_log_integrity());
    std::fs::write(&log, &original)?;
    std::fs::remove_file(&head)?;
    println!("head_missing={:?}", manager.verify_log_integrity());
    std::fs::create_dir(&head)?;
    println!("head_unreadable_directory={:?}", manager.verify_log_integrity());
    drop(manager);
    std::fs::remove_dir_all(dir)?;
    Ok(())
}

use anyhow::Result;
use redb::{Database, ReadableTable, TableDefinition};
use std::sync::Arc;

const PQ_CODE_LEN: usize = 16;
const INVERTED_LISTS_TABLE: TableDefinition<u32, &[u8]> = TableDefinition::new("inverted_lists");

/// Entry format: [id_len:2][id:N][pq_codes:16][arena_offset:8]
pub(crate) struct InvertedListEntry {
    pub id: String,
    pub pq_codes: [u8; PQ_CODE_LEN],
    /// Retained in serialized format for backward compatibility; not currently
    /// read outside of tests.
    #[allow(dead_code)]
    pub arena_offset: usize,
}

pub(crate) struct InvertedLists {
    db: Arc<Database>,
}

impl InvertedLists {
    pub fn new(db: Arc<Database>) -> Result<Self> {
        // Ensure table exists
        let write_tx = db.begin_write()?;
        {
            let _ = write_tx.open_table(INVERTED_LISTS_TABLE)?;
        }
        write_tx.commit()?;

        Ok(Self { db })
    }

    /// Append an entry to the inverted list for `cluster_id`.
    pub fn append(
        &self,
        cluster_id: usize,
        id: &str,
        pq_codes: &[u8; PQ_CODE_LEN],
        arena_offset: usize,
    ) -> Result<()> {
        let write_tx = self.db.begin_write()?;
        {
            let mut table = write_tx.open_table(INVERTED_LISTS_TABLE)?;
            let key = cluster_id as u32;

            let id_bytes = id.as_bytes();
            // Guard against silent truncation of ID length to u16.
            if id_bytes.len() > u16::MAX as usize {
                return Err(anyhow::anyhow!(
                    "ID too long for inverted list: {} bytes (max {})",
                    id_bytes.len(),
                    u16::MAX
                ));
            }
            let id_len = (id_bytes.len() as u16).to_le_bytes();

            let mut entry = Vec::with_capacity(2 + id_bytes.len() + PQ_CODE_LEN + 8);
            entry.extend_from_slice(&id_len);
            entry.extend_from_slice(id_bytes);
            entry.extend_from_slice(pq_codes);
            entry.extend_from_slice(&arena_offset.to_le_bytes());

            let mut data = match table.get(key)? {
                Some(d) => d.value().to_vec(),
                None => Vec::new(),
            };
            data.extend_from_slice(&entry);
            table.insert(key, data.as_slice())?;
        }
        write_tx.commit()?;
        Ok(())
    }

    /// Read all entries from the inverted list for `cluster_id`.
    pub fn read(&self, cluster_id: usize) -> Result<Vec<InvertedListEntry>> {
        let read_tx = self.db.begin_read()?;
        let table = read_tx.open_table(INVERTED_LISTS_TABLE)?;
        let key = cluster_id as u32;

        let data = match table.get(key)? {
            Some(d) => d.value().to_vec(),
            None => return Ok(Vec::new()),
        };

        parse_entries(&data)
    }

    /// Clear all inverted lists. Critical for correct re-training.
    pub fn clear_all(&self) -> Result<()> {
        let write_tx = self.db.begin_write()?;
        {
            let mut table = write_tx.open_table(INVERTED_LISTS_TABLE)?;
            // redb doesn't have a truncate but we can delete all keys
            let keys: Vec<u32> = table.iter()?.map(|r| r.unwrap().0.value()).collect();
            for k in keys {
                table.remove(k)?;
            }
        }
        write_tx.commit()?;
        Ok(())
    }
}

fn parse_entries(data: &[u8]) -> Result<Vec<InvertedListEntry>> {
    let mut entries = Vec::new();
    let mut pos = 0;

    while pos + 2 <= data.len() {
        let id_len = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;

        // Use checked arithmetic to prevent overflow with crafted id_len values.
        let Some(id_end) = pos.checked_add(id_len) else {
            break;
        };
        let Some(pq_end) = id_end.checked_add(PQ_CODE_LEN) else {
            break;
        };
        let Some(offset_end) = pq_end.checked_add(8) else {
            break;
        };

        if offset_end > data.len() {
            break;
        }

        let id = String::from_utf8_lossy(&data[pos..id_end]).to_string();
        let mut pq_codes = [0u8; PQ_CODE_LEN];
        pq_codes.copy_from_slice(&data[id_end..pq_end]);
        // Bounds already checked: offset_end <= data.len() is guarded above.
        let arena_offset = usize::from_le_bytes(
            data[pq_end..offset_end]
                .try_into()
                .map_err(|_| anyhow::anyhow!("corrupted inverted list entry at offset {}", pos))?,
        );

        entries.push(InvertedListEntry {
            id,
            pq_codes,
            arena_offset,
        });
        pos = offset_end;
    }

    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inverted_list_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let db = Arc::new(Database::create(dir.path().join("test.redb")).unwrap());
        let lists = InvertedLists::new(db).unwrap();

        let codes1 = [1u8; 16];
        let codes2 = [2u8; 16];

        lists.append(0, "vec_a", &codes1, 1000).unwrap();
        lists.append(0, "vec_b", &codes2, 2000).unwrap();
        lists.append(1, "vec_c", &codes1, 3000).unwrap();

        let entries0 = lists.read(0).unwrap();
        assert_eq!(entries0.len(), 2);
        assert_eq!(entries0[0].id, "vec_a");
        assert_eq!(entries0[0].pq_codes, codes1);
        assert_eq!(entries0[0].arena_offset, 1000);
        assert_eq!(entries0[1].id, "vec_b");
        assert_eq!(entries0[1].pq_codes, codes2);
        assert_eq!(entries0[1].arena_offset, 2000);

        let entries1 = lists.read(1).unwrap();
        assert_eq!(entries1.len(), 1);
        assert_eq!(entries1[0].id, "vec_c");

        let entries2 = lists.read(2).unwrap();
        assert!(entries2.is_empty());
    }
}

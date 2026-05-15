pub(crate) mod inverted_list;
pub(crate) mod kmeans;
pub(crate) mod nystrom;
pub(crate) mod pq;
pub(crate) mod query;
pub(crate) mod training;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::entangled::EntangledHVec;
use inverted_list::InvertedLists;
use kmeans::KMeansNystrom;
use nystrom::NystromProjector;
use pq::PQEncoder;

#[derive(Serialize, Deserialize)]
pub(crate) struct IVFIndex {
    pub(crate) projector: NystromProjector,
    pub(crate) kmeans: KMeansNystrom,
    pub(crate) pq: PQEncoder,
    #[serde(skip)]
    pub(crate) lists: Option<InvertedLists>,
    pub(crate) n_clusters: usize,
    pub(crate) dim: usize,
    pub(crate) trained: bool,
}

impl IVFIndex {
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    pub fn insert(&mut self, id: &str, vector: &EntangledHVec, arena_offset: usize) -> Result<()> {
        if !self.trained {
            return Ok(());
        }

        let projected = self.projector.project(vector);
        let cluster = self.kmeans.assign(&projected);
        let pq_codes = self.pq.encode(vector);

        if let Some(ref lists) = self.lists {
            lists.append(cluster, id, &pq_codes, arena_offset)
        } else {
            Err(anyhow::anyhow!("Inverted lists database not connected"))
        }
    }
}

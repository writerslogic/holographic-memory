#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct BridgeSnapshot {
    n: usize,
    dim: usize,
    padded: usize,
    bits: u8,
    cbytes: usize,
    stride: usize,
    degree: usize,
    mean: Vec<f32>,
    adj: Vec<u32>,
    adj3: Vec<u8>,
    codes: Vec<i8>,
    scales: Vec<f32>,
    res_bits: u8,
    res_stride: usize,
    residual: Vec<i8>,
    res_scales: Vec<f32>,
    entry: u32,
    upper_ids: Vec<u32>,
    layers: Vec<Vec<Vec<u32>>>,
    ids: Vec<u32>,
    graph_hash: String,
}

const BRIDGE_MAGIC: &[u8; 8] = b"HMSNAT01";
const BRIDGE_MAX_BYTES: u64 = 2_147_483_648;

fn bridge_io(error: impl std::fmt::Display) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string())
}

impl VGraph {
    pub fn bridge_save(&self, path: &std::path::Path, graph_hash: &str) -> std::io::Result<()> {
        use std::io::Write;
        if self.has_screen() {
            return Err(bridge_io(
                "screen encoding is outside the frozen native stack",
            ));
        }
        let state = BridgeSnapshot {
            n: self.n,
            dim: self.dim,
            padded: self.padded,
            bits: self.bits,
            cbytes: self.cbytes,
            stride: self.stride,
            degree: self.degree,
            mean: self.mean.clone(),
            adj: self.adj.clone(),
            adj3: self.adj3.clone(),
            codes: self.codes.bytes().to_vec(),
            scales: self.scales.clone(),
            res_bits: self.res_bits,
            res_stride: self.res_stride,
            residual: self.residual.clone(),
            res_scales: self.res_scales.clone(),
            entry: self.entry,
            upper_ids: self.upper_ids.clone(),
            layers: self.layers.clone(),
            ids: self.ids.clone(),
            graph_hash: graph_hash.to_owned(),
        };
        let config = bincode::config::standard().with_limit::<2147483648>();
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        writer.write_all(BRIDGE_MAGIC)?;
        bincode::serde::encode_into_std_write(state, &mut writer, config).map_err(bridge_io)?;
        writer.flush()?;
        writer.get_ref().sync_all()
    }

    pub fn bridge_load(path: &std::path::Path) -> std::io::Result<(Self, usize, String)> {
        use std::io::Read;
        let file = std::fs::File::open(path)?;
        if file.metadata()?.len() > BRIDGE_MAX_BYTES {
            return Err(bridge_io("index exceeds frozen byte limit"));
        }
        let mut reader = std::io::BufReader::new(file);
        let mut magic = [0; 8];
        reader.read_exact(&mut magic)?;
        if &magic != BRIDGE_MAGIC {
            return Err(bridge_io("index magic differs"));
        }
        let config = bincode::config::standard().with_limit::<2147483648>();
        let s: BridgeSnapshot =
            bincode::serde::decode_from_std_read(&mut reader, config).map_err(bridge_io)?;
        let mut extra = [0];
        if reader.read(&mut extra)? != 0 {
            return Err(bridge_io("trailing index bytes"));
        }
        if s.n == 0
            || s.n >= u32::MAX as usize
            || !(1..=4096).contains(&s.dim)
            || !matches!(s.bits, 4 | 8)
            || !matches!(s.res_bits, 0 | 4 | 8)
            || s.degree == 0
            || s.degree > 4096
            || !s.degree.is_multiple_of(2)
            || s.padded != s.dim.next_multiple_of(if s.bits == 4 { 32 } else { 16 })
            || s.cbytes != if s.bits == 4 { s.padded / 2 } else { s.padded }
            || s.stride < s.cbytes
            || s.stride > s.cbytes.next_multiple_of(128)
            || s.entry as usize >= s.n
            || s.graph_hash.len() != 16
            || !s.graph_hash.bytes().all(|c| c.is_ascii_hexdigit())
        {
            return Err(bridge_io("index header invariant failed"));
        }
        let links =
            s.n.checked_mul(s.degree)
                .ok_or_else(|| bridge_io("index size overflow"))?;
        let code_size =
            s.n.checked_mul(s.stride)
                .ok_or_else(|| bridge_io("code size overflow"))?;
        let res_stride = match s.res_bits {
            4 => s.padded / 2,
            8 => s.padded,
            _ => 0,
        };
        let res_size =
            s.n.checked_mul(res_stride)
                .ok_or_else(|| bridge_io("residual size overflow"))?;
        if s.codes.len() != code_size
            || s.scales.len() != s.n
            || s.mean.len() != s.padded
            || s.res_stride != res_stride
            || s.residual.len() != res_size
            || s.res_scales.len() != if s.res_bits == 0 { 0 } else { s.n }
            || !((s.adj.len() == links && s.adj3.is_empty())
                || (s.adj.is_empty()
                    && s.n < NONE3 as usize
                    && s.adj3.len()
                        == links
                            .checked_mul(3)
                            .ok_or_else(|| bridge_io("adjacency size overflow"))?))
            || s.mean
                .iter()
                .chain(&s.scales)
                .chain(&s.res_scales)
                .any(|v| !v.is_finite())
            || s.adj.iter().any(|&v| v != NONE && v as usize >= s.n)
            || s.adj3.chunks_exact(3).any(|v| {
                let id = u32::from(v[0]) | (u32::from(v[1]) << 8) | (u32::from(v[2]) << 16);
                id != NONE3 && id as usize >= s.n
            })
            || s.upper_ids.iter().any(|&v| v as usize >= s.n)
            || (!s.layers.is_empty() && s.layers[0].len() != s.upper_ids.len())
            || s.layers
                .windows(2)
                .any(|pair| pair[1].len() >= pair[0].len())
            || s.layers.iter().any(|layer| {
                layer.is_empty() || layer.iter().flatten().any(|&v| v as usize >= layer.len())
            })
        {
            return Err(bridge_io("index array invariant failed"));
        }
        if !s.ids.is_empty() {
            if s.ids.len() != s.n {
                return Err(bridge_io("id map size differs"));
            }
            let mut seen = vec![false; s.n];
            for &id in &s.ids {
                if id as usize >= s.n || std::mem::replace(&mut seen[id as usize], true) {
                    return Err(bridge_io("id map is not a permutation"));
                }
            }
        }
        let mut codes = Lines::zeroed(code_size);
        codes.bytes_mut().copy_from_slice(&s.codes);
        let (dim, hash) = (s.dim, s.graph_hash);
        Ok((
            Self {
                n: s.n,
                dim,
                padded: s.padded,
                bits: s.bits,
                cbytes: s.cbytes,
                stride: s.stride,
                degree: s.degree,
                mean: s.mean,
                adj: s.adj,
                adj3: s.adj3,
                codes,
                scales: s.scales,
                res_bits: s.res_bits,
                res_stride: s.res_stride,
                residual: s.residual,
                res_scales: s.res_scales,
                entry: s.entry,
                upper_ids: s.upper_ids,
                layers: s.layers,
                ids: s.ids,
                screen: Vec::new(),
                screen_words: 0,
                screen_rot: None,
                screen_sigma: 0.0,
            },
            dim,
            hash,
        ))
    }

    pub fn bridge_allocated_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.mean.capacity() * 4
            + self.adj.capacity() * 4
            + self.adj3.capacity()
            + self.codes.lines.capacity() * 128
            + self.scales.capacity() * 4
            + self.residual.capacity()
            + self.res_scales.capacity() * 4
            + self.upper_ids.capacity() * 4
            + self.ids.capacity() * 4
            + self.screen.capacity() * 8
            + self.layers.capacity() * std::mem::size_of::<Vec<Vec<u32>>>()
            + self
                .layers
                .iter()
                .map(|layer| {
                    layer.capacity() * std::mem::size_of::<Vec<u32>>()
                        + layer.iter().map(|row| row.capacity() * 4).sum::<usize>()
                })
                .sum::<usize>()
    }
}

#[cfg(test)]
mod bridge_snapshot_tests {
    use super::*;
    use std::io::{Read, Write};

    #[test]
    fn bridge_snapshot_rejects_corrupted_real_train_layers() {
        let data_path = std::path::PathBuf::from(
            std::env::var_os("HMS_NATIVE_TEST_TRAIN").expect("real cached glove train.f32 path"),
        );
        let dim = 100;
        let n = 16384;
        let mut bytes = vec![0; n * dim * 4];
        std::fs::File::open(data_path)
            .unwrap()
            .read_exact(&mut bytes)
            .unwrap();
        let rows: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let graph = pool.install(|| {
            Graph::build(
                &rows,
                dim,
                &BuildParams {
                    degree: 8,
                    build_ef: 32,
                    ..BuildParams::default()
                },
            )
            .unwrap()
        });
        let index =
            pool.install(|| VGraph::from_graph_with(&graph, 8, 8, true, false, 3, false).unwrap());
        let directory = std::path::PathBuf::from(
            std::env::var_os("HMS_NATIVE_TEST_DIR").expect("durable test directory"),
        );
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join(format!("snapshot-{}.index", std::process::id()));
        let hash = format!("{:016x}", graph.structure_hash());
        index.bridge_save(&path, &hash).unwrap();
        let (loaded, loaded_dim, loaded_hash) = VGraph::bridge_load(&path).unwrap();
        assert_eq!(loaded_dim, dim);
        assert_eq!(loaded_hash, hash);
        let mut a = index.searcher();
        let mut b = loaded.searcher();
        let mut oa = Vec::new();
        let mut ob = Vec::new();
        for q in rows.chunks_exact(dim).take(16) {
            let params = VSearchParams {
                ef: 96,
                rerank: 32,
                screen: false,
                ..VSearchParams::default()
            };
            a.search(q, 10, params, &mut oa).unwrap();
            b.search(q, 10, params, &mut ob).unwrap();
            assert_eq!(oa, ob);
        }
        let config = bincode::config::standard().with_limit::<2147483648>();
        let mut file = std::fs::File::open(&path).unwrap();
        let mut magic = [0; 8];
        file.read_exact(&mut magic).unwrap();
        let snapshot: BridgeSnapshot =
            bincode::serde::decode_from_std_read(&mut file, config).unwrap();
        assert!(snapshot.layers.len() >= 2);
        let mut states = Vec::new();
        let mut bad = snapshot.clone();
        bad.dim = 4097;
        states.push(bad);
        let mut bad = snapshot.clone();
        let invalid = bad.layers[1].len() as u32;
        assert!(invalid < bad.upper_ids.len() as u32);
        bad.layers[1][0].push(invalid);
        states.push(bad);
        let mut bad = snapshot.clone();
        bad.layers[0].pop();
        states.push(bad);
        let mut bad = snapshot.clone();
        bad.layers[1].clear();
        states.push(bad);
        for state in states {
            let mut file = std::fs::File::create(&path).unwrap();
            file.write_all(BRIDGE_MAGIC).unwrap();
            bincode::serde::encode_into_std_write(state, &mut file, config).unwrap();
            file.flush().unwrap();
            assert!(VGraph::bridge_load(&path).is_err());
        }
        std::fs::remove_file(path).unwrap();
    }
}

impl VSearcher<'_> {
    pub fn bridge_workspace_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.visited.marks.capacity()
            + self.qf.capacity() * 4
            + self.qc.capacity()
            + self.qtmp.capacity()
            + self.qrot.capacity() * 4
            + self.qs.planes.capacity() * 8
            + self.fresh.capacity() * 4
            + self.survivors.capacity() * 4
            + self.cand.capacity() * 8
            + self.best.capacity() * 8
            + self.keys.capacity() * 8
            + self.topk.capacity() * 8
            + self.results.capacity() * std::mem::size_of::<(f32, u32)>()
    }
}

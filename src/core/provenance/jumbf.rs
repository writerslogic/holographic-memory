// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// JUMBF box type identifiers per ISO 19566-5.
pub const JUMBF_BOX_TYPE: [u8; 4] = *b"jumb";
pub const DESCRIPTION_BOX_TYPE: [u8; 4] = *b"jumd";
pub const JSON_BOX_TYPE: [u8; 4] = *b"json";
pub const CBOR_BOX_TYPE: [u8; 4] = *b"cbor";
pub const UUID_BOX_TYPE: [u8; 4] = *b"uuid";
pub const EMBEDDED_FILE_BOX_TYPE: [u8; 4] = *b"bfdb";
pub const CODESTREAM_BOX_TYPE: [u8; 4] = *b"jp2c";

/// C2PA manifest JUMBF content type UUID (per C2PA spec 2.1).
pub const C2PA_MANIFEST_UUID: [u8; 16] = [
    0x63, 0x32, 0x70, 0x61, 0x00, 0x11, 0x00, 0x10, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];

/// C2PA assertion store content type UUID.
pub const C2PA_ASSERTION_STORE_UUID: [u8; 16] = [
    0x63, 0x32, 0x61, 0x73, 0x00, 0x11, 0x00, 0x10, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];

/// C2PA claim content type UUID.
pub const C2PA_CLAIM_UUID: [u8; 16] = [
    0x63, 0x32, 0x63, 0x6C, 0x00, 0x11, 0x00, 0x10, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];

/// C2PA claim signature content type UUID.
pub const C2PA_SIGNATURE_UUID: [u8; 16] = [
    0x63, 0x32, 0x63, 0x73, 0x00, 0x11, 0x00, 0x10, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
];

/// A JUMBF superbox or leaf box.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JumbfBox {
    pub box_type: [u8; 4],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<DescriptionBox>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<BoxContent>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<JumbfBox>,
}

/// JUMBF description box (jumd) metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DescriptionBox {
    pub uuid: [u8; 16],
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requestable: Option<bool>,
}

/// Content payload of a leaf JUMBF box.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BoxContent {
    Json(serde_json::Value),
    Cbor(Vec<u8>),
    Binary(Vec<u8>),
}

impl JumbfBox {
    pub fn superbox(label: &str, uuid: [u8; 16]) -> Self {
        Self {
            box_type: JUMBF_BOX_TYPE,
            description: Some(DescriptionBox {
                uuid,
                label: label.to_string(),
                requestable: Some(true),
            }),
            content: None,
            children: Vec::new(),
        }
    }

    pub fn json_box(label: &str, content: serde_json::Value) -> Self {
        Self {
            box_type: JUMBF_BOX_TYPE,
            description: Some(DescriptionBox {
                uuid: [0; 16],
                label: label.to_string(),
                requestable: None,
            }),
            content: Some(BoxContent::Json(content)),
            children: Vec::new(),
        }
    }

    pub fn cbor_box(label: &str, content: Vec<u8>) -> Self {
        Self {
            box_type: JUMBF_BOX_TYPE,
            description: Some(DescriptionBox {
                uuid: [0; 16],
                label: label.to_string(),
                requestable: None,
            }),
            content: Some(BoxContent::Cbor(content)),
            children: Vec::new(),
        }
    }

    pub fn add_child(&mut self, child: JumbfBox) {
        self.children.push(child);
    }

    pub fn find_by_label(&self, label: &str) -> Option<&JumbfBox> {
        if self.description.as_ref().is_some_and(|d| d.label == label) {
            return Some(self);
        }
        for child in &self.children {
            if let Some(found) = child.find_by_label(label) {
                return Some(found);
            }
        }
        None
    }

    /// Encode this box hierarchy to JUMBF binary format (ISO 19566-5).
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut buf = Vec::new();
        encode_box(self, &mut buf)?;
        Ok(buf)
    }

    /// Decode a JUMBF box hierarchy from binary data.
    pub fn decode(data: &[u8]) -> Result<Self> {
        let (jbox, consumed) = decode_box(data)?;
        if consumed != data.len() {
            return Err(anyhow!(
                "trailing data: consumed {consumed}, total {}",
                data.len()
            ));
        }
        Ok(jbox)
    }
}

fn encode_box(jbox: &JumbfBox, buf: &mut Vec<u8>) -> Result<()> {
    let start = buf.len();
    buf.extend_from_slice(&[0u8; 4]);
    buf.extend_from_slice(&jbox.box_type);

    if let Some(desc) = &jbox.description {
        encode_description_box(desc, buf)?;
    }

    match &jbox.content {
        Some(BoxContent::Json(val)) => {
            let json_bytes =
                serde_json::to_vec(val).map_err(|e| anyhow!("JSON serialization failed: {e}"))?;
            buf.extend_from_slice(&box_len(json_bytes.len())?.to_be_bytes());
            buf.extend_from_slice(&JSON_BOX_TYPE);
            buf.extend_from_slice(&json_bytes);
        }
        Some(BoxContent::Cbor(data)) => {
            buf.extend_from_slice(&box_len(data.len())?.to_be_bytes());
            buf.extend_from_slice(&CBOR_BOX_TYPE);
            buf.extend_from_slice(data);
        }
        Some(BoxContent::Binary(data)) => {
            buf.extend_from_slice(&box_len(data.len())?.to_be_bytes());
            buf.extend_from_slice(&EMBEDDED_FILE_BOX_TYPE);
            buf.extend_from_slice(data);
        }
        None => {}
    }

    for child in &jbox.children {
        encode_box(child, buf)?;
    }

    let total_len = box_len(buf.len() - start - 8)?;
    buf[start..start + 4].copy_from_slice(&total_len.to_be_bytes());

    Ok(())
}

/// The 32-bit length field of a box with an 8-byte header and `payload` bytes after it.
fn box_len(payload: usize) -> Result<u32> {
    payload
        .checked_add(8)
        .and_then(|len| u32::try_from(len).ok())
        .ok_or_else(|| anyhow!("JUMBF box of {payload} bytes exceeds the 32-bit length field"))
}

/// Description box toggle bits (ISO 19566-5): requestable, and a label follows the toggles.
const TOGGLE_REQUESTABLE: u8 = 0x01;
const TOGGLE_LABEL: u8 = 0x02;

fn encode_description_box(desc: &DescriptionBox, buf: &mut Vec<u8>) -> Result<()> {
    let start = buf.len();
    buf.extend_from_slice(&[0u8; 4]);
    buf.extend_from_slice(&DESCRIPTION_BOX_TYPE);
    buf.extend_from_slice(&desc.uuid);

    // The label is always written, so the label bit is always set: a reader that honors the
    // toggles would otherwise take the label bytes for the optional fields after it.
    let toggle = if desc.requestable.unwrap_or(false) {
        TOGGLE_LABEL | TOGGLE_REQUESTABLE
    } else {
        TOGGLE_LABEL
    };
    buf.push(toggle);

    buf.extend_from_slice(desc.label.as_bytes());
    buf.push(0);

    let total_len = box_len(buf.len() - start - 8)?;
    buf[start..start + 4].copy_from_slice(&total_len.to_be_bytes());

    Ok(())
}

/// Maximum JUMBF box nesting depth. Bounds recursion in `decode_box` so a
/// crafted deeply-nested superbox (a few KB) cannot overflow the stack.
const MAX_JUMBF_DEPTH: usize = 64;

fn decode_box(data: &[u8]) -> Result<(JumbfBox, usize)> {
    decode_box_depth(data, 0)
}

fn decode_box_depth(data: &[u8], depth: usize) -> Result<(JumbfBox, usize)> {
    if depth > MAX_JUMBF_DEPTH {
        return Err(anyhow!(
            "JUMBF nesting exceeds maximum depth of {MAX_JUMBF_DEPTH}"
        ));
    }
    if data.len() < 8 {
        return Err(anyhow!("insufficient data for box header"));
    }
    let box_len = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;
    if box_len < 8 || box_len > data.len() {
        return Err(anyhow!("invalid box length: {box_len}"));
    }
    let box_type: [u8; 4] = [data[4], data[5], data[6], data[7]];
    let box_data = &data[8..box_len];

    let mut offset = 0;
    let mut description = None;
    let mut content = None;
    let mut children = Vec::new();

    // Every byte of the box must belong to a well-formed inner box: a truncated or
    // overlong inner header is an error, not the end of the box, so nothing after it can be
    // silently dropped. A superbox has one description box, first, and one content box.
    while offset < box_data.len() {
        if box_data.len() - offset < 8 {
            return Err(anyhow!("truncated inner box header at offset {offset}"));
        }
        let inner_len = u32::from_be_bytes([
            box_data[offset],
            box_data[offset + 1],
            box_data[offset + 2],
            box_data[offset + 3],
        ]) as usize;
        if inner_len < 8 || inner_len > box_data.len() - offset {
            return Err(anyhow!(
                "invalid inner box length {inner_len} at offset {offset}"
            ));
        }
        let inner_type: [u8; 4] = [
            box_data[offset + 4],
            box_data[offset + 5],
            box_data[offset + 6],
            box_data[offset + 7],
        ];
        let inner_data = &box_data[offset + 8..offset + inner_len];

        match inner_type {
            DESCRIPTION_BOX_TYPE => {
                if offset != 0 {
                    return Err(anyhow!("description box is not the first box"));
                }
                description = Some(decode_description_box(inner_data)?);
            }
            _ if inner_type != JUMBF_BOX_TYPE && content.is_some() => {
                return Err(anyhow!("box has more than one content box"));
            }
            JSON_BOX_TYPE => {
                let val = serde_json::from_slice(inner_data)
                    .map_err(|e| anyhow!("JSON decode failed: {e}"))?;
                content = Some(BoxContent::Json(val));
            }
            CBOR_BOX_TYPE => {
                content = Some(BoxContent::Cbor(inner_data.to_vec()));
            }
            JUMBF_BOX_TYPE => {
                let (child, _) = decode_box_depth(&box_data[offset..], depth + 1)?;
                children.push(child);
            }
            _ => {
                content = Some(BoxContent::Binary(inner_data.to_vec()));
            }
        }

        offset += inner_len;
    }

    Ok((
        JumbfBox {
            box_type,
            description,
            content,
            children,
        },
        box_len,
    ))
}

fn decode_description_box(data: &[u8]) -> Result<DescriptionBox> {
    if data.len() < 17 {
        return Err(anyhow!("description box too short"));
    }
    let mut uuid = [0u8; 16];
    uuid.copy_from_slice(&data[..16]);
    let toggle = data[16];
    let requestable = Some(toggle & TOGGLE_REQUESTABLE != 0);

    // Writers before the label bit was set wrote toggles 0 followed by a label.
    if toggle & TOGGLE_LABEL == 0 && (toggle != 0 || data.len() == 17) {
        return Ok(DescriptionBox {
            uuid,
            label: String::new(),
            requestable,
        });
    }
    let label_data = &data[17..];
    let label_end = label_data
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(label_data.len());
    let label = String::from_utf8(label_data[..label_end].to_vec())
        .map_err(|e| anyhow!("invalid label UTF-8: {e}"))?;

    Ok(DescriptionBox {
        uuid,
        label,
        requestable,
    })
}

/// Build a C2PA manifest store JUMBF superbox containing claim, signature, and assertions.
pub fn build_c2pa_manifest_box(
    label: &str,
    claim_json: serde_json::Value,
    signature_cbor: Vec<u8>,
    assertions: Vec<(&str, serde_json::Value)>,
) -> JumbfBox {
    let mut manifest = JumbfBox::superbox(label, C2PA_MANIFEST_UUID);

    let mut assertion_store = JumbfBox::superbox("c2pa.assertions", C2PA_ASSERTION_STORE_UUID);
    for (assertion_label, assertion_value) in assertions {
        assertion_store.add_child(JumbfBox::json_box(assertion_label, assertion_value));
    }
    manifest.add_child(assertion_store);
    manifest.add_child(JumbfBox::json_box("c2pa.claim", claim_json));
    manifest.add_child(JumbfBox::cbor_box("c2pa.signature", signature_cbor));

    manifest
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrip() {
        let mut manifest = JumbfBox::superbox("test-manifest", C2PA_MANIFEST_UUID);
        manifest.add_child(JumbfBox::json_box(
            "c2pa.claim",
            serde_json::json!({"title": "test"}),
        ));
        manifest.add_child(JumbfBox::cbor_box("c2pa.signature", vec![0xA0]));

        let encoded = manifest.encode().unwrap();
        let decoded = JumbfBox::decode(&encoded).unwrap();

        assert_eq!(decoded.box_type, JUMBF_BOX_TYPE);
        let desc = decoded.description.as_ref().unwrap();
        assert_eq!(desc.label, "test-manifest");
        assert_eq!(desc.uuid, C2PA_MANIFEST_UUID);
    }

    #[test]
    fn c2pa_manifest_box_structure() {
        let claim = serde_json::json!({
            "dc:title": "HMS Knowledge Store",
            "claim_generator": "HMS/0.1"
        });
        let signature = vec![0xD2, 0x84, 0x43];
        let assertions = vec![
            (
                "c2pa.hash.data",
                serde_json::json!({"alg": "sha256", "hash": "abc123"}),
            ),
            (
                "c2pa.actions",
                serde_json::json!({"actions": [{"action": "c2pa.created"}]}),
            ),
        ];

        let manifest = build_c2pa_manifest_box("urn:hms:manifest:1", claim, signature, assertions);

        assert!(manifest.find_by_label("c2pa.assertions").is_some());
        assert!(manifest.find_by_label("c2pa.claim").is_some());
        assert!(manifest.find_by_label("c2pa.signature").is_some());
        assert!(manifest.find_by_label("c2pa.hash.data").is_some());
        assert!(manifest.find_by_label("c2pa.actions").is_some());
    }

    #[test]
    fn binary_encode_valid_structure() {
        let jbox = JumbfBox::json_box("test", serde_json::json!({"k": "v"}));
        let mut parent = JumbfBox::superbox("parent", [0; 16]);
        parent.add_child(jbox);

        let encoded = parent.encode().unwrap();
        assert!(encoded.len() > 16);
        assert_eq!(&encoded[4..8], &JUMBF_BOX_TYPE);

        let decoded = JumbfBox::decode(&encoded).unwrap();
        assert_eq!(decoded.children.len(), 1);
        assert!(decoded.find_by_label("test").is_some());
        assert!(decoded.find_by_label("nonexistent").is_none());
    }

    #[test]
    fn malformed_or_ambiguous_inner_boxes_are_rejected() {
        let mut parent = JumbfBox::superbox("parent", [0; 16]);
        parent.add_child(JumbfBox::json_box("child", serde_json::json!({"k": "v"})));
        let encoded = parent.encode().unwrap();

        // Trailing bytes inside the outer box that do not form a box.
        for tail in [&[0u8, 0, 0][..], &[0, 0, 0, 3, b'x', b'x', b'x', b'x'][..]] {
            let mut bad = encoded.clone();
            bad.extend_from_slice(tail);
            let len = u32::try_from(bad.len()).unwrap();
            bad[..4].copy_from_slice(&len.to_be_bytes());
            assert!(JumbfBox::decode(&bad).is_err(), "{tail:?}");
        }

        // A second description box would relabel the superbox.
        let mut relabeled = encoded.clone();
        let jumd_len =
            usize::try_from(u32::from_be_bytes(encoded[8..12].try_into().unwrap())).unwrap();
        relabeled.extend_from_slice(&encoded[8..8 + jumd_len]);
        let len = u32::try_from(relabeled.len()).unwrap();
        relabeled[..4].copy_from_slice(&len.to_be_bytes());
        assert!(JumbfBox::decode(&relabeled).is_err());
    }

    #[test]
    fn description_box_flags_its_label() {
        let encoded = JumbfBox::json_box("c2pa.claim", serde_json::json!({}))
            .encode()
            .unwrap();
        // jumb header (8), jumd header (8), uuid (16), then the toggles.
        assert_eq!(encoded[32] & 0x02, 0x02);

        // Boxes written with toggles 0 and a label still read back with it.
        let mut legacy = encoded.clone();
        legacy[32] = 0;
        let decoded = JumbfBox::decode(&legacy).unwrap();
        assert_eq!(decoded.description.unwrap().label, "c2pa.claim");
    }

    #[test]
    fn nested_encode_decode() {
        let claim = serde_json::json!({"title": "roundtrip test"});
        let manifest = build_c2pa_manifest_box("urn:test:1", claim, vec![0xA0, 0xB1], vec![]);

        let encoded = manifest.encode().unwrap();
        let decoded = JumbfBox::decode(&encoded).unwrap();

        let found_claim = decoded.find_by_label("c2pa.claim").unwrap();
        match &found_claim.content {
            Some(BoxContent::Json(v)) => assert_eq!(v["title"], "roundtrip test"),
            _ => panic!("expected JSON content in claim box"),
        }
    }
}

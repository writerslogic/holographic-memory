// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

#[allow(dead_code)]
mod fixture {
    #[derive(Clone, Debug)]
    struct Block {
        offset: usize,
        first: usize,
        last: usize,
        count: u16,
        doc_bits: u8,
        frequency_bits: u8,
        max_frequency: u32,
        min_words: u32,
        frontier: Option<Box<[(u32, u32)]>>,
    }

    #[derive(Default)]
    pub(super) struct PostingList {
        data: Vec<u8>,
        blocks: Vec<Block>,
        count: u32,
        positive_count: u32,
        last: u64,
    }

    pub(super) fn check() {
        let mut list = PostingList::default();
        list.blocks.push(Block {
            offset: 0,
            first: 0,
            last: 0,
            count: 1,
            doc_bits: 0,
            frequency_bits: 1,
            max_frequency: 1,
            min_words: 1,
            frontier: None,
        });
        let original_capacity = list.blocks.capacity();
        let mut reserved = PostingList::default();
        reserved.blocks.reserve_exact(1);
        reserved.blocks.push(list.blocks.pop().unwrap());
        assert_eq!(std::mem::size_of::<Block>(), 56);
        assert_eq!(std::mem::size_of::<PostingList>(), 64);
        assert_eq!(original_capacity, 4);
        assert_eq!(reserved.blocks.capacity(), 1);
        println!("{{\"block_size\":{},\"posting_list_size\":{},\"initial_push_capacity\":{original_capacity},\"reserved_capacity\":{}}}", std::mem::size_of::<Block>(), std::mem::size_of::<PostingList>(), reserved.blocks.capacity());
    }
}
fn main() {
    fixture::check();
}

// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use super::types::TextMetrics;
use rayon::prelude::*;

const TEXT_CHUNK_SIZE: usize = 32768;

pub struct TextProcessor;

/// Returns true if byte `b` starts a non-spacing character (CJK, Thai, etc.)
fn is_non_spacing_byte(b: u8) -> bool {
    // 0xE0: Thai, Tibetan
    // 0xE1: Myanmar, Khmer
    // 0xE3: Hiragana, Katakana
    // 0xE4..=0xE9: CJK Unified
    // 0xEA..=0xED: Hangul
    // 0xEF: Halfwidth/Fullwidth forms
    (0xE0..=0xED).contains(&b) || b == 0xEF
}

/// Returns true if byte `b` starts a word-continuing character (letter or non-ASCII spacing char).
fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphabetic() || (b >= 0x80 && !is_non_spacing_byte(b))
}

// Chunk result: (words, sentences, syllables, vowels, consonants, punctuation, in_word_at_end, starts_with_word_char)
type ChunkResult = (u32, u32, u32, u32, u32, u32, bool, bool);

impl TextProcessor {
    /// Parallel text analyzer. Splits on ASCII whitespace/punctuation.
    /// CJK/Thai/Myanmar detection is a first-byte heuristic. Syllable
    /// count is approximated by counting vowel groups.
    pub fn analyze(text: &str) -> TextMetrics {
        // Multi-threaded processing with UTF-8 safety: split only at char boundaries
        let mut chunk_starts = Vec::new();
        let mut pos = 0;
        while pos < text.len() {
            chunk_starts.push(pos);
            pos += TEXT_CHUNK_SIZE;
            while pos < text.len() && !text.is_char_boundary(pos) {
                pos += 1;
            }
        }
        chunk_starts.push(text.len());

        let metrics: ChunkResult = chunk_starts
            .par_windows(2)
            .map(|win| {
                let chunk_str = &text[win[0]..win[1]];
                let chunk = chunk_str.as_bytes();
                let mut words = 0u32;
                let mut sentences = 0u32;
                let mut syllables = 0u32;
                let mut vowels = 0u32;
                let mut consonants = 0u32;
                let mut punctuation = 0u32;
                let mut in_word = false;
                let mut last_was_vowel = false;
                let starts_with_word_char = !chunk.is_empty() && is_word_byte(chunk[0]);

                // Fast ASCII/UTF8 character class scanner
                let mut i = 0;
                while i < chunk.len() {
                    let b = chunk[i];

                    // Identify character category
                    if b < 128 {
                        // ASCII Path
                        match b {
                            b'a' | b'e' | b'i' | b'o' | b'u' | b'A' | b'E' | b'I' | b'O' | b'U' => {
                                vowels += 1;
                                if !last_was_vowel {
                                    syllables += 1;
                                }
                                last_was_vowel = true;
                                in_word = true;
                            }
                            // y is consonant at word start ("yes", "yellow"),
                            // vowel elsewhere ("gym", "myth", "baby")
                            b'y' | b'Y' => {
                                if in_word {
                                    vowels += 1;
                                    if !last_was_vowel {
                                        syllables += 1;
                                    }
                                    last_was_vowel = true;
                                } else {
                                    consonants += 1;
                                    last_was_vowel = false;
                                }
                                in_word = true;
                            }
                            b'a'..=b'z' | b'A'..=b'Z' => {
                                consonants += 1;
                                in_word = true;
                                last_was_vowel = false;
                            }
                            b' ' | b'\t' | b'\n' | b'\r' => {
                                if in_word {
                                    words += 1;
                                    in_word = false;
                                }
                                last_was_vowel = false;
                            }
                            b'.' | b'!' | b'?' => {
                                punctuation += 1;
                                sentences += 1;
                                if in_word {
                                    words += 1;
                                    in_word = false;
                                }
                                last_was_vowel = false;
                            }
                            b',' | b';' | b':' | b'(' | b')' | b'[' | b']' | b'{' | b'}' | b'"'
                            | b'\'' | b'-' => {
                                punctuation += 1;
                                if in_word {
                                    words += 1;
                                    in_word = false;
                                }
                                last_was_vowel = false;
                            }
                            _ => {
                                last_was_vowel = false;
                            }
                        }
                        i += 1;
                    } else {
                        // Multi-byte UTF-8 Path
                        // Basic heuristic: Non-ASCII non-punctuation are word parts
                        let char_len = if b & 0b11100000 == 0b11000000 {
                            2
                        } else if b & 0b11110000 == 0b11100000 {
                            3
                        } else if b & 0b11111000 == 0b11110000 {
                            4
                        } else {
                            1
                        };

                        // Non-spacing characters are often their own words
                        if is_non_spacing_byte(b) {
                            if in_word {
                                words += 1;
                            }
                            words += 1;
                            in_word = false;
                            // Approximating 1 syllable per non-spacing char
                            syllables += 1;
                        } else {
                            in_word = true;
                        }
                        last_was_vowel = false;
                        i += char_len;
                    }
                }
                (
                    words,
                    sentences,
                    syllables,
                    vowels,
                    consonants,
                    punctuation,
                    in_word,
                    starts_with_word_char,
                )
            })
            .reduce(
                || (0, 0, 0, 0, 0, 0, false, false),
                |a, b| {
                    let mut words = a.0 + b.0;
                    // If chunk A ended mid-word but chunk B doesn't continue it,
                    // the trailing word from A was never counted.
                    if a.6 && !b.7 {
                        words += 1;
                    }
                    (
                        words,
                        a.1 + b.1,
                        a.2 + b.2,
                        a.3 + b.3,
                        a.4 + b.4,
                        a.5 + b.5,
                        b.6,
                        a.7,
                    )
                },
            );

        let (mut words, sentences, syllables, vowels, consonants, punctuation, last_in_word, _) =
            metrics;
        if last_in_word {
            words += 1;
        }

        TextMetrics {
            word_count: words,
            sentence_count: sentences,
            syllable_count: syllables,
            vowel_count: vowels,
            consonant_count: consonants,
            punctuation_count: punctuation,
        }
    }

    pub fn calculate_readability(metrics: &TextMetrics) -> f64 {
        if metrics.word_count == 0 || metrics.sentence_count == 0 {
            return 0.0;
        }
        let asw = metrics.word_count as f64 / metrics.sentence_count as f64;
        let asy = metrics.syllable_count as f64 / metrics.word_count as f64;
        206.835 - (1.015 * asw) - (84.6 * asy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syllable_counting() {
        let metrics = TextProcessor::analyze("queue");
        assert_eq!(metrics.syllable_count, 1, "queue should be 1 syllable");

        let metrics = TextProcessor::analyze("hello");
        assert_eq!(metrics.syllable_count, 2, "hello should be 2 syllables");

        let metrics = TextProcessor::analyze("beauty");
        assert_eq!(metrics.syllable_count, 2, "beauty should be 2 syllables");
    }

    #[test]
    fn test_non_spacing_scripts() {
        // CJK
        let metrics = TextProcessor::analyze("你好世界");
        assert_eq!(metrics.word_count, 4);
        assert_eq!(metrics.syllable_count, 4);

        // Thai (3 characters)
        let metrics = TextProcessor::analyze("สวัสดี");
        assert_eq!(metrics.word_count, 6);
    }

    #[test]
    fn y_consonant_at_word_start() {
        let metrics = TextProcessor::analyze("yes");
        assert_eq!(
            metrics.consonant_count, 2,
            "'y' and 's' are consonants in 'yes'"
        );
        assert_eq!(metrics.vowel_count, 1, "only 'e' is vowel in 'yes'");

        let metrics = TextProcessor::analyze("gym");
        assert_eq!(metrics.vowel_count, 1, "'y' is vowel in 'gym'");
        assert_eq!(metrics.syllable_count, 1);
    }

    #[test]
    fn empty_input_returns_zeros() {
        let metrics = TextProcessor::analyze("");
        assert_eq!(metrics.word_count, 0);
        assert_eq!(metrics.sentence_count, 0);
        assert_eq!(metrics.syllable_count, 0);

        let score = TextProcessor::calculate_readability(&metrics);
        assert_eq!(score, 0.0, "readability of empty input should be 0");
    }

    #[test]
    fn readability_simple_text() {
        let metrics = TextProcessor::analyze("The cat sat on the mat.");
        let score = TextProcessor::calculate_readability(&metrics);
        // Simple short sentence: high readability (Flesch score > 80)
        assert!(
            score > 80.0,
            "Simple text should score > 80, got {:.1}",
            score
        );
    }

    #[test]
    fn readability_complex_text() {
        let metrics = TextProcessor::analyze(
            "The implementation of sophisticated algorithmic methodologies \
             necessitates comprehensive understanding of computational complexity. \
             Furthermore, the juxtaposition of theoretical abstractions with \
             practical considerations illuminates fundamental architectural decisions.",
        );
        let score = TextProcessor::calculate_readability(&metrics);
        // Complex text: lower readability (Flesch score < 40)
        assert!(
            score < 40.0,
            "Complex text should score < 40, got {:.1}",
            score
        );
    }

    #[test]
    fn word_count_across_chunk_boundary() {
        let mut text = String::with_capacity(40000);
        for _ in 0..5461 {
            text.push_str("hello ");
        }
        text.push_str("ab cd");
        let metrics = TextProcessor::analyze(&text);
        assert_eq!(
            metrics.word_count, 5463,
            "Should count word at chunk boundary correctly"
        );
    }
}

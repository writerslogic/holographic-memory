// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use fxhash::FxHashSet;

#[derive(Clone, Debug)]
pub struct MeaningUnit {
    pub subject: String,
    pub relation: String,
    pub object: String,
}

pub struct Decomposer {
    prepositions: FxHashSet<String>,
    relation_verbs: FxHashSet<String>,
}

impl Decomposer {
    pub fn new() -> Self {
        let preps = [
            "about",
            "above",
            "across",
            "after",
            "against",
            "along",
            "amid",
            "among",
            "around",
            "as",
            "at",
            "before",
            "behind",
            "below",
            "beneath",
            "beside",
            "between",
            "beyond",
            "by",
            "concerning",
            "despite",
            "down",
            "during",
            "except",
            "for",
            "from",
            "in",
            "inside",
            "into",
            "like",
            "near",
            "of",
            "off",
            "on",
            "onto",
            "opposite",
            "out",
            "outside",
            "over",
            "past",
            "per",
            "regarding",
            "since",
            "than",
            "through",
            "throughout",
            "till",
            "to",
            "toward",
            "under",
            "unlike",
            "until",
            "up",
            "upon",
            "via",
            "with",
            "within",
            "without",
        ];
        let verbs = [
            "is", "are", "was", "were", "has", "have", "had", "makes", "gives", "takes", "gets",
            "keeps", "puts", "says", "goes", "comes", "knows", "thinks", "sees", "finds", "shows",
            "means", "becomes", "contains", "includes", "requires", "provides", "creates",
            "causes", "supports",
        ];
        Self {
            prepositions: preps.iter().map(|s| s.to_string()).collect(),
            relation_verbs: verbs.iter().map(|s| s.to_string()).collect(),
        }
    }

    pub fn decompose(&self, text: &str) -> Vec<MeaningUnit> {
        let mut results = Vec::new();
        for sentence in text.split(|c: char| c == '.' || c == '!' || c == '?') {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }
            if let Some(unit) = self.try_extract(sentence) {
                results.push(unit);
            }
        }
        results
    }

    fn try_extract(&self, sentence: &str) -> Option<MeaningUnit> {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        if words.len() < 3 {
            return None;
        }

        // Pattern: NP "is" NP → is_a
        if let Some(is_pos) = words
            .iter()
            .position(|&w| w.eq_ignore_ascii_case("is") || w.eq_ignore_ascii_case("are"))
        {
            if is_pos > 0 && is_pos + 1 < words.len() {
                let subject = self.extract_np(&words[..is_pos]);
                let object = self.extract_np(&words[is_pos + 1..]);
                if !subject.is_empty() && !object.is_empty() {
                    return Some(MeaningUnit {
                        subject,
                        relation: "is_a".to_string(),
                        object,
                    });
                }
            }
        }

        // Pattern: NP verb prep NP → verb_prep
        for (i, &word) in words.iter().enumerate() {
            let lower = word.to_lowercase();
            if self.relation_verbs.contains(&lower) && i > 0 {
                if i + 2 < words.len() {
                    let next_lower = words[i + 1].to_lowercase();
                    if self.prepositions.contains(&next_lower) {
                        let subject = self.extract_np(&words[..i]);
                        let object = self.extract_np(&words[i + 2..]);
                        if !subject.is_empty() && !object.is_empty() {
                            return Some(MeaningUnit {
                                subject,
                                relation: format!("{}_{}", lower, next_lower),
                                object,
                            });
                        }
                    }
                }

                // Pattern: NP verb NP
                if i + 1 < words.len() {
                    let subject = self.extract_np(&words[..i]);
                    let object = self.extract_np(&words[i + 1..]);
                    if !subject.is_empty() && !object.is_empty() {
                        return Some(MeaningUnit {
                            subject,
                            relation: lower,
                            object,
                        });
                    }
                }
            }
        }

        None
    }

    fn extract_np(&self, words: &[&str]) -> String {
        let determiners = [
            "the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her", "its",
            "our", "their",
        ];
        let filtered: Vec<&str> = words
            .iter()
            .filter(|w| !determiners.contains(&w.to_lowercase().as_str()))
            .copied()
            .collect();
        if filtered.is_empty() {
            return String::new();
        }
        filtered.join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decompose_is_a() {
        let d = Decomposer::new();
        let units = d.decompose("Paris is the capital of France");
        assert!(!units.is_empty());
        assert_eq!(units[0].subject, "Paris");
        assert_eq!(units[0].relation, "is_a");
    }

    #[test]
    fn test_decompose_verb_prep() {
        let d = Decomposer::new();
        let units = d.decompose("The cat goes to the park");
        assert!(!units.is_empty());
    }

    #[test]
    fn test_decompose_empty() {
        let d = Decomposer::new();
        assert!(d.decompose("").is_empty());
        assert!(d.decompose("hi").is_empty());
    }

    #[test]
    fn test_decompose_multiple_sentences() {
        let d = Decomposer::new();
        let units = d.decompose("Paris is a city. Berlin is a city.");
        assert_eq!(units.len(), 2);
    }
}

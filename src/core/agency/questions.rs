// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

//! Template-based question generation.
//!
//! Given knowledge gaps or hypotheses, generates natural language questions.
//! Questions are ranked by goal-relevance: overlap between the question's
//! vector and the highest-priority active goal.

use crate::core::agency::goals::GoalStore;
use crate::core::atom_memory::AtomMemory;
use crate::core::cognition::gaps::KnowledgeGap;
use crate::core::cognition::hypothesis::Hypothesis;
use crate::core::entangled::EntangledHVec;

/// A generated question with its relevance score.
#[derive(Clone, Debug)]
pub struct Question {
    pub text: String,
    pub source: QuestionSource,
    pub goal_relevance: f64,
}

/// What triggered the question.
#[derive(Clone, Debug)]
pub enum QuestionSource {
    Gap {
        entity: String,
        relation: String,
    },
    Hypothesis {
        entity: String,
        relation: String,
        proposed: String,
    },
    Exploration {
        relation: String,
    },
}

/// Generates questions from gaps, hypotheses, and exploration targets.
pub struct QuestionGenerator;

impl QuestionGenerator {
    /// Generate questions from knowledge gaps.
    pub fn from_gaps(
        gaps: &[KnowledgeGap],
        atom_memory: &AtomMemory,
        goal_store: &GoalStore,
    ) -> Vec<Question> {
        gaps.iter()
            .map(|gap| {
                let text = format!("What is the {} of {}?", gap.missing_relation, gap.entity);
                let relevance = Self::score_relevance(&text, atom_memory, goal_store);
                Question {
                    text,
                    source: QuestionSource::Gap {
                        entity: gap.entity.clone(),
                        relation: gap.missing_relation.clone(),
                    },
                    goal_relevance: relevance,
                }
            })
            .collect()
    }

    /// Generate confirmation questions from hypotheses.
    pub fn from_hypotheses(
        hypotheses: &[Hypothesis],
        atom_memory: &AtomMemory,
        goal_store: &GoalStore,
    ) -> Vec<Question> {
        hypotheses
            .iter()
            .map(|hyp| {
                let text = format!(
                    "Is {} the {} of {}? (confidence: {:.0}%)",
                    hyp.proposed_filler,
                    hyp.relation,
                    hyp.entity,
                    hyp.confidence * 100.0
                );
                let relevance = Self::score_relevance(&text, atom_memory, goal_store);
                Question {
                    text,
                    source: QuestionSource::Hypothesis {
                        entity: hyp.entity.clone(),
                        relation: hyp.relation.clone(),
                        proposed: hyp.proposed_filler.clone(),
                    },
                    goal_relevance: relevance,
                }
            })
            .collect()
    }

    /// Generate exploration questions for under-represented relations.
    pub fn from_relations(
        relations: &[String],
        atom_memory: &AtomMemory,
        goal_store: &GoalStore,
    ) -> Vec<Question> {
        relations
            .iter()
            .map(|rel| {
                let text = format!("What entities have the {} relation?", rel);
                let relevance = Self::score_relevance(&text, atom_memory, goal_store);
                Question {
                    text,
                    source: QuestionSource::Exploration {
                        relation: rel.clone(),
                    },
                    goal_relevance: relevance,
                }
            })
            .collect()
    }

    /// Rank questions by goal relevance (highest first).
    pub fn prioritize(mut questions: Vec<Question>) -> Vec<Question> {
        questions.sort_by(|a, b| {
            b.goal_relevance
                .partial_cmp(&a.goal_relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        questions
    }

    fn score_relevance(text: &str, atom_memory: &AtomMemory, goal_store: &GoalStore) -> f64 {
        // Encode the question text as a vector by bundling its word atoms
        let words: Vec<&str> = text.split_whitespace().collect();
        let vecs: Vec<EntangledHVec> = words
            .iter()
            .filter_map(|w| atom_memory.get(&w.to_lowercase()))
            .collect();
        if vecs.is_empty() {
            return 0.0;
        }
        let question_vec = EntangledHVec::bundle(&vecs);
        if question_vec.indices().is_empty() {
            return 0.0;
        }
        goal_store.goal_relevance(&question_vec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::agency::goals::Goal;
    use crate::core::cognition::gaps::KnowledgeGap;
    use crate::core::cognition::hypothesis::Hypothesis;

    fn make_test_env() -> (AtomMemory, GoalStore) {
        let dim = 16384;
        let atom_mem = AtomMemory::new(dim, 3.0);
        let goal_store = GoalStore::new();

        atom_mem.get_or_insert("located_in");
        atom_mem.get_or_insert("tokyo");
        atom_mem.get_or_insert("europe");

        goal_store.add(Goal {
            name: "find_locations".to_string(),
            description: "Find where things are located".to_string(),
            vector: atom_mem.get("located_in").unwrap(),
            relevance: 1.0,
            urgency: 1.0,
            cost: 0.0,
            active: true,
        });

        (atom_mem, goal_store)
    }

    #[test]
    fn test_from_gaps() {
        let (atom_mem, goal_store) = make_test_env();
        let gaps = vec![KnowledgeGap {
            entity: "tokyo".to_string(),
            missing_relation: "located_in".to_string(),
            peers_with_relation: vec!["paris".to_string()],
            peer_coverage: 1.0,
        }];

        let questions = QuestionGenerator::from_gaps(&gaps, &atom_mem, &goal_store);
        assert_eq!(questions.len(), 1);
        assert!(questions[0].text.contains("located_in"));
        assert!(questions[0].text.contains("tokyo"));
    }

    #[test]
    fn test_from_hypotheses() {
        let (atom_mem, goal_store) = make_test_env();
        let hypotheses = vec![Hypothesis {
            entity: "tokyo".to_string(),
            relation: "located_in".to_string(),
            proposed_filler: "asia".to_string(),
            confidence: 0.85,
            evidence_count: 3,
            confirmed: false,
        }];

        let questions = QuestionGenerator::from_hypotheses(&hypotheses, &atom_mem, &goal_store);
        assert_eq!(questions.len(), 1);
        assert!(questions[0].text.contains("asia"));
        assert!(questions[0].text.contains("85%"));
    }

    #[test]
    fn test_prioritize() {
        let q1 = Question {
            text: "low".to_string(),
            source: QuestionSource::Exploration {
                relation: "r".to_string(),
            },
            goal_relevance: 0.1,
        };
        let q2 = Question {
            text: "high".to_string(),
            source: QuestionSource::Exploration {
                relation: "r".to_string(),
            },
            goal_relevance: 0.9,
        };
        let ranked = QuestionGenerator::prioritize(vec![q1, q2]);
        assert_eq!(ranked[0].text, "high");
        assert_eq!(ranked[1].text, "low");
    }

    #[test]
    fn test_no_goals_zero_relevance() {
        let atom_mem = AtomMemory::new(16384, 3.0);
        let goal_store = GoalStore::new(); // empty
        let gaps = vec![KnowledgeGap {
            entity: "x".to_string(),
            missing_relation: "r".to_string(),
            peers_with_relation: vec![],
            peer_coverage: 1.0,
        }];
        let questions = QuestionGenerator::from_gaps(&gaps, &atom_mem, &goal_store);
        assert_eq!(questions[0].goal_relevance, 0.0);
    }
}

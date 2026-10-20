// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

use fxhash::{FxHashMap, FxHashSet};

#[derive(Clone, Debug)]
pub struct MeaningUnit {
    pub subject: String,
    pub relation: String,
    pub object: String,
}

pub struct Decomposer {
    prepositions: FxHashSet<&'static str>,
    irregular_verbs: FxHashMap<&'static str, &'static str>,
    auxiliaries: FxHashSet<&'static str>,
    determiners: FxHashSet<&'static str>,
    coordinating_conj: FxHashSet<&'static str>,
    adverbs: FxHashSet<&'static str>,
}

impl Decomposer {
    pub fn new() -> Self {
        let prepositions: FxHashSet<&'static str> = [
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
            "towards",
            "under",
            "unlike",
            "until",
            "up",
            "upon",
            "via",
            "with",
            "within",
            "without",
        ]
        .into_iter()
        .collect();

        // Irregular verbs: inflected form → lemma.
        // Covers the ~80 most common irregular English verbs.
        let irregular_verbs: FxHashMap<&'static str, &'static str> = [
            // be
            ("is", "be"),
            ("are", "be"),
            ("was", "be"),
            ("were", "be"),
            ("am", "be"),
            ("been", "be"),
            ("being", "be"),
            // have
            ("has", "have"),
            ("had", "have"),
            ("having", "have"),
            // do
            ("does", "do"),
            ("did", "do"),
            ("done", "do"),
            // say
            ("says", "say"),
            ("said", "say"),
            // make
            ("makes", "make"),
            ("made", "make"),
            ("making", "make"),
            // go
            ("goes", "go"),
            ("went", "go"),
            ("gone", "go"),
            ("going", "go"),
            // take
            ("takes", "take"),
            ("took", "take"),
            ("taken", "take"),
            ("taking", "take"),
            // come
            ("comes", "come"),
            ("came", "come"),
            ("coming", "come"),
            // see
            ("sees", "see"),
            ("saw", "see"),
            ("seen", "see"),
            ("seeing", "see"),
            // know
            ("knows", "know"),
            ("knew", "know"),
            ("known", "know"),
            // get
            ("gets", "get"),
            ("got", "get"),
            ("gotten", "get"),
            ("getting", "get"),
            // give
            ("gives", "give"),
            ("gave", "give"),
            ("given", "give"),
            ("giving", "give"),
            // find
            ("finds", "find"),
            ("found", "find"),
            ("finding", "find"),
            // think
            ("thinks", "think"),
            ("thought", "think"),
            ("thinking", "think"),
            // tell
            ("tells", "tell"),
            ("told", "tell"),
            ("telling", "tell"),
            // become
            ("becomes", "become"),
            ("became", "become"),
            // leave
            ("leaves", "leave"),
            ("left", "leave"),
            ("leaving", "leave"),
            // put
            ("puts", "put"),
            ("putting", "put"),
            // keep
            ("keeps", "keep"),
            ("kept", "keep"),
            ("keeping", "keep"),
            // let
            ("lets", "let"),
            ("letting", "let"),
            // begin
            ("begins", "begin"),
            ("began", "begin"),
            ("begun", "begin"),
            // show
            ("shows", "show"),
            ("showed", "show"),
            ("shown", "show"),
            ("showing", "show"),
            // hear
            ("hears", "hear"),
            ("heard", "hear"),
            // run
            ("runs", "run"),
            ("ran", "run"),
            ("running", "run"),
            // move
            ("moves", "move"),
            ("moved", "move"),
            ("moving", "move"),
            // live
            ("lives", "live"),
            ("lived", "live"),
            ("living", "live"),
            // believe
            ("believes", "believe"),
            ("believed", "believe"),
            // hold
            ("holds", "hold"),
            ("held", "hold"),
            ("holding", "hold"),
            // bring
            ("brings", "bring"),
            ("brought", "bring"),
            // write
            ("writes", "write"),
            ("wrote", "write"),
            ("written", "write"),
            ("writing", "write"),
            // sit
            ("sits", "sit"),
            ("sat", "sit"),
            ("sitting", "sit"),
            // stand
            ("stands", "stand"),
            ("stood", "stand"),
            ("standing", "stand"),
            // lose
            ("loses", "lose"),
            ("lost", "lose"),
            ("losing", "lose"),
            // pay
            ("pays", "pay"),
            ("paid", "pay"),
            // meet
            ("meets", "meet"),
            ("met", "meet"),
            // set
            ("sets", "set"),
            ("setting", "set"),
            // learn
            ("learns", "learn"),
            ("learnt", "learn"),
            ("learned", "learn"),
            // lead
            ("leads", "lead"),
            ("led", "lead"),
            ("leading", "lead"),
            // grow
            ("grows", "grow"),
            ("grew", "grow"),
            ("grown", "grow"),
            // build
            ("builds", "build"),
            ("built", "build"),
            ("building", "build"),
            // send
            ("sends", "send"),
            ("sent", "send"),
            // fall
            ("falls", "fall"),
            ("fell", "fall"),
            ("fallen", "fall"),
            // buy
            ("buys", "buy"),
            ("bought", "buy"),
            // feel
            ("feels", "feel"),
            ("felt", "feel"),
            // speak
            ("speaks", "speak"),
            ("spoke", "speak"),
            ("spoken", "speak"),
            // read
            ("reads", "read"),
            // spend
            ("spends", "spend"),
            ("spent", "spend"),
            // win
            ("wins", "win"),
            ("won", "win"),
            // teach
            ("teaches", "teach"),
            ("taught", "teach"),
            // eat
            ("eats", "eat"),
            ("ate", "eat"),
            ("eaten", "eat"),
            // drink
            ("drinks", "drink"),
            ("drank", "drink"),
            ("drunk", "drink"),
            // drive
            ("drives", "drive"),
            ("drove", "drive"),
            ("driven", "drive"),
            // break
            ("breaks", "break"),
            ("broke", "break"),
            ("broken", "break"),
            // catch
            ("catches", "catch"),
            ("caught", "catch"),
            // draw
            ("draws", "draw"),
            ("drew", "draw"),
            ("drawn", "draw"),
            // fly
            ("flies", "fly"),
            ("flew", "fly"),
            ("flown", "fly"),
            // mean
            ("means", "mean"),
            ("meant", "mean"),
            // contain
            ("contains", "contain"),
            ("contained", "contain"),
            // include
            ("includes", "include"),
            ("included", "include"),
            // require
            ("requires", "require"),
            ("required", "require"),
            // provide
            ("provides", "provide"),
            ("provided", "provide"),
            // create
            ("creates", "create"),
            ("created", "create"),
            ("creating", "create"),
            // cause
            ("causes", "cause"),
            ("caused", "cause"),
            // support
            ("supports", "support"),
            ("supported", "support"),
            // produce
            ("produces", "produce"),
            ("produced", "produce"),
            // use
            ("uses", "use"),
            ("used", "use"),
            ("using", "use"),
            // need
            ("needs", "need"),
            ("needed", "need"),
            // want
            ("wants", "want"),
            ("wanted", "want"),
            // call
            ("calls", "call"),
            ("called", "call"),
            // try
            ("tries", "try"),
            ("tried", "try"),
            // ask
            ("asks", "ask"),
            ("asked", "ask"),
            // work
            ("works", "work"),
            ("worked", "work"),
            // play
            ("plays", "play"),
            ("played", "play"),
            // kill
            ("kills", "kill"),
            ("killed", "kill"),
            // turn
            ("turns", "turn"),
            ("turned", "turn"),
            // help
            ("helps", "help"),
            ("helped", "help"),
            // start
            ("starts", "start"),
            ("started", "start"),
            // follow
            ("follows", "follow"),
            ("followed", "follow"),
            // stop
            ("stops", "stop"),
            ("stopped", "stop"),
            // open
            ("opens", "open"),
            ("opened", "open"),
            // close
            ("closes", "close"),
            ("closed", "close"),
            // carry
            ("carries", "carry"),
            ("carried", "carry"),
            // offer
            ("offers", "offer"),
            ("offered", "offer"),
            // remember
            ("remembers", "remember"),
            ("remembered", "remember"),
            // love
            ("loves", "love"),
            ("loved", "love"),
            // consider
            ("considers", "consider"),
            ("considered", "consider"),
            // appear
            ("appears", "appear"),
            ("appeared", "appear"),
            // allow
            ("allows", "allow"),
            ("allowed", "allow"),
            // serve
            ("serves", "serve"),
            ("served", "serve"),
            // expect
            ("expects", "expect"),
            ("expected", "expect"),
            // remain
            ("remains", "remain"),
            ("remained", "remain"),
            // suggest
            ("suggests", "suggest"),
            ("suggested", "suggest"),
            // raise
            ("raises", "raise"),
            ("raised", "raise"),
            // develop
            ("develops", "develop"),
            ("developed", "develop"),
            // describe
            ("describes", "describe"),
            ("described", "describe"),
            // own
            ("owns", "own"),
            ("owned", "own"),
            // define
            ("defines", "define"),
            ("defined", "define"),
            // connect
            ("connects", "connect"),
            ("connected", "connect"),
            // represent
            ("represents", "represent"),
            ("represented", "represent"),
            // involve
            ("involves", "involve"),
            ("involved", "involve"),
            // belong
            ("belongs", "belong"),
            ("belonged", "belong"),
        ]
        .into_iter()
        .collect();

        let auxiliaries: FxHashSet<&'static str> = [
            "is", "are", "was", "were", "am", "has", "have", "had", "been", "being", "gets", "got",
            "gotten",
        ]
        .into_iter()
        .collect();

        let determiners: FxHashSet<&'static str> = [
            "the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her", "its",
            "our", "their", "some", "any", "each", "every", "no",
        ]
        .into_iter()
        .collect();

        let coordinating_conj: FxHashSet<&'static str> =
            ["and", "or", "but", "nor"].into_iter().collect();

        let adverbs: FxHashSet<&'static str> = [
            "very",
            "really",
            "quite",
            "always",
            "never",
            "often",
            "sometimes",
            "usually",
            "also",
            "just",
            "still",
            "already",
            "only",
            "even",
            "quickly",
            "slowly",
            "carefully",
            "easily",
            "strongly",
            "highly",
            "deeply",
            "widely",
            "nearly",
            "recently",
            "currently",
            "generally",
            "simply",
            "actually",
            "probably",
            "certainly",
            "clearly",
            "directly",
            "exactly",
            "finally",
            "however",
            "perhaps",
            "possibly",
            "rapidly",
            "suddenly",
            "truly",
            "mostly",
        ]
        .into_iter()
        .collect();

        Self {
            prepositions,
            irregular_verbs,
            auxiliaries,
            determiners,
            coordinating_conj,
            adverbs,
        }
    }

    pub fn decompose(&self, text: &str) -> Vec<MeaningUnit> {
        let mut results = Vec::new();
        for sentence in text.split(['.', '!', '?', ';']) {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }
            // Split conjunctions into sub-clauses
            let clauses = self.split_conjunctions(sentence);
            for clause in &clauses {
                let tokens = self.tokenize(clause);
                if tokens.len() < 2 {
                    continue;
                }
                results.extend(self.extract_all(&tokens));
            }
        }
        results
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|w| {
                w.trim_matches(|c: char| c.is_ascii_punctuation() && c != '-' && c != '\'')
                    .to_string()
            })
            .filter(|w| !w.is_empty())
            .collect()
    }

    fn split_conjunctions(&self, sentence: &str) -> Vec<String> {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        if words.len() < 5 {
            return vec![sentence.to_string()];
        }

        // Pattern: "A and B verb ..." or "A, B, and C verb ..."
        // Look for coordinating conjunctions that split subjects
        for (i, &word) in words.iter().enumerate() {
            let lower = word
                .trim_matches(|c: char| c.is_ascii_punctuation())
                .to_lowercase();
            if !self.coordinating_conj.contains(lower.as_str()) || i == 0 || i >= words.len() - 1 {
                continue;
            }

            // Check if there's a shared verb after the conjunction subject
            // Pattern: "NP1 and NP2 verb ..." → "NP1 verb ..." + "NP2 verb ..."
            let after_conj = &words[i + 1..];
            if let Some(verb_offset) = self.find_verb_position(after_conj) {
                if verb_offset > 0 {
                    let verb_and_rest: String = after_conj[verb_offset..].join(" ");
                    let subj1: String = words[..i].join(" ");
                    let subj2: String = after_conj[..verb_offset].join(" ");
                    if !subj1.is_empty() && !subj2.is_empty() {
                        return vec![
                            format!("{} {}", subj1, verb_and_rest),
                            format!("{} {}", subj2, verb_and_rest),
                        ];
                    }
                }
            }
        }

        vec![sentence.to_string()]
    }

    fn find_verb_position(&self, words: &[&str]) -> Option<usize> {
        // First pass: prefer known irregular/auxiliary verbs (high confidence)
        for (i, &word) in words.iter().enumerate() {
            let lower = word
                .trim_matches(|c: char| c.is_ascii_punctuation())
                .to_lowercase();
            if self.irregular_verbs.contains_key(lower.as_str()) {
                return Some(i);
            }
        }
        // Second pass: morphological guess (may false-positive on noun plurals)
        for (i, &word) in words.iter().enumerate() {
            let lower = word
                .trim_matches(|c: char| c.is_ascii_punctuation())
                .to_lowercase();
            if self.lemmatize_verb(&lower).is_some() {
                return Some(i);
            }
        }
        None
    }

    fn extract_all(&self, tokens: &[String]) -> Vec<MeaningUnit> {
        let mut results = Vec::new();

        // Try passive voice first: "NP was/were VERBed by NP"
        if let Some(unit) = self.try_passive(tokens) {
            results.push(unit);
            return results;
        }

        // Try copula: "NP is/are NP"
        if let Some(unit) = self.try_copula(tokens) {
            results.push(unit);
        }

        // Try verb+prep and simple verb patterns
        results.extend(self.try_verb_patterns(tokens));

        // Deduplicate: if copula and verb both matched, keep both
        // (they represent different semantic relations)
        results
    }

    fn try_passive(&self, tokens: &[String]) -> Option<MeaningUnit> {
        // Pattern: "NP aux PAST-PARTICIPLE by NP"
        // e.g. "The book was written by John" → John write book
        for (i, token) in tokens.iter().enumerate() {
            let lower = token.to_lowercase();
            if !self.auxiliaries.contains(lower.as_str()) || i == 0 {
                continue;
            }

            // Look for past participle after auxiliary (possibly with adverb between)
            let mut pp_idx = i + 1;
            while pp_idx < tokens.len()
                && self
                    .adverbs
                    .contains(tokens[pp_idx].to_lowercase().as_str())
            {
                pp_idx += 1;
            }
            if pp_idx >= tokens.len() {
                continue;
            }

            let pp_lower = tokens[pp_idx].to_lowercase();
            let lemma = self.lemmatize_verb(&pp_lower)?;

            // Look for "by" after the participle
            let by_idx =
                (pp_idx + 1..tokens.len()).find(|&j| tokens[j].eq_ignore_ascii_case("by"))?;

            if by_idx + 1 >= tokens.len() {
                continue;
            }

            let patient = self.extract_np(&tokens[..i]);
            let agent = self.extract_np(&tokens[by_idx + 1..]);

            if !patient.is_empty() && !agent.is_empty() {
                return Some(MeaningUnit {
                    subject: agent,
                    relation: lemma,
                    object: patient,
                });
            }
        }
        None
    }

    fn try_copula(&self, tokens: &[String]) -> Option<MeaningUnit> {
        // Pattern: "NP is/are/was/were NP"
        let copulas = ["is", "are", "was", "were", "am"];
        for (i, token) in tokens.iter().enumerate() {
            let lower = token.to_lowercase();
            if !copulas.contains(&lower.as_str()) || i == 0 || i + 1 >= tokens.len() {
                continue;
            }
            // Skip if next word looks like a past participle (passive pattern)
            if i + 1 < tokens.len() {
                let next_lower = tokens[i + 1].to_lowercase();
                if self.lemmatize_verb(&next_lower).is_some()
                    && (next_lower.ends_with("ed")
                        || next_lower.ends_with("en")
                        || self.is_irregular_participle(&next_lower))
                {
                    continue;
                }
            }
            let subject = self.extract_np(&tokens[..i]);
            let object = self.extract_np(&tokens[i + 1..]);
            if !subject.is_empty() && !object.is_empty() {
                return Some(MeaningUnit {
                    subject,
                    relation: "is_a".to_string(),
                    object,
                });
            }
        }
        None
    }

    fn try_verb_patterns(&self, tokens: &[String]) -> Vec<MeaningUnit> {
        let mut results = Vec::new();

        for (i, token) in tokens.iter().enumerate() {
            if i == 0 {
                continue;
            }
            let lower = token.to_lowercase();
            let lemma = match self.lemmatize_verb(&lower) {
                Some(l) => l,
                None => continue,
            };

            // Skip copulas (handled separately)
            if lemma == "be" {
                continue;
            }

            // Pattern: NP verb prep NP → verb_prep
            if i + 2 < tokens.len() {
                let next_lower = tokens[i + 1].to_lowercase();
                if self.prepositions.contains(next_lower.as_str()) {
                    let subject = self.extract_np(&tokens[..i]);
                    let object = self.extract_np(&tokens[i + 2..]);
                    if !subject.is_empty() && !object.is_empty() {
                        results.push(MeaningUnit {
                            subject,
                            relation: format!("{}_{}", lemma, next_lower),
                            object,
                        });
                        break; // first verb match wins for this clause
                    }
                }
            }

            // Pattern: NP verb NP
            if i + 1 < tokens.len() {
                let subject = self.extract_np(&tokens[..i]);
                let object = self.extract_np(&tokens[i + 1..]);
                if !subject.is_empty() && !object.is_empty() {
                    results.push(MeaningUnit {
                        subject,
                        relation: lemma,
                        object,
                    });
                    break;
                }
            }
        }

        results
    }

    fn lemmatize_verb(&self, word: &str) -> Option<String> {
        // Check irregular verbs first
        if let Some(&lemma) = self.irregular_verbs.get(word) {
            return Some(lemma.to_string());
        }

        // Check if the word itself is a base-form irregular
        if self.irregular_verbs.values().any(|&v| v == word) {
            return Some(word.to_string());
        }

        // Regular morphology: try to stem and validate
        if word.len() < 3 {
            return None;
        }

        // -ing forms: running→run, making→make, creating→create
        if word.ends_with("ing") && word.len() > 4 {
            let stem = &word[..word.len() - 3];
            // Double consonant: running → run
            if stem.len() >= 2 && stem.as_bytes()[stem.len() - 1] == stem.as_bytes()[stem.len() - 2]
            {
                return Some(stem[..stem.len() - 1].to_string());
            }
            // -ting where base ends in -te: creating → create
            if stem.ends_with('t') {
                let with_e = format!("{}e", stem);
                return Some(with_e);
            }
            // -king, -ling, etc: walking → walk
            return Some(stem.to_string());
        }

        // -ed forms: created→create, walked→walk, stopped→stop
        if word.ends_with("ed") && word.len() > 3 {
            let stem = &word[..word.len() - 2];
            // -ied: carried → carry
            if word.ends_with("ied") && word.len() > 4 {
                return Some(format!("{}y", &word[..word.len() - 3]));
            }
            // Double consonant: stopped → stop
            if stem.len() >= 2 && stem.as_bytes()[stem.len() - 1] == stem.as_bytes()[stem.len() - 2]
            {
                return Some(stem[..stem.len() - 1].to_string());
            }
            // -ated, -eted, etc: try adding 'e' back
            if stem.ends_with('t') || stem.ends_with('d') || stem.ends_with('k') {
                return Some(stem.to_string());
            }
            // Default: try with -e (created → create)
            return Some(format!("{}e", stem));
        }

        // -es forms: teaches→teach, goes→go, creates→create
        if word.ends_with("es") && word.len() > 3 {
            // -ies: carries → carry
            if let Some(stem) = word.strip_suffix("ies") {
                return Some(format!("{}y", stem));
            }
            // -ches, -shes, -sses, -xes, -zes
            if word.ends_with("ches")
                || word.ends_with("shes")
                || word.ends_with("sses")
                || word.ends_with("xes")
                || word.ends_with("zes")
            {
                return Some(word[..word.len() - 2].to_string());
            }
            // -tes, -des, etc: creates → create
            return Some(word[..word.len() - 1].to_string());
        }

        // -s forms: runs→run, plays→play
        if word.ends_with('s') && !word.ends_with("ss") && word.len() > 2 {
            return Some(word[..word.len() - 1].to_string());
        }

        None
    }

    fn is_irregular_participle(&self, word: &str) -> bool {
        matches!(
            word,
            "been"
                | "done"
                | "gone"
                | "seen"
                | "known"
                | "given"
                | "taken"
                | "made"
                | "found"
                | "told"
                | "left"
                | "kept"
                | "begun"
                | "shown"
                | "heard"
                | "held"
                | "brought"
                | "written"
                | "sat"
                | "stood"
                | "lost"
                | "met"
                | "led"
                | "grown"
                | "built"
                | "sent"
                | "fallen"
                | "bought"
                | "felt"
                | "spoken"
                | "spent"
                | "won"
                | "taught"
                | "eaten"
                | "drunk"
                | "driven"
                | "broken"
                | "caught"
                | "drawn"
                | "flown"
                | "meant"
                | "thought"
                | "paid"
                | "run"
                | "put"
                | "set"
                | "read"
        )
    }

    fn extract_np(&self, tokens: &[String]) -> String {
        let filtered: Vec<&str> = tokens
            .iter()
            .map(|t| t.as_str())
            .filter(|w| {
                let lower = w.to_lowercase();
                !self.determiners.contains(lower.as_str()) && !self.adverbs.contains(lower.as_str())
            })
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
        assert_eq!(units[0].relation, "go_to");
        assert_eq!(units[0].subject, "cat");
        assert_eq!(units[0].object, "park");
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

    #[test]
    fn test_passive_voice() {
        let d = Decomposer::new();
        let units = d.decompose("The book was written by John");
        assert!(!units.is_empty(), "Should extract passive voice");
        assert_eq!(units[0].subject, "John");
        assert_eq!(units[0].relation, "write");
        assert_eq!(units[0].object, "book");
    }

    #[test]
    fn test_conjunction_splitting() {
        let d = Decomposer::new();
        let units = d.decompose("Dogs and cats are animals");
        assert_eq!(units.len(), 2, "Should split conjunction into 2 triples");
        assert!(units.iter().any(|u| u.subject == "Dogs"));
        assert!(units.iter().any(|u| u.subject == "cats"));
    }

    #[test]
    fn test_regular_verb_morphology() {
        let d = Decomposer::new();
        // -ed form
        let units = d.decompose("Alice created the project");
        assert!(!units.is_empty());
        assert_eq!(units[0].relation, "create");
        // -s form
        let units = d.decompose("John runs the company");
        assert!(!units.is_empty());
        assert_eq!(units[0].relation, "run");
    }

    #[test]
    fn test_verb_with_preposition() {
        let d = Decomposer::new();
        let units = d.decompose("Birds fly over the ocean");
        assert!(!units.is_empty());
        assert_eq!(units[0].relation, "fly_over");
    }

    #[test]
    fn test_punctuation_handling() {
        let d = Decomposer::new();
        let units = d.decompose("Paris is a city, indeed!");
        assert!(!units.is_empty());
        assert_eq!(units[0].subject, "Paris");
    }

    #[test]
    fn test_semicolon_sentence_split() {
        let d = Decomposer::new();
        let units = d.decompose("Paris is a city; Berlin is a city");
        assert_eq!(units.len(), 2);
    }

    #[test]
    fn test_two_word_extraction() {
        let d = Decomposer::new();
        // With morphological verb detection, "Dogs bark" should work
        // "bark" → "bark" (removes trailing "s" → "bar" which isn't right)
        // Actually "bark" doesn't end in s/ed/ing/es so it won't be recognized
        // as a verb by morphological rules alone. This is expected behavior
        // for the rule-based approach.
        let units = d.decompose("Dogs bark");
        // Two words with unrecognized verb → empty
        assert!(units.is_empty());
    }
}

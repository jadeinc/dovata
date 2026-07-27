//! Over-context routing helpers (pure, platform-independent).
//!
//! When a request's prompt exceeds the model's native context window, the daemon
//! answers it with the blended KV-composition engine instead of failing. That
//! engine needs two things the raw prompt doesn't separate: a **corpus** of
//! retrievable text blocks, and a **question** to retrieve against. This module
//! derives both from a protocol prompt string, with no tokenizer dependency (the
//! engine tokenizes the blocks), so it builds and unit-tests on any platform.
//!
//! Split policy:
//! * **messages envelope** (`_ovata_infer_protocol: "messages"`): the question is the LAST
//!   `user` turn; the corpus is the system prompt + every other turn's content,
//!   concatenated and chunked. This is the realistic shape — a large context (system
//!   / prior turns / pasted document) followed by the user's actual ask.
//! * **plain string**: the corpus is the whole string chunked; the question is the
//!   trailing paragraph (a question typically trails the document it's about).

use crate::prompt::extract_envelope;

/// Default chunk size in characters (~512 tokens at ~4 chars/token). The block size
/// is the main retrieval-recall knob; the daemon passes this to [`derive_rlm_input`].
pub const DEFAULT_CHUNK_CHARS: usize = 2048;

/// The corpus + question extracted from an over-context prompt.
#[derive(Debug, PartialEq)]
pub struct RlmInput {
    /// Retrievable context blocks (already chunked; fed to `BlockStore::from_texts`).
    pub corpus: Vec<String>,
    /// The question the engine retrieves against and answers.
    pub question: String,
}

/// Derive `(corpus blocks, question)` from a protocol prompt, chunking the corpus to
/// ~`target_chars` per block. See the module docs for the messages vs plain-string split.
pub fn derive_rlm_input(prompt_str: &str, target_chars: usize) -> RlmInput {
    match extract_envelope(prompt_str) {
        Some((system, messages)) => {
            // Question = the last `user` turn (search from the end). Everything else
            // — system prompt + all other turns — becomes the retrievable corpus.
            let q_idx = messages.iter().rposition(|m| m.role == "user");
            let question = q_idx
                .map(|i| messages[i].content.clone())
                .unwrap_or_default();

            let mut parts: Vec<&str> = Vec::new();
            if !system.is_empty() {
                parts.push(system.as_str());
            }
            for (i, m) in messages.iter().enumerate() {
                // skip the turn we chose as the question; keep all other content
                if Some(i) != q_idx && !m.content.is_empty() {
                    parts.push(m.content.as_str());
                }
            }
            let corpus_text = parts.join("\n\n");
            RlmInput { corpus: chunk_text(&corpus_text, target_chars), question }
        }
        None => {
            // Plain string: the trailing paragraph is the question; the whole string
            // (question included — harmless if also retrieved) is the corpus.
            let question = trailing_paragraph(prompt_str);
            RlmInput { corpus: chunk_text(prompt_str, target_chars), question }
        }
    }
}

/// The last non-empty paragraph of `text` (segments split on blank lines), trimmed.
/// Falls back to the last non-empty line, then the whole trimmed string.
fn trailing_paragraph(text: &str) -> String {
    // `Split<&str>` is not double-ended (substring search isn't reversible), so take
    // the last non-empty segment by filtering forward rather than `rfind`.
    if let Some(p) = text.split("\n\n").map(str::trim).filter(|s| !s.is_empty()).last() {
        return p.to_owned();
    }
    text.lines()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .last()
        .unwrap_or(text.trim())
        .to_owned()
}

/// Split `text` into chunks of at most ~`target_chars`, preferring natural boundaries.
///
/// Strategy: pack whole paragraphs (split on blank lines) greedily up to the budget;
/// a paragraph larger than the budget is itself split on sentence boundaries, and a
/// sentence still too large is hard-split on character boundaries (UTF-8 safe). An
/// empty/whitespace input yields no blocks.
pub fn chunk_text(text: &str, target_chars: usize) -> Vec<String> {
    let budget = target_chars.max(1);
    let mut chunks: Vec<String> = Vec::new();
    let mut cur = String::new();

    // push the accumulator as a finished chunk (if non-empty) and reset it.
    let flush = |cur: &mut String, chunks: &mut Vec<String>| {
        let t = cur.trim();
        if !t.is_empty() {
            chunks.push(t.to_owned());
        }
        cur.clear();
    };

    for para in text.split("\n\n") {
        let para = para.trim();
        if para.is_empty() {
            continue;
        }
        // a paragraph too big to ever fit is broken into sub-pieces first.
        let pieces = if para.len() > budget {
            split_oversized(para, budget)
        } else {
            vec![para.to_owned()]
        };
        for piece in pieces {
            // would appending overflow the current chunk? if so, flush first.
            if !cur.is_empty() && cur.len() + 2 + piece.len() > budget {
                flush(&mut cur, &mut chunks);
            }
            if !cur.is_empty() {
                cur.push_str("\n\n");
            }
            cur.push_str(&piece);
        }
    }
    flush(&mut cur, &mut chunks);
    chunks
}

/// Break a single oversized paragraph into ≤`budget` pieces: pack sentences first,
/// then hard-split any sentence that alone exceeds the budget on char boundaries.
fn split_oversized(para: &str, budget: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for sentence in split_sentences(para) {
        let units = if sentence.len() > budget {
            hard_split(&sentence, budget)
        } else {
            vec![sentence]
        };
        for u in units {
            if !cur.is_empty() && cur.len() + 1 + u.len() > budget {
                out.push(std::mem::take(&mut cur));
            }
            if !cur.is_empty() {
                cur.push(' ');
            }
            cur.push_str(&u);
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Split text into sentences on ". " / "! " / "? " boundaries, keeping the terminator.
fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let c = bytes[i];
        if (c == b'.' || c == b'!' || c == b'?') && i + 1 < bytes.len() && bytes[i + 1] == b' ' {
            out.push(text[start..=i].trim().to_owned());
            start = i + 1;
        }
        i += 1;
    }
    if start < text.len() {
        let tail = text[start..].trim();
        if !tail.is_empty() {
            out.push(tail.to_owned());
        }
    }
    out
}

/// Hard-split `s` into ≤`budget`-byte pieces on UTF-8 character boundaries.
fn hard_split(s: &str, budget: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in s.chars() {
        if cur.len() + ch.len_utf8() > budget && !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
        }
        cur.push(ch);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Built through the shared builder, not hand-written JSON — see the note on
    /// the matching helper in `prompt.rs`.
    fn proto(system: &str, messages: &[(&str, &str)]) -> String {
        let msgs: Vec<ovata_infer_protocol::Message> = messages
            .iter()
            .map(|(r, c)| ovata_infer_protocol::Message {
                role: (*r).to_owned(),
                content: (*c).to_owned(),
            })
            .collect();
        ovata_infer_protocol::envelope::build(system, &msgs).to_string()
    }

    #[test]
    fn messages_last_user_is_question_rest_is_corpus() {
        let json = proto(
            "You are a doc assistant.",
            &[
                ("user", "Here is a big document about turbines and pressure valves."),
                ("assistant", "Got it, I have read the document."),
                ("user", "What does the document say about pressure valves?"),
            ],
        );
        let r = derive_rlm_input(&json, 4096);
        assert_eq!(r.question, "What does the document say about pressure valves?");
        // corpus holds the system prompt + the earlier turns, not the question turn.
        let joined = r.corpus.join("\n");
        assert!(joined.contains("doc assistant"));
        assert!(joined.contains("turbines and pressure valves"));
        assert!(joined.contains("I have read the document"));
        assert!(!joined.contains("What does the document say"));
    }

    #[test]
    fn messages_with_no_user_turn_yields_empty_question() {
        let json = proto("sys", &[("assistant", "hello")]);
        let r = derive_rlm_input(&json, 4096);
        assert_eq!(r.question, "");
        assert!(r.corpus.join("\n").contains("hello"));
    }

    #[test]
    fn plain_string_trailing_paragraph_is_question() {
        let prompt = "Big background paragraph one.\n\nMore background two.\n\nWhat is the secret code?";
        let r = derive_rlm_input(prompt, 4096);
        assert_eq!(r.question, "What is the secret code?");
        assert!(r.corpus.join("\n").contains("Big background paragraph one"));
    }

    #[test]
    fn chunk_packs_paragraphs_under_budget() {
        let text = "aaaa\n\nbbbb\n\ncccc"; // three 4-char paras
        // budget 10 → "aaaa\n\nbbbb" (10) is the max that fits; "cccc" spills to a 2nd chunk.
        let chunks = chunk_text(text, 10);
        assert_eq!(chunks, vec!["aaaa\n\nbbbb".to_string(), "cccc".to_string()]);
    }

    #[test]
    fn chunk_splits_oversized_paragraph_on_sentences() {
        let text = "First sentence here. Second sentence here. Third one here.";
        let chunks = chunk_text(text, 25);
        assert!(chunks.len() >= 2, "oversized paragraph must split: {chunks:?}");
        // every chunk respects the budget
        assert!(chunks.iter().all(|c| c.len() <= 25), "chunk over budget: {chunks:?}");
        // content is preserved (no dropped words)
        let recombined: String = chunks.join(" ");
        assert!(recombined.contains("First sentence"));
        assert!(recombined.contains("Third one here"));
    }

    #[test]
    fn chunk_hard_splits_a_single_huge_token() {
        let text = "x".repeat(100); // one 100-char "word", no boundaries
        let chunks = chunk_text(&text, 30);
        assert!(chunks.iter().all(|c| c.len() <= 30));
        assert_eq!(chunks.iter().map(|c| c.len()).sum::<usize>(), 100);
    }

    #[test]
    fn chunk_empty_input_yields_nothing() {
        assert!(chunk_text("   \n\n  ", 100).is_empty());
        assert!(chunk_text("", 100).is_empty());
    }

    #[test]
    fn hard_split_respects_utf8_boundaries() {
        // multibyte chars must not be cut mid-codepoint
        let s = "héllo wörld ünïcödé";
        let pieces = hard_split(s, 5);
        assert!(pieces.iter().all(|p| p.len() <= 5));
        assert_eq!(pieces.concat(), s);
    }
}

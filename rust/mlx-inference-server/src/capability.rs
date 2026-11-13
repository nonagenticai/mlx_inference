//! What the engine can be asked to do.
//!
//! One enum, used in three places that must never disagree: the checkpoint flags
//! the sidecar is spawned with ([`crate::worker::mlxcel::MlxcelConfig`]), the
//! routes this server mounts ([`crate::http`]), and what `/health` reports.
//!
//! A capability is *enabled* by configuration, not by the engine: the same
//! binary serves chat, embeddings, reranking and audio, but a 16 GiB device only
//! holds a subset of the checkpoints, so each host names the ones it serves.

use std::fmt;

/// One engine capability.
///
/// The variants are the four inference surfaces the engine exposes besides chat;
/// `Chat` stays here so that a device serving no chat model (a reranker-only or
/// TTS-only host) can say so explicitly rather than by omission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Capability {
    Chat,
    Embedding,
    Rerank,
    Speech,
    Transcription,
}

impl Capability {
    /// Every capability, in the order `/health` reports them.
    pub const ALL: [Capability; 5] = [
        Capability::Chat,
        Capability::Embedding,
        Capability::Rerank,
        Capability::Speech,
        Capability::Transcription,
    ];

    /// The `MLXCEL_CAPABILITIES` spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Capability::Chat => "chat",
            Capability::Embedding => "embedding",
            Capability::Rerank => "rerank",
            Capability::Speech => "speech",
            Capability::Transcription => "transcription",
        }
    }

    /// The engine path this capability is forwarded to.
    ///
    /// These are the engine's own paths, not this server's: the forwarding layer
    /// deliberately does not re-model the wire contract, so a route that is
    /// spelled `/v1/rerank` here is `/v1/rerank` there too.
    pub fn engine_path(self) -> &'static str {
        match self {
            Capability::Chat => "/v1/chat/completions",
            Capability::Embedding => "/v1/embeddings",
            Capability::Rerank => "/v1/rerank",
            Capability::Speech => "/v1/audio/speech",
            Capability::Transcription => "/v1/audio/transcriptions",
        }
    }

    /// Parse one `MLXCEL_CAPABILITIES` entry. Case-insensitive.
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "chat" => Some(Capability::Chat),
            "embedding" | "embeddings" => Some(Capability::Embedding),
            "rerank" | "reranking" => Some(Capability::Rerank),
            "speech" | "tts" => Some(Capability::Speech),
            "transcription" | "transcribe" | "asr" => Some(Capability::Transcription),
            _ => None,
        }
    }

    /// True for the JSON-in/JSON-out capabilities whose `model` field this
    /// server owns (see `MlxcelWorker::forward`).
    pub fn takes_json_model(self) -> bool {
        matches!(self, Capability::Embedding | Capability::Rerank)
    }
}

impl fmt::Display for Capability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip() {
        for cap in Capability::ALL {
            assert_eq!(Capability::parse(cap.as_str()), Some(cap));
        }
        assert_eq!(
            Capability::parse(" Embeddings "),
            Some(Capability::Embedding)
        );
        assert_eq!(Capability::parse("tts"), Some(Capability::Speech));
        assert_eq!(Capability::parse("vision"), None);
    }

    #[test]
    fn engine_paths_are_the_engine_spelling_and_never_versioned_twice() {
        assert_eq!(Capability::Rerank.engine_path(), "/v1/rerank");
        for cap in Capability::ALL {
            assert!(
                !cap.engine_path().starts_with("/v2"),
                "{} must not introduce a second version namespace",
                cap
            );
        }
    }
}

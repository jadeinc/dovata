//! Daemon health report.
//!
//! Returned in response to an `InferenceRequest` with `health_only = true`: the
//! daemon sends one [`Frame::Json`](crate::Frame::Json) carrying this struct as
//! JSON, then a `Done { tokens_used: 0 }`. A cheap liveness/readiness probe that
//! never touches the model.
//!
//! Every field is `#[serde(default)]` so the schema can grow without breaking old
//! clients (a missing field decodes to its default) and so a client built against a
//! newer schema can still read an older daemon's report.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Health {
    /// Coarse readiness, e.g. `"ok"` or `"starting"`.
    #[serde(default)]
    pub status: String,
    /// The serving model's name (matches the `Meta` frame).
    #[serde(default)]
    pub model: String,
    /// Whether the model weights are loaded and ready to generate.
    #[serde(default)]
    pub model_loaded: bool,
    /// Seconds since the daemon started serving.
    #[serde(default)]
    pub uptime_secs: u64,
    /// Wire-protocol revision the daemon speaks ([`crate::PROTOCOL_VERSION`]).
    #[serde(default)]
    pub protocol_version: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_through_json() {
        let h = Health {
            status: "ok".into(),
            model: "Qwen3-Coder-30B".into(),
            model_loaded: true,
            uptime_secs: 42,
            protocol_version: crate::PROTOCOL_VERSION,
        };
        let s = serde_json::to_string(&h).unwrap();
        let back: Health = serde_json::from_str(&s).unwrap();
        assert_eq!(back.model, "Qwen3-Coder-30B");
        assert!(back.model_loaded);
        assert_eq!(back.uptime_secs, 42);
        assert_eq!(back.protocol_version, crate::PROTOCOL_VERSION);
    }

    #[test]
    fn missing_fields_default() {
        // An older/sparser daemon emits only some fields.
        let h: Health = serde_json::from_str(r#"{"status":"ok","model":"m"}"#).unwrap();
        assert_eq!(h.status, "ok");
        assert_eq!(h.model, "m");
        assert!(!h.model_loaded);
        assert_eq!(h.uptime_secs, 0);
        assert_eq!(h.protocol_version, 0);
    }

    #[test]
    fn empty_object_is_all_defaults() {
        let h: Health = serde_json::from_str("{}").unwrap();
        assert_eq!(h.status, "");
        assert!(!h.model_loaded);
    }
}

//! Persistent paper-trading qualification for live execution.
//!
//! Paper mode runs every read-only freshness, balance, economics, repricing,
//! and target-selection gate. Only the external OctoBot mutation is omitted.
//! Qualification belongs to the running build revision, so deploying new code
//! deliberately requires a fresh paper observation window.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PaperQualificationState {
    pub build_revision: String,
    pub started_at: Option<f64>,
    pub last_observation_at: Option<f64>,
    pub qualified_at: Option<f64>,
    /// Lifetime evaluation counter captured when this build is first observed.
    /// Only evaluations completed by the current build count toward qualification.
    pub evaluation_count_at_start: u64,
    pub observed_evaluations: u64,
    pub validated_intents: u64,
    pub duplicate_intents_rejected: u64,
    recent_intents: HashMap<String, f64>,
}

#[derive(Clone, Copy, Debug)]
pub struct PaperQualificationPolicy {
    pub min_evaluations: u64,
    pub min_validated_intents: u64,
    pub validity_seconds: f64,
    pub intent_lease_seconds: f64,
}

impl PaperQualificationState {
    /// Records an intent after every read-only execution gate has succeeded.
    /// Returns `true` for a new paper intent and `false` for a duplicate.
    pub fn observe_intent(
        &mut self,
        revision: &str,
        intent_key: &str,
        evaluation_count: u64,
        now: f64,
        policy: PaperQualificationPolicy,
    ) -> bool {
        self.reset_for_revision(revision, evaluation_count, now);
        if self
            .qualified_at
            .is_some_and(|qualified_at| now - qualified_at > policy.validity_seconds.max(1.0))
        {
            // Expired evidence may be renewed only by completing a fresh paper
            // window; historical counts never roll into the replacement window.
            self.start_window(revision, evaluation_count, now);
        }
        self.observed_evaluations = self
            .observed_evaluations
            .max(evaluation_count.saturating_sub(self.evaluation_count_at_start));
        self.last_observation_at = Some(now);
        self.recent_intents
            .retain(|_, expires_at| *expires_at > now);

        let is_new = !self.recent_intents.contains_key(intent_key);
        if is_new {
            self.validated_intents = self.validated_intents.saturating_add(1);
            self.recent_intents.insert(
                intent_key.to_string(),
                now + policy.intent_lease_seconds.max(1.0),
            );
        } else {
            self.duplicate_intents_rejected = self.duplicate_intents_rejected.saturating_add(1);
        }
        if self.meets_counts(policy) {
            self.qualified_at.get_or_insert(now);
        }
        is_new
    }

    pub fn is_qualified(&self, revision: &str, now: f64, policy: PaperQualificationPolicy) -> bool {
        self.build_revision == revision
            && self.meets_counts(policy)
            && self
                .qualified_at
                .is_some_and(|qualified_at| now - qualified_at <= policy.validity_seconds.max(1.0))
    }

    fn meets_counts(&self, policy: PaperQualificationPolicy) -> bool {
        self.observed_evaluations >= policy.min_evaluations
            && self.validated_intents >= policy.min_validated_intents
    }

    fn reset_for_revision(&mut self, revision: &str, evaluation_count: u64, now: f64) {
        if self.build_revision == revision {
            return;
        }
        self.start_window(revision, evaluation_count, now);
    }

    fn start_window(&mut self, revision: &str, evaluation_count: u64, now: f64) {
        *self = Self {
            build_revision: revision.to_string(),
            started_at: Some(now),
            evaluation_count_at_start: evaluation_count,
            ..Self::default()
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> PaperQualificationPolicy {
        PaperQualificationPolicy {
            min_evaluations: 3,
            min_validated_intents: 2,
            validity_seconds: 100.0,
            intent_lease_seconds: 10.0,
        }
    }

    #[test]
    fn qualification_requires_counts_and_rejects_duplicate_intents() {
        let mut state = PaperQualificationState::default();
        assert!(state.observe_intent("rev-a", "buy:x", 2, 10.0, policy()));
        assert!(!state.observe_intent("rev-a", "buy:x", 5, 11.0, policy()));
        assert_eq!(state.duplicate_intents_rejected, 1);
        assert!(!state.is_qualified("rev-a", 11.0, policy()));

        assert!(state.observe_intent("rev-a", "sell:y", 5, 12.0, policy()));
        assert!(state.is_qualified("rev-a", 12.0, policy()));
        assert!(!state.is_qualified("rev-a", 200.0, policy()));
    }

    #[test]
    fn new_build_revision_requires_fresh_paper_evidence() {
        let mut state = PaperQualificationState::default();
        state.observe_intent("rev-a", "buy:x", 3, 10.0, policy());
        state.observe_intent("rev-a", "sell:y", 6, 11.0, policy());
        assert!(state.is_qualified("rev-a", 11.0, policy()));

        state.observe_intent("rev-b", "buy:z", 10, 12.0, policy());
        assert_eq!(state.build_revision, "rev-b");
        assert_eq!(state.evaluation_count_at_start, 10);
        assert_eq!(state.observed_evaluations, 0);
        assert_eq!(state.validated_intents, 1);
        assert!(!state.is_qualified("rev-b", 12.0, policy()));

        state.observe_intent("rev-b", "sell:z", 13, 13.0, policy());
        assert_eq!(state.observed_evaluations, 3);
        assert!(state.is_qualified("rev-b", 13.0, policy()));
    }

    #[test]
    fn expired_qualification_requires_a_complete_fresh_window() {
        let mut state = PaperQualificationState::default();
        state.observe_intent("rev-a", "buy:x", 10, 10.0, policy());
        state.observe_intent("rev-a", "sell:y", 13, 11.0, policy());
        assert!(state.is_qualified("rev-a", 11.0, policy()));

        state.observe_intent("rev-a", "buy:z", 200, 112.0, policy());
        assert_eq!(state.evaluation_count_at_start, 200);
        assert_eq!(state.observed_evaluations, 0);
        assert_eq!(state.validated_intents, 1);
        assert!(!state.is_qualified("rev-a", 112.0, policy()));

        state.observe_intent("rev-a", "sell:z", 203, 113.0, policy());
        assert!(state.is_qualified("rev-a", 113.0, policy()));
    }
}

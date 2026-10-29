//! Bounded business task ingress over Gail's existing ELM gates and executor.
//! No model, training or qualification authority is delegated to Foresight.
use crate::{
    elm_config::FeatureSchema,
    errors::{GailError, Result},
    orchestration::GailService,
};
use axum::{Json, extract::State, http::HeaderMap};
use serde::Deserialize;
use serde_json::{Value, json};
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericalRequest {
    pub decision_id: String,
    pub task: String,
    pub tenant_scope: String,
    pub feature_schema: FeatureSchema,
    pub feature_values: Vec<f64>,
    pub feature_timestamp: u64,
    pub target_unit: String,
    pub horizon_seconds: u64,
    pub expected_model_digest: String,
}
impl NumericalRequest {
    fn validate(&self) -> Result<()> {
        if self.decision_id.is_empty()
            || self.decision_id.len() > 128
            || self.tenant_scope.is_empty()
            || !matches!(
                (self.task.as_str(), self.target_unit.as_str()),
                ("compute_demand", "cpu_cores") | ("job_duration", "seconds")
            )
            || self.feature_values.is_empty()
            || self.feature_values.len() > 256
            || self.feature_values.len() != self.feature_schema.names.len()
            || self.feature_values.len() != self.feature_schema.units.len()
            || self.feature_schema.version == 0
            || self.feature_schema.id.is_empty()
            || self.feature_values.iter().any(|v| !v.is_finite())
            || self.expected_model_digest.len() != 64
            || self
                .expected_model_digest
                .bytes()
                .any(|b| !b.is_ascii_hexdigit())
            || self.horizon_seconds == 0
            || self.horizon_seconds > 604800
        {
            return Err(GailError::bad_request("invalid numerical task contract"));
        }
        Ok(())
    }
}
pub async fn predict(
    State(service): State<GailService>,
    headers: HeaderMap,
    Json(input): Json<NumericalRequest>,
) -> Result<Json<Value>> {
    service.authorize(&headers, "neuromorphic")?;
    input.validate()?;
    // The namespace's tenant scope is set by deployment, not by a caller.
    let tenant = std::env::var("GAIL_BUSINESS_TENANT").unwrap_or_else(|_| "neuralmimicry".into());
    if input.tenant_scope != tenant {
        return Err(GailError::unauthorized());
    }
    let unavailable = json!({"task_id":input.task,"applied":false,"proposal":null,"uncertainty":null,"reason":"qualified_model_unavailable"});
    #[cfg(feature = "elm")]
    {
        // Confirm output horizon and exact candidate against the authoritative
        // registry before spending an inference slot. elm_advisory repeats all
        // current policy, schema, freshness, OOD and rollout gates.
        if !service.business_model_matches(
            &input.task,
            &input.expected_model_digest,
            input.horizon_seconds,
        ) {
            return Ok(Json(unavailable));
        }
    }
    let Some(advice) = service
        .elm_advisory(
            &input.decision_id,
            &input.task,
            &input.tenant_scope,
            &input.feature_schema,
            &input.feature_values,
            input.feature_timestamp,
            &input.target_unit,
            None,
        )
        .await
    else {
        return Ok(Json(unavailable));
    };
    if advice.model_digest.as_deref() != Some(&input.expected_model_digest) {
        return Ok(Json(unavailable));
    }
    Ok(Json(
        json!({"task_id":advice.task_id,"model_digest":advice.model_digest,
        "proposal":advice.proposal,"uncertainty":advice.uncertainty,"applied":advice.applied,
        "reason":advice.reason,"elapsed_ms":advice.elapsed_ms,"execution_authority":false}),
    ))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_typed_business_tasks_are_accepted() {
        let mut r = NumericalRequest {
            decision_id: "p1".into(),
            task: "compute_demand".into(),
            tenant_scope: "t".into(),
            feature_schema: FeatureSchema {
                id: "cpu-v1".into(),
                version: 1,
                names: vec!["cpu".into()],
                units: vec!["cpu_cores".into()],
            },
            feature_values: vec![1.0],
            feature_timestamp: 1,
            target_unit: "cpu_cores".into(),
            horizon_seconds: 3600,
            expected_model_digest: "a".repeat(64),
        };
        assert!(r.validate().is_ok());
        r.task = "trading_advisory".into();
        assert!(r.validate().is_err());
        r.task = "compute_demand".into();
        r.target_unit = "USD".into();
        assert!(r.validate().is_err());
    }
}

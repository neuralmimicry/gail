use std::time::Duration;

use reqwest::Client;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::{
    errors::{GailError, Result},
    models::{
        AerDecodeRequest, AerDecodeResponse, AerEncodeRequest, AerEncodeResponse,
        CompletionRequest, CompletionResponse, NeuromorphicAnalyzeRequest,
        NeuromorphicPredictRequest, NeuromorphicPredictResponse, ProviderCompletionRequest,
        SpecialistAnalysisResponse, TranscriptionResponse,
    },
};

#[derive(Clone, Debug)]
pub struct GailClient {
    client: Client,
    base_url: String,
    bearer_token: Option<String>,
}

impl GailClient {
    pub fn new(base_url: impl Into<String>) -> Result<Self> {
        let client = Client::builder()
            .use_rustls_tls()
            .timeout(Duration::from_secs(60))
            .build()?;
        Ok(Self {
            client,
            base_url: normalize_base_url(base_url.into())?,
            bearer_token: None,
        })
    }

    pub fn with_bearer_token(mut self, token: impl Into<String>) -> Self {
        self.bearer_token = Some(token.into());
        self
    }

    pub async fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse> {
        self.post_json("/v1/llm/complete", request).await
    }

    pub async fn direct_complete(
        &self,
        request: &ProviderCompletionRequest,
    ) -> Result<CompletionResponse> {
        self.post_json("/v1/llm/direct-complete", request).await
    }

    pub async fn analyze_neuromorphic(
        &self,
        request: &NeuromorphicAnalyzeRequest,
    ) -> Result<SpecialistAnalysisResponse> {
        self.post_json("/v1/neuromorphic/analyze", request).await
    }

    pub async fn predict_neuromorphic(
        &self,
        request: &NeuromorphicPredictRequest,
    ) -> Result<NeuromorphicPredictResponse> {
        self.post_json("/v1/neuromorphic/predict", request).await
    }

    pub async fn encode_aer(&self, request: &AerEncodeRequest) -> Result<AerEncodeResponse> {
        self.post_json("/v1/aer/encode", request).await
    }

    pub async fn decode_aer(&self, request: &AerDecodeRequest) -> Result<AerDecodeResponse> {
        self.post_json("/v1/aer/decode", request).await
    }

    pub async fn orchestration_status(&self, limit: usize) -> Result<Value> {
        self.get_json(&format!("/v1/status/orchestration?limit={limit}")).await
    }

    pub async fn transcribe_bytes(
        &self,
        provider: &str,
        filename: &str,
        mime_type: &str,
        bytes: Vec<u8>,
        model: Option<&str>,
    ) -> Result<TranscriptionResponse> {
        let mut form = reqwest::multipart::Form::new()
            .text("provider", provider.to_string())
            .part(
                "file",
                reqwest::multipart::Part::bytes(bytes)
                    .file_name(filename.to_string())
                    .mime_str(mime_type)
                    .map_err(|error| GailError::Multipart(error.to_string()))?,
            );
        if let Some(model) = model {
            form = form.text("model", model.to_string());
        }
        let mut builder = self
            .client
            .post(format!("{}{path}", self.base_url, path = "/v1/llm/transcribe"))
            .multipart(form);
        if let Some(token) = self.bearer_token.as_deref() {
            builder = builder.bearer_auth(token);
        }
        let response = builder.send().await?;
        self.decode_response(response).await
    }

    async fn post_json<T, R>(&self, path: &str, payload: &T) -> Result<R>
    where
        T: serde::Serialize + ?Sized,
        R: DeserializeOwned,
    {
        let mut builder = self.client.post(format!("{}{}", self.base_url, path)).json(payload);
        if let Some(token) = self.bearer_token.as_deref() {
            builder = builder.bearer_auth(token);
        }
        let response = builder.send().await?;
        self.decode_response(response).await
    }

    async fn get_json<R>(&self, path: &str) -> Result<R>
    where
        R: DeserializeOwned,
    {
        let mut builder = self.client.get(format!("{}{}", self.base_url, path));
        if let Some(token) = self.bearer_token.as_deref() {
            builder = builder.bearer_auth(token);
        }
        let response = builder.send().await?;
        self.decode_response(response).await
    }

    async fn decode_response<R>(&self, response: reqwest::Response) -> Result<R>
    where
        R: DeserializeOwned,
    {
        let status = response.status();
        if status.is_success() {
            return Ok(response.json::<R>().await?);
        }
        let payload = response
            .json::<Value>()
            .await
            .unwrap_or_else(|_| Value::Object(Default::default()));
        let message = payload
            .get("message")
            .and_then(Value::as_str)
            .or_else(|| payload.get("error").and_then(Value::as_str))
            .unwrap_or("Gail request failed");
        Err(GailError::upstream("gail", Some(status), message.to_string()))
    }
}

fn normalize_base_url(base_url: String) -> Result<String> {
    let trimmed = base_url.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return Err(GailError::bad_request("Gail base URL must not be empty"));
    }
    Ok(trimmed.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::{Mock, MockServer, ResponseTemplate, matchers::{method, path}};

    #[tokio::test]
    async fn orchestration_status_fetches_json() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/status/orchestration"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"ok": true})))
            .mount(&server)
            .await;
        let client = GailClient::new(server.uri()).expect("client");
        let status = client.orchestration_status(20).await.expect("status");
        assert_eq!(status.get("ok").and_then(Value::as_bool), Some(true));
    }
}

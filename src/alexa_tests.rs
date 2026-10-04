//! Alexa verification and conversation tests. No network access: signing
//! chains are seeded into the verifier cache and completions are mocked.
use super::*;
use axum::http::HeaderValue;
use std::sync::atomic::{AtomicUsize, Ordering};

const TEST_ROOT: &[u8] = include_bytes!("alexa_testdata/root.pem");
const TEST_CHAIN: &[u8] = include_bytes!("alexa_testdata/chain.pem");
pub(crate) const TEST_LEAF_KEY: &[u8] = include_bytes!("alexa_testdata/leaf.pk8");
const WRONG_SAN_CHAIN: &[u8] = include_bytes!("alexa_testdata/wrongsan-chain.pem");
const WRONG_SAN_KEY: &[u8] = include_bytes!("alexa_testdata/wrongsan.pk8");
const AMAZON_CHAIN_2023: &[u8] = include_bytes!("alexa_testdata/amazon-echo-api-cert-12.pem");

pub(crate) const CHAIN_URL: &str = "https://s3.amazonaws.com/echo.api/echo-api-cert-test.pem";
const WRONG_SAN_URL: &str = "https://s3.amazonaws.com/echo.api/wrong-san.pem";
pub(crate) const SKILL_ID: &str = "amzn1.ask.skill.00000000-0000-0000-0000-000000000000";

fn test_anchors() -> Vec<TrustAnchor<'static>> {
    let root = CertificateDer::from_pem_slice(TEST_ROOT).expect("root pem");
    vec![
        webpki::anchor_from_trusted_cert(&root)
            .expect("anchor")
            .to_owned(),
    ]
}

pub(crate) fn test_verifier() -> AlexaVerifier {
    let verifier = AlexaVerifier::with_trust_anchors(test_anchors(), Duration::from_secs(3600));
    verifier.insert_chain(CHAIN_URL, TEST_CHAIN).unwrap();
    verifier
        .insert_chain(WRONG_SAN_URL, WRONG_SAN_CHAIN)
        .unwrap();
    verifier
}

fn enabled_config() -> AlexaConfig {
    AlexaConfig {
        enabled: true,
        skill_ids: vec![SKILL_ID.to_string()],
        ..AlexaConfig::default()
    }
}

fn sign(key: &[u8], body: &[u8]) -> String {
    let pair = ring::signature::RsaKeyPair::from_pkcs8(key).expect("pkcs8");
    let mut signature = vec![0; pair.public().modulus_len()];
    pair.sign(
        &ring::signature::RSA_PKCS1_SHA256,
        &ring::rand::SystemRandom::new(),
        body,
        &mut signature,
    )
    .expect("sign");
    base64::engine::general_purpose::STANDARD.encode(signature)
}

pub(crate) fn iso_now_offset(offset_secs: i64) -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
        + offset_secs;
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    // civil_from_days (Hinnant)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

pub(crate) fn envelope(request: Value, timestamp: &str, app_id: &str) -> Value {
    let mut request = request;
    request["timestamp"] = json!(timestamp);
    request["requestId"] = json!("amzn1.echo-api.request.test");
    json!({
        "version": "1.0",
        "session": {
            "new": false,
            "sessionId": "amzn1.echo-api.session.test",
            "application": { "applicationId": app_id },
            "attributes": {},
        },
        "context": { "System": { "application": { "applicationId": app_id } } },
        "request": request,
    })
}

pub(crate) fn ask_request(query: &str) -> Value {
    json!({
        "type": "IntentRequest",
        "intent": { "name": "AskAaronIntent", "slots": { "query": { "name": "query", "value": query } } },
    })
}

fn intent(name: &str) -> Value {
    json!({ "type": "IntentRequest", "intent": { "name": name } })
}

pub(crate) fn signed_headers(body: &[u8], url: &str, key: &[u8]) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert("SignatureCertChainUrl", HeaderValue::from_str(url).unwrap());
    headers.insert(
        "Signature-256",
        HeaderValue::from_str(&sign(key, body)).unwrap(),
    );
    headers
}

pub(crate) fn ssml_of(response: &Value) -> &str {
    response["response"]["outputSpeech"]["ssml"]
        .as_str()
        .unwrap_or_default()
}

fn mock_asker(answer: &'static str, delay: Duration, calls: Arc<AtomicUsize>) -> Asker {
    Arc::new(move |messages: Vec<ChatMessage>| {
        calls.fetch_add(1, Ordering::SeqCst);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[0].flattened_text(), VOICE_SYSTEM_PROMPT);
        Box::pin(async move {
            tokio::time::sleep(delay).await;
            Ok(answer.to_string())
        })
    })
}

fn runtime(asker: Asker) -> AlexaRuntime {
    AlexaRuntime::with_verifier(enabled_config(), test_verifier(), asker)
}

// --- certificate URL -------------------------------------------------------

#[test]
fn cert_chain_url_accepts_amazon_examples() {
    for url in [
        "https://s3.amazonaws.com/echo.api/echo-api-cert.pem",
        "https://s3.amazonaws.com:443/echo.api/echo-api-cert.pem",
        "https://s3.amazonaws.com/echo.api/../echo.api/echo-api-cert.pem",
        "HTTPS://S3.AMAZONAWS.COM/echo.api/echo-api-cert.pem",
    ] {
        assert!(
            validate_cert_chain_url(url).is_ok(),
            "{url} should be valid"
        );
    }
}

#[test]
fn cert_chain_url_rejects_amazon_counter_examples() {
    for url in [
        "http://s3.amazonaws.com/echo.api/echo-api-cert.pem",
        "https://notamazon.com/echo.api/echo-api-cert.pem",
        "https://s3.amazonaws.com/EcHo.aPi/echo-api-cert.pem",
        "https://s3.amazonaws.com/invalid.path/echo-api-cert.pem",
        "https://s3.amazonaws.com:563/echo.api/echo-api-cert.pem",
        "https://s3.amazonaws.com/echo.api/../evil/cert.pem",
        "https://user:pw@s3.amazonaws.com/echo.api/echo-api-cert.pem",
        "https://s3.amazonaws.com.evil.com/echo.api/echo-api-cert.pem",
        "not a url",
    ] {
        let error = validate_cert_chain_url(url).expect_err(url);
        assert_eq!(error.status, StatusCode::BAD_REQUEST, "{url}");
    }
}

// --- chain and signature ---------------------------------------------------

#[test]
fn self_signed_test_chain_verifies_and_signature_matches() {
    let certs = parse_pem_chain(TEST_CHAIN).unwrap();
    verify_chain(&certs, &test_anchors(), UnixTime::now()).unwrap();
    let body = br#"{"hello":"alexa"}"#;
    let raw = base64::engine::general_purpose::STANDARD
        .decode(sign(TEST_LEAF_KEY, body))
        .unwrap();
    verify_body_signature(&certs[0], body, SignatureKind::Sha256, &raw).unwrap();
    let tampered = verify_body_signature(&certs[0], b"{}", SignatureKind::Sha256, &raw);
    assert_eq!(tampered.unwrap_err().reason, "signature_mismatch");
}

#[test]
fn legacy_sha1_signature_is_supported_as_fallback() {
    // ring cannot create SHA-1 signatures, so the fixture was produced with
    // `openssl dgst -sha1 -sign` using the test leaf key.
    let certs = parse_pem_chain(TEST_CHAIN).unwrap();
    let body = include_bytes!("alexa_testdata/legacy-body.json");
    let raw = include_bytes!("alexa_testdata/legacy-body.sha1.sig");
    verify_body_signature(&certs[0], body, SignatureKind::LegacySha1, raw).unwrap();
    assert!(verify_body_signature(&certs[0], body, SignatureKind::Sha256, raw).is_err());
    assert!(verify_body_signature(&certs[0], b"other", SignatureKind::LegacySha1, raw).is_err());
}

#[test]
fn signature_256_is_preferred_over_legacy_header() {
    let mut headers = HeaderMap::new();
    headers.insert("Signature", HeaderValue::from_static("AAAA"));
    assert_eq!(
        select_signature(&headers).unwrap().0,
        SignatureKind::LegacySha1
    );
    headers.insert("Signature-256", HeaderValue::from_static("AQID"));
    let (kind, bytes) = select_signature(&headers).unwrap();
    assert_eq!(kind, SignatureKind::Sha256);
    assert_eq!(bytes, vec![1, 2, 3]);
    assert_eq!(
        select_signature(&HeaderMap::new()).unwrap_err().reason,
        "missing_signature"
    );
}

#[test]
fn chain_rejections_untrusted_wrong_san_and_expired() {
    let certs = parse_pem_chain(TEST_CHAIN).unwrap();
    // Not anchored in the public roots.
    let untrusted = verify_chain(&certs, webpki_roots::TLS_SERVER_ROOTS, UnixTime::now());
    assert_eq!(untrusted.unwrap_err().reason, "untrusted_certificate_chain");
    // Valid chain, but the leaf does not name echo-api.amazon.com.
    let wrong = parse_pem_chain(WRONG_SAN_CHAIN).unwrap();
    let mismatch = verify_chain(&wrong, &test_anchors(), UnixTime::now());
    assert_eq!(mismatch.unwrap_err().reason, "certificate_subject_mismatch");
    // Expired: evaluate the 100-year test chain in the 22nd century.
    let future = UnixTime::since_unix_epoch(Duration::from_secs(5_000_000_000));
    let expired = verify_chain(&certs, &test_anchors(), future);
    assert_eq!(expired.unwrap_err().reason, "certificate_expired");
    assert!(parse_pem_chain(b"garbage").is_err());
}

#[test]
fn real_amazon_chain_verifies_against_webpki_roots_when_it_was_valid() {
    let certs = parse_pem_chain(AMAZON_CHAIN_2023).unwrap();
    // 2023-06-01T00:00:00Z, inside the certificate's validity window.
    let then = UnixTime::since_unix_epoch(Duration::from_secs(1_685_577_600));
    verify_chain(&certs, webpki_roots::TLS_SERVER_ROOTS, then).unwrap();
    let now = verify_chain(&certs, webpki_roots::TLS_SERVER_ROOTS, UnixTime::now());
    assert_eq!(now.unwrap_err().reason, "certificate_expired");
}

// --- timestamp and application id ------------------------------------------

#[test]
fn timestamp_parsing_and_skew() {
    assert_eq!(
        parse_timestamp(&json!("2026-10-04T12:00:00Z")),
        Some(1_791_115_200)
    );
    assert_eq!(
        parse_timestamp(&json!("2026-10-04T13:00:00.123+01:00")),
        Some(1_791_115_200)
    );
    assert_eq!(
        parse_timestamp(&json!(1_791_115_200_000_i64)),
        Some(1_791_115_200)
    );
    assert_eq!(parse_timestamp(&json!("yesterday")), None);
    let now = 1_791_115_200;
    let request = json!({"request": {"timestamp": "2026-10-04T12:02:00Z"}});
    assert!(check_timestamp(&request, now, 150).is_ok());
    let stale = json!({"request": {"timestamp": "2026-10-04T11:57:29Z"}});
    assert_eq!(
        check_timestamp(&stale, now, 150).unwrap_err().reason,
        "timestamp_out_of_range"
    );
    // Configured tolerance cannot exceed Amazon's 150 s.
    assert!(check_timestamp(&stale, now, 10_000).is_err());
}

#[test]
fn application_id_must_match_configuration() {
    let allowed = vec![SKILL_ID.to_string()];
    let good = envelope(json!({"type": "LaunchRequest"}), "x", SKILL_ID);
    assert!(check_application_id(&good, &allowed).is_ok());
    let bad = envelope(
        json!({"type": "LaunchRequest"}),
        "x",
        "amzn1.ask.skill.other",
    );
    assert_eq!(
        check_application_id(&bad, &allowed).unwrap_err().reason,
        "application_id_mismatch"
    );
    assert!(check_application_id(&good, &[]).is_err());
    let mut mixed = good.clone();
    mixed["context"]["System"]["application"]["applicationId"] = json!("amzn1.ask.skill.other");
    assert!(check_application_id(&mixed, &allowed).is_err());
    assert!(check_application_id(&json!({}), &allowed).is_err());
}

#[test]
fn config_overrides_and_clamps() {
    let mut config = AlexaConfig::default();
    assert!(!config.enabled);
    config.timestamp_tolerance_seconds = 9999;
    config.answer_deadline_ms = 60_000;
    config.apply_overrides(
        Some("true".into()),
        Some(" amzn1.ask.skill.a, amzn1.ask.skill.b ".into()),
    );
    assert!(config.enabled);
    assert_eq!(
        config.skill_ids,
        vec!["amzn1.ask.skill.a", "amzn1.ask.skill.b"]
    );
    assert_eq!(config.timestamp_tolerance_seconds, 150);
    assert_eq!(config.answer_deadline_ms, 7500);
    config.apply_overrides(Some("0".into()), None);
    assert!(!config.enabled);
}

// --- full request verification ---------------------------------------------

#[tokio::test]
async fn verify_request_end_to_end_and_rejections() {
    let verifier = test_verifier();
    let config = enabled_config();
    let body = envelope(
        json!({"type": "LaunchRequest"}),
        &iso_now_offset(0),
        SKILL_ID,
    )
    .to_string();
    let headers = signed_headers(body.as_bytes(), CHAIN_URL, TEST_LEAF_KEY);
    let now = SystemTime::now();
    assert!(
        verifier
            .verify_request(&headers, body.as_bytes(), now, &config)
            .await
            .is_ok()
    );

    // Tampered body.
    let tampered = body.replace("LaunchRequest", "SessionEndedRequest");
    let err = verifier
        .verify_request(&headers, tampered.as_bytes(), now, &config)
        .await
        .unwrap_err();
    assert_eq!(
        (err.status, err.reason),
        (StatusCode::UNAUTHORIZED, "signature_mismatch")
    );

    // Missing headers.
    let err = verifier
        .verify_request(&HeaderMap::new(), body.as_bytes(), now, &config)
        .await
        .unwrap_err();
    assert_eq!(err.status, StatusCode::BAD_REQUEST);

    // Bad cert URL never triggers a download.
    let mut bad_url = headers.clone();
    bad_url.insert(
        "SignatureCertChainUrl",
        HeaderValue::from_static("https://evil.example.com/echo.api/cert.pem"),
    );
    let err = verifier
        .verify_request(&bad_url, body.as_bytes(), now, &config)
        .await
        .unwrap_err();
    assert_eq!(err.reason, "invalid_signature_cert_chain_url");

    // Signed by a key whose certificate lacks the echo-api SAN.
    let wrong = signed_headers(body.as_bytes(), WRONG_SAN_URL, WRONG_SAN_KEY);
    let err = verifier
        .verify_request(&wrong, body.as_bytes(), now, &config)
        .await
        .unwrap_err();
    assert_eq!(err.reason, "certificate_subject_mismatch");

    // Timestamp skew beyond 150 s.
    let stale = envelope(
        json!({"type": "LaunchRequest"}),
        &iso_now_offset(-200),
        SKILL_ID,
    )
    .to_string();
    let stale_headers = signed_headers(stale.as_bytes(), CHAIN_URL, TEST_LEAF_KEY);
    let err = verifier
        .verify_request(&stale_headers, stale.as_bytes(), now, &config)
        .await
        .unwrap_err();
    assert_eq!(
        (err.status, err.reason),
        (StatusCode::BAD_REQUEST, "timestamp_out_of_range")
    );

    // Different skill.
    let other = envelope(
        json!({"type": "LaunchRequest"}),
        &iso_now_offset(0),
        "amzn1.ask.skill.someone-else",
    )
    .to_string();
    let other_headers = signed_headers(other.as_bytes(), CHAIN_URL, TEST_LEAF_KEY);
    let err = verifier
        .verify_request(&other_headers, other.as_bytes(), now, &config)
        .await
        .unwrap_err();
    assert_eq!(err.reason, "application_id_mismatch");
}

#[tokio::test]
async fn disabled_runtime_returns_not_found() {
    let calls = Arc::new(AtomicUsize::new(0));
    let rt = AlexaRuntime::with_verifier(
        AlexaConfig::default(),
        test_verifier(),
        mock_asker("x", Duration::ZERO, calls),
    );
    let response = rt.handle(&HeaderMap::new(), b"{}").await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

// --- conversation ----------------------------------------------------------

#[tokio::test]
async fn launch_keeps_session_open_with_reprompt() {
    let calls = Arc::new(AtomicUsize::new(0));
    let rt = runtime(mock_asker("x", Duration::ZERO, calls.clone()));
    let response = rt
        .respond(&envelope(json!({"type": "LaunchRequest"}), "", SKILL_ID))
        .await;
    assert_eq!(response["version"], "1.0");
    assert_eq!(response["response"]["shouldEndSession"], false);
    assert!(ssml_of(&response).starts_with("<speak>Hi, I&apos;m Aaron"));
    assert_eq!(response["response"]["outputSpeech"]["type"], "SSML");
    assert!(response["response"]["reprompt"]["outputSpeech"]["ssml"].is_string());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn ask_intent_answers_through_gail_and_keeps_history() {
    let calls = Arc::new(AtomicUsize::new(0));
    let rt = runtime(mock_asker(
        "**Ben Nevis** is the tallest <mountain> & it's 1345 m.",
        Duration::ZERO,
        calls.clone(),
    ));
    let response = rt
        .respond(&envelope(
            ask_request("what is the tallest mountain in scotland"),
            "",
            SKILL_ID,
        ))
        .await;
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        ssml_of(&response),
        "<speak>Ben Nevis is the tallest &lt;mountain&gt; &amp; it&apos;s 1345 m.</speak>"
    );
    assert_eq!(response["response"]["shouldEndSession"], false);
    assert!(response["response"]["reprompt"].is_object());
    let history = &response["sessionAttributes"]["history"];
    assert_eq!(history[0]["q"], "what is the tallest mountain in scotland");

    // Follow-up carries history into the transcript.
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen_clone = seen.clone();
    let asker: Asker = Arc::new(move |messages: Vec<ChatMessage>| {
        *seen_clone.lock().unwrap() = messages;
        Box::pin(async { Ok("About four hours.".to_string()) })
    });
    let rt = runtime(asker);
    let mut follow_up = envelope(ask_request("how long to climb it"), "", SKILL_ID);
    follow_up["session"]["attributes"] = response["sessionAttributes"].clone();
    rt.respond(&follow_up).await;
    let messages = seen.lock().unwrap().clone();
    assert_eq!(messages.len(), 4);
    assert_eq!(messages[1].role, "user");
    assert_eq!(messages[2].role, "assistant");
    assert_eq!(messages[3].flattened_text(), "how long to climb it");
}

#[tokio::test]
async fn built_in_intents_and_session_end() {
    let calls = Arc::new(AtomicUsize::new(0));
    let rt = runtime(mock_asker("x", Duration::ZERO, calls.clone()));
    for name in ["AMAZON.StopIntent", "AMAZON.CancelIntent"] {
        let response = rt.respond(&envelope(intent(name), "", SKILL_ID)).await;
        assert_eq!(response["response"]["shouldEndSession"], true, "{name}");
        assert_eq!(ssml_of(&response), "<speak>Goodbye.</speak>");
    }
    for name in ["AMAZON.HelpIntent", "AMAZON.FallbackIntent", "Unknown"] {
        let response = rt.respond(&envelope(intent(name), "", SKILL_ID)).await;
        assert_eq!(response["response"]["shouldEndSession"], false, "{name}");
        assert!(response["response"]["reprompt"].is_object(), "{name}");
    }
    let empty = rt.respond(&envelope(ask_request("  "), "", SKILL_ID)).await;
    assert!(ssml_of(&empty).contains("What would you like to ask?"));
    let ended = rt
        .respond(&envelope(
            json!({"type": "SessionEndedRequest"}),
            "",
            SKILL_ID,
        ))
        .await;
    assert_eq!(ended, json!({"version": "1.0", "response": {}}));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn timeout_says_still_thinking_and_yes_delivers_answer() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut config = enabled_config();
    config.answer_deadline_ms = 500; // minimum allowed
    let rt = AlexaRuntime::with_verifier(
        config,
        test_verifier(),
        mock_asker("Forty two.", Duration::from_millis(1500), calls.clone()),
    );
    let started = Instant::now();
    let response = rt
        .respond(&envelope(ask_request("the meaning of life"), "", SKILL_ID))
        .await;
    assert!(started.elapsed() < Duration::from_millis(1400));
    assert!(ssml_of(&response).contains("still thinking"));
    assert_eq!(response["response"]["shouldEndSession"], false);
    assert!(response["response"]["reprompt"].is_object());

    // The completion keeps running in the background; let it finish so the
    // Yes turn does not depend on scheduler timing.
    tokio::time::sleep(Duration::from_millis(1300)).await;
    let response = rt
        .respond(&envelope(intent("AMAZON.YesIntent"), "", SKILL_ID))
        .await;
    assert_eq!(ssml_of(&response), "<speak>Forty two.</speak>");
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let response = rt
        .respond(&envelope(intent("AMAZON.YesIntent"), "", SKILL_ID))
        .await;
    assert!(ssml_of(&response).contains("don&apos;t have an answer waiting"));
}

#[tokio::test]
async fn governance_block_and_failure_are_spoken() {
    for (error, expected) in [
        (AskError::Blocked, "can&apos;t help with that"),
        (AskError::Failed, "couldn&apos;t get an answer"),
    ] {
        let asker: Asker = Arc::new(move |_| Box::pin(async move { Err(error) }));
        let response = runtime(asker)
            .respond(&envelope(ask_request("anything"), "", SKILL_ID))
            .await;
        assert!(ssml_of(&response).contains(expected), "{response}");
        assert_eq!(response["response"]["shouldEndSession"], false);
    }
    let blocked = GailError::upstream(
        "aria",
        Some(StatusCode::FORBIDDEN),
        "governance did not permit the operation",
    );
    assert_eq!(AskError::from_gail(&blocked), AskError::Blocked);
    let unavailable = GailError::upstream("aria", Some(StatusCode::SERVICE_UNAVAILABLE), "x");
    assert_eq!(AskError::from_gail(&unavailable), AskError::Failed);
}

#[tokio::test]
async fn progressive_response_is_sent_for_slow_answers() {
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{body_string_contains, header, method, path},
    };
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/directives"))
        .and(header("authorization", "Bearer test-api-access-token"))
        .and(body_string_contains("VoicePlayer.Speak"))
        .and(body_string_contains("amzn1.echo-api.request.test"))
        .respond_with(ResponseTemplate::new(204))
        .expect(1)
        .mount(&server)
        .await;
    let calls = Arc::new(AtomicUsize::new(0));
    let rt = runtime(mock_asker("Done.", Duration::from_millis(300), calls))
        .with_test_directives(Duration::from_millis(10));
    let mut request = envelope(ask_request("something slow"), "", SKILL_ID);
    request["context"]["System"]["apiAccessToken"] = json!("test-api-access-token");
    request["context"]["System"]["apiEndpoint"] = json!(server.uri());
    let response = rt.respond(&request).await;
    assert_eq!(ssml_of(&response), "<speak>Done.</speak>");
    server.verify().await;
}

#[test]
fn directive_endpoint_must_be_amazon() {
    assert!(directive_url("https://api.amazonalexa.com", false).is_some());
    assert_eq!(
        directive_url("https://api.eu.amazonalexa.com", false)
            .unwrap()
            .as_str(),
        "https://api.eu.amazonalexa.com/v1/directives"
    );
    assert!(directive_url("https://evil.example.com", false).is_none());
    assert!(directive_url("http://api.amazonalexa.com", false).is_none());
    assert!(directive_url("https://amazonalexa.com.evil.com", false).is_none());
}

// --- SSML ------------------------------------------------------------------

#[test]
fn ssml_escaping_and_length_limit() {
    assert_eq!(
        escape_ssml(r#"a & b < c > d "e" 'f'"#),
        "a &amp; b &lt; c &gt; d &quot;e&quot; &apos;f&apos;"
    );
    assert_eq!(escape_ssml("line\nbreak\u{0007}"), "line break");
    assert_eq!(to_ssml("<speak>"), "<speak>&lt;speak&gt;</speak>");
    let long = "word & ".repeat(3000);
    let ssml = to_ssml(&long);
    assert!(ssml.chars().count() <= MAX_SSML_CHARS);
    assert!(ssml.starts_with("<speak>") && ssml.ends_with("...</speak>"));
    assert!(!ssml.contains("&am...")); // never cuts inside an entity
}

#[test]
fn spoken_text_strips_markdown() {
    assert_eq!(
        spoken_text("# Title\n- **one** [link](https://x.y)\n* `two`\n```rust\ncode\n```"),
        "Title one link two rust code"
    );
}

#[test]
fn interaction_models_match_question_intent_table() {
    for model in [
        include_str!("../integrations/alexa/skill-package/interactionModels/custom/en-GB.json"),
        include_str!("../integrations/alexa/skill-package/interactionModels/custom/en-US.json"),
    ] {
        let model: Value = serde_json::from_str(model).unwrap();
        let language = &model["interactionModel"]["languageModel"];
        assert_eq!(language["invocationName"], "aaron");
        let intents = language["intents"].as_array().unwrap();
        let mut ask_intents = 0;
        for intent in intents {
            let name = intent["name"].as_str().unwrap();
            if name.starts_with("AMAZON.") {
                continue;
            }
            ask_intents += 1;
            assert!(question_prefix(name).is_some(), "{name} missing from table");
            assert_eq!(intent["slots"][0]["type"], "AMAZON.SearchQuery");
            for sample in intent["samples"].as_array().unwrap() {
                let sample = sample.as_str().unwrap();
                // SearchQuery needs a carrier phrase and must end the utterance.
                assert!(
                    sample.ends_with("{query}") && sample != "{query}",
                    "{sample}"
                );
                let carrier = sample.trim_end_matches("{query}").trim();
                let prefix = question_prefix(name).unwrap();
                if !prefix.is_empty() && !carrier.contains('\'') && name != "AskAaronWhetherIntent"
                {
                    assert!(
                        prefix.starts_with(carrier.split(' ').next().unwrap()),
                        "{name}: {sample}"
                    );
                }
            }
        }
        assert_eq!(ask_intents, QUESTION_INTENTS.len());
        for builtin in [
            "AMAZON.HelpIntent",
            "AMAZON.StopIntent",
            "AMAZON.CancelIntent",
            "AMAZON.FallbackIntent",
        ] {
            assert!(intents.iter().any(|i| i["name"] == builtin), "{builtin}");
        }
    }
}

#[tokio::test]
async fn question_intents_restore_their_carrier_words() {
    let seen = Arc::new(Mutex::new(String::new()));
    let seen_clone = seen.clone();
    let asker: Asker = Arc::new(move |messages: Vec<ChatMessage>| {
        *seen_clone.lock().unwrap() = messages.last().unwrap().flattened_text();
        Box::pin(async { Ok("Ok.".to_string()) })
    });
    let rt = runtime(asker);
    let mut request = ask_request("is the tallest mountain");
    request["intent"]["name"] = json!("AskAaronWhatIntent");
    rt.respond(&envelope(request, "", SKILL_ID)).await;
    assert_eq!(*seen.lock().unwrap(), "what is the tallest mountain");
    let mut request = ask_request("black holes");
    request["intent"]["name"] = json!("AskAaronIntent");
    rt.respond(&envelope(request, "", SKILL_ID)).await;
    assert_eq!(*seen.lock().unwrap(), "black holes");
}

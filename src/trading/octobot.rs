/// OctoBot REST API client.
///
/// OctoBot exposes its web interface on port 5001.  Current OctoBot web API
/// routes are Flask endpoints under `/api/*` plus dashboard/backtesting routes.
/// The cluster deployment keeps native OctoBot web auth disabled behind
/// shared ingress auth, so the bridge treats `/api/ping` as the session probe.
///
/// Endpoint coverage:
///   Portfolio, open orders, trade history, exchange/symbol listings,
///   order cancellation, market snapshots, and general status. Live order
///   placement is attempted through known `/api/orders` and `/api/user_command`
///   variants to support native OctoBot and custom bridge extensions.
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Duration,
};

use once_cell::sync::Lazy;
use regex::Regex;
use reqwest::{Client, ClientBuilder, StatusCode, header::CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::Mutex;
use tokio::task::JoinSet;
use tokio::time::sleep;
use tracing::{debug, warn};
use url::form_urlencoded;

use crate::{
    adaptive_schema::{self, AdaptiveApiSchema},
    api_issues,
};

const OCTOBOT_API: &str = "octobot";
const MAX_PARALLEL_MARKET_SNAPSHOT_REQUESTS: usize = 8;
const MARKET_SNAPSHOT_CANDIDATE_MULTIPLIER: usize = 1;
const MARKET_SNAPSHOT_UNKNOWN_PROBE_LIMIT_PER_EXCHANGE: usize = 8;
const MARKET_SNAPSHOT_UNAVAILABLE_COOLDOWN_SECONDS: f64 = 3_600.0;
const TRADING_PAIR_RESTART_COOLDOWN_SECONDS: f64 = 180.0;

// ---------------------------------------------------------------------------
// Domain models
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct OctobotPortfolio {
    /// Map of currency symbol → balance entry
    pub currencies: std::collections::HashMap<String, CurrencyBalance>,
    pub total_value_usd: Option<f64>,
    /// Optional per-exchange balances when OctoBot provides exchange-scoped holdings.
    #[serde(default)]
    pub exchange_currencies:
        std::collections::HashMap<String, std::collections::HashMap<String, CurrencyBalance>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct CurrencyBalance {
    pub free: f64,
    pub locked: f64,
    pub total: f64,
    pub value_usd: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OctobotOrder {
    pub id: String,
    pub exchange: String,
    pub symbol: String,
    pub side: String,
    pub order_type: String,
    pub amount: f64,
    pub price: Option<f64>,
    pub status: String,
    pub timestamp: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OctobotTrade {
    pub id: Option<String>,
    pub exchange: String,
    pub symbol: String,
    pub side: String,
    pub amount: f64,
    pub price: f64,
    pub cost: f64,
    pub fee: Option<f64>,
    pub timestamp: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OctobotExchange {
    pub name: String,
    pub enabled: bool,
    pub symbols: Vec<String>,
}

#[derive(Clone, Debug)]
struct TradingPageSymbolStatusRow {
    exchange_id: String,
    exchange_name: String,
    symbol: String,
}

#[derive(Clone, Debug)]
struct ExchangeInfoEntry {
    name: String,
    enabled: bool,
    symbols: Vec<String>,
    exchange_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OctobotStatus {
    pub running: bool,
    pub version: Option<String>,
    pub uptime_seconds: Option<f64>,
    pub trading_enabled: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OctobotOrderResult {
    pub order_id: String,
    pub symbol: String,
    pub side: String,
    pub amount: f64,
    pub price: Option<f64>,
    pub status: String,
}

#[derive(Clone, Debug, Default)]
pub struct TradingPairActivationStatus {
    pub ready: bool,
    pub restart_required: bool,
    pub changed: bool,
    pub message: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct OctobotLogEntry {
    pub time: Option<String>,
    pub level: String,
    pub source: String,
    pub message: String,
}

// ---------------------------------------------------------------------------
// Market data snapshot (assembled from various OctoBot API endpoints)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct MarketSnapshot {
    pub exchange: String,
    pub symbol: String,
    pub price: f64,
    pub price_change_pct_1h: Option<f64>,
    pub price_change_pct_24h: Option<f64>,
    pub volume_24h: Option<f64>,
    pub volume_change_pct: Option<f64>,
    pub high_24h: Option<f64>,
    pub low_24h: Option<f64>,
    pub fetched_at: f64,
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct OctobotClient {
    client: Client,
    base_url: String,
    password: Option<String>,
    api_schema: Arc<Mutex<AdaptiveApiSchema>>,

    /// Cached successful order-submission mode.
    ///
    /// Gail discovers this automatically by trying candidate modes in a safe
    /// order. Once one mode is positively acknowledged, future orders try it
    /// first, but can still fall back if that mode later fails.
    preferred_order_submission_mode: Arc<Mutex<Option<OctobotOrderSubmissionMode>>>,

    /// Exchange name → OctoBot exchange_id mapping discovered from API/HTML surfaces.
    exchange_id_by_name: Arc<Mutex<HashMap<String, String>>>,

    /// Rotating scan offsets used to poll different subsets of symbols over time.
    symbol_scan_offsets: Arc<Mutex<HashMap<String, usize>>>,

    /// Exchange+symbol pairs that recently returned no market data. Avoid
    /// immediately retrying these pairs on every cycle.
    market_snapshot_unavailable_until: Arc<Mutex<HashMap<String, f64>>>,

    /// Exchange+symbol pairs that have previously returned valid market data.
    /// These are prioritized over unproven symbols to reduce noisy probing.
    market_snapshot_available_symbols: Arc<Mutex<HashSet<String>>>,

    /// Exchange+symbol keys with a short restart cooldown to avoid repeatedly
    /// requesting OctoBot restart on every evaluation while a restart is
    /// already in progress for the same pair activation.
    trading_pair_restart_cooldown_until: Arc<Mutex<HashMap<String, f64>>>,
}

impl OctobotClient {
    pub fn new(base_url: &str, password: Option<&str>, timeout_seconds: f64) -> Self {
        Self::new_with_schema(
            base_url,
            password,
            timeout_seconds,
            AdaptiveApiSchema::default(),
        )
    }

    pub fn new_with_schema(
        base_url: &str,
        password: Option<&str>,
        timeout_seconds: f64,
        api_schema: AdaptiveApiSchema,
    ) -> Self {
        let client = ClientBuilder::new()
            .use_rustls_tls()
            .cookie_store(true)
            .timeout(Duration::from_secs_f64(timeout_seconds))
            .build()
            .unwrap_or_default();

        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            password: password.map(str::to_string),
            api_schema: Arc::new(Mutex::new(api_schema)),
            preferred_order_submission_mode: Arc::new(Mutex::new(None)),
            exchange_id_by_name: Arc::new(Mutex::new(HashMap::new())),
            symbol_scan_offsets: Arc::new(Mutex::new(HashMap::new())),
            market_snapshot_unavailable_until: Arc::new(Mutex::new(HashMap::new())),
            market_snapshot_available_symbols: Arc::new(Mutex::new(HashSet::new())),
            trading_pair_restart_cooldown_until: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub async fn api_schema_snapshot(&self) -> AdaptiveApiSchema {
        self.api_schema.lock().await.clone()
    }

    async fn get_json(&self, path: &str, label: &str) -> Result<Value, String> {
        let url = format!("{}{}", self.base_url, path);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("OctoBot {label} request failed: {e}"))?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            self.observe_failure("GET", path, label, Some(status.as_u16()), text.trim())
                .await;
            return Err(format!(
                "OctoBot {label} failed: HTTP {}: {}",
                status.as_u16(),
                text.trim()
            ));
        }
        let parsed: Value = match serde_json::from_str(&text) {
            Ok(parsed) => parsed,
            Err(err) => {
                let error = format!("OctoBot {label} parse failed: {err}");
                self.observe_failure("GET", path, label, Some(status.as_u16()), &error)
                    .await;
                return Err(error);
            }
        };
        self.observe_success("GET", path, label, &parsed).await;
        Ok(parsed)
    }

    async fn get_optional_json(&self, path: &str, label: &str) -> Result<Option<Value>, String> {
        if self.should_skip("GET", path).await {
            debug!(
                "trading: skipping OctoBot {} at {} due to adaptive schema backoff",
                label, path
            );
            return Ok(None);
        }
        let url = format!("{}{}", self.base_url, path);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("OctoBot {label} request failed: {e}"))?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if status == StatusCode::NOT_FOUND {
            self.observe_failure("GET", path, label, Some(status.as_u16()), text.trim())
                .await;
            return Ok(None);
        }
        if !status.is_success() {
            self.observe_failure("GET", path, label, Some(status.as_u16()), text.trim())
                .await;
            return Err(format!(
                "OctoBot {label} failed: HTTP {}: {}",
                status.as_u16(),
                text.trim()
            ));
        }
        let parsed = match serde_json::from_str(&text) {
            Ok(parsed) => parsed,
            Err(err) => {
                let error = format!("OctoBot {label} parse failed: {err}");
                self.observe_failure("GET", path, label, Some(status.as_u16()), &error)
                    .await;
                return Err(error);
            }
        };
        self.observe_success("GET", path, label, &parsed).await;
        Ok(Some(parsed))
    }

    async fn get_optional_text(&self, path: &str, label: &str) -> Result<Option<String>, String> {
        if self.should_skip("GET", path).await {
            debug!(
                "trading: skipping OctoBot {} at {} due to adaptive schema backoff",
                label, path
            );
            return Ok(None);
        }
        let url = format!("{}{}", self.base_url, path);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("OctoBot {label} request failed: {e}"))?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if status == StatusCode::NOT_FOUND {
            self.observe_failure("GET", path, label, Some(status.as_u16()), text.trim())
                .await;
            return Ok(None);
        }
        if !status.is_success() {
            self.observe_failure("GET", path, label, Some(status.as_u16()), text.trim())
                .await;
            return Err(format!(
                "OctoBot {label} failed: HTTP {}: {}",
                status.as_u16(),
                text.trim()
            ));
        }
        self.observe_success("GET", path, label, &json!({ "body": "text" }))
            .await;
        Ok(Some(text))
    }

    async fn post_json_with_status(
        &self,
        path: &str,
        body: &Value,
        label: &str,
    ) -> Result<(StatusCode, Value), String> {
        let url = format!("{}{}", self.base_url, path);
        let resp = self
            .client
            .post(&url)
            .json(body)
            .send()
            .await
            .map_err(|e| format!("OctoBot {label} request failed: {e}"))?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        let parsed = if text.trim().is_empty() {
            Value::Null
        } else {
            serde_json::from_str(&text).unwrap_or_else(|_| json!({ "message": text }))
        };
        if status.is_success() {
            self.observe_success("POST", path, label, &parsed).await;
        } else {
            self.observe_failure("POST", path, label, Some(status.as_u16()), text.trim())
                .await;
        }
        Ok((status, parsed))
    }

    async fn should_skip(&self, method: &str, path: &str) -> bool {
        self.api_schema.lock().await.should_skip(method, path)
    }

    async fn observe_success(&self, method: &str, path: &str, label: &str, body: &Value) {
        self.api_schema
            .lock()
            .await
            .observe_success(method, path, label, body);
        adaptive_schema::observe_success(OCTOBOT_API, method, path, label, body).await;
        api_issues::observe_api_recovery(OCTOBOT_API, method, path, label).await;
    }

    async fn observe_failure(
        &self,
        method: &str,
        path: &str,
        label: &str,
        status: Option<u16>,
        error: &str,
    ) {
        self.api_schema
            .lock()
            .await
            .observe_failure(method, path, label, status, error);
        adaptive_schema::observe_failure(OCTOBOT_API, method, path, label, status, error).await;
        api_issues::observe_api_failure(OCTOBOT_API, method, path, label, status, error).await;
    }

    /// Authenticate with OctoBot and establish a session cookie.
    pub async fn login(&self) -> Result<(), String> {
        match self.get_json("/api/ping", "ping").await {
            Ok(_) => {
                debug!("trading: OctoBot web API is reachable");
                Ok(())
            }
            Err(err) if self.password.is_some() => Err(format!(
                "{err}. OctoBot does not expose a JSON password-login endpoint; disable octobot_password for shared-auth cluster deployments or expose an authenticated API session."
            )),
            Err(err) => Err(err),
        }
    }

    // -----------------------------------------------------------------------
    // Status / health
    // -----------------------------------------------------------------------

    pub async fn get_status(&self) -> Result<OctobotStatus, String> {
        let ping = self.get_json("/api/ping", "ping").await?;
        let version = self
            .get_optional_json("/api/version", "version")
            .await?
            .and_then(|body| body.as_str().map(str::to_string));
        Ok(OctobotStatus {
            running: true,
            version,
            uptime_seconds: None,
            trading_enabled: ping.as_str().is_some_and(|value| value.contains("Running"))
                || !ping.is_null(),
        })
    }

    // -----------------------------------------------------------------------
    // Portfolio
    // -----------------------------------------------------------------------

    pub async fn get_portfolio(&self) -> Result<OctobotPortfolio, String> {
        // OctoBot builds differ: some expose `/api/portfolio`, others only the
        // HTML `/portfolio` page.
        let api_portfolio = match self.get_optional_json("/api/portfolio", "portfolio").await {
            Ok(body) => body,
            Err(err) if err.contains("parse failed") => {
                warn!(
                    "trading: OctoBot /api/portfolio returned a non-JSON body; falling back to /portfolio HTML: {}",
                    err
                );
                None
            }
            Err(err) => return Err(err),
        };

        if let Some(body) = api_portfolio {
            let mut portfolio = parse_portfolio_json(&body);
            if portfolio.total_value_usd.is_none() {
                self.enrich_portfolio_total_from_history(&mut portfolio)
                    .await?;
            }
            if !portfolio.currencies.is_empty() || portfolio.total_value_usd.is_some() {
                return Ok(portfolio);
            }
        }

        if let Some(page) = self
            .get_optional_text("/portfolio", "portfolio page")
            .await?
            && let Some(mut portfolio) = parse_portfolio_html(&page)
        {
            if portfolio.total_value_usd.is_none() {
                self.enrich_portfolio_total_from_history(&mut portfolio)
                    .await?;
            }
            return Ok(portfolio);
        }

        let mut portfolio = OctobotPortfolio::default();
        self.enrich_portfolio_total_from_history(&mut portfolio)
            .await?;
        Ok(portfolio)
    }

    async fn enrich_portfolio_total_from_history(
        &self,
        portfolio: &mut OctobotPortfolio,
    ) -> Result<(), String> {
        if portfolio.total_value_usd.is_some() {
            return Ok(());
        }
        if let Some(body) = self
            .get_optional_json(
                "/api/historical_portfolio_value?currency=USDT",
                "historical portfolio",
            )
            .await?
        {
            portfolio.total_value_usd = parse_latest_portfolio_value(&body);
        }
        Ok(())
    }

    /// Ask OctoBot to refresh exchange balances before reading portfolio data.
    ///
    /// This is useful right before sell decisions: OctoBot can lag briefly
    /// after fills/transfers, and stale balances can lead to avoidable
    /// "balance unavailable" skips in Gail.
    pub async fn refresh_portfolio(&self) -> Result<(), String> {
        let (status, body) = self
            .post_json_with_status("/api/refresh_portfolio", &json!({}), "refresh portfolio")
            .await?;
        if status.is_success() {
            return Ok(());
        }

        let detail =
            extract_attempt_message(&body).unwrap_or_else(|| summarize_order_attempt_body(&body));
        Err(format!(
            "OctoBot refresh portfolio failed: HTTP {}: {detail}",
            status.as_u16()
        ))
    }

    // -----------------------------------------------------------------------
    // Trading pair activation / market status
    // -----------------------------------------------------------------------

    pub async fn ensure_trading_pair_active_for_order(
        &self,
        exchange: &str,
        symbol: &str,
    ) -> Result<TradingPairActivationStatus, String> {
        let Some(exchange_key) = normalize_exchange_name(exchange) else {
            return Err(format!(
                "Invalid exchange `{exchange}` while checking OctoBot trading pair activation"
            ));
        };
        let Some(normalized_symbol) = normalize_trading_symbol(symbol) else {
            return Err(format!(
                "Invalid symbol `{symbol}` while checking OctoBot trading pair activation"
            ));
        };

        if self
            .market_status_contains_pair(&exchange_key, &normalized_symbol)
            .await?
        {
            self.clear_trading_pair_restart_cooldown(&exchange_key, &normalized_symbol)
                .await;
            return Ok(TradingPairActivationStatus {
                ready: true,
                restart_required: false,
                changed: false,
                message: format!(
                    "{exchange_key}/{normalized_symbol} already active in OctoBot market status"
                ),
            });
        }

        let config_changed = self.ensure_symbol_in_config(&normalized_symbol).await?;
        self.ensure_symbol_in_watchlist(&normalized_symbol).await?;

        let restart_key = trading_pair_restart_key(&exchange_key, &normalized_symbol);
        if self.trading_pair_restart_on_cooldown(&restart_key).await {
            return Ok(TradingPairActivationStatus {
                ready: false,
                restart_required: true,
                changed: config_changed,
                message: format!(
                    "{exchange_key}/{normalized_symbol} not active yet; restart already requested recently, waiting for OctoBot restart to apply pair changes"
                ),
            });
        }

        self.request_restart().await?;
        self.record_trading_pair_restart_request(&restart_key).await;

        Ok(TradingPairActivationStatus {
            ready: false,
            restart_required: true,
            changed: config_changed,
            message: format!(
                "{exchange_key}/{normalized_symbol} was not active in market status; requested OctoBot restart to apply trading pair changes"
            ),
        })
    }

    pub async fn remove_trading_pair_configuration(
        &self,
        symbol: &str,
    ) -> Result<TradingPairActivationStatus, String> {
        let Some(normalized_symbol) = normalize_trading_symbol(symbol) else {
            return Err(format!(
                "Invalid symbol `{symbol}` while removing OctoBot trading pair"
            ));
        };

        let config_changed = self.remove_symbol_from_config(&normalized_symbol).await?;
        self.remove_symbol_from_watchlist(&normalized_symbol)
            .await?;

        if !config_changed {
            return Ok(TradingPairActivationStatus {
                ready: true,
                restart_required: false,
                changed: false,
                message: format!(
                    "{normalized_symbol} not present in configured currencies; no restart needed"
                ),
            });
        }

        self.request_restart().await?;
        Ok(TradingPairActivationStatus {
            ready: false,
            restart_required: true,
            changed: true,
            message: format!(
                "{normalized_symbol} removed from configured currencies; requested OctoBot restart to apply removal"
            ),
        })
    }

    async fn market_status_contains_pair(
        &self,
        exchange: &str,
        symbol: &str,
    ) -> Result<bool, String> {
        let Some(html) = self.get_optional_text("/trading", "trading page").await? else {
            return Ok(false);
        };
        let rows = parse_trading_page_symbol_status_rows(&html);
        self.observe_trading_page_rows(&rows).await;
        let Some(exchange_key) = normalize_exchange_name(exchange) else {
            return Ok(false);
        };

        Ok(rows.iter().any(|row| {
            normalize_exchange_name(&row.exchange_name).is_some_and(|name| name == exchange_key)
                && row.symbol.eq_ignore_ascii_case(symbol)
        }))
    }

    async fn observe_trading_page_rows(&self, rows: &[TradingPageSymbolStatusRow]) {
        if rows.is_empty() {
            return;
        }
        let mut cache = self.exchange_id_by_name.lock().await;
        for row in rows {
            if let Some(exchange_key) = normalize_exchange_name(&row.exchange_name) {
                cache
                    .entry(exchange_key)
                    .or_insert_with(|| row.exchange_id.clone());
            }
        }
    }

    async fn ensure_symbol_in_config(&self, symbol: &str) -> Result<bool, String> {
        let configured = self.get_configured_currency_map().await?;
        if configured_currency_map_contains_symbol(&configured, symbol) {
            return Ok(false);
        }

        let currency_key = best_currency_key_for_symbol(&configured, symbol).unwrap_or_else(|| {
            symbol_base_asset(symbol)
                .unwrap_or(symbol)
                .to_ascii_uppercase()
        });

        let mut currencies_payload = serde_json::Map::new();
        currencies_payload.insert(
            currency_key.clone(),
            json!({
                "enabled": true,
                "pairs": [symbol]
            }),
        );
        let payload = json!({
            "action": "update",
            "currencies": Value::Object(currencies_payload)
        });
        let (status, body) = self
            .post_json_with_status("/api/set_config_currency", &payload, "set config currency")
            .await?;
        if !status.is_success() {
            let detail = extract_attempt_message(&body)
                .unwrap_or_else(|| summarize_order_attempt_body(&body));
            return Err(format!(
                "OctoBot set_config_currency failed for {symbol}: HTTP {}: {detail}",
                status.as_u16()
            ));
        }

        Ok(true)
    }

    async fn remove_symbol_from_config(&self, symbol: &str) -> Result<bool, String> {
        let configured = self.get_configured_currency_map().await?;
        if configured.is_empty() {
            return Ok(false);
        }

        let mut changed = false;
        let mut updated = serde_json::Map::new();

        for (currency, entry) in configured {
            let enabled = entry
                .get("enabled")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            let pairs = configured_currency_pairs(&entry);
            let filtered_pairs = pairs
                .into_iter()
                .filter(|pair| !pair.eq_ignore_ascii_case(symbol))
                .collect::<Vec<_>>();
            if filtered_pairs.is_empty() {
                if configured_currency_contains_symbol(&entry, symbol) {
                    changed = true;
                }
                continue;
            }

            if filtered_pairs.len() != configured_currency_pairs(&entry).len() {
                changed = true;
            }

            updated.insert(
                currency,
                json!({
                    "enabled": enabled,
                    "pairs": filtered_pairs
                }),
            );
        }

        if !changed {
            return Ok(false);
        }

        let payload = json!({
            "action": "replace",
            "currencies": Value::Object(updated)
        });
        let (status, body) = self
            .post_json_with_status("/api/set_config_currency", &payload, "set config currency")
            .await?;
        if status.is_success() {
            return Ok(true);
        }

        let detail =
            extract_attempt_message(&body).unwrap_or_else(|| summarize_order_attempt_body(&body));
        Err(format!(
            "OctoBot set_config_currency replace failed while removing {symbol}: HTTP {}: {detail}",
            status.as_u16()
        ))
    }

    async fn get_configured_currency_map(&self) -> Result<serde_json::Map<String, Value>, String> {
        let body = self
            .get_optional_json("/api/get_config_currency", "configured currencies")
            .await?
            .unwrap_or_else(|| json!({}));
        Ok(body.as_object().cloned().unwrap_or_default())
    }

    async fn ensure_symbol_in_watchlist(&self, symbol: &str) -> Result<(), String> {
        let payload = json!({
            "symbol": symbol,
            "action": "add"
        });
        let (status, body) = self
            .post_json_with_status("/watched_symbols", &payload, "watched symbols")
            .await?;
        if status.is_success() {
            return Ok(());
        }
        let detail =
            extract_attempt_message(&body).unwrap_or_else(|| summarize_order_attempt_body(&body));
        Err(format!(
            "OctoBot watched_symbols add failed for {symbol}: HTTP {}: {detail}",
            status.as_u16()
        ))
    }

    async fn remove_symbol_from_watchlist(&self, symbol: &str) -> Result<(), String> {
        let payload = json!({
            "symbol": symbol,
            "action": "remove"
        });
        let (status, body) = self
            .post_json_with_status("/watched_symbols", &payload, "watched symbols")
            .await?;
        if status.is_success() {
            return Ok(());
        }
        let detail =
            extract_attempt_message(&body).unwrap_or_else(|| summarize_order_attempt_body(&body));
        Err(format!(
            "OctoBot watched_symbols remove failed for {symbol}: HTTP {}: {detail}",
            status.as_u16()
        ))
    }

    async fn request_restart(&self) -> Result<(), String> {
        let (status, body) = self
            .post_json_with_status("/commands/restart", &json!({}), "restart command")
            .await?;
        if status.is_success() {
            return Ok(());
        }
        let detail =
            extract_attempt_message(&body).unwrap_or_else(|| summarize_order_attempt_body(&body));
        Err(format!(
            "OctoBot restart command failed: HTTP {}: {detail}",
            status.as_u16()
        ))
    }

    async fn trading_pair_restart_on_cooldown(&self, pair_key: &str) -> bool {
        let now = current_unix_timestamp_f64();
        self.trading_pair_restart_cooldown_until
            .lock()
            .await
            .get(pair_key)
            .is_some_and(|until| *until > now)
    }

    async fn record_trading_pair_restart_request(&self, pair_key: &str) {
        let mut cooldown = self.trading_pair_restart_cooldown_until.lock().await;
        cooldown.insert(
            pair_key.to_string(),
            current_unix_timestamp_f64() + TRADING_PAIR_RESTART_COOLDOWN_SECONDS,
        );
    }

    async fn clear_trading_pair_restart_cooldown(&self, exchange: &str, symbol: &str) {
        let key = trading_pair_restart_key(exchange, symbol);
        self.trading_pair_restart_cooldown_until
            .lock()
            .await
            .remove(&key);
    }

    // -----------------------------------------------------------------------
    // Orders
    // -----------------------------------------------------------------------

    pub async fn get_open_orders(&self) -> Result<Vec<OctobotOrder>, String> {
        let body = self.get_json("/api/orders", "open orders").await?;
        parse_orders_array(&body)
    }

    pub async fn get_trade_history(&self, limit: usize) -> Result<Vec<OctobotTrade>, String> {
        let body = self.get_json("/api/trades", "trade history").await?;
        let mut trades = parse_trades_array(&body)?;
        trades.truncate(limit);
        Ok(trades)
    }

    pub async fn place_buy_order(
        &self,
        exchange: &str,
        symbol: &str,
        amount_usd: f64,
    ) -> Result<OctobotOrderResult, String> {
        self.place_order(exchange, symbol, "buy", amount_usd).await
    }

    pub async fn place_sell_order(
        &self,
        exchange: &str,
        symbol: &str,
        amount_usd: f64,
    ) -> Result<OctobotOrderResult, String> {
        self.place_order(exchange, symbol, "sell", amount_usd).await
    }

    async fn place_order(
        &self,
        exchange: &str,
        symbol: &str,
        side: &str,
        amount_usd: f64,
    ) -> Result<OctobotOrderResult, String> {
        let normalized_side = normalize_order_side(side)
            .ok_or_else(|| format!("Unsupported trade side `{side}`: expected buy or sell"))?;

        if !amount_usd.is_finite() || amount_usd <= 0.0 {
            return Err(format!(
                "Invalid trade amount `{amount_usd}`: expected a positive finite USD value"
            ));
        }

        let rounded_amount = ((amount_usd * 100.0).round() / 100.0).max(0.01);

        let baseline = self.capture_order_baseline().await;
        let request_started_at = current_unix_timestamp_f64();

        let preferred_mode = { *self.preferred_order_submission_mode.lock().await };

        let modes = ordered_submission_modes(preferred_mode);
        let mut attempts = Vec::new();
        let mut submission_amount = rounded_amount;
        let mut sell_retry_attempts_remaining: usize =
            if normalized_side == "sell" { 2 } else { 0 };

        loop {
            let canonical_order = json!({
                "exchange": exchange,
                "symbol": symbol,
                "side": normalized_side,
                "type": "market",
                "order_type": "market",
                "amount": submission_amount,
                "amount_usd": submission_amount,
                "price": serde_json::Value::Null,
            });

            let mut saw_portfolio_negative_rejection = false;
            let mut suggested_retry_amount_usd: Option<f64> = None;

            for mode in modes.iter().copied() {
                let (path, payload, label) = build_order_submission(
                    mode,
                    exchange,
                    symbol,
                    normalized_side,
                    submission_amount,
                    &canonical_order,
                );

                match self
                    .attempt_order_submission(
                        mode,
                        path,
                        &payload,
                        label,
                        exchange,
                        symbol,
                        normalized_side,
                        submission_amount,
                        &baseline,
                        request_started_at,
                        &mut attempts,
                    )
                    .await?
                {
                    OrderSubmissionAttempt::Accepted(result) => {
                        {
                            let mut preferred = self.preferred_order_submission_mode.lock().await;
                            *preferred = Some(mode);
                        }

                        debug!(
                            ?mode,
                            exchange = %exchange,
                            symbol = %symbol,
                            side = %normalized_side,
                            amount_usd = submission_amount,
                            "trading: selected OctoBot order submission mode"
                        );

                        return Ok(result);
                    }

                    OrderSubmissionAttempt::Rejected => {
                        debug!(
                            ?mode,
                            exchange = %exchange,
                            symbol = %symbol,
                            side = %normalized_side,
                            amount_usd = submission_amount,
                            "trading: OctoBot order submission mode rejected order; trying next candidate"
                        );
                    }

                    OrderSubmissionAttempt::RejectedPortfolioNegative { retry_amount_usd } => {
                        saw_portfolio_negative_rejection = true;
                        if let Some(candidate) =
                            retry_amount_usd.filter(|value| value.is_finite() && *value > 0.0)
                        {
                            suggested_retry_amount_usd = Some(
                                suggested_retry_amount_usd
                                    .map_or(candidate, |current| current.min(candidate)),
                            );
                        }
                        warn!(
                            ?mode,
                            exchange = %exchange,
                            symbol = %symbol,
                            side = %normalized_side,
                            amount_usd = submission_amount,
                            retry_amount_usd = suggested_retry_amount_usd,
                            "trading: OctoBot rejected sell order due to portfolio precision/availability; retrying with a smaller amount"
                        );
                        break;
                    }

                    OrderSubmissionAttempt::RejectedNonPositiveQuantity => {
                        warn!(
                            ?mode,
                            exchange = %exchange,
                            symbol = %symbol,
                            side = %normalized_side,
                            amount_usd = submission_amount,
                            "trading: OctoBot rejected order due to non-positive adapted quantity; skipping fallback endpoints"
                        );
                        return Err(format!(
                            "OctoBot rejected {normalized_side} order for {exchange} {symbol} ${submission_amount:.2}: quantity became non-positive after market adaptation. Tried: {}",
                            attempts.join(" | ")
                        ));
                    }

                    OrderSubmissionAttempt::AmbiguousAccepted => {
                        warn!(
                            ?mode,
                            exchange = %exchange,
                            symbol = %symbol,
                            side = %normalized_side,
                            amount_usd = submission_amount,
                            "trading: OctoBot order endpoint returned success without acknowledgement; refusing fallback to avoid duplicate order"
                        );

                        return Err(format!(
                            "OctoBot order submission via {mode:?} returned HTTP success but no order acknowledgement or observable side-effect. Refusing to try additional mutating endpoints to avoid duplicate orders. Tried: {}",
                            attempts.join(" | ")
                        ));
                    }
                }
            }

            if sell_retry_attempts_remaining > 0
                && saw_portfolio_negative_rejection
                && let Some(retry_amount) = suggested_retry_amount_usd
                    .or_else(|| reduced_sell_retry_amount(submission_amount))
            {
                attempts.push(format!(
                    "sell-retry amount_usd={submission_amount:.2}->{retry_amount:.2} after portfolio-negative rejection"
                ));
                warn!(
                    exchange = %exchange,
                    symbol = %symbol,
                    side = %normalized_side,
                    amount_usd = submission_amount,
                    retry_amount_usd = retry_amount,
                    "trading: retrying sell order with reduced amount after portfolio-negative rejection"
                );
                submission_amount = retry_amount;
                sell_retry_attempts_remaining = sell_retry_attempts_remaining.saturating_sub(1);
                continue;
            }

            return Err(format!(
                "OctoBot order placement failed for {exchange} {symbol} {normalized_side} ${submission_amount:.2}. Tried: {}",
                attempts.join(" | ")
            ));
        }
    }

    pub async fn cancel_order(&self, order_id: &str) -> Result<(), String> {
        let body = json!({ "id": order_id });
        let (status, _) = self
            .post_json_with_status("/api/orders?action=cancel_order", &body, "cancel order")
            .await?;
        if status.is_success() || status == StatusCode::NOT_FOUND {
            Ok(())
        } else {
            Err(format!(
                "OctoBot cancel order failed: HTTP {}",
                status.as_u16()
            ))
        }
    }

    async fn attempt_order_submission(
        &self,
        mode: OctobotOrderSubmissionMode,
        path: &str,
        payload: &Value,
        label: &str,
        exchange: &str,
        symbol: &str,
        side: &str,
        amount_usd: f64,
        baseline: &OrderPlacementBaseline,
        request_started_at: f64,
        attempts: &mut Vec<String>,
    ) -> Result<OrderSubmissionAttempt, String> {
        let (status, body) = self.post_json_with_status(path, payload, label).await?;

        if !status.is_success() {
            let summarized = summarize_order_attempt_body(&body);
            attempts.push(format!(
                "{mode:?} {path} => HTTP {} {}",
                status.as_u16(),
                summarized
            ));

            // If the cached preferred mode starts failing, forget it so future
            // calls can re-discover a working mode.
            let mut preferred = self.preferred_order_submission_mode.lock().await;
            if *preferred == Some(mode) {
                *preferred = None;
            }

            if normalize_order_side(side) == Some("sell")
                && body_has_portfolio_negative_rejection(&body)
            {
                return Ok(OrderSubmissionAttempt::RejectedPortfolioNegative {
                    retry_amount_usd: retry_amount_from_portfolio_negative_rejection(
                        &body, amount_usd,
                    ),
                });
            }

            if body_has_non_positive_quantity_rejection(&body) {
                return Ok(OrderSubmissionAttempt::RejectedNonPositiveQuantity);
            }

            return Ok(OrderSubmissionAttempt::Rejected);
        }

        if let Some(result) =
            parse_order_result_body(&body, exchange, symbol, side, amount_usd, path)
        {
            attempts.push(format!(
                "{mode:?} {path} => acknowledged ({})",
                summarize_order_attempt_body(&body)
            ));

            return Ok(OrderSubmissionAttempt::Accepted(result));
        }

        if let Some(result) = self
            .wait_for_order_side_effects(
                exchange,
                symbol,
                side,
                amount_usd,
                baseline,
                request_started_at,
            )
            .await
        {
            attempts.push(format!(
                "{mode:?} {path} => accepted (observed order side-effects)"
            ));

            return Ok(OrderSubmissionAttempt::Accepted(result));
        }

        attempts.push(format!(
            "{mode:?} {path} => HTTP {} without order acknowledgement ({})",
            status.as_u16(),
            summarize_order_attempt_body(&body)
        ));

        Ok(OrderSubmissionAttempt::AmbiguousAccepted)
    }

    async fn capture_order_baseline(&self) -> OrderPlacementBaseline {
        let mut baseline = OrderPlacementBaseline {
            captured_at: current_unix_timestamp_f64(),
            ..OrderPlacementBaseline::default()
        };

        if let Ok(orders) = self.get_open_orders().await {
            baseline.open_orders_captured = true;
            baseline
                .open_order_ids
                .extend(orders.into_iter().map(|order| order.id));
        }

        if let Ok(trades) = self.get_trade_history(50).await {
            baseline.trades_captured = true;
            baseline.latest_trade_ts = trades
                .iter()
                .filter_map(|trade| trade.timestamp)
                .reduce(f64::max);
            baseline.trade_ids.extend(
                trades
                    .into_iter()
                    .filter_map(|trade| trade.id)
                    .filter(|id| !id.trim().is_empty()),
            );
        }

        baseline
    }

    async fn wait_for_order_side_effects(
        &self,
        exchange: &str,
        symbol: &str,
        side: &str,
        amount_usd: f64,
        baseline: &OrderPlacementBaseline,
        request_started_at: f64,
    ) -> Option<OctobotOrderResult> {
        const ORDER_POLL_ATTEMPTS: usize = 6;
        const ORDER_POLL_DELAY_MS: u64 = 500;

        for _ in 0..ORDER_POLL_ATTEMPTS {
            if let Ok(open_orders) = self.get_open_orders().await
                && let Some(order) = open_orders.into_iter().find(|order| {
                    order.symbol.eq_ignore_ascii_case(symbol)
                        && side_matches(&order.side, side)
                        && is_new_open_order(order, baseline, request_started_at)
                })
            {
                return Some(OctobotOrderResult {
                    order_id: order.id,
                    symbol: if order.symbol.is_empty() {
                        symbol.to_string()
                    } else {
                        order.symbol
                    },
                    side: if order.side.is_empty() {
                        side.to_string()
                    } else {
                        order.side
                    },
                    amount: if order.amount > 0.0 {
                        order.amount
                    } else {
                        amount_usd
                    },
                    price: order.price,
                    status: if order.status.is_empty() {
                        "submitted".to_string()
                    } else {
                        order.status
                    },
                });
            }

            if let Ok(trades) = self.get_trade_history(30).await
                && let Some(trade) = trades.into_iter().find(|trade| {
                    trade.symbol.eq_ignore_ascii_case(symbol)
                        && side_matches(&trade.side, side)
                        && is_new_trade(trade, baseline, request_started_at)
                })
            {
                let ts = trade.timestamp.unwrap_or_else(current_unix_timestamp_f64);
                return Some(OctobotOrderResult {
                    order_id: trade.id.unwrap_or_else(|| format!("filled-{ts:.3}")),
                    symbol: if trade.symbol.is_empty() {
                        symbol.to_string()
                    } else {
                        trade.symbol
                    },
                    side: if trade.side.is_empty() {
                        side.to_string()
                    } else {
                        trade.side
                    },
                    amount: if trade.amount > 0.0 {
                        trade.amount
                    } else {
                        amount_usd
                    },
                    price: Some(trade.price),
                    status: "filled".to_string(),
                });
            }

            sleep(Duration::from_millis(ORDER_POLL_DELAY_MS)).await;
        }

        let _ = exchange;
        None
    }

    // -----------------------------------------------------------------------
    // Exchanges and symbols
    // -----------------------------------------------------------------------

    pub async fn get_exchange_info(&self) -> Result<Vec<OctobotExchange>, String> {
        let mut exchanges_by_key: HashMap<String, OctobotExchange> = HashMap::new();
        let mut exchange_ids_by_key: HashMap<String, String> = HashMap::new();
        let mut trading_symbols_by_exchange: HashMap<String, HashSet<String>> = HashMap::new();

        if let Some(exchange_body) = self
            .get_optional_json("/api/first_exchange_details", "first exchange details")
            .await?
        {
            let exchange_data = unwrap_octobot_data(&exchange_body);
            if let Some(exchange_name_key) = exchange_data
                .get("exchange_name")
                .or_else(|| exchange_data.get("name"))
                .and_then(Value::as_str)
                .and_then(normalize_exchange_name)
            {
                exchanges_by_key
                    .entry(exchange_name_key.clone())
                    .or_insert_with(|| OctobotExchange {
                        name: exchange_name_key.clone(),
                        enabled: true,
                        symbols: Vec::new(),
                    });
                if let Some(exchange_id) = exchange_data
                    .get("exchange_id")
                    .or_else(|| exchange_data.get("id"))
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                {
                    exchange_ids_by_key.insert(exchange_name_key, exchange_id.to_string());
                }
            }
        }

        if let Some(body) = self
            .get_optional_json("/api/exchanges", "exchanges")
            .await?
        {
            for entry in parse_exchange_info_entries(&body) {
                let Some(exchange_name_key) = normalize_exchange_name(&entry.name) else {
                    continue;
                };
                let exchange = exchanges_by_key
                    .entry(exchange_name_key.clone())
                    .or_insert_with(|| OctobotExchange {
                        name: exchange_name_key.clone(),
                        enabled: entry.enabled,
                        symbols: Vec::new(),
                    });
                exchange.enabled |= entry.enabled;
                if let Some(exchange_id) = entry.exchange_id {
                    exchange_ids_by_key
                        .entry(exchange_name_key.clone())
                        .or_insert(exchange_id);
                }
                for symbol in entry.symbols {
                    trading_symbols_by_exchange
                        .entry(exchange_name_key.clone())
                        .or_default()
                        .insert(symbol);
                }
            }
        }

        if let Some(trading_page_html) = self.get_optional_text("/trading", "trading page").await? {
            for row in parse_trading_page_symbol_status_rows(&trading_page_html) {
                let Some(exchange_name_key) = normalize_exchange_name(&row.exchange_name) else {
                    continue;
                };
                exchanges_by_key
                    .entry(exchange_name_key.clone())
                    .or_insert_with(|| OctobotExchange {
                        name: exchange_name_key.clone(),
                        enabled: true,
                        symbols: Vec::new(),
                    });
                exchange_ids_by_key
                    .entry(exchange_name_key.clone())
                    .or_insert(row.exchange_id);
                if let Some(symbol) = normalize_trading_symbol(&row.symbol) {
                    trading_symbols_by_exchange
                        .entry(exchange_name_key)
                        .or_default()
                        .insert(symbol);
                }
            }
        }

        if exchanges_by_key.is_empty() {
            return Ok(Vec::new());
        }

        for exchange in exchanges_by_key.values_mut() {
            let mut configured_symbols = self
                .configured_symbols(Some(exchange.name.as_str()))
                .await
                .unwrap_or_default();
            configured_symbols.sort();
            configured_symbols.dedup();

            let mut prioritized_trading_symbols = trading_symbols_by_exchange
                .get(&exchange.name)
                .map(|symbols| symbols.iter().cloned().collect::<Vec<_>>())
                .unwrap_or_default();
            prioritized_trading_symbols.sort();
            prioritized_trading_symbols.dedup();

            let mut symbols = Vec::new();
            let mut seen_symbols = HashSet::new();
            for symbol in prioritized_trading_symbols
                .into_iter()
                .chain(configured_symbols.into_iter())
            {
                let symbol_key = symbol.to_ascii_uppercase();
                if seen_symbols.insert(symbol_key) {
                    symbols.push(symbol);
                }
            }
            exchange.symbols = symbols;
        }

        {
            let mut cache = self.exchange_id_by_name.lock().await;
            for (exchange_name_key, exchange_id) in exchange_ids_by_key {
                cache.insert(exchange_name_key, exchange_id);
            }
        }

        let mut exchanges = exchanges_by_key.into_values().collect::<Vec<_>>();
        exchanges.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(exchanges)
    }

    async fn configured_symbols(&self, exchange_name: Option<&str>) -> Result<Vec<String>, String> {
        let body = self
            .get_optional_json("/api/get_config_currency", "configured currencies")
            .await?
            .unwrap_or(Value::Null);
        let mut symbols = Vec::new();
        if let Some(config) = body.as_object() {
            for data in config.values() {
                let enabled = data.get("enabled").and_then(Value::as_bool).unwrap_or(true);
                if !enabled {
                    continue;
                }
                if let Some(pairs) = data
                    .get("pairs")
                    .or_else(|| data.get("crypto-pairs"))
                    .and_then(Value::as_array)
                {
                    symbols.extend(
                        pairs
                            .iter()
                            .filter_map(Value::as_str)
                            .filter_map(normalize_trading_symbol),
                    );
                }
            }
        }
        symbols.sort();
        symbols.dedup();

        // Enrich configured pairs with exchange-listed symbols to broaden
        // candidate discovery beyond the currently watched subset.
        if let Some(exchange_name) = exchange_name {
            let configured_quotes: HashSet<String> = symbols
                .iter()
                .filter_map(|symbol| symbol_quote_asset(symbol))
                .map(str::to_string)
                .collect();

            let mut discovered_symbols = match self.exchange_symbols(exchange_name).await {
                Ok(symbols) => symbols,
                Err(err) => {
                    debug!(
                        "trading: exchange symbol enrichment failed for {}: {}",
                        exchange_name, err
                    );
                    Vec::new()
                }
            };

            if !configured_quotes.is_empty() {
                discovered_symbols.retain(|symbol| {
                    symbol_quote_asset(symbol)
                        .is_some_and(|quote| configured_quotes.contains(quote))
                });
            }

            let mut seen_symbols: HashSet<String> = symbols
                .iter()
                .map(|symbol| symbol.to_ascii_uppercase())
                .collect();
            for symbol in discovered_symbols {
                if seen_symbols.insert(symbol.to_ascii_uppercase()) {
                    symbols.push(symbol);
                }
            }
        }

        // Fallback order is intentionally progressive:
        // 1) configured pairs
        // 2) exchange market universe
        // 3) generic currency list
        // 4) dashboard first symbol
        //
        // This keeps symbols useful across OctoBot editions where any one
        // endpoint can be disabled, empty, or unavailable.
        if symbols.is_empty() {
            symbols.extend(self.currency_list_symbols().await?);
        }
        if symbols.is_empty()
            && let Some(first) = self
                .get_optional_json("/dashboard/first_symbol", "first symbol")
                .await?
            && let Some(symbol) = first
                .get("symbol")
                .and_then(Value::as_str)
                .and_then(normalize_trading_symbol)
        {
            symbols.push(symbol);
        }
        Ok(symbols)
    }

    async fn exchange_symbols(&self, exchange_name: &str) -> Result<Vec<String>, String> {
        let exchange_name = exchange_name.trim();
        if exchange_name.is_empty() {
            return Ok(Vec::new());
        }
        let path = format!("/api/get_all_symbols/{exchange_name}");
        let mut symbols = self
            .get_optional_json(&path, "exchange symbols")
            .await?
            .map(|body| parse_trading_symbol_candidates(&body))
            .unwrap_or_default();
        symbols.sort();
        symbols.dedup();
        Ok(symbols)
    }

    async fn currency_list_symbols(&self) -> Result<Vec<String>, String> {
        let mut symbols = self
            .get_optional_json("/api/currency_list", "currency list")
            .await?
            .map(|body| parse_trading_symbol_candidates(&body))
            .unwrap_or_default();
        symbols.sort();
        symbols.dedup();
        Ok(symbols)
    }

    /// Fetch market data (ticker) for a symbol on an exchange.
    pub async fn get_market_snapshot(
        &self,
        exchange: &str,
        symbol: &str,
    ) -> Result<MarketSnapshot, String> {
        // Prefer dashboard routes in current OctoBot builds.
        let web_symbol = symbol.replace('/', "|");
        if let Some((exchange_id, time_frame)) =
            self.watched_symbol_context(exchange, symbol).await?
        {
            let graph_path = format!(
                "/dashboard/currency_price_graph_update/{exchange_id}/{web_symbol}/{time_frame}/live?display_orders=false"
            );
            if let Some(graph) = self.get_optional_json(&graph_path, "price graph").await? {
                if let Some(snapshot) = parse_graph_snapshot(exchange, symbol, &graph) {
                    return Ok(snapshot);
                }
                if let Some(error) = graph_error_message(&graph) {
                    debug!(
                        "trading: dashboard graph had no usable data for {}/{}: {}",
                        exchange, symbol, error
                    );
                }
            }
        }

        // Legacy fallback.
        let legacy_path = format!("/api/market/ticker?exchange={exchange}&symbol={symbol}");
        if let Some(body) = self.get_optional_json(&legacy_path, "ticker").await? {
            return Ok(parse_ticker_snapshot(exchange, symbol, &body));
        }

        Err(format!(
            "OctoBot market snapshot unavailable for {exchange}/{symbol}: no dashboard or ticker endpoint returned data"
        ))
    }

    /// Fetch historical dashboard candle snapshots for a symbol/time frame.
    ///
    /// This is used for one-time datalake bootstrap/backfill after a new
    /// container build or schema change.
    pub async fn get_market_snapshot_history(
        &self,
        exchange: &str,
        symbol: &str,
        time_frame: &str,
    ) -> Result<Vec<MarketSnapshot>, String> {
        let web_symbol = symbol.replace('/', "|");
        let Some((exchange_id, default_time_frame)) =
            self.watched_symbol_context(exchange, symbol).await?
        else {
            return Ok(Vec::new());
        };
        let resolved_time_frame = if time_frame.trim().is_empty() {
            default_time_frame
        } else {
            time_frame.trim().to_string()
        };
        let graph_path = format!(
            "/dashboard/currency_price_graph_update/{exchange_id}/{web_symbol}/{resolved_time_frame}/history?display_orders=false"
        );
        let Some(graph) = self
            .get_optional_json(&graph_path, "price graph history")
            .await?
        else {
            return Ok(Vec::new());
        };
        Ok(parse_graph_history_snapshots(
            exchange,
            symbol,
            &resolved_time_frame,
            &graph,
        ))
    }

    async fn watched_symbol_context(
        &self,
        exchange: &str,
        symbol: &str,
    ) -> Result<Option<(String, String)>, String> {
        let exchange_id_hint = self.exchange_id_for_exchange_name(exchange).await?;
        let web_symbol = symbol.replace('/', "|");
        let watched_path = format!("/dashboard/watched_symbol/{web_symbol}");
        let mut watched_exchange_id = None;
        let mut time_frame = "1h".to_string();

        if let Some(watched) = self
            .get_optional_json(&watched_path, "watched symbol")
            .await?
        {
            watched_exchange_id = watched
                .get("exchange_id")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string);
            time_frame = watched
                .get("time_frame")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .unwrap_or("1h")
                .to_string();
        }

        let Some(exchange_id) = exchange_id_hint.or(watched_exchange_id) else {
            return Ok(None);
        };
        Ok(Some((exchange_id, time_frame)))
    }

    async fn exchange_id_for_exchange_name(
        &self,
        exchange_name: &str,
    ) -> Result<Option<String>, String> {
        let Some(exchange_name_key) = normalize_exchange_name(exchange_name) else {
            return Ok(None);
        };

        if let Some(exchange_id) = self
            .exchange_id_by_name
            .lock()
            .await
            .get(&exchange_name_key)
            .cloned()
        {
            return Ok(Some(exchange_id));
        }

        let _ = self.get_exchange_info().await?;
        Ok(self
            .exchange_id_by_name
            .lock()
            .await
            .get(&exchange_name_key)
            .cloned())
    }

    pub async fn get_recent_logs(&self, limit: usize) -> Result<Vec<OctobotLogEntry>, String> {
        let limit = limit.clamp(1, 1000);
        let candidate_paths = [
            format!("/logs?format=json&limit={limit}"),
            "/logs".to_string(),
            format!("/api/logs?limit={limit}"),
        ];
        let mut last_error = None;
        for path in candidate_paths {
            match self.get_optional_text(&path, "logs").await {
                Ok(Some(text)) => {
                    let logs = parse_octobot_logs(&text, limit);
                    if !logs.is_empty() {
                        self.observe_log_entries(&logs).await;
                        return Ok(logs);
                    }
                }
                Ok(None) => {}
                Err(err) => last_error = Some(err),
            }
        }
        if let Some(err) = last_error {
            Err(err)
        } else {
            Ok(Vec::new())
        }
    }

    pub async fn observe_log_entries(&self, logs: &[OctobotLogEntry]) {
        for entry in logs {
            {
                let mut schema = self.api_schema.lock().await;
                schema.observe_log_entry(&entry.level, &entry.source, &entry.message);
            }
            adaptive_schema::observe_log_entry(
                OCTOBOT_API,
                &entry.level,
                &entry.source,
                &entry.message,
            )
            .await;
        }
    }
}

#[derive(Clone, Debug, Default)]
struct OrderPlacementBaseline {
    captured_at: f64,
    open_order_ids: HashSet<String>,
    open_orders_captured: bool,
    trade_ids: HashSet<String>,
    latest_trade_ts: Option<f64>,
    trades_captured: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OctobotOrderSubmissionMode {
    DirectCreateOrder,
    DirectCreateOrders,
    DirectOrdersActionBody,
    DirectOrdersCanonicalBody,
    UserCommandTrading,
    UserCommandGailTrading,
    UserCommandTradingBridge,
}

#[derive(Debug)]
enum OrderSubmissionAttempt {
    /// The endpoint positively acknowledged the order, either directly in the
    /// response body or by producing an observable order/trade side-effect.
    Accepted(OctobotOrderResult),

    /// The endpoint clearly rejected the order, usually via non-2xx HTTP.
    Rejected,

    /// The endpoint rejected a sell request with a portfolio precision/
    /// availability error. This is safe to retry once with a smaller amount.
    RejectedPortfolioNegative { retry_amount_usd: Option<f64> },

    /// The endpoint rejected the order because quantity adaptation collapsed
    /// to a non-positive value. Retrying equivalent endpoints is unnecessary.
    RejectedNonPositiveQuantity,

    /// The endpoint returned HTTP 2xx but did not provide a parseable order
    /// acknowledgement and no side-effect was observed within the polling
    /// window.
    ///
    /// This is intentionally treated as unsafe to continue, because retrying
    /// another mutating endpoint could create a duplicate order.
    AmbiguousAccepted,
}

fn default_order_submission_mode_candidates() -> Vec<OctobotOrderSubmissionMode> {
    vec![
        // Prefer native trading endpoints first: they synchronously return
        // explicit success/failure payloads for order creation.
        OctobotOrderSubmissionMode::DirectCreateOrder,
        OctobotOrderSubmissionMode::DirectCreateOrders,
        OctobotOrderSubmissionMode::DirectOrdersActionBody,
        OctobotOrderSubmissionMode::DirectOrdersCanonicalBody,
        // `/api/user_command` is asynchronous and often returns only an echo
        // payload with HTTP 200, so treat it as a fallback path.
        OctobotOrderSubmissionMode::UserCommandTrading,
        OctobotOrderSubmissionMode::UserCommandGailTrading,
        OctobotOrderSubmissionMode::UserCommandTradingBridge,
    ]
}

fn ordered_submission_modes(
    preferred: Option<OctobotOrderSubmissionMode>,
) -> Vec<OctobotOrderSubmissionMode> {
    let mut modes = Vec::new();

    if let Some(mode) = preferred {
        modes.push(mode);
    }

    for mode in default_order_submission_mode_candidates() {
        if Some(mode) != preferred {
            modes.push(mode);
        }
    }

    modes
}

fn build_order_submission(
    mode: OctobotOrderSubmissionMode,
    exchange: &str,
    symbol: &str,
    side: &str,
    rounded_amount_usd: f64,
    canonical_order: &Value,
) -> (&'static str, Value, &'static str) {
    match mode {
        OctobotOrderSubmissionMode::DirectCreateOrder => (
            "/api/orders?action=create_order",
            canonical_order.clone(),
            "create order",
        ),

        OctobotOrderSubmissionMode::DirectCreateOrders => (
            "/api/orders?action=create_orders",
            json!([canonical_order.clone()]),
            "create orders",
        ),

        OctobotOrderSubmissionMode::DirectOrdersActionBody => (
            "/api/orders",
            json!({
                "action": "create_order",
                "exchange": exchange,
                "symbol": symbol,
                "side": side,
                "type": "market",
                "amount": rounded_amount_usd,
                "amount_usd": rounded_amount_usd,
            }),
            "create order",
        ),

        OctobotOrderSubmissionMode::DirectOrdersCanonicalBody => {
            ("/api/orders", canonical_order.clone(), "create order")
        }

        OctobotOrderSubmissionMode::UserCommandTrading => (
            "/api/user_command",
            json!({
                "subject": "trading",
                "action": "create_order",
                "data": canonical_order,
            }),
            "user command order",
        ),

        OctobotOrderSubmissionMode::UserCommandGailTrading => (
            "/api/user_command",
            json!({
                "subject": "gail_trading",
                "action": "create_order",
                "data": {
                    "exchange": exchange,
                    "symbol": symbol,
                    "side": side,
                    "amount_usd": rounded_amount_usd,
                    "type": "market",
                },
            }),
            "user command order",
        ),

        OctobotOrderSubmissionMode::UserCommandTradingBridge => (
            "/api/user_command",
            json!({
                "subject": "trading_bridge",
                "action": "create_order",
                "data": {
                    "exchange": exchange,
                    "symbol": symbol,
                    "side": side,
                    "amount": rounded_amount_usd,
                    "order_type": "market",
                },
            }),
            "user command order",
        ),
    }
}

fn normalize_order_side(raw: &str) -> Option<&'static str> {
    let lowered = raw.to_ascii_lowercase();
    if lowered.contains("buy") {
        Some("buy")
    } else if lowered.contains("sell") {
        Some("sell")
    } else {
        None
    }
}

fn side_matches(candidate: &str, expected: &str) -> bool {
    normalize_order_side(candidate) == normalize_order_side(expected)
}

fn is_new_open_order(
    order: &OctobotOrder,
    baseline: &OrderPlacementBaseline,
    request_started_at: f64,
) -> bool {
    if baseline.open_orders_captured {
        return !baseline.open_order_ids.contains(&order.id);
    }
    order
        .timestamp
        .is_some_and(|ts| ts + 1.0 >= baseline.captured_at || ts + 1.0 >= request_started_at)
}

fn is_new_trade(
    trade: &OctobotTrade,
    baseline: &OrderPlacementBaseline,
    request_started_at: f64,
) -> bool {
    if let Some(id) = trade.id.as_deref().filter(|value| !value.trim().is_empty()) {
        if baseline.trades_captured {
            return !baseline.trade_ids.contains(id);
        }
        return true;
    }

    if baseline.trades_captured {
        return match (trade.timestamp, baseline.latest_trade_ts) {
            (Some(ts), Some(last_ts)) => ts > last_ts + 0.001,
            (Some(_), None) => true,
            _ => false,
        };
    }

    trade
        .timestamp
        .is_some_and(|ts| ts + 1.0 >= baseline.captured_at || ts + 1.0 >= request_started_at)
}

fn summarize_order_attempt_body(body: &Value) -> String {
    match body {
        Value::Null => "empty-body".to_string(),
        Value::String(text) => {
            let compact = text.replace('\n', " ").trim().to_string();
            if compact.is_empty() {
                "empty-string".to_string()
            } else {
                format!("message={}", compact.chars().take(120).collect::<String>())
            }
        }
        Value::Object(object) => {
            if let Some(message) = extract_attempt_message(body) {
                let compact = message.replace('\n', " ").trim().to_string();
                if !compact.is_empty() {
                    return format!("message={}", compact.chars().take(120).collect::<String>());
                }
            }
            if object.is_empty() {
                "empty-object".to_string()
            } else {
                let keys = object.keys().take(6).cloned().collect::<Vec<_>>().join(",");
                format!("keys={keys}")
            }
        }
        Value::Array(array) => format!("array(len={})", array.len()),
        _ => format!("{}-body", body.as_str().unwrap_or("scalar")),
    }
}

fn extract_attempt_message(body: &Value) -> Option<String> {
    match body {
        Value::String(text) => Some(text.clone()),
        Value::Object(object) => {
            for key in ["message", "error", "detail", "details", "reason"] {
                if let Some(value) = object.get(key)
                    && let Some(message) = extract_attempt_message(value)
                    && !message.trim().is_empty()
                {
                    return Some(message);
                }
            }
            None
        }
        Value::Array(items) => items.iter().find_map(extract_attempt_message),
        _ => None,
    }
}

fn body_has_portfolio_negative_rejection(body: &Value) -> bool {
    extract_attempt_message(body).is_some_and(|message| {
        let lowered = message.to_ascii_lowercase();
        lowered.contains("portfolionegativevalueerror")
            || (lowered.contains("trying to update") && lowered.contains("quantity was"))
    })
}

fn body_has_non_positive_quantity_rejection(body: &Value) -> bool {
    extract_attempt_message(body).is_some_and(|message| {
        let lowered = message.to_ascii_lowercase();
        lowered.contains("order quantity became non-positive after market adaptation")
            || (lowered.contains("quantity became non-positive")
                && lowered.contains("market adaptation"))
    })
}

fn retry_amount_from_portfolio_negative_rejection(
    body: &Value,
    current_amount_usd: f64,
) -> Option<f64> {
    extract_attempt_message(body)
        .and_then(|message| {
            retry_amount_from_portfolio_negative_message(message.as_str(), current_amount_usd)
        })
        .or_else(|| reduced_sell_retry_amount(current_amount_usd))
}

fn retry_amount_from_portfolio_negative_message(
    message: &str,
    current_amount_usd: f64,
) -> Option<f64> {
    static PORTFOLIO_NEGATIVE_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(
            r"(?i)trying to update\s+\S+\s+with\s+(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+but quantity was\s+(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
        )
        .expect("valid portfolio negative regex")
    });

    let captures = PORTFOLIO_NEGATIVE_RE.captures(message)?;
    let attempted_quantity = captures.get(1)?.as_str().parse::<f64>().ok()?.abs();
    let available_quantity = captures.get(2)?.as_str().parse::<f64>().ok()?.abs();

    if !attempted_quantity.is_finite()
        || !available_quantity.is_finite()
        || attempted_quantity <= 0.0
        || available_quantity <= 0.0
    {
        return None;
    }

    if available_quantity + f64::EPSILON >= attempted_quantity {
        return None;
    }

    let scaled_amount =
        floor_usd_to_cents(current_amount_usd * (available_quantity / attempted_quantity) * 0.995);
    if scaled_amount >= 0.01 && scaled_amount + f64::EPSILON < current_amount_usd {
        Some(scaled_amount)
    } else {
        None
    }
}

fn floor_usd_to_cents(value: f64) -> f64 {
    if !value.is_finite() {
        return 0.0;
    }
    ((value.max(0.0)) * 100.0).floor() / 100.0
}

fn reduced_sell_retry_amount(current_amount_usd: f64) -> Option<f64> {
    if !current_amount_usd.is_finite() || current_amount_usd <= 0.01 {
        return None;
    }

    let reduced = ((current_amount_usd * 0.95) * 100.0).round() / 100.0;
    if reduced >= 0.01 && reduced + f64::EPSILON < current_amount_usd {
        Some(reduced)
    } else {
        None
    }
}

fn parse_order_result_body(
    body: &Value,
    default_exchange: &str,
    default_symbol: &str,
    default_side: &str,
    default_amount: f64,
    source_hint: &str,
) -> Option<OctobotOrderResult> {
    match body {
        Value::Array(items) => items.iter().find_map(|entry| {
            parse_order_result_body(
                entry,
                default_exchange,
                default_symbol,
                default_side,
                default_amount,
                source_hint,
            )
        }),
        Value::Object(object) => parse_order_result_object(
            object,
            default_exchange,
            default_symbol,
            default_side,
            default_amount,
            source_hint,
        ),
        _ => None,
    }
}

fn parse_order_result_object(
    object: &serde_json::Map<String, Value>,
    default_exchange: &str,
    default_symbol: &str,
    default_side: &str,
    default_amount: f64,
    source_hint: &str,
) -> Option<OctobotOrderResult> {
    for nested_key in [
        "order",
        "result",
        "data",
        "created_order",
        "created_orders",
        "payload",
    ] {
        if let Some(nested) = object.get(nested_key)
            && let Some(result) = parse_order_result_body(
                nested,
                default_exchange,
                default_symbol,
                default_side,
                default_amount,
                source_hint,
            )
        {
            return Some(result);
        }
    }

    if let Some(entries) = object.get("orders").and_then(Value::as_array)
        && let Some(result) = entries.iter().find_map(|entry| {
            parse_order_result_body(
                entry,
                default_exchange,
                default_symbol,
                default_side,
                default_amount,
                source_hint,
            )
        })
    {
        return Some(result);
    }

    let order_id = order_result_string_field(object, &["order_id", "id", "exchange_order_id"])?;
    let symbol = order_result_string_field(object, &["symbol", "pair", "market"])
        .map(|value| value.replace('|', "/"))
        .unwrap_or_else(|| default_symbol.to_string());
    let side = order_result_string_field(object, &["side", "order_side", "type"])
        .and_then(|value| normalize_order_side(&value).map(str::to_string))
        .unwrap_or_else(|| default_side.to_string());
    let amount = order_result_numeric_field(
        object,
        &["amount", "quantity", "size", "cost", "amount_usd"],
    )
    .unwrap_or(default_amount)
    .max(0.0);
    let price = order_result_numeric_field(
        object,
        &["price", "avg_price", "average_price", "filled_price"],
    );
    let status = order_result_string_field(object, &["status", "state", "result"])
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| format!("submitted via {}", source_hint));

    Some(OctobotOrderResult {
        order_id,
        symbol,
        side,
        amount,
        price,
        status,
    })
}

fn order_result_string_field(
    object: &serde_json::Map<String, Value>,
    keys: &[&str],
) -> Option<String> {
    keys.iter().find_map(|key| {
        object.get(*key).and_then(|value| {
            value
                .as_str()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
        })
    })
}

fn order_result_numeric_field(
    object: &serde_json::Map<String, Value>,
    keys: &[&str],
) -> Option<f64> {
    keys.iter().find_map(|key| {
        object.get(*key).and_then(|value| {
            value.as_f64().or_else(|| {
                value
                    .as_str()
                    .and_then(|text| text.trim().parse::<f64>().ok())
            })
        })
    })
}

fn parse_exchange_info_entries(body: &Value) -> Vec<ExchangeInfoEntry> {
    let mut entries = Vec::new();

    if let Some(array) = body
        .as_array()
        .or_else(|| body.get("exchanges").and_then(Value::as_array))
        .or_else(|| unwrap_octobot_data(body).as_array())
    {
        for entry in array {
            if let Some(parsed) = parse_exchange_info_entry(entry, None) {
                entries.push(parsed);
            }
        }
        return entries;
    }

    if let Some(object) = body
        .as_object()
        .or_else(|| unwrap_octobot_data(body).as_object())
    {
        for (fallback_name, entry) in object {
            if let Some(parsed) = parse_exchange_info_entry(entry, Some(fallback_name)) {
                entries.push(parsed);
            }
        }
    }

    entries
}

fn parse_exchange_info_entry(
    value: &Value,
    fallback_name: Option<&str>,
) -> Option<ExchangeInfoEntry> {
    let name = value
        .get("name")
        .or_else(|| value.get("exchange_name"))
        .and_then(Value::as_str)
        .or(fallback_name)
        .map(str::trim)
        .filter(|value| !value.is_empty())?
        .to_string();
    let enabled = value
        .get("enabled")
        .or_else(|| value.get("is_enabled"))
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let mut symbols = value
        .get("symbols")
        .or_else(|| value.get("markets"))
        .map(parse_trading_symbol_candidates)
        .unwrap_or_default();
    if symbols.is_empty() {
        symbols = parse_trading_symbol_candidates(value);
    }
    symbols.sort();
    symbols.dedup();

    let exchange_id = value
        .get("exchange_id")
        .or_else(|| value.get("exchangeId"))
        .or_else(|| value.get("uuid"))
        .or_else(|| value.get("id"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);

    Some(ExchangeInfoEntry {
        name,
        enabled,
        symbols,
        exchange_id,
    })
}

fn normalize_trading_symbol(raw: &str) -> Option<String> {
    let candidate = raw.trim().replace('|', "/");
    if candidate.is_empty() || !is_supported_spot_symbol(&candidate) {
        return None;
    }
    Some(candidate)
}

fn parse_trading_symbol_candidates(body: &Value) -> Vec<String> {
    let mut symbols = Vec::new();
    collect_trading_symbol_candidates(body, &mut symbols);
    symbols.sort();
    symbols.dedup();
    symbols
}

fn collect_trading_symbol_candidates(value: &Value, symbols: &mut Vec<String>) {
    match value {
        Value::String(text) => {
            if let Some(symbol) = normalize_trading_symbol(text) {
                symbols.push(symbol);
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_trading_symbol_candidates(item, symbols);
            }
        }
        Value::Object(object) => {
            for key in ["symbol", "symbols", "pair", "pairs", "market", "markets"] {
                if let Some(candidate) = object.get(key) {
                    collect_trading_symbol_candidates(candidate, symbols);
                }
            }
        }
        _ => {}
    }
}

fn unwrap_octobot_data(body: &Value) -> &Value {
    body.get("data").unwrap_or(body)
}

fn parse_ticker_snapshot(exchange: &str, symbol: &str, body: &Value) -> MarketSnapshot {
    let now = current_unix_timestamp_f64();
    MarketSnapshot {
        exchange: exchange.to_string(),
        symbol: symbol.to_string(),
        price: body
            .get("last")
            .or_else(|| body.get("close"))
            .and_then(Value::as_f64)
            .unwrap_or(0.0),
        price_change_pct_1h: body.get("change_1h").and_then(Value::as_f64),
        price_change_pct_24h: body
            .get("change_24h")
            .or_else(|| body.get("percentage"))
            .and_then(Value::as_f64),
        volume_24h: body
            .get("baseVolume")
            .or_else(|| body.get("volume"))
            .and_then(Value::as_f64),
        volume_change_pct: body.get("volume_change_pct").and_then(Value::as_f64),
        high_24h: body.get("high").and_then(Value::as_f64),
        low_24h: body.get("low").and_then(Value::as_f64),
        fetched_at: now,
    }
}

fn parse_graph_snapshot(exchange: &str, symbol: &str, body: &Value) -> Option<MarketSnapshot> {
    let candles = body.get("candles").unwrap_or(&Value::Null);
    let closes = value_array(candles, "close");
    if closes.as_ref().is_none_or(Vec::is_empty) {
        return None;
    }
    let highs = value_array(candles, "high");
    let lows = value_array(candles, "low");
    let volumes = value_array(candles, "volume").or_else(|| value_array(candles, "vol"));
    let price = closes
        .as_ref()
        .and_then(|values| values.last().copied())
        .unwrap_or(0.0);
    let first_close = closes.as_ref().and_then(|values| values.first().copied());
    let price_change_pct_24h = first_close
        .filter(|first| first.abs() > f64::EPSILON)
        .map(|first| ((price - first) / first) * 100.0);
    Some(MarketSnapshot {
        exchange: exchange.to_string(),
        symbol: symbol.to_string(),
        price,
        price_change_pct_1h: None,
        price_change_pct_24h,
        volume_24h: volumes.as_ref().map(|items| items.iter().sum()),
        volume_change_pct: None,
        high_24h: highs
            .as_ref()
            .and_then(|items| items.iter().copied().reduce(f64::max)),
        low_24h: lows
            .as_ref()
            .and_then(|items| items.iter().copied().reduce(f64::min)),
        fetched_at: current_unix_timestamp_f64(),
    })
}

fn parse_graph_history_snapshots(
    exchange: &str,
    symbol: &str,
    time_frame: &str,
    body: &Value,
) -> Vec<MarketSnapshot> {
    let candles = body.get("candles").unwrap_or(&Value::Null);
    let closes = value_array(candles, "close").unwrap_or_default();
    if closes.is_empty() {
        return Vec::new();
    }
    let highs = value_array(candles, "high").unwrap_or_default();
    let lows = value_array(candles, "low").unwrap_or_default();
    let volumes = value_array(candles, "volume")
        .or_else(|| value_array(candles, "vol"))
        .unwrap_or_default();
    let times = value_string_array(candles, "time").unwrap_or_default();
    let fallback_now = current_unix_timestamp_f64();
    let fallback_step = time_frame_to_seconds(time_frame).max(1) as f64;
    let fallback_start = fallback_now - fallback_step * closes.len().saturating_sub(1) as f64;
    let first_close = closes.first().copied();

    let mut snapshots = Vec::with_capacity(closes.len());
    for (idx, close) in closes.iter().copied().enumerate() {
        let fetched_at = times
            .get(idx)
            .and_then(|value| parse_dashboard_time_to_ts(value))
            .unwrap_or(fallback_start + fallback_step * idx as f64);
        let price_change_pct_24h = first_close
            .filter(|first| first.abs() > f64::EPSILON)
            .map(|first| ((close - first) / first) * 100.0);
        snapshots.push(MarketSnapshot {
            exchange: exchange.to_string(),
            symbol: symbol.to_string(),
            price: close,
            price_change_pct_1h: None,
            price_change_pct_24h,
            volume_24h: volumes.get(idx).copied(),
            volume_change_pct: None,
            high_24h: highs.get(idx).copied(),
            low_24h: lows.get(idx).copied(),
            fetched_at,
        });
    }
    snapshots
}

fn value_array(body: &Value, key: &str) -> Option<Vec<f64>> {
    body.get(key).and_then(Value::as_array).map(|values| {
        values
            .iter()
            .filter_map(|value| {
                value
                    .as_f64()
                    .or_else(|| value.as_str().and_then(|text| text.parse::<f64>().ok()))
            })
            .collect::<Vec<_>>()
    })
}

fn split_symbol_assets(symbol: &str) -> Option<(&str, &str)> {
    let (base, quote) = symbol.split_once('/')?;
    if base.contains('/') || quote.contains('/') {
        return None;
    }
    Some((base, quote))
}

fn symbol_base_asset(symbol: &str) -> Option<&str> {
    split_symbol_assets(symbol).map(|(base, _)| base)
}

fn symbol_quote_asset(symbol: &str) -> Option<&str> {
    split_symbol_assets(symbol).map(|(_, quote)| quote)
}

fn configured_currency_pairs(entry: &Value) -> Vec<String> {
    entry
        .get("pairs")
        .or_else(|| entry.get("crypto-pairs"))
        .and_then(Value::as_array)
        .map(|pairs| {
            pairs
                .iter()
                .filter_map(Value::as_str)
                .filter_map(normalize_trading_symbol)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn configured_currency_contains_symbol(entry: &Value, symbol: &str) -> bool {
    configured_currency_pairs(entry)
        .iter()
        .any(|candidate| candidate.eq_ignore_ascii_case(symbol))
}

fn configured_currency_map_contains_symbol(
    configured: &serde_json::Map<String, Value>,
    symbol: &str,
) -> bool {
    configured
        .values()
        .filter(|entry| {
            entry
                .get("enabled")
                .and_then(Value::as_bool)
                .unwrap_or(true)
        })
        .any(|entry| configured_currency_contains_symbol(entry, symbol))
}

fn best_currency_key_for_symbol(
    configured: &serde_json::Map<String, Value>,
    symbol: &str,
) -> Option<String> {
    let base_asset = symbol_base_asset(symbol)?;

    for (currency, entry) in configured {
        if configured_currency_pairs(entry).iter().any(|pair| {
            symbol_base_asset(pair)
                .is_some_and(|configured_base| configured_base.eq_ignore_ascii_case(base_asset))
        }) {
            return Some(currency.clone());
        }
    }

    configured
        .keys()
        .find(|currency| currency.eq_ignore_ascii_case(base_asset))
        .cloned()
}

fn trading_pair_restart_key(exchange: &str, symbol: &str) -> String {
    format!(
        "{}|{}",
        exchange.to_ascii_lowercase(),
        symbol.to_ascii_uppercase()
    )
}

fn is_supported_spot_symbol(symbol: &str) -> bool {
    let Some((base, quote)) = split_symbol_assets(symbol) else {
        return false;
    };
    if base.is_empty() || quote.is_empty() || quote.len() < 3 {
        return false;
    }
    base.chars().all(|ch| ch.is_ascii_alphanumeric())
        && quote.chars().all(|ch| ch.is_ascii_alphanumeric())
}

fn graph_error_message(body: &Value) -> Option<String> {
    body.get("error")
        .or_else(|| body.get("message"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|message| !message.is_empty())
        .map(str::to_string)
}

fn value_string_array(body: &Value, key: &str) -> Option<Vec<String>> {
    body.get(key).and_then(Value::as_array).map(|values| {
        values
            .iter()
            .filter_map(|value| value.as_str().map(str::to_string))
            .collect::<Vec<_>>()
    })
}

fn parse_dashboard_time_to_ts(value: &str) -> Option<f64> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Ok(timestamp) = trimmed.parse::<f64>() {
        return Some(timestamp);
    }
    let mut parts = trimmed.split(' ');
    let date = parts.next()?;
    let time = parts.next()?;
    if parts.next().is_some() {
        return None;
    }
    let mut d = date.split('-');
    let year_short = d.next()?.parse::<i64>().ok()?;
    let month = d.next()?.parse::<u32>().ok()?;
    let day = d.next()?.parse::<u32>().ok()?;
    if d.next().is_some() {
        return None;
    }
    let mut t = time.split(':');
    let hour = t.next()?.parse::<u32>().ok()?;
    let minute = t.next()?.parse::<u32>().ok()?;
    let second = t.next()?.parse::<u32>().ok()?;
    if t.next().is_some() {
        return None;
    }
    let year = 2000 + year_short;
    unix_timestamp_from_ymd_hms(year, month, day, hour, minute, second)
}

fn unix_timestamp_from_ymd_hms(
    year: i64,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
) -> Option<f64> {
    if !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || hour > 23
        || minute > 59
        || second > 59
    {
        return None;
    }
    let y = year - if month <= 2 { 1 } else { 0 };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let month_i = month as i64;
    let day_i = day as i64;
    let doy = (153 * (month_i + if month_i > 2 { -3 } else { 9 }) + 2) / 5 + day_i - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days_since_epoch = era * 146_097 + doe - 719_468;
    let seconds_of_day = hour as i64 * 3600 + minute as i64 * 60 + second as i64;
    Some((days_since_epoch * 86_400 + seconds_of_day) as f64)
}

fn time_frame_to_seconds(time_frame: &str) -> u64 {
    let trimmed = time_frame.trim();
    if trimmed.len() < 2 {
        return 3_600;
    }
    let (value, unit) = trimmed.split_at(trimmed.len() - 1);
    let amount = value.parse::<u64>().unwrap_or(1).max(1);
    match unit.to_ascii_lowercase().as_str() {
        "m" => amount.saturating_mul(60),
        "h" => amount.saturating_mul(3_600),
        "d" => amount.saturating_mul(86_400),
        "w" => amount.saturating_mul(7 * 86_400),
        _ => 3_600,
    }
}

fn current_unix_timestamp_f64() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn parse_octobot_logs(raw: &str, limit: usize) -> Vec<OctobotLogEntry> {
    if let Ok(value) = serde_json::from_str::<Value>(raw) {
        let mut logs = parse_octobot_log_value(&value);
        logs.truncate(limit);
        if !logs.is_empty() {
            return logs;
        }
    }

    let text = html_table_to_text(raw);
    let mut logs = Vec::new();
    for line in text.lines() {
        if logs.len() >= limit {
            break;
        }
        let cols = line
            .split('\t')
            .map(str::trim)
            .filter(|item| !item.is_empty())
            .collect::<Vec<_>>();
        if cols.len() >= 4 && looks_like_log_time(cols[0]) {
            logs.push(OctobotLogEntry {
                time: Some(cols[0].to_string()),
                level: cols[1].to_string(),
                source: cols[2].to_string(),
                message: cols[3..].join(" "),
            });
        }
    }
    logs
}

fn parse_octobot_log_value(value: &Value) -> Vec<OctobotLogEntry> {
    let entries = value
        .as_array()
        .or_else(|| value.get("logs").and_then(Value::as_array))
        .or_else(|| value.get("data").and_then(Value::as_array));
    let Some(entries) = entries else {
        return Vec::new();
    };
    entries
        .iter()
        .filter_map(|entry| {
            let object = entry.as_object()?;
            let message = object
                .get("message")
                .or_else(|| object.get("msg"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim();
            if message.is_empty() {
                return None;
            }
            Some(OctobotLogEntry {
                time: object
                    .get("time")
                    .or_else(|| object.get("timestamp"))
                    .or_else(|| object.get("date"))
                    .and_then(|value| {
                        value
                            .as_str()
                            .map(str::to_string)
                            .or_else(|| value.as_f64().map(|number| number.to_string()))
                    }),
                level: object
                    .get("level")
                    .and_then(Value::as_str)
                    .unwrap_or("INFO")
                    .to_string(),
                source: object
                    .get("source")
                    .or_else(|| object.get("logger"))
                    .and_then(Value::as_str)
                    .unwrap_or("OctoBot")
                    .to_string(),
                message: message.to_string(),
            })
        })
        .collect()
}

fn html_table_to_text(raw: &str) -> String {
    let mut output = String::new();
    let mut chars = raw.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '<' {
            output.push(ch);
            continue;
        }
        let mut tag = String::new();
        for tag_ch in chars.by_ref() {
            if tag_ch == '>' {
                break;
            }
            tag.push(tag_ch);
        }
        let tag_name = tag
            .trim()
            .trim_start_matches('/')
            .split_whitespace()
            .next()
            .unwrap_or_default()
            .to_ascii_lowercase();
        if tag_name == "td" || tag_name == "th" {
            output.push('\t');
        } else if tag_name == "tr" && tag.trim_start().starts_with('/') {
            output.push('\n');
        }
    }
    decode_basic_html_entities(&output)
}

fn decode_basic_html_entities(value: &str) -> String {
    value
        .replace("&#10;", "\n")
        .replace("&#xA;", "\n")
        .replace("&#xa;", "\n")
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
}

fn looks_like_log_time(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() >= 19
        && bytes
            .get(0..4)
            .is_some_and(|year| year.iter().all(|item| item.is_ascii_digit()))
        && bytes.get(4) == Some(&b'-')
        && bytes.get(7) == Some(&b'-')
        && bytes.get(10) == Some(&b' ')
        && bytes.get(13) == Some(&b':')
        && bytes.get(16) == Some(&b':')
}

fn parse_latest_portfolio_value(body: &Value) -> Option<f64> {
    body.as_array()
        .and_then(|items| items.last())
        .and_then(|item| item.get("value"))
        .and_then(|value| {
            value
                .as_f64()
                .or_else(|| value.as_str().and_then(|text| text.parse::<f64>().ok()))
        })
}

fn parse_portfolio_json(body: &Value) -> OctobotPortfolio {
    let mut portfolio = OctobotPortfolio::default();
    let Some(entries) = body.as_object() else {
        return portfolio;
    };

    for (top_level_key, data) in entries {
        if top_level_key.eq_ignore_ascii_case("total_value_usd") {
            portfolio.total_value_usd = json_f64(data);
            continue;
        }

        if let Some(balance) = parse_currency_balance(data) {
            // Currency-first payload:
            // {
            //   "DOGE": {"free": 10, "locked": 0, "total": 10, "exchanges": {...}}
            // }
            merge_currency_balance(&mut portfolio.currencies, top_level_key, &balance);
            collect_exchange_balances_for_currency(
                &mut portfolio.exchange_currencies,
                top_level_key,
                data,
            );
            continue;
        }

        // Exchange-first payload:
        // {
        //   "bitget": {"DOGE": {"free": 10, ...}, "USDT": {"free": 20, ...}}
        // }
        let Some(assets) = data.as_object() else {
            continue;
        };
        for (asset, asset_value) in assets {
            let Some(balance) = parse_currency_balance(asset_value) else {
                continue;
            };
            merge_currency_balance(&mut portfolio.currencies, asset, &balance);
            merge_exchange_currency_balance(
                &mut portfolio.exchange_currencies,
                top_level_key,
                asset,
                &balance,
            );
        }
    }

    portfolio
}

fn collect_exchange_balances_for_currency(
    exchange_currencies: &mut std::collections::HashMap<
        String,
        std::collections::HashMap<String, CurrencyBalance>,
    >,
    currency: &str,
    data: &Value,
) {
    let Some(exchange_entries) = data
        .as_object()
        .and_then(|obj| obj.get("exchanges"))
        .and_then(Value::as_object)
    else {
        return;
    };

    for (exchange, exchange_balance_data) in exchange_entries {
        let Some(balance) = parse_currency_balance(exchange_balance_data) else {
            continue;
        };
        merge_exchange_currency_balance(exchange_currencies, exchange, currency, &balance);
    }
}

fn merge_exchange_currency_balance(
    exchange_currencies: &mut std::collections::HashMap<
        String,
        std::collections::HashMap<String, CurrencyBalance>,
    >,
    exchange: &str,
    currency: &str,
    balance: &CurrencyBalance,
) {
    let Some(normalized_exchange) = normalize_exchange_name(exchange) else {
        return;
    };
    let exchange_balances = exchange_currencies.entry(normalized_exchange).or_default();
    merge_currency_balance(exchange_balances, currency, balance);
}

fn merge_currency_balance(
    balances: &mut std::collections::HashMap<String, CurrencyBalance>,
    currency: &str,
    incoming: &CurrencyBalance,
) {
    let entry = balances.entry(currency.to_string()).or_default();
    entry.free += incoming.free;
    entry.locked += incoming.locked;
    entry.total += incoming.total;
    entry.value_usd = match (entry.value_usd, incoming.value_usd) {
        (Some(existing), Some(next)) => Some(existing + next),
        (Some(existing), None) => Some(existing),
        (None, Some(next)) => Some(next),
        (None, None) => None,
    };
}

fn parse_currency_balance(value: &Value) -> Option<CurrencyBalance> {
    if let Some(amount) = json_f64(value) {
        if !amount.is_finite() {
            return None;
        }
        let normalized = amount.max(0.0);
        return Some(CurrencyBalance {
            free: normalized,
            locked: 0.0,
            total: normalized,
            value_usd: None,
        });
    }

    let object = value.as_object()?;
    let free = coalesce_json_f64(object, &["free", "available"]);
    let locked = coalesce_json_f64(object, &["locked", "used", "in_order", "in_orders"]);
    let value_usd = coalesce_json_f64(object, &["value_usd", "usd_value", "value"]);
    let total =
        coalesce_json_f64(object, &["total", "amount", "balance", "quantity"]).or_else(|| {
            match (free, locked) {
                (Some(free), Some(locked)) => Some(free + locked),
                (Some(free), None) => Some(free),
                (None, Some(locked)) => Some(locked),
                (None, None) => None,
            }
        });
    let Some(total) = total else {
        return None;
    };

    let normalized_total = total.max(0.0);
    let normalized_locked = locked
        .unwrap_or_else(|| (normalized_total - free.unwrap_or(0.0)).max(0.0))
        .max(0.0);
    let normalized_free = free
        .unwrap_or_else(|| (normalized_total - normalized_locked).max(0.0))
        .max(0.0);

    Some(CurrencyBalance {
        free: normalized_free,
        locked: normalized_locked,
        total: normalized_total,
        value_usd,
    })
}

fn coalesce_json_f64(object: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<f64> {
    keys.iter()
        .filter_map(|key| object.get(*key))
        .find_map(json_f64)
}

fn json_f64(value: &Value) -> Option<f64> {
    value.as_f64().or_else(|| {
        value
            .as_str()
            .and_then(|text| text.trim().parse::<f64>().ok())
    })
}

fn parse_portfolio_html(raw: &str) -> Option<OctobotPortfolio> {
    let text = html_table_to_text(raw);
    let mut portfolio = OctobotPortfolio::default();
    portfolio.total_value_usd = parse_portfolio_total_from_text(&text);

    for row in html_table_rows(raw) {
        if row.len() < 5 {
            continue;
        }
        if row[0].text.eq_ignore_ascii_case("asset") {
            continue;
        }
        let Some(total) = parse_first_number(&row[1].text) else {
            continue;
        };
        let Some(symbol) = extract_portfolio_symbol(&row[0].text) else {
            continue;
        };
        let free = parse_first_number(&row[3].text).unwrap_or(total);
        let locked = parse_first_number(&row[4].text).unwrap_or((total - free).max(0.0));
        let value_usd = parse_first_number(&row[2].text);
        portfolio.currencies.insert(
            symbol.clone(),
            CurrencyBalance {
                free,
                locked,
                total,
                value_usd,
            },
        );
        collect_exchange_balances_from_html_row(&mut portfolio.exchange_currencies, &symbol, &row);
    }

    if portfolio.currencies.is_empty() && portfolio.total_value_usd.is_none() {
        None
    } else {
        Some(portfolio)
    }
}

fn collect_exchange_balances_from_html_row(
    exchange_currencies: &mut std::collections::HashMap<
        String,
        std::collections::HashMap<String, CurrencyBalance>,
    >,
    symbol: &str,
    row: &[HtmlTableCell],
) {
    let mut per_exchange: std::collections::HashMap<String, CurrencyBalance> =
        std::collections::HashMap::new();

    for (column_index, cell) in row.iter().enumerate() {
        let Some(title) = cell.title.as_deref() else {
            continue;
        };
        for (exchange, amount) in parse_exchange_tooltip_balances(title) {
            let entry = per_exchange.entry(exchange).or_default();
            match column_index {
                // Total
                1 => entry.total = entry.total.max(amount),
                // Value in USDT
                2 => {
                    let existing = entry.value_usd.unwrap_or(0.0);
                    entry.value_usd = Some(existing.max(amount));
                }
                // Available
                3 => entry.free = entry.free.max(amount),
                // Locked in orders
                4 => entry.locked = entry.locked.max(amount),
                _ => {}
            }
        }
    }

    for (exchange, mut balance) in per_exchange {
        if balance.total <= 0.0 {
            balance.total = (balance.free + balance.locked).max(0.0);
        }
        if balance.free <= 0.0 && balance.total > 0.0 && balance.locked <= 0.0 {
            balance.free = balance.total;
        }
        merge_exchange_currency_balance(exchange_currencies, &exchange, symbol, &balance);
    }
}

fn parse_exchange_tooltip_balances(value: &str) -> Vec<(String, f64)> {
    static EXCHANGE_BALANCE_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(
            r"(?i)([a-z0-9][a-z0-9 ._-]{0,31})\s*:\s*([-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)",
        )
        .expect("valid exchange tooltip regex")
    });

    EXCHANGE_BALANCE_RE
        .captures_iter(value)
        .filter_map(|capture| {
            let exchange = capture
                .get(1)
                .map(|m| m.as_str())
                .and_then(normalize_exchange_name)?;
            let amount = capture
                .get(2)
                .and_then(|m| parse_first_number(m.as_str()))
                .map(|parsed| parsed.max(0.0))?;
            Some((exchange, amount))
        })
        .collect()
}

fn normalize_exchange_name(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.to_ascii_lowercase())
}

fn parse_trading_page_symbol_status_rows(raw: &str) -> Vec<TradingPageSymbolStatusRow> {
    static SYMBOL_STATUS_LINK_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(
            r#"(?is)<a[^>]+href\s*=\s*["']([^"']*symbol_market_status\?[^"']*)["'][^>]*>(.*?)</a>"#,
        )
        .expect("valid symbol market status link regex")
    });

    let mut rows = Vec::new();
    let mut seen = HashSet::new();
    for captures in SYMBOL_STATUS_LINK_RE.captures_iter(raw) {
        let Some(href_raw) = captures.get(1).map(|match_| match_.as_str()) else {
            continue;
        };
        let Some((exchange_id, symbol)) = parse_symbol_status_href(href_raw) else {
            continue;
        };
        let exchange_name = captures
            .get(2)
            .and_then(|match_| parse_trading_exchange_name_label(match_.as_str()));
        let Some(exchange_name) = exchange_name else {
            continue;
        };
        let Some(exchange_name_key) = normalize_exchange_name(&exchange_name) else {
            continue;
        };
        let dedupe_key = format!(
            "{}|{}|{}",
            exchange_id.to_ascii_lowercase(),
            exchange_name_key,
            symbol.to_ascii_uppercase()
        );
        if !seen.insert(dedupe_key) {
            continue;
        }
        rows.push(TradingPageSymbolStatusRow {
            exchange_id,
            exchange_name,
            symbol,
        });
    }
    rows
}

fn parse_trading_exchange_name_label(raw: &str) -> Option<String> {
    let mut label = strip_html_markup(raw);
    if label.is_empty() || normalize_trading_symbol(&label).is_some() {
        return None;
    }

    if let Some((head, _)) = label.split_once(':') {
        label = head.trim().to_string();
    }

    // Some OctoBot templates append suffixes like "(Indexing 6 coins)".
    // Strip trailing parenthetical notes to keep the canonical exchange name.
    while label.ends_with(')') {
        let Some(start) = label.rfind(" (") else {
            break;
        };
        label.truncate(start);
        label = label.trim().to_string();
    }

    let compact = label.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.is_empty() || normalize_trading_symbol(&compact).is_some() {
        return None;
    }
    Some(compact)
}

fn parse_symbol_status_href(href: &str) -> Option<(String, String)> {
    let decoded_href = decode_basic_html_entities(href);
    let query = decoded_href
        .split_once('?')
        .map(|(_, query)| query)
        .unwrap_or(decoded_href.as_str())
        .split('#')
        .next()
        .unwrap_or_default();

    let mut exchange_id = None;
    let mut symbol = None;
    for (key, value) in form_urlencoded::parse(query.as_bytes()) {
        if key.eq_ignore_ascii_case("exchange_id") {
            exchange_id = Some(value.trim().to_string());
        } else if key.eq_ignore_ascii_case("symbol") {
            symbol = normalize_trading_symbol(value.replace('|', "/").as_str());
        }
    }

    let exchange_id = exchange_id
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())?;
    let symbol = symbol?;
    Some((exchange_id, symbol))
}

fn strip_html_markup(value: &str) -> String {
    static HTML_TAG_RE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"(?is)<[^>]+>").expect("valid html tag regex"));
    decode_basic_html_entities(HTML_TAG_RE.replace_all(value, " ").as_ref())
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[derive(Clone, Debug, Default)]
struct HtmlTableCell {
    text: String,
    title: Option<String>,
}

fn html_table_rows(raw: &str) -> Vec<Vec<HtmlTableCell>> {
    let mut rows = Vec::new();
    let mut current_row: Vec<HtmlTableCell> = Vec::new();
    let mut current_cell = String::new();
    let mut current_cell_title: Option<String> = None;
    let mut in_row = false;
    let mut in_cell = false;
    let mut chars = raw.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch != '<' {
            if in_cell {
                current_cell.push(ch);
            }
            continue;
        }

        let mut tag = String::new();
        for tag_ch in chars.by_ref() {
            if tag_ch == '>' {
                break;
            }
            tag.push(tag_ch);
        }
        let trimmed = tag.trim();
        if trimmed.is_empty() {
            continue;
        }
        let is_closing = trimmed.starts_with('/');
        let tag_name = trimmed
            .trim_start_matches('/')
            .split_whitespace()
            .next()
            .unwrap_or_default()
            .to_ascii_lowercase();

        match (is_closing, tag_name.as_str()) {
            (false, "tr") => {
                if in_row && !current_row.is_empty() {
                    rows.push(std::mem::take(&mut current_row));
                }
                in_row = true;
                in_cell = false;
                current_cell.clear();
            }
            (true, "tr") => {
                if in_cell {
                    finalize_html_cell(
                        &mut current_row,
                        &mut current_cell,
                        &mut current_cell_title,
                    );
                    in_cell = false;
                }
                if !current_row.is_empty() {
                    rows.push(std::mem::take(&mut current_row));
                }
                in_row = false;
            }
            (false, "td") | (false, "th") => {
                if in_row {
                    in_cell = true;
                    current_cell.clear();
                    current_cell_title = parse_html_cell_title(trimmed);
                }
            }
            (true, "td") | (true, "th") => {
                if in_row && in_cell {
                    finalize_html_cell(
                        &mut current_row,
                        &mut current_cell,
                        &mut current_cell_title,
                    );
                    in_cell = false;
                }
            }
            _ => {}
        }
    }

    if in_row && in_cell {
        finalize_html_cell(&mut current_row, &mut current_cell, &mut current_cell_title);
    }
    if !current_row.is_empty() {
        rows.push(current_row);
    }

    rows
}

fn finalize_html_cell(row: &mut Vec<HtmlTableCell>, cell: &mut String, title: &mut Option<String>) {
    let normalized = decode_basic_html_entities(cell)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let cell_title = title.take().filter(|value| !value.is_empty());
    if !normalized.is_empty() || cell_title.is_some() {
        row.push(HtmlTableCell {
            text: normalized,
            title: cell_title,
        });
    }
    cell.clear();
}

fn parse_html_cell_title(tag: &str) -> Option<String> {
    extract_html_attribute(tag, "title")
        .or_else(|| extract_html_attribute(tag, "data-original-title"))
        .or_else(|| extract_html_attribute(tag, "data-bs-original-title"))
        .map(|value| decode_basic_html_entities(&value))
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn extract_html_attribute(tag: &str, attribute: &str) -> Option<String> {
    static HTML_ATTRIBUTE_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r#"(?i)([a-z_:][-a-z0-9_:.]*)\s*=\s*("([^"]*)"|'([^']*)'|([^\s>]+))"#)
            .expect("valid html attribute regex")
    });

    HTML_ATTRIBUTE_RE.captures_iter(tag).find_map(|capture| {
        let name = capture.get(1)?.as_str();
        if !name.eq_ignore_ascii_case(attribute) {
            return None;
        }
        capture
            .get(3)
            .or_else(|| capture.get(4))
            .or_else(|| capture.get(5))
            .map(|value| value.as_str().to_string())
    })
}

fn parse_portfolio_total_from_text(text: &str) -> Option<f64> {
    let marker = "portfolio:";
    let lowered = text.to_ascii_lowercase();
    let idx = lowered.find(marker)?;
    let tail = &text[idx + marker.len()..];
    for token in tail.split_whitespace().take(8) {
        if let Some(value) = parse_first_number(token) {
            return Some(value);
        }
    }
    None
}

fn parse_first_number(value: &str) -> Option<f64> {
    static NUMBER_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?").expect("valid numeric regex")
    });
    let matched = NUMBER_RE.find(value)?.as_str().replace(',', "");
    matched.parse::<f64>().ok()
}

fn extract_portfolio_symbol(value: &str) -> Option<String> {
    static SYMBOL_RE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"\b([A-Z0-9][A-Z0-9._-]{1,15})\b").expect("valid symbol regex"));
    SYMBOL_RE
        .captures_iter(value)
        .filter_map(|capture| capture.get(1).map(|m| m.as_str()))
        .last()
        .map(str::to_string)
}

impl OctobotClient {
    // -----------------------------------------------------------------------
    // Exchanges and symbols continued
    // -----------------------------------------------------------------------

    #[allow(dead_code)]
    async fn get_legacy_exchange_info(&self) -> Result<Vec<OctobotExchange>, String> {
        let body = self.get_json("/api/exchanges", "exchanges").await?;
        let mut exchanges = Vec::new();
        if let Some(arr) = body.as_array() {
            for entry in arr {
                let exchange = OctobotExchange {
                    name: entry
                        .get("name")
                        .or_else(|| entry.get("id"))
                        .and_then(Value::as_str)
                        .unwrap_or("unknown")
                        .to_string(),
                    enabled: entry
                        .get("enabled")
                        .and_then(Value::as_bool)
                        .unwrap_or(true),
                    symbols: entry
                        .get("symbols")
                        .and_then(Value::as_array)
                        .map(|a| {
                            a.iter()
                                .filter_map(Value::as_str)
                                .map(str::to_string)
                                .collect()
                        })
                        .unwrap_or_default(),
                };
                exchanges.push(exchange);
            }
        }
        Ok(exchanges)
    }

    /// Fetch tickers for all available symbols on enabled exchanges.
    pub async fn get_all_market_snapshots(
        &self,
        target_exchanges: &[String],
        target_currencies: &[String],
        limit: usize,
    ) -> Vec<MarketSnapshot> {
        #[derive(Clone, Debug)]
        struct ExchangeTargetState {
            exchange_name: String,
            exchange_key: String,
            symbols: Vec<String>,
            base_offset: usize,
            rotation_span: usize,
            consumed: usize,
        }

        #[derive(Clone, Debug)]
        struct MarketTarget {
            exchange_name: String,
            exchange_key: String,
            symbol: String,
        }

        let limit = limit.max(1);
        let candidate_limit = limit
            .saturating_mul(MARKET_SNAPSHOT_CANDIDATE_MULTIPLIER)
            .max(limit);
        let exchanges = match self.get_exchange_info().await {
            Ok(exs) => exs,
            Err(err) => {
                warn!("trading: failed to get exchange info: {}", err);
                return Vec::new();
            }
        };
        let now = current_unix_timestamp_f64();
        let unavailable_snapshot = {
            let mut cache = self.market_snapshot_unavailable_until.lock().await;
            cache.retain(|_, retry_after| *retry_after > now);
            cache.clone()
        };
        let available_snapshot = self.market_snapshot_available_symbols.lock().await.clone();
        let offset_snapshot = self.symbol_scan_offsets.lock().await.clone();

        let mut states: Vec<ExchangeTargetState> = Vec::new();
        for exchange in exchanges.iter().filter(|exchange| exchange.enabled) {
            if !target_exchanges.is_empty()
                && !target_exchanges
                    .iter()
                    .any(|candidate| candidate.eq_ignore_ascii_case(&exchange.name))
            {
                continue;
            }
            let Some(exchange_key) = normalize_exchange_name(&exchange.name) else {
                continue;
            };
            let mut symbols = exchange
                .symbols
                .iter()
                .filter(|symbol| {
                    target_currencies.is_empty()
                        || target_currencies
                            .iter()
                            .any(|candidate| candidate.eq_ignore_ascii_case(symbol))
                })
                .filter(|symbol| {
                    !market_snapshot_is_cooling_down(
                        &unavailable_snapshot,
                        &exchange_key,
                        symbol,
                        now,
                    )
                })
                .cloned()
                .collect::<Vec<_>>();
            if symbols.is_empty() {
                continue;
            }
            symbols.sort();
            symbols.dedup();
            if symbols.is_empty() {
                continue;
            }
            let rotation_span = symbols.len();
            let base_offset =
                offset_snapshot.get(&exchange_key).copied().unwrap_or(0) % rotation_span;

            let mut selected = Vec::new();
            let mut unknown_probe_count = 0usize;
            for index in 0..rotation_span {
                let symbol = symbols[(base_offset + index) % rotation_span].clone();
                if market_snapshot_is_known_available(&available_snapshot, &exchange_key, &symbol) {
                    selected.push(symbol);
                    continue;
                }
                if unknown_probe_count < MARKET_SNAPSHOT_UNKNOWN_PROBE_LIMIT_PER_EXCHANGE {
                    selected.push(symbol);
                    unknown_probe_count += 1;
                }
            }
            if selected.is_empty() {
                continue;
            }
            states.push(ExchangeTargetState {
                exchange_name: exchange.name.clone(),
                exchange_key,
                symbols: selected,
                base_offset,
                rotation_span,
                consumed: 0,
            });
        }

        let mut targets: Vec<MarketTarget> = Vec::new();
        while targets.len() < candidate_limit {
            let mut progressed = false;
            for state in &mut states {
                if targets.len() >= candidate_limit || state.consumed >= state.symbols.len() {
                    continue;
                }
                targets.push(MarketTarget {
                    exchange_name: state.exchange_name.clone(),
                    exchange_key: state.exchange_key.clone(),
                    symbol: state.symbols[state.consumed].clone(),
                });
                state.consumed += 1;
                progressed = true;
            }
            if !progressed {
                break;
            }
        }

        if targets.is_empty() {
            return Vec::new();
        }

        let mut requested_by_exchange: HashMap<String, usize> = HashMap::new();
        let mut snapshots = Vec::new();
        let mut join_set: JoinSet<(String, String, String, Result<MarketSnapshot, String>)> =
            JoinSet::new();
        let mut pending = targets.into_iter();
        let max_parallel = MAX_PARALLEL_MARKET_SNAPSHOT_REQUESTS.max(1);
        for _ in 0..max_parallel {
            let Some(target) = pending.next() else {
                break;
            };
            *requested_by_exchange
                .entry(target.exchange_key.clone())
                .or_insert(0) += 1;
            let client = self.clone();
            join_set.spawn(async move {
                let result = client
                    .get_market_snapshot(&target.exchange_name, &target.symbol)
                    .await;
                (
                    target.exchange_name,
                    target.exchange_key,
                    target.symbol,
                    result,
                )
            });
        }

        while let Some(result) = join_set.join_next().await {
            match result {
                Ok((_exchange, exchange_key, symbol, Ok(snapshot))) => {
                    self.mark_market_snapshot_available(&exchange_key, &symbol)
                        .await;
                    self.clear_market_snapshot_unavailable(&exchange_key, &symbol)
                        .await;
                    snapshots.push(snapshot);
                    if snapshots.len() >= limit {
                        join_set.abort_all();
                        break;
                    }
                    if let Some(next_target) = pending.next() {
                        *requested_by_exchange
                            .entry(next_target.exchange_key.clone())
                            .or_insert(0) += 1;
                        let client = self.clone();
                        join_set.spawn(async move {
                            let result = client
                                .get_market_snapshot(
                                    &next_target.exchange_name,
                                    &next_target.symbol,
                                )
                                .await;
                            (
                                next_target.exchange_name,
                                next_target.exchange_key,
                                next_target.symbol,
                                result,
                            )
                        });
                    }
                }
                Ok((exchange, exchange_key, symbol, Err(err))) => {
                    if is_market_snapshot_unavailable_error(&err) {
                        self.clear_market_snapshot_available(&exchange_key, &symbol)
                            .await;
                        self.mark_market_snapshot_unavailable(&exchange_key, &symbol)
                            .await;
                    }
                    warn!("trading: ticker {}/{} failed: {}", exchange, symbol, err);
                    if let Some(next_target) = pending.next() {
                        *requested_by_exchange
                            .entry(next_target.exchange_key.clone())
                            .or_insert(0) += 1;
                        let client = self.clone();
                        join_set.spawn(async move {
                            let result = client
                                .get_market_snapshot(
                                    &next_target.exchange_name,
                                    &next_target.symbol,
                                )
                                .await;
                            (
                                next_target.exchange_name,
                                next_target.exchange_key,
                                next_target.symbol,
                                result,
                            )
                        });
                    }
                }
                Err(error) => {
                    warn!("trading: market snapshot task failed: {}", error);
                    if let Some(next_target) = pending.next() {
                        *requested_by_exchange
                            .entry(next_target.exchange_key.clone())
                            .or_insert(0) += 1;
                        let client = self.clone();
                        join_set.spawn(async move {
                            let result = client
                                .get_market_snapshot(
                                    &next_target.exchange_name,
                                    &next_target.symbol,
                                )
                                .await;
                            (
                                next_target.exchange_name,
                                next_target.exchange_key,
                                next_target.symbol,
                                result,
                            )
                        });
                    }
                }
            }
        }

        {
            let mut offsets = self.symbol_scan_offsets.lock().await;
            for state in states {
                if state.rotation_span == 0 {
                    continue;
                }
                let requested = requested_by_exchange
                    .get(&state.exchange_key)
                    .copied()
                    .unwrap_or(0)
                    .min(state.rotation_span);
                let next_offset = (state.base_offset + requested) % state.rotation_span;
                offsets.insert(state.exchange_key, next_offset);
            }
        }

        snapshots
    }

    async fn mark_market_snapshot_unavailable(&self, exchange_key: &str, symbol: &str) {
        let key = market_snapshot_symbol_key(exchange_key, symbol);
        let retry_after =
            current_unix_timestamp_f64() + MARKET_SNAPSHOT_UNAVAILABLE_COOLDOWN_SECONDS;
        self.market_snapshot_unavailable_until
            .lock()
            .await
            .insert(key, retry_after);
    }

    async fn clear_market_snapshot_unavailable(&self, exchange_key: &str, symbol: &str) {
        let key = market_snapshot_symbol_key(exchange_key, symbol);
        self.market_snapshot_unavailable_until
            .lock()
            .await
            .remove(&key);
    }

    async fn mark_market_snapshot_available(&self, exchange_key: &str, symbol: &str) {
        let key = market_snapshot_symbol_key(exchange_key, symbol);
        self.market_snapshot_available_symbols
            .lock()
            .await
            .insert(key);
    }

    async fn clear_market_snapshot_available(&self, exchange_key: &str, symbol: &str) {
        let key = market_snapshot_symbol_key(exchange_key, symbol);
        self.market_snapshot_available_symbols
            .lock()
            .await
            .remove(&key);
    }
}

fn market_snapshot_symbol_key(exchange_key: &str, symbol: &str) -> String {
    format!(
        "{}|{}",
        exchange_key.to_ascii_lowercase(),
        symbol.to_ascii_uppercase()
    )
}

fn market_snapshot_is_cooling_down(
    unavailable: &HashMap<String, f64>,
    exchange_key: &str,
    symbol: &str,
    now: f64,
) -> bool {
    let key = market_snapshot_symbol_key(exchange_key, symbol);
    unavailable
        .get(&key)
        .is_some_and(|retry_after| *retry_after > now)
}

fn market_snapshot_is_known_available(
    available: &HashSet<String>,
    exchange_key: &str,
    symbol: &str,
) -> bool {
    let key = market_snapshot_symbol_key(exchange_key, symbol);
    available.contains(&key)
}

fn is_market_snapshot_unavailable_error(error: &str) -> bool {
    error.contains("no dashboard or ticker endpoint returned data")
}

// ---------------------------------------------------------------------------
// Backtesting API
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DataCollectorStartRequest {
    exchange: String,
    symbols: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    time_frames: Option<Vec<String>>,
    #[serde(rename = "startTimestamp", skip_serializing_if = "Option::is_none")]
    start_timestamp: Option<i64>,
    #[serde(rename = "endTimestamp", skip_serializing_if = "Option::is_none")]
    end_timestamp: Option<i64>,
}

/// Request body for starting an OctoBot backtesting run.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct BacktestStartRequest {
    /// Paths to OctoBot `.data` files (relative to OctoBot root), e.g.
    /// `"user/backtesting/collector/binance_BTC_USDT_1h.data"`.
    pub files: Vec<String>,
    /// Optional start time bound (ms epoch).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_timestamp: Option<i64>,
    /// Optional end time bound (ms epoch).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_timestamp: Option<i64>,
    /// Whether to emit verbose logs from the backtesting run.
    pub enable_logs: bool,
}

/// Parsed report returned by `GET /backtesting?update_type=backtesting_report`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BacktestRunReport {
    /// Overall profitability % per exchange (e.g. `{"binance": 12.34}`).
    pub profitability: std::collections::HashMap<String, f64>,
    /// Market buy-and-hold profitability % for the same period.
    pub market_average_profitability: std::collections::HashMap<String, f64>,
    /// Reference market currency (e.g. `"USDT"`).
    pub reference_market: String,
    /// Trading mode used (e.g. `"DipAnalyserTradingMode"`).
    pub trading_mode: String,
    /// Starting portfolio value per exchange.
    pub starting_portfolio: std::collections::HashMap<String, f64>,
    /// Ending portfolio value per exchange.
    pub end_portfolio: std::collections::HashMap<String, f64>,
    /// Number of trades executed during the backtest.
    pub total_trades: usize,
    /// Symbols covered, e.g. `["BTC/USDT", "ETH/USDT"]`.
    pub symbols: Vec<String>,
    /// Number of runtime errors encountered.
    pub errors_count: usize,
    /// The raw JSON from OctoBot (stored for diagnostics).
    pub raw: serde_json::Value,
}

impl BacktestRunReport {
    /// Returns the first available overall profitability value, if any.
    pub fn best_profitability(&self) -> Option<f64> {
        self.profitability.values().copied().next()
    }
    /// Returns the first available market-average profitability value, if any.
    pub fn best_market_avg(&self) -> Option<f64> {
        self.market_average_profitability.values().copied().next()
    }
}

impl OctobotClient {
    /// Start a backtesting run against the OctoBot instance.
    ///
    /// Maps to: `POST /backtesting?action_type=start_backtesting&source=backtesting`
    pub async fn start_backtest(&self, request: &BacktestStartRequest) -> Result<(), String> {
        let url = format!(
            "{}/backtesting?action_type=start_backtesting&source=backtesting&run_on_common_part_only=true",
            self.base_url
        );
        let path = "/backtesting?action_type=start_backtesting&source=backtesting&run_on_common_part_only=true";
        let resp = match self.client.post(&url).json(request).send().await {
            Ok(resp) => resp,
            Err(err) => {
                self.observe_failure("POST", path, "start backtest", None, &err.to_string())
                    .await;
                return Err(format!("OctoBot start_backtest request failed: {err}"));
            }
        };
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if status.is_success() {
            self.observe_success("POST", path, "start backtest", &json!({ "body": text }))
                .await;
            debug!("trading: OctoBot backtesting started: {}", text.trim());
            Ok(())
        } else {
            self.observe_failure(
                "POST",
                path,
                "start backtest",
                Some(status.as_u16()),
                text.trim(),
            )
            .await;
            Err(format!(
                "OctoBot start_backtest rejected HTTP {}: {}",
                status.as_u16(),
                text.trim()
            ))
        }
    }

    /// Stop any running backtest.
    ///
    /// Maps to: `POST /backtesting?action_type=stop_backtesting`
    pub async fn stop_backtest(&self) -> Result<(), String> {
        let url = format!("{}/backtesting?action_type=stop_backtesting", self.base_url);
        let path = "/backtesting?action_type=stop_backtesting";
        let resp = match self
            .client
            .post(&url)
            .json(&serde_json::json!({}))
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(err) => {
                self.observe_failure("POST", path, "stop backtest", None, &err.to_string())
                    .await;
                return Err(format!("OctoBot stop_backtest request failed: {err}"));
            }
        };
        if resp.status().is_success() {
            self.observe_success(
                "POST",
                path,
                "stop backtest",
                &adaptive_schema::endpoint_status_body(resp.status().as_u16()),
            )
            .await;
            Ok(())
        } else {
            self.observe_failure(
                "POST",
                path,
                "stop backtest",
                Some(resp.status().as_u16()),
                resp.status().as_str(),
            )
            .await;
            Err(format!(
                "OctoBot stop_backtest failed: HTTP {}",
                resp.status().as_u16()
            ))
        }
    }

    /// Trigger OctoBot historical data collection for backtesting.
    ///
    /// Maps to: `POST /data_collector?action_type=start_collector`
    pub async fn start_data_collector(
        &self,
        exchange: &str,
        symbols: &[String],
        time_frames: &[String],
        start_timestamp: Option<i64>,
        end_timestamp: Option<i64>,
    ) -> Result<(), String> {
        let exchange = exchange.trim();
        if exchange.is_empty() {
            return Err("OctoBot start_data_collector requires a non-empty exchange".to_string());
        }
        let symbols = symbols
            .iter()
            .map(|symbol| symbol.trim().to_string())
            .filter(|symbol| !symbol.is_empty())
            .collect::<Vec<_>>();
        if symbols.is_empty() {
            return Err("OctoBot start_data_collector requires at least one symbol".to_string());
        }
        let time_frames = time_frames
            .iter()
            .map(|time_frame| time_frame.trim().to_string())
            .filter(|time_frame| !time_frame.is_empty())
            .collect::<Vec<_>>();

        let request = DataCollectorStartRequest {
            exchange: exchange.to_string(),
            symbols,
            time_frames: if time_frames.is_empty() {
                None
            } else {
                Some(time_frames)
            },
            start_timestamp,
            end_timestamp,
        };

        let url = format!(
            "{}/data_collector?action_type=start_collector",
            self.base_url
        );
        let path = "/data_collector?action_type=start_collector";
        let resp = match self.client.post(&url).json(&request).send().await {
            Ok(resp) => resp,
            Err(err) => {
                self.observe_failure("POST", path, "start data collector", None, &err.to_string())
                    .await;
                return Err(format!(
                    "OctoBot start_data_collector request failed: {err}"
                ));
            }
        };
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if status.is_success() {
            self.observe_success(
                "POST",
                path,
                "start data collector",
                &json!({
                    "status": status.as_u16(),
                    "message": text.trim(),
                }),
            )
            .await;
            Ok(())
        } else {
            self.observe_failure(
                "POST",
                path,
                "start data collector",
                Some(status.as_u16()),
                text.trim(),
            )
            .await;
            Err(format!(
                "OctoBot start_data_collector failed: HTTP {}: {}",
                status.as_u16(),
                text.trim()
            ))
        }
    }

    /// Poll for the backtesting report.  Returns `None` if no report is
    /// available yet (backtest still running or not started).
    ///
    /// Maps to: `GET /backtesting?update_type=backtesting_report&source=backtesting`
    pub async fn get_backtest_report(&self) -> Result<Option<BacktestRunReport>, String> {
        let url = format!(
            "{}/backtesting?update_type=backtesting_report&source=backtesting",
            self.base_url
        );
        let path = "/backtesting?update_type=backtesting_report&source=backtesting";
        let resp = match self.client.get(&url).send().await {
            Ok(resp) => resp,
            Err(err) => {
                self.observe_failure("GET", path, "backtest report", None, &err.to_string())
                    .await;
                return Err(format!("OctoBot get_backtest_report request failed: {err}"));
            }
        };
        let status = resp.status();
        let body: Value = match resp.json().await {
            Ok(body) => body,
            Err(err) => {
                let message = format!("OctoBot get_backtest_report parse failed: {err}");
                self.observe_failure(
                    "GET",
                    path,
                    "backtest report",
                    Some(status.as_u16()),
                    &message,
                )
                .await;
                return Err(message);
            }
        };
        if !status.is_success() {
            self.observe_failure(
                "GET",
                path,
                "backtest report",
                Some(status.as_u16()),
                &body.to_string(),
            )
            .await;
            return Err(format!(
                "OctoBot get_backtest_report failed: HTTP {}",
                status.as_u16()
            ));
        }
        self.observe_success("GET", path, "backtest report", &body)
            .await;

        // OctoBot returns `{}` when no report is ready.
        if body.as_object().map(|o| o.is_empty()).unwrap_or(false) {
            return Ok(None);
        }
        Ok(Some(parse_backtest_report(body)))
    }

    /// Get the latest backtesting run ID.
    ///
    /// Maps to: `GET /backtesting_run_id`
    pub async fn get_backtest_run_id(&self) -> Result<Option<u64>, String> {
        let url = format!("{}/backtesting_run_id", self.base_url);
        let path = "/backtesting_run_id";
        let resp = match self.client.get(&url).send().await {
            Ok(resp) => resp,
            Err(err) => {
                self.observe_failure("GET", path, "backtest run id", None, &err.to_string())
                    .await;
                return Err(format!("OctoBot get_backtest_run_id request failed: {err}"));
            }
        };
        let status = resp.status();
        let body: Value = match resp.json().await {
            Ok(body) => body,
            Err(err) => {
                let message = format!("OctoBot get_backtest_run_id parse failed: {err}");
                self.observe_failure(
                    "GET",
                    path,
                    "backtest run id",
                    Some(status.as_u16()),
                    &message,
                )
                .await;
                return Err(message);
            }
        };
        if !status.is_success() {
            self.observe_failure(
                "GET",
                path,
                "backtest run id",
                Some(status.as_u16()),
                &body.to_string(),
            )
            .await;
            return Err(format!(
                "OctoBot get_backtest_run_id failed: HTTP {}",
                status.as_u16()
            ));
        }
        self.observe_success("GET", path, "backtest run id", &body)
            .await;
        Ok(body
            .get("backtesting_id")
            .and_then(Value::as_u64)
            .or_else(|| body.as_u64()))
    }

    /// List available `.data` files that can be used for backtesting.
    ///
    /// Maps to: `GET /backtesting?update_type=backtesting_data_files&source=backtesting`
    pub async fn list_backtest_data_files(&self) -> Result<Vec<String>, String> {
        let candidates = [
            ("/data_collector", "data collector page"),
            (
                "/backtesting?update_type=backtesting_data_files&source=backtesting",
                "backtest data files",
            ),
        ];
        let mut last_error = None;
        let mut saw_successful_endpoint = false;

        for (path, label) in candidates {
            match self.list_backtest_data_files_from_path(path, label).await {
                Ok(files) => {
                    saw_successful_endpoint = true;
                    if !files.is_empty() {
                        return Ok(files);
                    }
                }
                Err(err) => {
                    last_error = Some(err);
                }
            }
        }

        if saw_successful_endpoint {
            Ok(Vec::new())
        } else {
            Err(last_error
                .unwrap_or_else(|| "OctoBot list_backtest_data_files failed: no compatible endpoint responded successfully".to_string()))
        }
    }

    async fn list_backtest_data_files_from_path(
        &self,
        path: &str,
        label: &str,
    ) -> Result<Vec<String>, String> {
        let url = format!("{}{}", self.base_url, path);
        let resp = match self.client.get(&url).send().await {
            Ok(resp) => resp,
            Err(err) => {
                self.observe_failure("GET", path, label, None, &err.to_string())
                    .await;
                return Err(format!(
                    "OctoBot list_backtest_data_files request failed via {path}: {err}"
                ));
            }
        };
        let status = resp.status();
        let content_type = resp
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_string();
        let text = resp.text().await.unwrap_or_default();

        if !status.is_success() {
            self.observe_failure("GET", path, label, Some(status.as_u16()), text.trim())
                .await;
            return Err(format!(
                "OctoBot list_backtest_data_files failed via {path}: HTTP {}",
                status.as_u16()
            ));
        }

        if let Ok(body) = serde_json::from_str::<Value>(&text) {
            self.observe_success("GET", path, label, &body).await;
            return Ok(extract_backtest_data_files_from_json(&body));
        }

        let files = extract_backtest_data_files_from_text(&text);
        self.observe_success(
            "GET",
            path,
            label,
            &json!({
                "status": status.as_u16(),
                "content_type": content_type,
                "files_discovered": files.len(),
                "mode": "text",
            }),
        )
        .await;
        Ok(files)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_backtest_report(body: Value) -> BacktestRunReport {
    let report_obj = body.get("report").unwrap_or(&body);
    let bot_report = report_obj.get("bot_report").unwrap_or(&Value::Null);

    // profitability: {"exchange_name": pct}
    let profitability = extract_f64_map(bot_report.get("profitability"));
    let market_avg = extract_f64_map(bot_report.get("market_average_profitability"));

    let reference_market = bot_report
        .get("reference_market")
        .and_then(Value::as_str)
        .unwrap_or("USDT")
        .to_string();
    let trading_mode = bot_report
        .get("trading_mode")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    // Portfolio values: try to sum all currency value_usd entries per exchange.
    let starting_portfolio = extract_portfolio_totals(bot_report.get("starting_portfolio"));
    let end_portfolio = extract_portfolio_totals(bot_report.get("end_portfolio"));

    // Symbols from chart_identifiers.
    let symbols = report_obj
        .get("chart_identifiers")
        .and_then(Value::as_array)
        .map(|ids| {
            ids.iter()
                .filter_map(|id| id.get("symbol").and_then(Value::as_str))
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();

    // Trade count from trades array.
    let total_trades = body
        .get("trades")
        .and_then(Value::as_array)
        .map(|a| a.len())
        .unwrap_or(0);

    let errors_count = report_obj
        .get("errors_count")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;

    BacktestRunReport {
        profitability,
        market_average_profitability: market_avg,
        reference_market,
        trading_mode,
        starting_portfolio,
        end_portfolio,
        total_trades,
        symbols,
        errors_count,
        raw: body,
    }
}

fn extract_f64_map(value: Option<&Value>) -> std::collections::HashMap<String, f64> {
    let Some(obj) = value.and_then(Value::as_object) else {
        return std::collections::HashMap::new();
    };
    obj.iter()
        .filter_map(|(k, v)| v.as_f64().map(|f| (k.clone(), f)))
        .collect()
}

/// Extract total portfolio value per exchange from a portfolio object like:
/// `{"binance": {"BTC": 0.05, "USDT": 1200.0}}`.
/// Returns `{"binance": 1200.0 + BTC_value ...}` — here we just sum numeric values.
fn extract_portfolio_totals(value: Option<&Value>) -> std::collections::HashMap<String, f64> {
    let Some(obj) = value.and_then(Value::as_object) else {
        return std::collections::HashMap::new();
    };
    obj.iter()
        .filter_map(|(exchange, holdings)| {
            let total: f64 = holdings
                .as_object()?
                .values()
                .filter_map(Value::as_f64)
                .sum();
            Some((exchange.clone(), total))
        })
        .collect()
}

fn extract_backtest_data_files_from_json(body: &Value) -> Vec<String> {
    let mut files = Vec::new();
    collect_backtest_data_files(body, &mut files);
    dedupe_backtest_data_file_paths(files)
}

fn collect_backtest_data_files(value: &Value, output: &mut Vec<String>) {
    match value {
        Value::String(text) => {
            if is_backtest_data_file_path(text) {
                output.push(text.to_string());
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_backtest_data_files(item, output);
            }
        }
        Value::Object(map) => {
            for value in map.values() {
                collect_backtest_data_files(value, output);
            }
        }
        _ => {}
    }
}

fn extract_backtest_data_files_from_text(text: &str) -> Vec<String> {
    static DATA_FILE_PATH_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r#"([A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*\.data)($|["'<>,\s])"#)
            .expect("valid backtest data-file regex")
    });

    let files = DATA_FILE_PATH_RE
        .captures_iter(text)
        .filter_map(|caps| caps.get(1).map(|matched| matched.as_str().to_string()))
        .filter(|candidate| is_backtest_data_file_path(candidate))
        .collect::<Vec<_>>();
    dedupe_backtest_data_file_paths(files)
}

fn dedupe_backtest_data_file_paths(files: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut deduped = Vec::new();
    for file in files {
        let trimmed = file.trim();
        if trimmed.is_empty() {
            continue;
        }
        if seen.insert(trimmed.to_string()) {
            deduped.push(trimmed.to_string());
        }
    }
    deduped
}

fn is_backtest_data_file_path(value: &str) -> bool {
    let trimmed = value.trim();
    if trimmed.is_empty()
        || trimmed.contains('<')
        || trimmed.contains('>')
        || trimmed.contains('?')
        || trimmed.contains('#')
        || trimmed.contains('\\')
        || trimmed.contains("://")
        || trimmed.starts_with("//")
    {
        return false;
    }

    static SAFE_BACKTEST_DATA_FILE_PATH_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"^/?[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*\.data$")
            .expect("valid safe backtest data-file path regex")
    });

    if !SAFE_BACKTEST_DATA_FILE_PATH_RE.is_match(trimmed) {
        return false;
    }

    // Reject CDN-like or URL host-like captures such as
    // `cdn.example.net/path/to/file.data`.
    if trimmed.contains('/') {
        let normalized = trimmed.trim_start_matches('/');
        let first_segment = normalized.split('/').next().unwrap_or_default();
        if first_segment.contains('.') {
            return false;
        }
    } else {
        // Plain `.data` names from OctoBot are collector artifacts. Reject
        // generic tokens (for example CDN host fragments like `cdn.data`).
        let lower = trimmed.to_ascii_lowercase();
        if !lower.contains("collector") && !lower.contains("backtest") {
            return false;
        }
    }

    true
}

fn parse_orders_array(body: &Value) -> Result<Vec<OctobotOrder>, String> {
    let arr = match body.as_array() {
        Some(a) => a,
        None => {
            // OctoBot sometimes returns `{"orders": [...]}`.
            if let Some(inner) = body.get("orders").and_then(Value::as_array) {
                return parse_orders_array(&Value::Array(inner.clone()));
            }
            return Ok(Vec::new());
        }
    };
    Ok(arr
        .iter()
        .filter_map(|entry| {
            let order_type = entry
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("market")
                .to_string();
            Some(OctobotOrder {
                id: entry.get("id").and_then(Value::as_str)?.to_string(),
                exchange: entry
                    .get("exchange")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string(),
                symbol: entry
                    .get("symbol")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string(),
                side: entry
                    .get("side")
                    .or_else(|| entry.get("order_side"))
                    .and_then(Value::as_str)
                    .or_else(|| infer_side(order_type.as_str()))
                    .unwrap_or("unknown")
                    .to_string(),
                order_type,
                amount: entry.get("amount").and_then(Value::as_f64).unwrap_or(0.0),
                price: entry.get("price").and_then(Value::as_f64),
                status: entry
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string(),
                timestamp: entry
                    .get("timestamp")
                    .or_else(|| entry.get("time"))
                    .and_then(Value::as_f64),
            })
        })
        .collect())
}

fn parse_trades_array(body: &Value) -> Result<Vec<OctobotTrade>, String> {
    let arr = match body.as_array() {
        Some(a) => a,
        None => {
            if let Some(inner) = body.get("trades").and_then(Value::as_array) {
                return parse_trades_array(&Value::Array(inner.clone()));
            }
            return Ok(Vec::new());
        }
    };
    Ok(arr
        .iter()
        .filter_map(|entry| {
            let trade_type = entry.get("type").and_then(Value::as_str).unwrap_or("");
            Some(OctobotTrade {
                id: entry.get("id").and_then(Value::as_str).map(str::to_string),
                exchange: entry
                    .get("exchange")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string(),
                symbol: entry
                    .get("symbol")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string(),
                side: entry
                    .get("side")
                    .or_else(|| entry.get("order_side"))
                    .and_then(Value::as_str)
                    .or_else(|| infer_side(trade_type))
                    .unwrap_or("unknown")
                    .to_string(),
                amount: entry.get("amount").and_then(Value::as_f64).unwrap_or(0.0),
                price: entry.get("price").and_then(Value::as_f64).unwrap_or(0.0),
                cost: entry.get("cost").and_then(Value::as_f64).unwrap_or(0.0),
                fee: entry.get("fee").and_then(Value::as_f64),
                timestamp: entry
                    .get("timestamp")
                    .or_else(|| entry.get("time"))
                    .and_then(Value::as_f64),
            })
        })
        .collect())
}

fn infer_side(value: &str) -> Option<&'static str> {
    let lowered = value.to_ascii_lowercase();
    if lowered.contains("buy") {
        Some("buy")
    } else if lowered.contains("sell") {
        Some("sell")
    } else {
        None
    }
}

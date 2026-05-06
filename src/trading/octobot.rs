/// OctoBot REST API client.
///
/// OctoBot exposes its web interface on port 5001.  Current OctoBot web API
/// routes are Flask endpoints under `/api/*` plus dashboard/backtesting routes.
/// The cluster deployment keeps native OctoBot web auth disabled behind
/// shared ingress auth, so the bridge treats `/api/ping` as the session probe.
///
/// Endpoint coverage:
///   Portfolio, open orders, trade history, exchange/symbol listings,
///   order cancellation, market snapshots, and general status. Direct market
///   order placement is intentionally refused until a supported OctoBot
///   trading-mode or user-command bridge is configured.
use std::{sync::Arc, time::Duration};

use reqwest::{Client, ClientBuilder, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::Mutex;
use tracing::{debug, warn};

use crate::{
    adaptive_schema::{self, AdaptiveApiSchema},
    api_issues,
};

const OCTOBOT_API: &str = "octobot";

// ---------------------------------------------------------------------------
// Domain models
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct OctobotPortfolio {
    /// Map of currency symbol → balance entry
    pub currencies: std::collections::HashMap<String, CurrencyBalance>,
    pub total_value_usd: Option<f64>,
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
        let body = match self
            .get_optional_json("/api/portfolio", "portfolio")
            .await?
        {
            Some(body) => body,
            None => {
                let mut portfolio = OctobotPortfolio::default();
                if let Some(body) = self
                    .get_optional_json(
                        "/api/historical_portfolio_value?currency=USDT",
                        "historical portfolio",
                    )
                    .await?
                {
                    portfolio.total_value_usd = parse_latest_portfolio_value(&body);
                }
                return Ok(portfolio);
            }
        };
        let mut portfolio = OctobotPortfolio::default();
        if let Some(currencies) = body.as_object() {
            for (symbol, data) in currencies {
                if symbol == "total_value_usd" {
                    portfolio.total_value_usd = data.as_f64();
                    continue;
                }
                let balance = CurrencyBalance {
                    free: data.get("free").and_then(Value::as_f64).unwrap_or(0.0),
                    locked: data.get("locked").and_then(Value::as_f64).unwrap_or(0.0),
                    total: data.get("total").and_then(Value::as_f64).unwrap_or(0.0),
                    value_usd: data.get("value_usd").and_then(Value::as_f64),
                };
                portfolio.currencies.insert(symbol.clone(), balance);
            }
        }
        Ok(portfolio)
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
        let _ = (exchange, symbol, side, amount_usd);
        Err("OctoBot's web API does not expose direct market order placement; configure an OctoBot trading mode or user-command bridge before enabling Gail live execution".to_string())
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

    // -----------------------------------------------------------------------
    // Exchanges and symbols
    // -----------------------------------------------------------------------

    pub async fn get_exchange_info(&self) -> Result<Vec<OctobotExchange>, String> {
        if let Some(body) = self
            .get_optional_json("/api/exchanges", "exchanges")
            .await?
        {
            let exchanges = parse_exchange_info_array(&body);
            if !exchanges.is_empty() {
                return Ok(exchanges);
            }
        }

        let exchange_body = self
            .get_optional_json("/api/first_exchange_details", "first exchange details")
            .await?;
        let Some(exchange_body) = exchange_body else {
            return Ok(Vec::new());
        };
        let exchange_data = unwrap_octobot_data(&exchange_body);
        let exchange_name = exchange_data
            .get("exchange_name")
            .or_else(|| exchange_data.get("name"))
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string();
        let symbols = self.configured_symbols().await.unwrap_or_default();
        Ok(vec![OctobotExchange {
            name: exchange_name,
            enabled: true,
            symbols,
        }])
    }

    async fn configured_symbols(&self) -> Result<Vec<String>, String> {
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
                    symbols.extend(pairs.iter().filter_map(Value::as_str).map(str::to_string));
                }
            }
        }
        if symbols.is_empty()
            && let Some(first) = self
                .get_optional_json("/dashboard/first_symbol", "first symbol")
                .await?
        {
            if let Some(symbol) = first.get("symbol").and_then(Value::as_str) {
                symbols.push(symbol.replace('|', "/"));
            }
        }
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
        let legacy_path = format!("/api/market/ticker?exchange={}&symbol={}", exchange, symbol);
        if let Some(body) = self.get_optional_json(&legacy_path, "ticker").await? {
            return Ok(parse_ticker_snapshot(exchange, symbol, &body));
        }

        let web_symbol = symbol.replace('/', "|");
        let watched_path = format!("/dashboard/watched_symbol/{web_symbol}");
        let watched = self.get_json(&watched_path, "watched symbol").await?;
        let exchange_id = watched
            .get("exchange_id")
            .and_then(Value::as_str)
            .ok_or_else(|| "OctoBot watched symbol response missing exchange_id".to_string())?;
        let time_frame = watched
            .get("time_frame")
            .and_then(Value::as_str)
            .unwrap_or("1h");
        let graph_path = format!(
            "/dashboard/currency_price_graph_update/{exchange_id}/{web_symbol}/{time_frame}/live?display_orders=false"
        );
        let graph = self.get_json(&graph_path, "price graph").await?;
        Ok(parse_graph_snapshot(exchange, symbol, &graph))
    }

    pub async fn get_recent_logs(&self, limit: usize) -> Result<Vec<OctobotLogEntry>, String> {
        let limit = limit.clamp(1, 1000);
        let candidate_paths = [
            format!("/api/logs?limit={limit}"),
            format!("/logs?format=json&limit={limit}"),
            "/logs".to_string(),
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

fn parse_exchange_info_array(body: &Value) -> Vec<OctobotExchange> {
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
    exchanges
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

fn parse_graph_snapshot(exchange: &str, symbol: &str, body: &Value) -> MarketSnapshot {
    let candles = body.get("candles").unwrap_or(&Value::Null);
    let closes = value_array(candles, "close");
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
    MarketSnapshot {
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
    }
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
        let exchanges = match self.get_exchange_info().await {
            Ok(exs) => exs,
            Err(err) => {
                warn!("trading: failed to get exchange info: {}", err);
                return Vec::new();
            }
        };
        let mut snapshots = Vec::new();
        'outer: for exchange in exchanges.iter().filter(|e| e.enabled) {
            if !target_exchanges.is_empty()
                && !target_exchanges
                    .iter()
                    .any(|t| t.eq_ignore_ascii_case(&exchange.name))
            {
                continue;
            }
            for symbol in exchange.symbols.iter().take(limit) {
                if !target_currencies.is_empty()
                    && !target_currencies
                        .iter()
                        .any(|t| t.eq_ignore_ascii_case(symbol))
                {
                    continue;
                }
                match self.get_market_snapshot(&exchange.name, symbol).await {
                    Ok(snap) => snapshots.push(snap),
                    Err(err) => {
                        warn!(
                            "trading: ticker {}/{} failed: {}",
                            exchange.name, symbol, err
                        );
                    }
                }
                if snapshots.len() >= limit {
                    break 'outer;
                }
            }
        }
        snapshots
    }
}

// ---------------------------------------------------------------------------
// Backtesting API
// ---------------------------------------------------------------------------

/// Request body for starting an OctoBot backtesting run.
#[derive(Clone, Debug, Serialize, Deserialize)]
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

impl Default for BacktestStartRequest {
    fn default() -> Self {
        Self {
            files: Vec::new(),
            start_timestamp: None,
            end_timestamp: None,
            enable_logs: false,
        }
    }
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
        let url = format!(
            "{}/backtesting?update_type=backtesting_data_files&source=backtesting",
            self.base_url
        );
        let path = "/backtesting?update_type=backtesting_data_files&source=backtesting";
        let resp = match self.client.get(&url).send().await {
            Ok(resp) => resp,
            Err(err) => {
                self.observe_failure("GET", path, "backtest data files", None, &err.to_string())
                    .await;
                return Err(format!(
                    "OctoBot list_backtest_data_files request failed: {err}"
                ));
            }
        };
        let status = resp.status();
        let body: Value = match resp.json().await {
            Ok(body) => body,
            Err(err) => {
                let message = format!("OctoBot list_backtest_data_files parse failed: {err}");
                self.observe_failure(
                    "GET",
                    path,
                    "backtest data files",
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
                "backtest data files",
                Some(status.as_u16()),
                &body.to_string(),
            )
            .await;
            return Err(format!(
                "OctoBot list_backtest_data_files failed: HTTP {}",
                status.as_u16()
            ));
        }
        self.observe_success("GET", path, "backtest data files", &body)
            .await;
        // Response may be an array directly or wrapped in {"data_files": [...]}.
        let files = body
            .as_array()
            .or_else(|| body.get("data_files").and_then(Value::as_array))
            .map(|arr| {
                arr.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
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

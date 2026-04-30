/// OctoBot REST API client.
///
/// OctoBot exposes its web interface on port 5001.  Authentication is done
/// via a session login (POST /api/accounts/login) that returns a session
/// cookie, which is then attached to subsequent requests.  We also support
/// the unofficial `Authorization: Basic` header approach as a fallback.
///
/// Endpoint coverage:
///   Portfolio, open orders, trade history, exchange/symbol listings,
///   order placement and cancellation, and general status.
use std::time::Duration;

use reqwest::{Client, ClientBuilder, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tracing::{debug, warn};

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
}

impl OctobotClient {
    pub fn new(base_url: &str, password: Option<&str>, timeout_seconds: f64) -> Self {
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
        }
    }

    /// Authenticate with OctoBot and establish a session cookie.
    pub async fn login(&self) -> Result<(), String> {
        let Some(ref password) = self.password else {
            return Ok(()); // no auth configured — assume open instance
        };
        let url = format!("{}/api/accounts/login", self.base_url);
        let body = json!({ "password": password });
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("OctoBot login request failed: {e}"))?;
        if resp.status().is_success() {
            debug!("trading: OctoBot session established");
            Ok(())
        } else {
            Err(format!(
                "OctoBot login rejected: HTTP {}",
                resp.status().as_u16()
            ))
        }
    }

    // -----------------------------------------------------------------------
    // Status / health
    // -----------------------------------------------------------------------

    pub async fn get_status(&self) -> Result<OctobotStatus, String> {
        let url = format!("{}/api/status", self.base_url);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("OctoBot status request failed: {e}"))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("OctoBot status parse failed: {e}"))?;
        Ok(OctobotStatus {
            running: body
                .get("running")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            version: body
                .get("version")
                .and_then(Value::as_str)
                .map(str::to_string),
            uptime_seconds: body.get("uptime").and_then(Value::as_f64),
            trading_enabled: body
                .get("trading_enabled")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        })
    }

    // -----------------------------------------------------------------------
    // Portfolio
    // -----------------------------------------------------------------------

    pub async fn get_portfolio(&self) -> Result<OctobotPortfolio, String> {
        let url = format!("{}/api/portfolio", self.base_url);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("OctoBot portfolio request failed: {e}"))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("OctoBot portfolio parse failed: {e}"))?;
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
        let url = format!("{}/api/orders", self.base_url);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("OctoBot open orders request failed: {e}"))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("OctoBot open orders parse failed: {e}"))?;
        parse_orders_array(&body)
    }

    pub async fn get_trade_history(&self, limit: usize) -> Result<Vec<OctobotTrade>, String> {
        let url = format!("{}/api/trades?limit={}", self.base_url, limit);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("OctoBot trade history request failed: {e}"))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("OctoBot trade history parse failed: {e}"))?;
        parse_trades_array(&body)
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
        let url = format!("{}/api/orders", self.base_url);
        let body = json!({
            "exchange": exchange,
            "symbol": symbol,
            "side": side,
            "amount": amount_usd,
            "type": "market"
        });
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("OctoBot place order request failed: {e}"))?;
        let status = resp.status();
        let data: Value = resp
            .json()
            .await
            .map_err(|e| format!("OctoBot place order response parse failed: {e}"))?;
        if status.is_success() {
            Ok(OctobotOrderResult {
                order_id: data
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string(),
                symbol: symbol.to_string(),
                side: side.to_string(),
                amount: amount_usd,
                price: data.get("price").and_then(Value::as_f64),
                status: data
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("placed")
                    .to_string(),
            })
        } else {
            Err(format!(
                "OctoBot order rejected (HTTP {}): {}",
                status.as_u16(),
                data.get("error")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown error")
            ))
        }
    }

    pub async fn cancel_order(&self, order_id: &str) -> Result<(), String> {
        let url = format!("{}/api/orders/{}", self.base_url, order_id);
        let resp = self
            .client
            .delete(&url)
            .send()
            .await
            .map_err(|e| format!("OctoBot cancel order request failed: {e}"))?;
        if resp.status().is_success() || resp.status() == StatusCode::NOT_FOUND {
            Ok(())
        } else {
            Err(format!(
                "OctoBot cancel order failed: HTTP {}",
                resp.status().as_u16()
            ))
        }
    }

    // -----------------------------------------------------------------------
    // Exchanges and symbols
    // -----------------------------------------------------------------------

    pub async fn get_exchange_info(&self) -> Result<Vec<OctobotExchange>, String> {
        let url = format!("{}/api/exchanges", self.base_url);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("OctoBot exchanges request failed: {e}"))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("OctoBot exchanges parse failed: {e}"))?;
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

    /// Fetch market data (ticker) for a symbol on an exchange.
    pub async fn get_market_snapshot(
        &self,
        exchange: &str,
        symbol: &str,
    ) -> Result<MarketSnapshot, String> {
        let url = format!(
            "{}/api/market/ticker?exchange={}&symbol={}",
            self.base_url, exchange, symbol
        );
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("OctoBot ticker request failed: {e}"))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("OctoBot ticker parse failed: {e}"))?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        Ok(MarketSnapshot {
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
        })
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
        let resp = self
            .client
            .post(&url)
            .json(request)
            .send()
            .await
            .map_err(|e| format!("OctoBot start_backtest request failed: {e}"))?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if status.is_success() {
            debug!("trading: OctoBot backtesting started: {}", text.trim());
            Ok(())
        } else {
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
        let resp = self
            .client
            .post(&url)
            .json(&serde_json::json!({}))
            .send()
            .await
            .map_err(|e| format!("OctoBot stop_backtest request failed: {e}"))?;
        if resp.status().is_success() {
            Ok(())
        } else {
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
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("OctoBot get_backtest_report request failed: {e}"))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("OctoBot get_backtest_report parse failed: {e}"))?;

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
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("OctoBot get_backtest_run_id request failed: {e}"))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("OctoBot get_backtest_run_id parse failed: {e}"))?;
        Ok(body.get("backtesting_id").and_then(Value::as_u64))
    }

    /// List available `.data` files that can be used for backtesting.
    ///
    /// Maps to: `GET /backtesting?update_type=backtesting_data_files&source=backtesting`
    pub async fn list_backtest_data_files(&self) -> Result<Vec<String>, String> {
        let url = format!(
            "{}/backtesting?update_type=backtesting_data_files&source=backtesting",
            self.base_url
        );
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("OctoBot list_backtest_data_files request failed: {e}"))?;
        let body: Value = resp
            .json()
            .await
            .map_err(|e| format!("OctoBot list_backtest_data_files parse failed: {e}"))?;
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
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string(),
                order_type: entry
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("market")
                    .to_string(),
                amount: entry.get("amount").and_then(Value::as_f64).unwrap_or(0.0),
                price: entry.get("price").and_then(Value::as_f64),
                status: entry
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string(),
                timestamp: entry.get("timestamp").and_then(Value::as_f64),
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
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string(),
                amount: entry.get("amount").and_then(Value::as_f64).unwrap_or(0.0),
                price: entry.get("price").and_then(Value::as_f64).unwrap_or(0.0),
                cost: entry.get("cost").and_then(Value::as_f64).unwrap_or(0.0),
                fee: entry.get("fee").and_then(Value::as_f64),
                timestamp: entry.get("timestamp").and_then(Value::as_f64),
            })
        })
        .collect())
}

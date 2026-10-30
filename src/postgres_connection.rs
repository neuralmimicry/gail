//! Bounded PostgreSQL operations shared by Gail's ledger, provider registry,
//! API issue writer and market mirror. Dropping a timed-out Rust query future
//! does not cancel a statement already queued at PostgreSQL: an abandoned DDL
//! lock can otherwise block a backup and every later reader indefinitely.
//! Server deadlines bound that wait even after the caller or pod disappears.

use std::time::Duration;
use tokio_postgres::{Client, Config, Error, NoTls};

fn bounded_config(dsn: &str, component: &str) -> Result<Config, Error> {
    let mut config: Config = dsn.parse()?;
    let inherited = config.get_options().unwrap_or("");
    // These apply to this Gail connection only. Preserve other DSN options;
    // never change the platform role, backup connection or global server.
    config.options(format!("{inherited} -c statement_timeout=25000 -c lock_timeout=3000 -c idle_in_transaction_session_timeout=30000"));
    config.application_name(component);
    config.connect_timeout(Duration::from_secs(10));
    Ok(config)
}

pub(crate) async fn connect(dsn: &str, component: &'static str) -> Result<Client, Error> {
    let (client, connection) = bounded_config(dsn, component)?.connect(NoTls).await?;
    // The driver drains already-sent work on client drop. Server deadlines
    // above ensure it cannot retain an abandoned lock/query indefinitely.
    tokio::spawn(async move {
        if let Err(error) = connection.await {
            tracing::debug!(component, error = %error, "Gail Postgres connection ended");
        }
    });
    Ok(client)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deadlines_are_connection_scoped_and_preserve_other_dsn_options() {
        let config = bounded_config(
            "host=127.0.0.1 dbname=gail_test_history options='-c timezone=UTC'",
            "gail-market-mirror",
        )
        .unwrap();
        assert_eq!(config.get_application_name(), Some("gail-market-mirror"));
        assert_eq!(config.get_connect_timeout(), Some(&Duration::from_secs(10)));
        assert_eq!(
            config.get_options(),
            Some(
                "-c timezone=UTC -c statement_timeout=25000 -c lock_timeout=3000 -c idle_in_transaction_session_timeout=30000"
            )
        );
    }

    #[tokio::test]
    #[ignore = "requires an isolated loopback PostgreSQL gail_test_history database"]
    async fn abandoned_lock_wait_expires_on_server_and_releases_connection() {
        let dsn = std::env::var("GAIL_TEST_POSTGRES_DSN").expect("isolated test DSN");
        let parsed: Config = dsn.parse().unwrap();
        assert_eq!(parsed.get_dbname(), Some("gail_test_history"));
        assert!(parsed.get_hosts().iter().all(
            |host| matches!(host, tokio_postgres::config::Host::Tcp(name) if name == "127.0.0.1")
        ));
        let owner = connect(&dsn, "gail-lock-test-owner").await.unwrap();
        assert_eq!(
            owner
                .query_one("SHOW statement_timeout", &[])
                .await
                .unwrap()
                .get::<_, String>(0),
            "25s"
        );
        assert_eq!(
            owner
                .query_one("SHOW lock_timeout", &[])
                .await
                .unwrap()
                .get::<_, String>(0),
            "3s"
        );
        owner
            .batch_execute("CREATE TABLE IF NOT EXISTS gail_timeout_probe(id INTEGER)")
            .await
            .unwrap();
        owner
            .batch_execute("BEGIN; LOCK TABLE gail_timeout_probe IN ACCESS EXCLUSIVE MODE")
            .await
            .unwrap();
        let waiting = connect(&dsn, "gail-abandoned-lock-test").await.unwrap();
        assert!(
            tokio::time::timeout(
                Duration::from_millis(100),
                waiting.query("SELECT * FROM gail_timeout_probe", &[])
            )
            .await
            .is_err()
        );
        drop(waiting);
        // The caller is gone, while PostgreSQL still sees its queued query.
        let remaining: i64 = owner.query_one("SELECT count(*) FROM pg_stat_activity WHERE application_name='gail-abandoned-lock-test'", &[]).await.unwrap().get(0);
        assert_eq!(remaining, 1);
        tokio::time::timeout(Duration::from_secs(6), async {
            loop {
                let remaining: i64 = owner.query_one("SELECT count(*) FROM pg_stat_activity WHERE application_name='gail-abandoned-lock-test'", &[]).await.unwrap().get(0);
                if remaining == 0 { break; }
                // In a transaction, clear the cached statistics snapshot so
                // each poll observes the actual backend rather than its first read.
                owner.simple_query("SELECT pg_stat_clear_snapshot()").await.unwrap();
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
        }).await.expect("server lock timeout must release an abandoned client without operator cancellation");
        owner
            .batch_execute("ROLLBACK; DROP TABLE gail_timeout_probe")
            .await
            .unwrap();
        connect(&dsn, "gail-reconnected-test")
            .await
            .unwrap()
            .simple_query("SELECT 1")
            .await
            .unwrap();
    }
}

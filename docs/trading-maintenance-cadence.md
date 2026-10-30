# Trading evaluation cadence and maintenance reviews

Discovery and pruning previously awaited entire candidate batches inside the
primary evaluation loop. With twenty candidates, bounded parallelism and long
provider timeouts, a maintenance review could defer the next configured
three-minute evaluation for over an hour. Healthy HTTP handlers and independently
scheduled backtests did not establish a healthy decision cadence.

The bridge now keeps one independent read-only review in flight. Due work is
coalesced; the oldest due discovery/pruning review is selected next. Candidate
research retains its existing bounded parallelism and advisor deadlines. A
completed plan returns to the primary loop, which remains the single owner of
order execution and position-limit checks. The planner has no order executor.
Pauses, pending overrides and changed model fingerprints discard completed
plans; the existing freshness, repricing, economics, paper, profitability and
durable execution-lease checks remain in force. Shutdown aborts the planner.

The `maintenance` log category records review start/completion and discarded or
failed plans. A selected plan or accepted paper intent is not a venue fill.
The active LLM policy still requires genuine qualification; a native quant
backtest does not demonstrate the LLM policy's historical performance. This
cadence repair does not lower financial thresholds, change the active strategy,
fabricate paper observations or declare trading restored.

Regression tests hold provider planning behind a semaphore and verify that
primary-loop polling remains immediate across repeated ticks, duplicate work
coalesces, one result is returned, and shutdown cancels planning before it can
publish. Run the complete existing libtorch-free trading gate with
`scripts/test-trading-ci-safe.sh`.

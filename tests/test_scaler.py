from agentic_finops.mag.agents import ScalerAgent, ScalerRecommendation


def test_scaler_observe_builds_ewma_history() -> None:
    s = ScalerAgent()
    s.observe("workload-nlp", 10_000)
    s.observe("workload-nlp", 20_000)
    s.observe("workload-nlp", 30_000)
    forecast = s.forecast_tpw("workload-nlp")
    # EWMA should weight recent samples; forecast should be > first sample
    assert forecast > 10_000
    assert forecast < 30_000  # but not equal to last sample either


def test_scaler_recommends_no_change_when_within_bounds() -> None:
    s = ScalerAgent(capacity_tokens_per_replica_per_window=100_000)
    s.observe("w", 50_000)  # 50% pressure
    rec = s.recommend(
        workload_id="w", latency_ms=500, slo_latency_ms=2000, uci_per_token=0.0001
    )
    assert rec.action == "no_change"
    assert rec.replicas_recommended == rec.replicas_current
    assert rec.cost_delta_usd == 0.0


def test_scaler_recommends_scale_up_on_high_pressure() -> None:
    s = ScalerAgent(capacity_tokens_per_replica_per_window=100_000)
    # Push EWMA above the upper threshold
    for _ in range(10):
        s.observe("w", 95_000)
    rec = s.recommend(
        workload_id="w", latency_ms=500, slo_latency_ms=2000, uci_per_token=0.0001
    )
    assert rec.action == "scale_up"
    assert rec.replicas_recommended > rec.replicas_current
    assert rec.cost_delta_usd > 0
    assert rec.pressure_ratio >= 0.85


def test_scaler_recommends_scale_up_on_latency_breach() -> None:
    s = ScalerAgent()
    s.observe("w", 100)  # tiny demand → low pressure
    # Latency > 1.25× SLO must override low pressure
    rec = s.recommend(
        workload_id="w", latency_ms=3_000, slo_latency_ms=2_000, uci_per_token=0.0001
    )
    assert rec.action == "scale_up"
    assert rec.latency_ratio >= 1.25


def test_scaler_recommends_scale_down_when_under_utilized() -> None:
    s = ScalerAgent(capacity_tokens_per_replica_per_window=100_000)
    # Bootstrap to 3 replicas so scale_down has somewhere to go
    s._replicas["w"] = 3
    for _ in range(10):
        s.observe("w", 1_000)  # very low pressure
    rec = s.recommend(
        workload_id="w", latency_ms=200, slo_latency_ms=2_000, uci_per_token=0.0002
    )
    assert rec.action == "scale_down"
    assert rec.replicas_recommended == 2
    # Cost delta must be NEGATIVE (savings) when scaling down
    assert rec.cost_delta_usd < 0


def test_scaler_apply_commits_replica_change() -> None:
    s = ScalerAgent(capacity_tokens_per_replica_per_window=100_000)
    for _ in range(10):
        s.observe("w", 95_000)
    rec = s.recommend(
        workload_id="w", latency_ms=500, slo_latency_ms=2000, uci_per_token=0.0001
    )
    assert s.current_replicas("w") == 1
    s.apply(rec)
    assert s.current_replicas("w") == rec.replicas_recommended


def test_scaler_records_recommendation_history() -> None:
    s = ScalerAgent()
    s.observe("a", 10_000)
    s.observe("b", 20_000)
    s.recommend(workload_id="a", latency_ms=300, slo_latency_ms=2000, uci_per_token=0.0)
    s.recommend(workload_id="b", latency_ms=300, slo_latency_ms=2000, uci_per_token=0.0)
    history = s.recent(limit=10)
    assert len(history) == 2
    assert {r.workload_id for r in history} == {"a", "b"}
    assert all(isinstance(r, ScalerRecommendation) for r in history)


def test_scaler_cost_delta_tracks_uci_per_token() -> None:
    s = ScalerAgent(capacity_tokens_per_replica_per_window=100_000)
    for _ in range(10):
        s.observe("w", 95_000)
    rec_cheap = s.recommend(
        workload_id="w", latency_ms=500, slo_latency_ms=2000, uci_per_token=0.0001
    )
    # Reset state for an apples-to-apples comparison
    s2 = ScalerAgent(capacity_tokens_per_replica_per_window=100_000)
    for _ in range(10):
        s2.observe("w", 95_000)
    rec_expensive = s2.recommend(
        workload_id="w", latency_ms=500, slo_latency_ms=2000, uci_per_token=0.001
    )
    # Same replica delta + same capacity, 10× per-token cost → 10× cost delta
    assert rec_expensive.cost_delta_usd > rec_cheap.cost_delta_usd
    ratio = rec_expensive.cost_delta_usd / max(1e-12, rec_cheap.cost_delta_usd)
    assert 9.0 < ratio < 11.0


def test_scaler_observe_clamps_negative_values() -> None:
    s = ScalerAgent()
    s.observe("w", -500)
    assert s.forecast_tpw("w") >= 0


def test_legacy_scale_recommendation_still_returns_string() -> None:
    s = ScalerAgent()
    assert s.scale_recommendation(latency_ms=3000, slo_latency_ms=2000) == "scale_up"
    assert s.scale_recommendation(latency_ms=200, slo_latency_ms=2000) == "scale_down"
    assert s.scale_recommendation(latency_ms=1500, slo_latency_ms=2000) == "no_change"

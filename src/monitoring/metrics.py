"""Custom Prometheus metrics for fraud detection monitoring."""

from prometheus_client import Counter, Histogram

# Prediction counters by decision
PREDICTIONS_TOTAL = Counter(
    "fraud_predictions_total",
    "Total number of fraud predictions",
    ["decision"],
)

# Prediction latency histogram
PREDICTION_LATENCY = Histogram(
    "fraud_prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0],
)

# Fraud probability distribution
FRAUD_PROBABILITY = Histogram(
    "fraud_probability_distribution",
    "Distribution of predicted fraud probabilities",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


def track_prediction(decision: str, latency_ms: float, fraud_prob: float = 0.0):
    """Record a prediction in Prometheus metrics."""
    PREDICTIONS_TOTAL.labels(decision=decision).inc()
    PREDICTION_LATENCY.observe(latency_ms / 1000)
    FRAUD_PROBABILITY.observe(fraud_prob)

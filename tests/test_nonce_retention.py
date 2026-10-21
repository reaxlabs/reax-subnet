"""rc10.post2: the validator's nonce store keeps nonces for at least NONCE_RETENTION_MIN_S, so the capacity probe's late
signature check (one slot after the other, after the slowest window) accepts answers that are minutes old instead of
counting the probe 0 ("auth: StaleRequest" for every slot checked late, 21 Oct 2026)."""

from reax_subnet import validator as V
from reax_subnet.auth import http_auth


def test_minimum_retention_constant():
    assert V.NONCE_RETENTION_MIN_S >= 600.0


def test_probe_age_cap_follows_the_retention():
    # s1validator.check caps max_age at retention - allowed skew; a 5-minute check must still fit
    retention = max(20.0 + 60.0, V.NONCE_RETENTION_MIN_S) - http_auth.DEFAULT_ALLOWED_SKEW
    assert retention > 300.0

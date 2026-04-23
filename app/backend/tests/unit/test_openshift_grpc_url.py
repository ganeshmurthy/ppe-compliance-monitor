"""model_url_to_ovms_grpc: HTTP Service URL to OVMS gRPC host:port."""

import pytest

from openshift_grpc_url import model_url_to_ovms_grpc


@pytest.mark.parametrize(
    "model_url, expected",
    [
        ("http://ppe-predictor", "ppe-predictor:9000"),
        ("http://ppe-predictor:80", "ppe-predictor:9000"),
        ("https://ppe-predictor:443", "ppe-predictor:9000"),
        ("https://ppe-predictor", "ppe-predictor:9000"),
        ("ppe-predictor", "ppe-predictor:9000"),
        ("http://ppe-predictor:9000", "ppe-predictor:9000"),
        ("http://ovms:8080", "ovms:8080"),
    ],
)
def test_openshift_grpc_url_mapping(model_url: str, expected: str) -> None:
    assert model_url_to_ovms_grpc(model_url) == expected


def test_openshift_grpc_url_brackets_ipv6() -> None:
    out = model_url_to_ovms_grpc("http://[::1]")
    assert out == "[::1]:9000"
    out2 = model_url_to_ovms_grpc("http://[::1]:80")
    assert out2 == "[::1]:9000"

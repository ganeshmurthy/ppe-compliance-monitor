"""Map app_config HTTP model_url to OVMS gRPC host:port on OpenShift (KServe predictor)."""

from __future__ import annotations

from urllib.parse import urlparse


def _grpc_host_port(host: str, port: int) -> str:
    """Build host:port for gRPC; bracket IPv6 literals."""
    if not host:
        return ""
    if ":" in host:
        return f"[{host}]:{port}"
    return f"{host}:{port}"


def model_url_to_ovms_grpc(model_url: str) -> str:
    """
    Map app_config model_url to host:port for ovmsclient on OpenShift.

    The KServe predictor Service exposes HTTP on default port 80; OVMS gRPC is on 9000.
    Handles explicit :80, omitted port (``http://host`` / RFC default 80), and bare
    ``host`` the same way. Non-default or explicit gRPC-style ports (e.g. 9000) are kept.
    """
    raw = (model_url or "").strip()
    parse_src = raw if "://" in raw else f"http://{raw}"
    p = urlparse(parse_src)
    if p.hostname and p.scheme in ("http", "https"):
        port = p.port
        if port is not None and port not in (80, 443):
            return _grpc_host_port(p.hostname, port)
        return _grpc_host_port(p.hostname, 9000)
    stripped = raw.replace("https://", "").replace("http://", "")
    if stripped.endswith(":80"):
        return stripped[: -len(":80")] + ":9000"
    if ":" not in stripped:
        return f"{stripped}:9000"
    return stripped

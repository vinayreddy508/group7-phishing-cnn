"""URL-only parsing, normalization, and heuristic auditing. No live URL requests."""

import re
from functools import lru_cache
from urllib.parse import urlsplit

_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_WWW = re.compile(r"^www\.", re.IGNORECASE)


def _split_url(url: str):
    text = str(url).strip()
    try:
        return urlsplit(text if "://" in text else "//" + text)
    except ValueError:
        return urlsplit("//")


def get_hostname(url: str) -> str:
    try:
        return (_split_url(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return ""


@lru_cache(maxsize=1)
def _extractor():
    import tldextract
    return tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)


def domain_group(url: str) -> str:
    hostname = get_hostname(url)
    if not hostname:
        return ""
    ext = _extractor()(hostname)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return hostname


def no_prefix(url: str) -> str:
    """Remove URL scheme and leading www only; keep host, path, query, fragment."""
    text = _SCHEME.sub("", str(url).strip())
    return _WWW.sub("", text)


def host_only(url: str) -> str:
    return _WWW.sub("", get_hostname(url))


def url_representation(url: str, representation: str) -> str:
    if representation == "raw":
        return str(url).strip()
    if representation == "no_prefix":
        return no_prefix(url)
    if representation == "host_only":
        return host_only(url)
    raise ValueError(f"Unknown representation: {representation}")


def force_prefix(url: str, prefix: str) -> str:
    """Textual prefix perturbation; does NOT establish equivalent real-world safety."""
    if prefix == "https_www":
        return "https://www." + no_prefix(url)
    if prefix == "http_no_www":
        return "http://" + no_prefix(url)
    raise ValueError(prefix)


def shortcut_features(url: str) -> dict:
    text = str(url).strip()
    low = text.lower()
    parsed = _split_url(text)
    host = get_hostname(text)
    return {
        "starts_https_www": int(low.startswith("https://www.")),
        "starts_https": int(low.startswith("https://")),
        "starts_http": int(low.startswith("http://")),
        "host_starts_www": int(host.startswith("www.")),
        "has_path": int(bool(parsed.path and parsed.path != "/")),
        "has_query": int(bool(parsed.query)),
        "has_fragment": int(bool(parsed.fragment)),
        "url_length": len(text),
        "digit_count": sum(ch.isdigit() for ch in text),
        "hyphen_count": text.count("-"),
        "at_count": text.count("@"),
    }

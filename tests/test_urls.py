import pytest
from phishcnn.urls import no_prefix, force_prefix, get_hostname, shortcut_features, host_only


def test_no_prefix_removes_only_scheme_and_leading_www():
    assert no_prefix("https://www.example.co.uk/a?q=7") == "example.co.uk/a?q=7"
    assert no_prefix("http://sub.example.org/p") == "sub.example.org/p"
    assert no_prefix("https://example.com/www.a") == "example.com/www.a"


def test_perturbations_share_normalized_string():
    original = "https://example.com/a"
    assert no_prefix(force_prefix(original, "https_www")) == no_prefix(original)
    assert no_prefix(force_prefix(original, "http_no_www")) == no_prefix(original)


def test_host_only_and_shortcuts():
    assert get_hostname("https://WWW.Example.Com/login") == "www.example.com"
    assert host_only("https://WWW.Example.Com/login") == "example.com"
    f = shortcut_features("https://www.example.com/login?u=a")
    assert f["starts_https_www"] == 1
    assert f["has_path"] == 1 and f["has_query"] == 1


def test_unusual_malformed_url():
    assert get_hostname("http://[invalid") == ""


def test_bad_prefix_rejected():
    with pytest.raises(ValueError):
        force_prefix("https://example.com", "unknown")

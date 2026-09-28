"""
Unit tests for URL enrichment: SSRF guard, domain skipping, and timeouts.
No real network access — DNS and HTTP are mocked.
"""

import socket
import time
from unittest.mock import Mock, patch

import pytest

from candidate_recommender.core import enricher


def _resolve_to(ip):
    """Fake getaddrinfo that resolves every host to `ip`."""
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    return lambda host, port, *a, **k: [(family, socket.SOCK_STREAM, 6, "", (ip, 0))]


class TestIsPublicUrl:
    @pytest.mark.parametrize(
        "ip",
        [
            "127.0.0.1",
            "10.0.0.5",
            "192.168.1.1",
            "172.16.0.1",
            "169.254.169.254",
            "0.0.0.0",
            "::1",
            "fd00::1",
        ],
    )
    def test_rejects_non_public_addresses(self, ip):
        with patch("candidate_recommender.core.enricher.socket.getaddrinfo", _resolve_to(ip)):
            assert not enricher.is_public_url("http://example.com/")

    def test_accepts_public_address(self):
        with patch(
            "candidate_recommender.core.enricher.socket.getaddrinfo", _resolve_to("93.184.216.34")
        ):
            assert enricher.is_public_url("https://example.com/portfolio")

    @pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com", "http://"])
    def test_rejects_bad_schemes_and_hosts(self, url):
        assert not enricher.is_public_url(url)

    def test_rejects_unresolvable_host(self):
        with patch(
            "candidate_recommender.core.enricher.socket.getaddrinfo", side_effect=socket.gaierror
        ):
            assert not enricher.is_public_url("http://nope.invalid/")


class TestFetchWebpage:
    def test_redirect_to_private_address_is_not_followed(self):
        redirect = Mock(is_redirect=True, headers={"Location": "http://internal.test/"})

        def resolve(host, port, *a, **k):
            ip = "10.0.0.1" if host == "internal.test" else "93.184.216.34"
            return _resolve_to(ip)(host, port)

        with (
            patch("candidate_recommender.core.enricher.socket.getaddrinfo", resolve),
            patch(
                "candidate_recommender.core.enricher._requests.get", return_value=redirect
            ) as get,
        ):
            assert enricher.fetch_webpage_text("http://example.com/") == ""
        assert get.call_count == 1  # never requested the internal host

    def test_skips_blocked_domains_by_hostname(self):
        with patch("candidate_recommender.core.enricher._requests.get") as get:
            assert enricher.fetch_webpage_text("https://www.linkedin.com/in/jane") == ""
        get.assert_not_called()


def test_domain_skip_matches_hostname_not_substring():
    assert enricher._is_skipped_domain("www.linkedin.com")
    assert not enricher._is_skipped_domain("notlinkedin.com.example.org")


def test_enrich_candidate_deadline_returns_instead_of_raising():
    def slow(_):
        time.sleep(0.5)
        return "late"

    with (
        patch.object(enricher, "_ENRICH_DEADLINE", 0.05),
        patch.object(enricher, "fetch_github_info", slow),
    ):
        start = time.monotonic()
        result = enricher.enrich_candidate("", {"github": "github.com/janedoe"})
        elapsed = time.monotonic() - start

    assert result == ""
    assert elapsed < 0.4  # didn't wait for the straggler

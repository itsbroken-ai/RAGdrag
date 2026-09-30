"""Origin scoping and normalization for target profiles."""

import pytest

from ragdrag.engine.models import ImpactLevel, RequestBudget
from ragdrag.engine.profile import TargetProfile, canonical_origin


def test_canonical_origin_includes_effective_port():
    assert canonical_origin("https://Example.COM/chat") == "https://example.com:443"
    assert canonical_origin("https://example.com:443/chat") == "https://example.com:443"
    assert canonical_origin("http://example.com:8000/v1") == "http://example.com:8000"


def test_explicit_zero_port_is_rejected_before_origin_lookup():
    with pytest.raises(ValueError, match="port zero"):
        canonical_origin("https://example.com:0/x")

    profile = TargetProfile.from_cli(
        "https://example.com/chat", headers={"Authorization": "Bearer secret"}
    )
    assert profile.headers_for("https://example.com:443/x") == {
        "Authorization": "Bearer secret"
    }
    assert profile.headers_for("https://example.com:8443/x") == {}
    with pytest.raises(ValueError, match="port zero"):
        profile.headers_for("https://example.com:0/x")
    with pytest.raises(ValueError, match="port zero"):
        TargetProfile.from_cli("https://example.com:0/x")


def test_ipv6_canonical_origin_is_bracketed_and_idempotent():
    assert canonical_origin("http://[::1]/") == "http://[::1]:80"
    assert canonical_origin("http://[::1]:80/") == "http://[::1]:80"
    assert canonical_origin("http://[::1]:8080/") == "http://[::1]:8080"
    assert canonical_origin("https://[2001:DB8::1]/") == "https://[2001:db8::1]:443"
    origin = canonical_origin("https://[2001:DB8::1]/")
    assert canonical_origin(origin) == origin


def test_ipv6_profile_constructs_and_isolates_credentials_by_port():
    profile = TargetProfile.from_cli(
        "http://[::1]/",
        headers={"Authorization": "Bearer primary"},
        additional_origin_headers={"http://[::1]:8080": {"X-Api-Key": "secondary"}},
    )
    assert profile.approved_origins == frozenset({"http://[::1]:80", "http://[::1]:8080"})
    assert profile.headers_for("http://[::1]:80/chat") == {
        "Authorization": "Bearer primary"
    }
    assert profile.headers_for("http://[::1]:8080/chat") == {
        "X-Api-Key": "secondary"
    }
    assert profile.headers_for("http://[::1]:9090/chat") == {}


def test_cli_headers_are_scoped_only_to_primary_origin():
    profile = TargetProfile.from_cli(
        "https://example.com/chat",
        headers={"Authorization": "Bearer secret"},
        cookie="sid=secret",
    )
    assert profile.headers_for("https://example.com/chat") == {
        "Authorization": "Bearer secret",
        "Cookie": "sid=secret",
    }
    assert profile.headers_for("https://example.com:8443/") == {}
    assert profile.headers_for("http://example.com/chat") == {}


def test_profile_rejects_unapproved_header_origin():
    with pytest.raises(ValueError, match="not approved"):
        TargetProfile(
            target_url="https://example.com/chat",
            approved_origins=frozenset({"https://example.com:443"}),
            headers_by_origin={"https://other.example:443": {"X-Key": "secret"}},
            budget=RequestBudget(),
            impact_ceiling=ImpactLevel.ACTIVE,
        )


def test_explicit_additional_origin_gets_only_its_own_key():
    profile = TargetProfile.from_cli(
        "https://chat.example/api",
        headers={"Authorization": "Bearer chat"},
        additional_origin_headers={
            "https://vectors.example:6333": {"X-Api-Key": "vector-key"},
        },
    )
    assert profile.headers_for("https://vectors.example:6333/collections") == {
        "X-Api-Key": "vector-key"
    }
    assert "Authorization" not in profile.headers_for("https://vectors.example:6333/collections")


def test_profile_copies_scoped_headers():
    source = {"Authorization": "Bearer original"}
    profile = TargetProfile.from_cli("https://example.com/chat", headers=source)
    source["Authorization"] = "Bearer changed"
    returned = profile.headers_for("https://example.com/chat")
    returned["Authorization"] = "Bearer changed"
    assert profile.headers_for("https://example.com/chat") == {
        "Authorization": "Bearer original"
    }


@pytest.mark.parametrize("url", ["ftp://example.com/file", "example.com", "https:///chat"])
def test_canonical_origin_rejects_non_http_targets(url):
    with pytest.raises(ValueError, match="Invalid HTTP target"):
        canonical_origin(url)

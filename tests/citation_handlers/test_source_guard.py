"""Tests for the shared prompt-injection guard helper."""

from local_deep_research.citation_handlers.source_guard import (
    UNTRUSTED_SOURCES_GUARD,
    guard_untrusted_sources,
)


class TestGuardUntrustedSources:
    def test_wraps_text_in_fence_with_guard(self):
        result = guard_untrusted_sources("[1] some web content")

        assert UNTRUSTED_SOURCES_GUARD in result
        assert "<sources>" in result
        assert "</sources>" in result
        assert "[1] some web content" in result

    def test_preserves_inner_content_between_tags(self):
        result = guard_untrusted_sources("hello")

        inner = result.split("<sources>\n", 1)[1].rsplit("\n</sources>", 1)[0]
        assert inner == "hello"

    def test_empty_string_returns_empty(self):
        assert guard_untrusted_sources("") == ""

    def test_whitespace_only_returns_empty(self):
        assert guard_untrusted_sources("   \n\t  ") == ""

    def test_none_returns_empty(self):
        assert guard_untrusted_sources(None) == ""

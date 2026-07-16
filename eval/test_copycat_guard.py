"""Unit tests for copycat reference PR selection."""

import copycat_guard as cg


def test_list_reference_prs_filters_drafts(monkeypatch):
    payload = [
        {"number": 10, "author": {"login": "alice"}, "isDraft": False},
        {"number": 11, "author": {"login": "bob"}, "isDraft": True},
        {"number": 12, "author": {"login": "carol"}, "isDraft": False},
    ]

    def fake_gh(args):
        class R:
            stdout = __import__("json").dumps(payload)
        return R

    monkeypatch.setattr(cg, "gh", fake_gh)
    out = cg.list_reference_prs("owner/repo", limit=50)
    assert [p["number"] for p in out] == [10, 12]


def test_list_reference_prs_uses_open_state(monkeypatch):
    seen = {}

    def fake_gh(args):
        seen["args"] = args
        class R:
            stdout = "[]"
        return R

    monkeypatch.setattr(cg, "gh", fake_gh)
    cg.list_reference_prs("owner/repo")
    assert "--state" in seen["args"]
    assert seen["args"][seen["args"].index("--state") + 1] == "open"

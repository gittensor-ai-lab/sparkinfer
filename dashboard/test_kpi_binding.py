#!/usr/bin/env python3
"""Regression tests for the dashboard KPI strip binding (issue #646).

The KPI strip's "Frontier decode" tile and the SOTA card must publish the SAME
decode number: the scored 128-tok competition frontier (`qwen36.frontier_tps`).
A prior revision bound the KPI to the 512-context entry, so the page showed two
disagreeing "frontier decode" figures and computed the vs-llama.cpp lead off the
wrong context.

`frontier_tps` is the 128-context field by the eval bot's own schema --
see CTX_SERIES in eval/pr_eval_bot.py, where 128 maps to status "frontier_tps"
while 512 maps to "longctx_512_tps".

The dashboard is a no-build static page (index.html + data.js), so these tests
assert on the binding expressions in the source plus the data invariant they
rely on, rather than driving a browser.

Run from the repo root:
  python3 dashboard/test_kpi_binding.py
"""
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "dashboard" / "index.html"
DATA = ROOT / "dashboard" / "data.json"


def read_index():
    return INDEX.read_text(encoding="utf-8")


def const_rhs(html, name):
    """Right-hand side of `const <name> = ...;` as written in index.html."""
    m = re.search(r"^[ \t]*const[ \t]+" + re.escape(name) + r"[ \t]*=[ \t]*(.+?);[ \t]*$",
                  html, re.M)
    assert m, "const %s assignment not found in %s" % (name, INDEX.name)
    return m.group(1).strip()


def kpi_block(html):
    """The `$("kpis").innerHTML = [ ... ]` array literal, including tile notes."""
    start = html.index('$("kpis").innerHTML')
    end = html.index("].map(", start)
    return html[start:end]


def ctx_entry(model, label):
    for row in model.get("ctx", []):
        if row.get("label") == label:
            return row
    raise AssertionError("no ctx entry labelled %r" % label)


# `const q36ctx128 = (q36.ctx||[]).find(x=>x.label==="128");` and friends.
CTX_CONST_RE = re.compile(
    r'const[ \t]+(q3\dctx\w+)[ \t]*=[ \t]*\((q3\d)\.ctx\|\|\[\]\)\.find\(x=>x\.label==="([^"]+)"\)')


def binding_namespace(html, data):
    """The names a KPI expression may reference, resolved as index.html defines them.

    Reads the per-context consts out of the page rather than assuming which ones
    exist, so the same helper resolves the expression before and after a rebind.
    """
    models = {"q36": data.get("qwen36") or {}, "q35": data.get("qwen35") or {}}
    ns = dict(models, s=data.get("status") or {})
    for name, model, label in CTX_CONST_RE.findall(html):
        rows = models.get(model, {}).get("ctx") or []
        ns[name] = next((r for r in rows if r.get("label") == label), None)
    return ns


def resolve_or_chain(expr, ns):
    """Evaluate a `a.b || c.d` fallback chain (optional chaining allowed) against ns.

    Mirrors JS `||` semantics: the first truthy operand wins, so a 0 or missing
    value falls through exactly as the page would.
    """
    for operand in expr.split("||"):
        parts = operand.strip().replace("?.", ".").split(".")
        value = ns.get(parts[0])
        for part in parts[1:]:
            value = value.get(part) if isinstance(value, dict) else None
        if value:
            return value
    return None


class KpiBindingTest(unittest.TestCase):
    """The KPI tiles must read the scored frontier, not a per-context row."""

    def setUp(self):
        self.html = read_index()

    def test_kpi_tps_binds_to_scored_frontier(self):
        rhs = const_rhs(self.html, "kpiTps")
        self.assertTrue(rhs.startswith("q36.frontier_tps"),
                        "kpiTps must bind to the scored 128-tok frontier, got: %s" % rhs)

    def test_kpi_ref_binds_to_scored_frontier_reference(self):
        rhs = const_rhs(self.html, "kpiRef")
        self.assertTrue(rhs.startswith("q36.ref_tps"),
                        "kpiRef must bind to the 128-tok llama.cpp reference, got: %s" % rhs)

    def test_kpi_keeps_status_fallback(self):
        """Backward compatibility: the page still renders when D.qwen36 is absent."""
        self.assertIn("s.frontier_tps", const_rhs(self.html, "kpiTps"))
        self.assertIn("s.ref_tps", const_rhs(self.html, "kpiRef"))

    def test_no_per_context_row_feeds_the_kpi(self):
        """A ctx-row lookup in the KPI is the exact defect from #646."""
        for expr in (const_rhs(self.html, "kpiTps"), const_rhs(self.html, "kpiRef")):
            self.assertNotRegex(expr, r"q3\dctx\d",
                                "KPI must not bind to a per-context row: %s" % expr)

    def test_kpi_notes_do_not_claim_512_context(self):
        block = kpi_block(self.html)
        self.assertNotIn("512-context", block,
                         "KPI tile notes still advertise 512-context")
        self.assertIn("128-tok", block,
                      "KPI tile notes should state the 128-tok context")


class FrontierIdentityTest(unittest.TestCase):
    """The data invariant that makes the KPI and the SOTA card agree."""

    @classmethod
    def setUpClass(cls):
        cls.html = read_index()
        cls.data = json.loads(DATA.read_text(encoding="utf-8"))
        cls.q36 = cls.data["qwen36"]

    def test_fixture_discriminates_128_from_512(self):
        """Guard: if the two rows ever held equal tok/s these tests would pass vacuously."""
        self.assertNotEqual(ctx_entry(self.q36, "128")["tps"],
                            ctx_entry(self.q36, "512")["tps"],
                            "fixture no longer distinguishes the 128 and 512 rows")

    def test_frontier_tps_is_the_128_tok_row(self):
        self.assertEqual(self.q36["frontier_tps"], ctx_entry(self.q36, "128")["tps"])

    def test_frontier_reference_is_the_128_tok_reference(self):
        """The vs-llama.cpp percentage must be computed off the 128-tok reference."""
        self.assertEqual(self.q36["ref_tps"], ctx_entry(self.q36, "128")["ref_tps"])

    def test_kpi_and_sota_resolve_to_one_decode_number(self):
        """One 'frontier decode' number on the page, not two (#646).

        Resolves the expression as written in index.html against the live data,
        so a rebind to any other context fails here and not just on the source
        assertions above.
        """
        ns = binding_namespace(self.html, self.data)
        kpi_tps = resolve_or_chain(const_rhs(self.html, "kpiTps"), ns)
        kpi_ref = resolve_or_chain(const_rhs(self.html, "kpiRef"), ns)
        sota_tps = self.q36["frontier_tps"]          # rendered as "128-tok decode"

        self.assertEqual(kpi_tps, sota_tps,
                         "KPI decode number disagrees with the SOTA card")
        self.assertEqual(kpi_ref, ctx_entry(self.q36, "128")["ref_tps"],
                         "vs-llama.cpp %% is computed off the wrong context")
        self.assertNotEqual(kpi_tps, ctx_entry(self.q36, "512")["tps"],
                            "KPI still resolves to the 512-context row")


if __name__ == "__main__":
    unittest.main(verbosity=2)

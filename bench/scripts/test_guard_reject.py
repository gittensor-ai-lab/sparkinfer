"""No-regression gates: any failed guard rejects — gains elsewhere do not excuse it.

Mirrors evaluate.sh: reject when EVAL_MODE != short and ALL_GUARDS_PASS != true.
PR #562 scored eval:L with +13.6% @64k pp while 128k pp collapsed (287 vs 17020)
because a verified mid-ctx gain previously skipped the REJECT path.
"""
import unittest


def should_reject_for_guards(all_guards_pass: bool, eval_mode: str = "longctx") -> bool:
    if eval_mode == "short":
        return False
    return not all_guards_pass


class GuardRejectTests(unittest.TestCase):
    def test_pr562_128k_pp_fail_with_64k_gain_rejects(self):
        # Score path had HAS_VERIFIED_CONTEXT_GAIN from 64k pp +13.6%; 128k pp failed.
        all_guards_pass = False  # guard_128k_pp_pass == false
        self.assertTrue(should_reject_for_guards(all_guards_pass))

    def test_all_guards_pass_does_not_reject(self):
        self.assertFalse(should_reject_for_guards(True))

    def test_short_mode_skips_guard_reject(self):
        self.assertFalse(should_reject_for_guards(False, eval_mode="short"))


if __name__ == "__main__":
    unittest.main()

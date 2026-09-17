#!/usr/bin/env python3
"""Unit tests for the noise-ban eval penalty.

Run from the repo root:
  python3 eval/test_noise_penalty.py
"""
import datetime
import os
import unittest

import noise_penalty as np


D = datetime.date.fromisoformat


class LabelArithmeticTest(unittest.TestCase):
    def test_parks_every_eval_family(self):
        for label, parked in [
            ("eval:XL", "eval:XL-p"),
            ("eval:S", "eval:S-p"),
            ("eval:REJECT", "eval:REJECT-p"),
            ("eval:BASELINE", "eval:BASELINE-p"),
            ("eval:infra-error", "eval:infra-error-p"),
            ("eval-prefill:M", "eval-prefill:M-p"),
            ("eval-dflash:XS", "eval-dflash:XS-p"),
            ("eval-museglimmer:L", "eval-museglimmer:L-p"),
            ("eval-qwen38:XL", "eval-qwen38:XL-p"),
        ]:
            self.assertEqual(np.penalty_name(label), parked)
            self.assertEqual(np.original_name(parked), label, "restore must be exact")

    def test_none_is_exempt_in_every_family(self):
        for label in ("eval:none", "eval-prefill:none", "eval-dflash:none",
                      "eval-museglimmer:none", "eval-qwen38:none"):
            self.assertIsNone(np.penalty_name(label))

    def test_non_eval_labels_are_untouched(self):
        for label in ("area:kernels", "4k-context", "regression-4k", "penalty",
                      "flagged:gaming", "qwen38-merge-first", "eval", "evaluation:XL", ""):
            self.assertIsNone(np.penalty_name(label))
            self.assertIsNone(np.original_name(label))

    def test_parking_twice_is_a_no_op(self):
        self.assertIsNone(np.penalty_name("eval:XL-p"))
        add, remove = np.plan_labels(["eval:XL-p"], penalize=True)
        self.assertEqual((add, remove), ([], []))

    def test_restore_ignores_unparked_labels(self):
        self.assertIsNone(np.original_name("eval:XL"))
        self.assertIsNone(np.original_name("eval:none"))

    def test_plan_parks_then_restores_round_trip(self):
        labels = ["eval:XL", "eval-qwen38:M", "eval:none", "area:kernels", "4k-context"]
        add, remove = np.plan_labels(labels, penalize=True)
        self.assertEqual(add, ["eval:XL-p", "eval-qwen38:M-p"])
        self.assertEqual(remove, ["eval:XL", "eval-qwen38:M"])

        after = [l for l in labels if l not in remove] + add
        back_add, back_remove = np.plan_labels(after, penalize=False)
        self.assertEqual(sorted(back_add), ["eval-qwen38:M", "eval:XL"])
        self.assertEqual(sorted(back_remove), ["eval-qwen38:M-p", "eval:XL-p"])
        self.assertIn("eval:none", after, "the exempt tier never moves")

    def test_plan_is_empty_when_there_is_nothing_to_do(self):
        self.assertEqual(np.plan_labels(["eval:none", "area:moe"], penalize=True), ([], []))
        self.assertEqual(np.plan_labels(["eval:XL"], penalize=False), ([], []))


class BanWindowTest(unittest.TestCase):
    ENTRY = {"login": "widecloud", "start": D("2026-09-17"), "level": "ban",
             "reason": "", "line": 1}

    def test_three_day_window(self):
        self.assertEqual(np.BAN_DAYS, 3)
        self.assertEqual(np.lift_date(D("2026-09-17")), D("2026-09-20"))

    def test_active_for_exactly_three_days(self):
        for day in ("2026-09-17", "2026-09-18", "2026-09-19"):
            self.assertEqual(np.ban_status(self.ENTRY, D(day))[0], "active", day)
        for day in ("2026-09-20", "2026-09-21", "2027-01-01"):
            self.assertEqual(np.ban_status(self.ENTRY, D(day))[0], "expired", day)

    def test_future_start_is_pending_not_active(self):
        self.assertEqual(np.ban_status(self.ENTRY, D("2026-09-16"))[0], "pending")

    def test_month_boundary(self):
        entry = dict(self.ENTRY, start=D("2026-09-30"))
        self.assertEqual(np.lift_date(entry["start"]), D("2026-10-03"))
        self.assertEqual(np.ban_status(entry, D("2026-10-02"))[0], "active")
        self.assertEqual(np.ban_status(entry, D("2026-10-03"))[0], "expired")


class BanListParsingTest(unittest.TestCase):
    def test_parses_login_date_and_reason(self):
        entries, problems = np.parse_ban_list(
            "# header\n\nwidecloud  2026-09-17  ban   # sustained off-topic noise\n")
        self.assertEqual(problems, [])
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["login"], "widecloud")
        self.assertEqual(entries[0]["start"], D("2026-09-17"))
        self.assertEqual(entries[0]["level"], "ban")
        self.assertEqual(entries[0]["reason"], "sustained off-topic noise")

    def test_login_is_lowercased(self):
        entries, _ = np.parse_ban_list("WideCloud 2026-09-17\n")
        self.assertEqual(entries[0]["login"], "widecloud")

    def test_missing_or_bad_date_is_a_problem_not_a_guess(self):
        for text in ("widecloud\n", "widecloud soon\n", "widecloud 2026-13-01\n",
                     "widecloud 17-09-2026\n"):
            entries, problems = np.parse_ban_list(text)
            self.assertEqual(entries, [], text)
            self.assertEqual(len(problems), 1, text)

    def test_trailing_text_must_be_a_comment(self):
        entries, problems = np.parse_ban_list("widecloud 2026-09-17 ban spamming\n")
        self.assertEqual(entries, [])
        self.assertIn("trailing text", problems[0])

    def test_unknown_level_is_rejected(self):
        entries, problems = np.parse_ban_list("widecloud 2026-09-17 banned\n")
        self.assertEqual(entries, [])
        self.assertIn("unknown level", problems[0])

    def test_duplicate_login_is_rejected(self):
        entries, problems = np.parse_ban_list(
            "widecloud 2026-09-17 ban\nwidecloud 2026-09-18 ban\n")
        self.assertEqual(len(entries), 1)
        self.assertIn("listed twice", problems[0])

    def test_missing_file_is_a_problem_not_a_crash(self):
        entries, problems = np.parse_ban_list("")
        self.assertEqual((entries, problems), ([], []))
        entries, problems = np.load_ban_list("/nonexistent/noise-ban-list.txt")
        self.assertEqual(entries, [])
        self.assertEqual(len(problems), 1)


class OneLifeTest(unittest.TestCase):
    """A first offence is a warning: on the record, nothing parked."""

    WARN = {"login": "someone", "start": D("2026-09-17"), "level": "warn",
            "reason": "", "line": 1}

    def test_a_warning_never_becomes_active(self):
        for day in ("2026-09-16", "2026-09-17", "2026-09-18", "2026-09-19", "2026-09-30"):
            self.assertEqual(np.ban_status(self.WARN, D(day))[0], "warned", day)

    def test_same_dates_park_under_ban_but_not_under_warn(self):
        ban = dict(self.WARN, level="ban")
        self.assertEqual(np.ban_status(ban, D("2026-09-18"))[0], "active")
        self.assertEqual(np.ban_status(self.WARN, D("2026-09-18"))[0], "warned")

    def test_level_defaults_to_warn(self):
        entries, problems = np.parse_ban_list("someone 2026-09-17\n")
        self.assertEqual(problems, [])
        self.assertEqual(entries[0]["level"], "warn")
        self.assertEqual(np.ban_status(entries[0], D("2026-09-17"))[0], "warned",
                         "an unmarked line must never penalise anyone")

    def test_missing_level_field_defaults_rather_than_erroring(self):
        self.assertEqual(np.DEFAULT_LEVEL, "warn")
        self.assertEqual(np.ban_status({"login": "x", "start": D("2026-09-17")},
                                       D("2026-09-17"))[0], "warned")


class ShippedBanListTest(unittest.TestCase):
    """The list that actually ships must parse, and must still name the first ban."""

    def test_shipped_list_is_valid(self):
        entries, problems = np.load_ban_list()
        self.assertEqual(problems, [], f"ban list has problems: {problems}")
        self.assertTrue(entries, "ban list parsed to nothing")

    def test_widecloud_is_listed_as_a_warning_not_a_ban(self):
        entries, _ = np.load_ban_list()
        by_login = {e["login"]: e for e in entries}
        self.assertIn("widecloud", by_login)
        self.assertEqual(by_login["widecloud"]["level"], "warn")

    def test_nothing_in_the_shipped_list_is_currently_parking(self):
        """Guards the amnesty: no shipped entry may park a label on any future date."""
        entries, _ = np.load_ban_list()
        for e in entries:
            for day in ("2026-09-17", "2026-09-18", "2026-09-19", "2026-09-20"):
                self.assertNotEqual(np.ban_status(e, D(day))[0], "active",
                                    f"{e['login']} would still be parked on {day}")

    def test_noise_list_and_gaming_denylist_stay_separate(self):
        """A noise ban is a 3-day timeout; the denylist is permanent. Never both."""
        noise = {e["login"] for e in np.load_ban_list()[0]}
        denylist = set()
        path = os.path.join(np.ROOT, ".github", "blocked-contributors.txt")
        with open(path) as f:
            for line in f:
                s = line.split("#", 1)[0].strip().lower()
                if s:
                    denylist.add(s)
        self.assertEqual(noise & denylist, set(),
                         "an account blocked for gaming must not also carry a noise timeout")


class PrHelpersTest(unittest.TestCase):
    def test_reads_gh_json_shape(self):
        pr = {"number": 7, "labels": [{"name": "eval:XL"}, {"name": "area:kernels"}],
              "author": {"login": "WideCloud"}}
        self.assertEqual(np.label_names(pr), ["eval:XL", "area:kernels"])
        self.assertEqual(np.pr_author(pr), "widecloud")

    def test_tolerates_missing_fields(self):
        self.assertEqual(np.label_names({}), [])
        self.assertEqual(np.pr_author({}), "")
        self.assertEqual(np.pr_author({"author": None}), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)

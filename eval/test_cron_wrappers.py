"""The cron wrappers (eval/run_*_cron.sh + eval/cron_common.sh), run for real in a sandbox.

Each test builds a throwaway origin and working copy holding these wrappers and fake bots that
report where and how they were run, points HOME, the lock file and the bot tree into the sandbox,
and runs a wrapper the way cron does. Nothing touches the controller, GitHub or the eval box: the
fake GPU check fails fast (no instance configured), so every tick runs --labels-only.
"""
import fcntl
import json
import os
import shutil
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
WRAPPERS = ("cron_common.sh", "run_bonsai_pr_cron.sh", "run_museglimmer_cron.sh",
            "run_qwen38_cron.sh", "run_sync_cron.sh")

FAKE_BOT = """import json, os, sys, time
print("BOT " + json.dumps({"bot": os.path.basename(__file__), "cwd": os.getcwd(),
                          "marker": open("marker.txt").read().strip(), "args": sys.argv[1:],
                          "lock_held": os.environ.get("SPARKINFER_BOT_LOCK_HELD"),
                          "token": os.environ.get("GH_TOKEN")}))
sys.stdout.flush()
if os.environ.get("FAKE_BOT_SLEEP"):
    time.sleep(float(os.environ["FAKE_BOT_SLEEP"]))
"""
FAKE_ARB = """import os, time
def close_exhausted_eval_prs(r):
    print("SYNC close_exhausted", r, flush=True)
    if os.environ.get("FAKE_BOT_SLEEP"):
        time.sleep(float(os.environ["FAKE_BOT_SLEEP"]))
def reconcile_merge_labels(r):
    print("SYNC reconcile", r, "automerge=" + os.environ.get("SPARKINFER_AUTOMERGE", ""), "cwd=" + os.getcwd())
"""
FAKE_GH = """#!/bin/sh
# `gh api user --jq .login`, answered from FAKE_GH_LOGIN.
echo "$FAKE_GH_LOGIN"
"""


def git(*args, cwd=None):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True,
                   capture_output=True, text=True)


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.mkdtemp(prefix="sparkinfer-wrappers-")
        self.addCleanup(shutil.rmtree, self.t, True)
        self.origin, self.seed = os.path.join(self.t, "origin.git"), os.path.join(self.t, "seed")
        self.repo, self.home = os.path.join(self.t, "repo"), os.path.join(self.t, "home")
        self.tree, self.lock = os.path.join(self.home, "tree"), os.path.join(self.t, "bot.lock")
        os.makedirs(self.home)
        git("init", "-q", "--bare", "-b", "main", self.origin)
        git("clone", "-q", self.origin, self.seed)
        os.makedirs(os.path.join(self.seed, "eval"))
        for w in WRAPPERS:
            shutil.copy(os.path.join(HERE, w), os.path.join(self.seed, "eval", w))
        for b in ("pr_bonsai_bot.py", "pr_museglimmer_bot.py", "pr_qwen38_bot.py"):
            with open(os.path.join(self.seed, "eval", b), "w") as f:
                f.write(FAKE_BOT)
        with open(os.path.join(self.seed, "eval", "pr_eval_bot.py"), "w") as f:
            f.write(FAKE_ARB)
        self.commit("v1")
        git("clone", "-q", self.origin, self.repo)
        self.fake_gh = os.path.join(self.t, "gh")
        with open(self.fake_gh, "w") as f:
            f.write(FAKE_GH)
        os.chmod(self.fake_gh, 0o755)

    def commit(self, marker, cwd=None):
        cwd = cwd or self.seed
        with open(os.path.join(cwd, "marker.txt"), "w") as f:
            f.write(marker + "\n")
        git("add", "-A", cwd=cwd)
        git("commit", "-q", "-m", marker, cwd=cwd)
        if cwd == self.seed:
            git("checkout", "-q", "-B", "main", cwd=cwd)
            git("push", "-q", "origin", "main", cwd=cwd)

    def tick(self, wrapper="run_bonsai_pr_cron.sh", token="t0ken", path=None, **env):
        e = {"HOME": self.home, "PATH": "/usr/bin:/bin", "SPARKINFER_LOCK_FILE": self.lock,
             "SPARKINFER_BOT_TREE": self.tree, "SPARKINFER_LOCK_WAIT_S": "2"}
        if token:
            e["GH_TOKEN"] = token
        e.update(env)
        r = subprocess.run(["bash", path or os.path.join(self.repo, "eval", wrapper)], env=e,
                           capture_output=True, text=True, timeout=180)
        bots = [json.loads(l[4:]) for l in r.stdout.splitlines() if l.startswith("BOT ")]
        return r, bots

    def test_no_token_is_refused_and_a_run_of_refusals_is_loud(self):
        for w in ("run_museglimmer_cron.sh", "run_qwen38_cron.sh", "run_sync_cron.sh"):
            r, bots = self.tick(wrapper=w, token="")
            self.assertEqual((r.returncode, bots), (1, []), w)
            self.assertIn("GH_TOKEN is empty", r.stderr, w)
            self.assertNotIn("SYNC", r.stdout, w)
        for i in range(3):
            r, bots = self.tick(token="")
            self.assertEqual(r.returncode, 1)
            self.assertEqual(bots, [])
            self.assertIn("GH_TOKEN is empty", r.stderr)
        self.assertIn("EVAL NOT RUNNING: 3 consecutive ticks", r.stderr)
        r, bots = self.tick()
        self.assertEqual(len(bots), 1)
        self.assertFalse(os.path.exists(os.path.join(self.home, ".sparkinfer_bonsai_refused")))

    def test_the_bot_runs_origin_main_from_its_own_tree_whatever_the_working_copy_holds(self):
        r, bots = self.tick()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(bots[0]["cwd"], self.tree)
        self.assertEqual(bots[0]["marker"], "v1")
        self.assertIn("--labels-only", bots[0]["args"])            # no GPU configured in the sandbox
        self.assertEqual(bots[0]["lock_held"], "1")                 # arb.hold_bot_lock must not wait
        self.commit("v2")
        git("checkout", "-q", "-b", "feature", cwd=self.repo)
        self.commit("branch-code", cwd=self.repo)
        with open(os.path.join(self.tree, "marker.txt"), "w") as f:
            f.write("dirty\n")
        r, bots = self.tick(wrapper="run_museglimmer_cron.sh")
        self.assertEqual((bots[0]["cwd"], bots[0]["marker"]), (self.tree, "v2"))
        self.assertIn("is not on main", r.stderr)
        with open(os.path.join(self.repo, "marker.txt")) as f:
            self.assertEqual(f.read().strip(), "branch-code")     # the branch itself is left alone

    def test_every_bot_runs_from_the_tree(self):
        for w in ("run_bonsai_pr_cron.sh", "run_museglimmer_cron.sh", "run_qwen38_cron.sh"):
            r, bots = self.tick(wrapper=w)
            self.assertEqual(bots[0]["cwd"], self.tree, w)
        r, _ = self.tick(wrapper="run_sync_cron.sh")
        self.assertIn("cwd=" + self.tree, r.stdout)

    def test_a_working_copy_on_main_follows_origin(self):
        self.tick()
        self.commit("v2")
        r, bots = self.tick(wrapper="run_qwen38_cron.sh")
        self.assertEqual((bots[0]["marker"], bots[0]["cwd"]), ("v2", self.tree))
        with open(os.path.join(self.repo, "marker.txt")) as f:
            self.assertEqual(f.read().strip(), "v2")

    def test_a_broken_tree_is_made_again(self):
        self.tick()
        admin = os.path.join(self.repo, ".git", "worktrees")
        # A working copy re-cloned under the tree: its admin dir is gone.
        shutil.rmtree(admin)
        r, bots = self.tick()
        self.assertEqual((r.returncode, bots[0]["marker"]), (0, "v1"), r.stderr)
        # A checkout killed part-way leaves the tree's index locked.
        (name,) = os.listdir(admin)
        open(os.path.join(admin, name, "index.lock"), "w").close()
        r, bots = self.tick()
        self.assertEqual((r.returncode, bots[0]["marker"]), (0, "v1"), r.stderr)

    def test_a_healthy_tree_is_reused_not_made_again(self):
        r, _ = self.tick()
        self.assertIn("(origin/main, new)", r.stdout)
        keep = os.path.join(self.tree, "build-cache")
        os.makedirs(keep)
        with open(os.path.join(keep, "x"), "w") as f:              # ignored by nothing: clean -fd drops it
            f.write("x")
        ino = os.stat(os.path.join(self.tree, "marker.txt")).st_ino
        r, bots = self.tick()
        self.assertNotIn("new", r.stdout.split("bot tree:")[1].splitlines()[0])
        self.assertNotIn("making it again", r.stderr)
        self.assertEqual(os.stat(os.path.join(self.tree, "marker.txt")).st_ino, ino)
        self.assertFalse(os.path.exists(keep))                      # but it is clean

    def test_someone_elses_worktree_is_refused_and_left_exactly_as_it_was(self):
        # A worktree of the same repository that the wrappers did not make: a dev checkout, say.
        git("worktree", "add", "-q", "-b", "dev", self.tree, cwd=self.repo)
        notes = os.path.join(self.tree, "notes.txt")
        with open(notes, "w") as f:
            f.write("uncommitted work\n")
        r, bots = self.tick()
        self.assertEqual((r.returncode, bots), (1, []))
        self.assertIn("not the bot tree these wrappers made", r.stderr)
        self.assertTrue(os.path.exists(notes))
        out = subprocess.run(["git", "-C", self.tree, "branch", "--show-current"], capture_output=True, text=True)
        self.assertEqual(out.stdout.strip(), "dev")

    def test_a_tree_whose_admin_dir_is_gone_is_moved_aside_not_deleted(self):
        self.tick()
        open(os.path.join(self.tree, "left-behind"), "w").close()
        shutil.rmtree(os.path.join(self.repo, ".git", "worktrees"))
        r, bots = self.tick()
        self.assertEqual((r.returncode, bots[0]["marker"]), (0, "v1"), r.stderr)
        aside = [d for d in os.listdir(self.home) if d.startswith("tree.orphan-")]
        self.assertEqual(len(aside), 1)
        self.assertTrue(os.path.exists(os.path.join(self.home, aside[0], "left-behind")))

    def test_a_token_of_another_account_is_refused(self):
        env = {"SPARKINFER_BOT_LOGIN": "bot-account", "SPARKINFER_GH": self.fake_gh}
        r, bots = self.tick(FAKE_GH_LOGIN="someone-else", **env)
        self.assertEqual((r.returncode, bots), (1, []))
        self.assertIn("belongs to someone-else", r.stderr)
        r, bots = self.tick(FAKE_GH_LOGIN="bot-account", **env)
        self.assertEqual(len(bots), 1)
        self.assertIn("acting as GitHub account: bot-account", r.stdout)
        r, bots = self.tick(FAKE_GH_LOGIN="", **env)                 # GitHub did not answer
        self.assertEqual(len(bots), 1)

    def test_the_sync_reads_env_eval_too_and_never_merges(self):
        with open(os.path.join(self.repo, ".env.eval"), "w") as f:
            f.write("SPARKINFER_BOT_LOGIN=bot-account\nSPARKINFER_AUTOMERGE=1\n")
        r, _ = self.tick(wrapper="run_sync_cron.sh", FAKE_GH_LOGIN="someone-else", SPARKINFER_GH=self.fake_gh)
        self.assertEqual(r.returncode, 1)
        self.assertNotIn("SYNC", r.stdout)
        r, _ = self.tick(wrapper="run_sync_cron.sh", FAKE_GH_LOGIN="bot-account", SPARKINFER_GH=self.fake_gh)
        self.assertIn("SYNC reconcile", r.stdout)
        self.assertIn("automerge= ", r.stdout)

    def test_the_sync_is_short(self):
        import time as _t
        with open(self.lock, "a") as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            t0 = _t.monotonic()
            r, _ = self.tick(wrapper="run_sync_cron.sh", SPARKINFER_LOCK_WAIT_S="30")
            self.assertLess(_t.monotonic() - t0, 10)                 # never waits for a round
            self.assertEqual(r.returncode, 0)
        r, _ = self.tick(wrapper="run_sync_cron.sh", FAKE_BOT_SLEEP="30", SPARKINFER_SYNC_TIMEOUT_S="1")
        self.assertIn("bot run stopped after 1s", r.stderr)

    def test_a_bot_that_fails_every_tick_is_loud(self):
        with open(os.path.join(self.seed, "eval", "pr_qwen38_bot.py"), "w") as f:
            f.write("raise SystemExit(3)\n")
        self.commit("broken")
        for _ in range(3):
            r, _ = self.tick(wrapper="run_qwen38_cron.sh")
            self.assertEqual(r.returncode, 3)
        self.assertIn("qwen38 BOT FAILING: 3 consecutive runs ended with exit 3", r.stderr)

    def test_a_tree_deleted_by_hand_is_made_again(self):
        self.tick()
        shutil.rmtree(self.tree)                                    # still registered with git
        r, bots = self.tick()
        self.assertEqual((r.returncode, bots[0]["marker"]), (0, "v1"), r.stderr)
        r, bots = self.tick(SPARKINFER_BOT_TREE=self.tree + "/")    # a trailing slash is the same tree
        self.assertEqual((r.returncode, bots[0]["cwd"]), (0, self.tree), r.stderr)

    def test_a_refused_tick_says_why_in_its_banner(self):
        os.makedirs(self.tree)
        open(os.path.join(self.tree, "x"), "w").close()
        for _ in range(3):
            r, _ = self.tick()
        self.assertIn("Last reason: " + self.tree + " is not the bot tree these wrappers made", r.stderr)

    def test_the_crontabs_token_wins_over_env_eval(self):
        with open(os.path.join(self.repo, ".env.eval"), "w") as f:
            f.write("GH_TOKEN=someone-elses\n")
        r, bots = self.tick(token="bot-token")
        self.assertEqual(bots[0]["token"], "bot-token")

    def test_a_lock_file_that_cannot_be_opened_is_named_as_such(self):
        os.makedirs(self.lock)                                       # not a file: exec 9> fails
        r, bots = self.tick()
        self.assertEqual((r.returncode, bots), (1, []))
        self.assertIn("cannot open the lock file", r.stderr)
        self.assertNotIn("another bot's round", r.stdout)

    def test_repeated_fetch_failures_are_loud(self):
        self.tick()
        shutil.move(self.origin, self.origin + ".gone")
        for _ in range(3):
            r, bots = self.tick()
            self.assertEqual(len(bots), 1)                          # still runs the last fetched main
        self.assertIn("BOTS RUNNING A STALE origin/main: 3 consecutive fetches failed", r.stderr)

    def test_a_directory_that_is_not_a_worktree_is_never_removed(self):
        os.makedirs(self.tree)
        keep = os.path.join(self.tree, "someone's file")
        open(keep, "w").close()
        r, bots = self.tick()
        self.assertEqual((r.returncode, bots), (1, []))
        self.assertIn("not the bot tree these wrappers made", r.stderr)
        self.assertTrue(os.path.exists(keep))

    def test_a_held_lock_skips_the_tick_and_a_run_of_skips_is_loud(self):
        with open(self.lock, "a") as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            for _ in range(3):
                r, bots = self.tick(SPARKINFER_LOCK_WAIT_S="1")
                self.assertEqual((r.returncode, bots), (0, []))
        self.assertIn("skipping bonsai tick (3 in a row)", r.stdout)
        self.assertIn("EVAL STARVED", r.stderr)
        r, bots = self.tick()
        self.assertEqual(len(bots), 1)
        self.assertFalse(os.path.exists(os.path.join(self.home, ".sparkinfer_bonsai_lock_skips")))
        # The sync never waits: it skips at once while a round holds the lock.
        with open(self.lock, "a") as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            r, _ = self.tick(wrapper="run_sync_cron.sh", SPARKINFER_LOCK_WAIT_S="30")
            self.assertEqual(r.returncode, 0)
            self.assertNotIn("SYNC", r.stdout)

    def test_a_run_that_hangs_is_stopped(self):
        r, bots = self.tick(FAKE_BOT_SLEEP="30", SPARKINFER_BOT_TIMEOUT_S="1")
        self.assertEqual(r.returncode, 124)
        self.assertIn("bot run stopped after 1s", r.stderr)

    def test_a_dead_box_is_loud_on_every_bot(self):
        for wrapper, name in (("run_qwen38_cron.sh", "QWEN38"), ("run_bonsai_pr_cron.sh", "BONSAI"),
                              ("run_museglimmer_cron.sh", "MUSE")):
            for _ in range(3):
                r, _ = self.tick(wrapper=wrapper)
            self.assertIn(f"{name} EVAL DEGRADED: 3 consecutive ticks", r.stderr, wrapper)

    def test_the_sync_runs_from_the_tree_too(self):
        r, _ = self.tick(wrapper="run_sync_cron.sh")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("SYNC close_exhausted", r.stdout)
        self.assertIn("SYNC reconcile", r.stdout)


if __name__ == "__main__":
    unittest.main()

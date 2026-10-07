"""Report-only mode (eval/report_only.py): GitHub writes are logged and skipped, reads still run.

Nothing here touches GitHub: the real subprocess.run is replaced by a recorder before install().
"""
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import report_only as ro  # noqa: E402

WRITES = [
    ["gh", "pr", "comment", "12", "-R", "o/r", "--body", "verdict"],
    ["gh", "pr", "edit", "12", "-R", "o/r", "--add-label", "eval:L"],
    ["gh", "pr", "merge", "12", "--squash"],
    ["gh", "pr", "close", "12", "-R", "o/r"],
    ["gh", "pr", "review", "12", "--approve"],
    ["gh", "pr", "reopen", "12"],
    ["gh", "issue", "comment", "3", "--body", "x"],
    ["gh", "label", "create", "eval:XL"],
    ["gh", "api", "-X", "POST", "repos/o/r/issues/12/labels", "-f", "labels[]=eval:S"],
    ["gh", "api", "repos/o/r/issues/12/comments", "-f", "body=hi"],
    ["gh", "api", "--method=DELETE", "repos/o/r/issues/12/labels/eval:S"],
    ["gh", "api", "graphql", "-f", "query=mutation { addLabelsToLabelable(input:{}) { clientMutationId } }"],
    ["gh", "something-new", "do"],
    ["git", "push", "-q", "origin", "main"],
    ["git", "-C", "/tmp/logs", "push", "-q"],
    ["/usr/bin/git", "-c", "user.name=x", "push"],
]
READS = [
    ["gh", "pr", "list", "-R", "o/r", "--json", "number"],
    ["gh", "pr", "view", "12", "--json", "labels"],
    ["gh", "pr", "checks", "12"],
    ["gh", "pr", "diff", "12"],
    ["gh", "api", "repos/o/r/pulls/12"],
    ["gh", "api", "-X", "GET", "repos/o/r/issues", "-f", "state=open"],
    ["gh", "api", "graphql", "-f", "query={ repository(owner:\"o\", name:\"r\") { id } }"],
    ["gh", "auth", "token"],
    ["gh", "label", "list"],
    ["git", "fetch", "-q", "origin", "main"],
    ["git", "-C", "/tmp/logs", "commit", "-q", "-m", "push the logs"],
    ["git", "rev-parse", "--short", "origin/main"],
    ["ssh", "-p", "1", "root@h", "git push"],
    "gh pr comment 1",   # a string command is not an argv: never interpreted
]


class Classify(unittest.TestCase):
    def test_writes(self):
        for cmd in WRITES:
            self.assertTrue(ro.is_github_write(cmd), cmd)

    def test_reads(self):
        for cmd in READS:
            self.assertFalse(ro.is_github_write(cmd), cmd)


class Install(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.real = subprocess.run
        self.tmp = tempfile.NamedTemporaryFile(delete=False)
        self.tmp.close()

        def recorder(*a, **k):
            self.calls.append(a[0] if a else k.get("args"))
            return subprocess.CompletedProcess(a[0] if a else k.get("args"), 0, "out", "")
        subprocess.run = recorder

    def tearDown(self):
        subprocess.run = self.real
        os.unlink(self.tmp.name)

    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {"SPARKINFER_REPORT_ONLY": "0"}):
            self.assertFalse(ro.install())
        subprocess.run(["gh", "pr", "merge", "1"])
        self.assertEqual(self.calls, [["gh", "pr", "merge", "1"]])

    def test_writes_blocked_reads_pass(self):
        with mock.patch.dict(os.environ, {"SPARKINFER_REPORT_ONLY": "1"}), \
                mock.patch.object(ro, "LOG", self.tmp.name):
            self.assertTrue(ro.install())
            self.assertTrue(ro.install())   # idempotent: no double wrap
            body = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False)
            body.write("## eval verdict\nL\n")
            body.close()
            r = subprocess.run(["gh", "pr", "comment", "7", "--body-file", body.name],
                               capture_output=True, text=True)
            self.assertEqual((r.returncode, r.stdout, r.stderr), (0, "", ""))
            r = subprocess.run(["git", "-C", "/x", "push", "-q"], check=False)
            self.assertEqual(r.returncode, 0)
            r = subprocess.run(["gh", "pr", "view", "7"], capture_output=True, text=True)
            self.assertEqual(r.stdout, "out")
            os.unlink(body.name)
        self.assertEqual(self.calls, [["gh", "pr", "view", "7"]])
        log = open(self.tmp.name).read()
        self.assertIn("gh\npr\ncomment\n7", log)
        self.assertIn("## eval verdict", log)
        self.assertIn("push", log)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env bash
# Entrypoint of the release container when it runs as a Gittensor compute-pool workload (spec:
# https://github.com/entrius/gittensor-compute-template, MANIFEST.md). Two jobs, then the normal entrypoint:
#
# 1. With SPARKINFER_NO_DOWNLOAD=1 (pre-staged weights, no egress) verify every `artifacts[]` entry of the manifest
#    on disk against its sha256 before serving. The pool's controller stages the files and bind-mounts the manifest
#    it signed over /manifest.yaml; a mismatch means the wrong weights or a partial stage, and the only right answer
#    is to refuse to serve (exit 1). Without SPARKINFER_NO_DOWNLOAD the container is somebody's own `docker run`:
#    nothing is verified and nothing changes.
# 2. The pool's manifest can set environment but not arguments, so SPARKINFER_MODE=serve-dspark selects the
#    drafter mode exactly as `docker run ... serve-dspark` does. Explicit arguments still win.
#
# Then exec entrypoint.sh, which execs sparkinfer_server: SIGTERM lands on the server and its drain runs as
# documented (SPARKINFER_DRAIN_GRACE_S).
set -euo pipefail

MANIFEST="${MANIFEST_PATH:-/manifest.yaml}"

verify_artifacts() {
    python3 - "$MANIFEST" <<'PY'
import hashlib, os, sys, yaml

# One definition of an artifact's sha256, shared with the pool's controller: a file hashes its bytes; a directory
# hashes "<relpath>\0<file sha256 hex>\n" over its files in sorted walk order, skipping top-level dotfiles and
# dot-directories (.revision, .gitattributes, .cache): markers and source metadata, not content.
def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

manifest = yaml.safe_load(open(sys.argv[1])) or {}
failed = False
for art in manifest.get('artifacts') or []:
    path, want = art['path'], art['sha256'].lower()
    if os.path.isdir(path):
        walk = []
        for root, dirs, files in os.walk(path):
            if root == path:
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                files = [f for f in files if not f.startswith('.')]
            walk.append((root, files))
        h = hashlib.sha256()
        for root, files in sorted(walk):
            for name in sorted(files):
                full = os.path.join(root, name)
                h.update(os.path.relpath(full, path).encode() + b'\0' + file_sha256(full).encode() + b'\n')
        got = h.hexdigest()
    elif os.path.isfile(path):
        got = file_sha256(path)
    else:
        print(f'[sparkinfer] FATAL: artifact {path} is missing (pre-staged by the controller; never downloaded here)', file=sys.stderr)
        failed = True
        continue
    if got != want:
        print(f'[sparkinfer] FATAL: artifact {path} sha256 MISMATCH (got {got}, want {want}); refusing to serve', file=sys.stderr)
        failed = True
    else:
        print(f'[sparkinfer] artifact {path} sha256 OK', file=sys.stderr)
sys.exit(1 if failed else 0)
PY
}

if [ "${SPARKINFER_NO_DOWNLOAD:-0}" = "1" ] && [ -f "$MANIFEST" ]; then
    verify_artifacts
fi

if [ $# -eq 0 ] && [ -n "${SPARKINFER_MODE:-}" ]; then
    set -- "$SPARKINFER_MODE"
fi
exec /opt/sparkinfer/entrypoint.sh "$@"

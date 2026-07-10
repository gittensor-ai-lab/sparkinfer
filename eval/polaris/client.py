#!/usr/bin/env python3
"""Polaris API client for submitting scoring workloads to Intel TDX enclaves.

Usage:
    from eval.polaris.client import PolarisClient

    client = PolarisClient()  # reads POLARIS_API_KEY from env
    receipt = client.attest_scoring(result_json, nonce, e2e_pubkey_b64)
    # receipt["verification"]["intel_verified"] is True when TDX attestation passes
"""

import base64
import hashlib
import json
import os
import urllib.request
import urllib.error

POLARIS_BASE = os.environ.get("POLARIS_API_BASE", "https://polaris.computer")


def _attest_endpoint() -> str:
    base = os.environ.get("POLARIS_API_BASE", POLARIS_BASE).rstrip("/")
    return f"{base}/v1/attest"

# The scoring script that runs inside the TDX enclave.
# Loaded from the sibling scoring.py module.
_SCORING_SOURCE = None


def _get_scoring_source() -> str:
    """Load the scoring script source from the sibling scoring.py file."""
    global _SCORING_SOURCE
    if _SCORING_SOURCE is not None:
        return _SCORING_SOURCE
    import os as _os
    _path = _os.path.join(_os.path.dirname(__file__), "scoring.py")
    with open(_path) as f:
        _SCORING_SOURCE = f.read()
    return _SCORING_SOURCE


class PolarisClient:
    """Client for the Polaris /v1/attest API.

    Submits workloads to Intel TDX enclaves and returns DCAP-quoted receipts.
    """

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.environ.get("POLARIS_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "POLARIS_API_KEY is required. Set the environment variable "
                "or pass api_key to the constructor."
            )

    def attest(
        self,
        workload: str,
        files: dict = None,
        nonce: str = "",
        e2e_pubkey_b64: str = "",
        image: str = "python:3.12-slim",
        egress: str = "none",
    ) -> dict:
        """Submit a workload to Polaris TDX and return the attestation response.

        Args:
            workload: Shell command to run inside the enclave.
            files: Dict mapping destination paths to base64-encoded content.
                   e.g. {"/submission/score.py": "cHJpbnQoJ2hlbGxvJyk="}
            nonce: Optional nonce bound into the DCAP quote (max 64 bytes).
            e2e_pubkey_b64: Optional end-to-end public key bound into the quote.
            image: Docker image to run (default: python:3.12-slim).
            egress: Network egress policy for the enclave ("none" or "full").

        Returns:
            The full Polaris attestation response dict with keys:
            tee_attestation, result_sha256, stdout_b64, verification, cost_usd.
        """
        body = {
            "workload": workload,
            "image": image,
            "egress": egress,
        }
        if files:
            body["files"] = files
        if nonce:
            body["nonce"] = nonce
        if e2e_pubkey_b64:
            body["e2e_pubkey_b64"] = e2e_pubkey_b64

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            _attest_endpoint(),
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "sparkinfer-eval/1.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body_text = ""
            try:
                body_text = e.read().decode("utf-8")
            except Exception:
                pass
            raise RuntimeError(
                f"Polaris attest failed: HTTP {e.code} — {body_text}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Polaris attest failed: {e.reason}") from e

    def attest_scoring(
        self,
        result_json: dict,
        nonce: str,
        e2e_pubkey_b64: str,
    ) -> dict:
        """Convenience method: submit the standard scoring workload to Polaris TDX.

        Encodes the scoring script + RESULT_JSON as base64 files, submits to
        the enclave, and returns the Polaris attestation response.

        Args:
            result_json: The merged RESULT_JSON dict (primary + guard fields).
            nonce: Nonce bound into the DCAP quote.
            e2e_pubkey_b64: SparkInfer's Ed25519 public key, bound into the quote.

        Returns:
            Full Polaris attestation response.
        """
        scoring_source = _get_scoring_source()
        result_bytes = json.dumps(result_json, sort_keys=True).encode("utf-8")

        files = {
            "/submission/score.py": base64.b64encode(
                scoring_source.encode("utf-8")
            ).decode("ascii"),
            "/submission/result.json": base64.b64encode(result_bytes).decode("ascii"),
        }

        # Compute expected hashes for later verification
        scoring_sha256 = hashlib.sha256(scoring_source.encode("utf-8")).hexdigest()
        result_sha256 = hashlib.sha256(result_bytes).hexdigest()

        response = self.attest(
            workload="python3 /submission/score.py",
            files=files,
            nonce=nonce,
            e2e_pubkey_b64=e2e_pubkey_b64,
        )

        # Attach our expected hashes for the receipt builder to use
        response["_expected"] = {
            "scoring_sha256": scoring_sha256,
            "result_sha256": result_sha256,
        }

        return response

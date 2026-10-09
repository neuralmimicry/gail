"""Focused tests for the secret boundary in Gail's Slurm model downloader."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "trainer" / "model_cache_download.py"
SPEC = importlib.util.spec_from_file_location("gail_model_cache_download", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
downloader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(downloader)


class ModelCacheDownloadTests(unittest.TestCase):
    def test_model_reference_requires_repository_and_immutable_commit(self) -> None:
        downloader.validate_model_reference(
            "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        )
        for model_id, revision in (
            ("../private/model", "a" * 40),
            ("Qwen/model", "main"),
            ("Qwen/model", "A" * 40),
        ):
            with self.subTest(model_id=model_id, revision=revision):
                with self.assertRaises(ValueError):
                    downloader.validate_model_reference(model_id, revision)

    def test_credential_response_returns_only_bounded_access_token(self) -> None:
        self.assertEqual(
            downloader.parse_credential_response(
                {
                    "provider": "huggingface",
                    "credential_present": True,
                    "access_token": "hf_test_token",
                    "username": "must-not-be-consumed",
                    "password": "must-not-be-consumed",
                }
            ),
            "hf_test_token",
        )
        self.assertIsNone(
            downloader.parse_credential_response(
                {"provider": "huggingface", "credential_present": False}
            )
        )
        with self.assertRaises(ValueError):
            downloader.parse_credential_response(
                {
                    "provider": "huggingface",
                    "credential_present": True,
                    "access_token": "x" * (downloader.MAX_PROVIDER_TOKEN_BYTES + 1),
                }
            )

    def test_safe_error_reporting_keeps_status_and_hides_unknown_provider_text(self) -> None:
        self.assertEqual(
            downloader.safe_failure_detail(
                downloader.SafeDownloaderError("Gail model credential request returned HTTP 401")
            ),
            "Gail model credential request returned HTTP 401",
        )

        class ProviderFailure(Exception):
            pass

        secret = "hf_must_not_be_logged"
        self.assertNotIn(secret, downloader.safe_failure_detail(ProviderFailure(secret)))

    def test_api_token_must_be_readable_only_by_its_service_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            token_file = Path(directory) / "api-token"
            token_file.write_text("gail-service-token\n", encoding="utf-8")
            token_file.chmod(0o600)
            self.assertEqual(downloader.read_api_token(token_file), "gail-service-token")
            token_file.chmod(0o644)
            with self.assertRaises(PermissionError):
                downloader.read_api_token(token_file)

    def test_provider_credential_request_uses_https_and_keeps_auth_out_of_url(self) -> None:
        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def geturl(self) -> str:
                return "https://gail.example/v1/internal/model-download-credential"

            def read(self, _size: int) -> bytes:
                return json.dumps(
                    {
                        "provider": "huggingface",
                        "credential_present": True,
                        "access_token": "hf_runtime_token",
                    }
                ).encode("utf-8")

        endpoint = "https://gail.example/v1/internal/model-download-credential"
        observed_handlers = []

        class FakeOpener:
            def open(self, request, timeout):
                self.request = request
                self.timeout = timeout
                return FakeResponse()

        fake_opener = FakeOpener()

        def build_opener(*handlers):
            observed_handlers.extend(handlers)
            return fake_opener

        with patch.object(
            downloader.urllib.request,
            "build_opener",
            side_effect=build_opener,
        ) as open_url:
            self.assertEqual(
                downloader.fetch_provider_token(endpoint, "machine-download-token"),
                "hf_runtime_token",
            )
        request = fake_opener.request
        self.assertEqual(request.get_header("Authorization"), "Bearer machine-download-token")
        self.assertNotIn("machine-download-token", request.full_url)
        self.assertEqual(open_url.call_count, 1)
        self.assertEqual(len(observed_handlers), 1)
        self.assertIsInstance(observed_handlers[0], downloader.RejectRedirectHandler)
        self.assertIsNone(
            observed_handlers[0].redirect_request(
                request,
                None,
                302,
                "Found",
                {},
                "https://attacker.example/collect",
            )
        )
        with self.assertRaises(ValueError):
            downloader.fetch_provider_token(endpoint.replace("https://", "http://"), "token")

    def test_credential_request_surfaces_redirect_without_following_it(self) -> None:
        endpoint = "https://gail.example/v1/internal/model-download-credential"
        observed_handlers = []
        opener_calls = []

        class FakeOpener:
            def open(self, request, timeout):
                opener_calls.append((request, timeout))
                raise downloader.urllib.error.HTTPError(
                    request.full_url,
                    302,
                    "Found",
                    {},
                    None,
                )

        def build_opener(*handlers):
            observed_handlers.extend(handlers)
            return FakeOpener()

        with patch.object(
            downloader.urllib.request,
            "build_opener",
            side_effect=build_opener,
        ):
            with self.assertRaises(downloader.SafeDownloaderError) as error:
                downloader.fetch_provider_token(endpoint, "machine-download-token")

        self.assertIn("HTTP 302", str(error.exception))
        self.assertEqual(len(opener_calls), 1)
        self.assertEqual(len(observed_handlers), 1)
        self.assertIsInstance(observed_handlers[0], downloader.RejectRedirectHandler)

    def test_provenance_is_atomic_non_secret_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "snapshot" / "base_model_provenance.json"
            downloader.write_provenance(
                target,
                {
                    "provider": "huggingface",
                    "model_id": "Qwen/Qwen3.5-4B",
                    "resolved_revision": "a" * 40,
                },
            )
            metadata = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(metadata["model_id"], "Qwen/Qwen3.5-4B")
            self.assertEqual(target.stat().st_mode & 0o777, 0o640)
            self.assertEqual(
                sorted(path.name for path in target.parent.iterdir()),
                ["base_model_provenance.json"],
            )


if __name__ == "__main__":
    unittest.main()

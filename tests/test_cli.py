"""CLI: end to end against the mock endpoint, including the awkward paths."""

import json
import os

import pytest

from model_fingerprint import __version__, cli, layers
from model_fingerprint.cli import build_parser, main, parse_headers, resolve_key


def base_args(mock, tmp_path, extra=()):
    return [
        "--base-url",
        mock.base_url,
        "--model",
        "mock-alpha",
        "--api",
        "openai",
        "--api-key",
        "k",
        "--quick",
        "--cells",
        "fav_number",
        "--out",
        str(tmp_path / "fp.json"),
        "--report",
        str(tmp_path / "fp.md"),
        "--quiet",
        *extra,
    ]


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--base-url", "u", "--model", "m", "--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_help_lists_core_flags(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--help"])
    out = capsys.readouterr().out
    for flag in (
        "--base-url",
        "--model",
        "--api-key-env",
        "--env-file",
        "--header",
        "--reasoning-effort",
        "--cell-budget",
        "--context-probe",
        "--report",
    ):
        assert flag in out


def test_parse_headers_ignores_junk():
    assert parse_headers(["A: 1", "B:two:three", "no-colon"]) == {"A": "1", "B": "two:three"}
    assert parse_headers(None) == {}


def test_resolve_key_precedence(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("FROM_FILE=filekey\n")
    args = build_parser().parse_args(["--base-url", "u", "--model", "m", "--api-key", "literal"])
    assert resolve_key(args) == "literal"

    args = build_parser().parse_args(
        ["--base-url", "u", "--model", "m", "--api-key-env", "FROM_ENV"]
    )
    monkeypatch.setenv("FROM_ENV", "envkey")
    assert resolve_key(args) == "envkey"

    monkeypatch.delenv("FROM_ENV")
    args = build_parser().parse_args(
        [
            "--base-url",
            "u",
            "--model",
            "m",
            "--api-key-env",
            "FROM_FILE",
            "--env-file",
            str(env_file),
        ]
    )
    assert resolve_key(args) == "filekey"

    args = build_parser().parse_args(["--base-url", "u", "--model", "m"])
    assert resolve_key(args) == ""


def test_main_writes_json_and_report(mock, tmp_path, capsys):
    code = main(base_args(mock, tmp_path))
    assert code == 0
    fingerprint = json.loads((tmp_path / "fp.json").read_text())
    for layer in ("transport", "errors", "tokenizer", "capabilities", "limits", "behaviour"):
        assert layer in fingerprint
        assert "unavailable" not in fingerprint[layer]
    assert fingerprint["meta"]["api"] == "openai"
    assert fingerprint["meta"]["depth"] == "quick"
    assert fingerprint["meta"]["model"] == "mock-alpha"
    assert fingerprint["requests"]
    assert fingerprint["tokenizer"]["probes_usable"] == fingerprint["tokenizer"]["probes_attempted"]
    report = (tmp_path / "fp.md").read_text()
    assert "## Grading worksheet" in report
    assert "mock-alpha" in report
    assert capsys.readouterr().out == ""  # --quiet


def test_main_prints_report_by_default(mock, tmp_path, capsys):
    args = [a for a in base_args(mock, tmp_path) if a != "--quiet"]
    assert main(args) == 0
    assert "# Fingerprint:" in capsys.readouterr().out


def test_report_path_defaults_to_out_stem(mock, tmp_path):
    args = [
        "--base-url",
        mock.base_url,
        "--model",
        "mock-alpha",
        "--api-key",
        "k",
        "--quick",
        "--cells",
        "fav_number",
        "--quiet",
        "--out",
        str(tmp_path / "nested.json"),
    ]
    assert main(args) == 0
    assert (tmp_path / "nested.md").exists()


def test_missing_key_warns_but_runs(mock, tmp_path, capsys):
    args = [
        "--base-url",
        mock.base_url,
        "--model",
        "mock-alpha",
        "--quick",
        "--cells",
        "fav_number",
        "--out",
        str(tmp_path / "fp.json"),
        "--report",
        str(tmp_path / "fp.md"),
        "--quiet",
    ]
    assert main(args) == 0
    assert "no API key" in capsys.readouterr().err


def test_dead_layer_is_recorded_not_hidden(mock, tmp_path, monkeypatch):
    def explode(*_args, **_kwargs):
        raise TimeoutError("behaviour probe timed out")

    monkeypatch.setattr(layers, "behaviour", explode)
    args = base_args(mock, tmp_path)
    assert main(args) == 0
    fingerprint = json.loads((tmp_path / "fp.json").read_text())
    assert "TimeoutError" in fingerprint["behaviour"]["unavailable"]
    assert "layer `behaviour` unavailable" in (tmp_path / "fp.md").read_text()


def test_unreachable_endpoint_still_produces_a_report(tmp_path, capsys):
    args = [
        "--base-url",
        "http://127.0.0.1:1/v1",
        "--model",
        "ghost",
        "--api-key",
        "k",
        "--quick",
        "--cells",
        "fav_number",
        "--out",
        str(tmp_path / "fp.json"),
        "--report",
        str(tmp_path / "fp.md"),
        "--quiet",
    ]
    assert main(args) == 0
    fingerprint = json.loads((tmp_path / "fp.json").read_text())
    assert fingerprint["tokenizer"]["probes_usable"] == 0
    assert all(r["status"] is None for r in fingerprint["requests"])
    assert "behaviour cells with real answers: 0/1" in (tmp_path / "fp.md").read_text()


def test_module_entry_point_runs(tmp_path, mock):
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        [sys.executable, "-m", "model_fingerprint", "--version"],
        check=False,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert __version__ in result.stdout


def test_console_script_equivalent_call(mock, tmp_path):
    """main() is what the console script calls - keep the signature stable."""
    assert callable(cli.main)
    assert main(base_args(mock, tmp_path)) == 0

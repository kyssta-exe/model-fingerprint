"""Packaging and dependency guards: the tool must stay stdlib-only and versioned."""

import ast
import pathlib
import re
import sys

import model_fingerprint

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "model_fingerprint"


def test_version_matches_pyproject():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match, "pyproject.toml has no static version"
    assert match.group(1) == model_fingerprint.__version__


def test_console_script_points_at_a_real_callable():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^model-fingerprint\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match
    module_name, _, attribute = match.group(1).partition(":")
    module = __import__(module_name, fromlist=[attribute])
    assert callable(getattr(module, attribute))


def test_no_third_party_imports():
    """Stdlib only: it has to run on a bare VPS with no pip access."""
    allowed = set(sys.stdlib_module_names)
    offenders = []
    for path in sorted(PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]] if node.level == 0 else []
            else:
                continue
            for name in names:
                if name and name not in allowed and name != "model_fingerprint":
                    offenders.append("%s: %s" % (path.name, name))
    assert not offenders


def test_package_has_no_import_side_effects(capsys):
    """Importing must not print or open sockets."""
    for name in list(sys.modules):
        if name.startswith("model_fingerprint"):
            del sys.modules[name]
    __import__("model_fingerprint")
    __import__("model_fingerprint.cli")
    assert capsys.readouterr() == ("", "")


def test_probe_corpus_is_well_formed():
    from model_fingerprint import probes

    names = [name for name, _ in probes.TOKEN_PROBES]
    assert len(names) == len(set(names))
    assert set(names) >= probes.QUICK_PROBES
    assert all(isinstance(text, str) and text for _, text in probes.TOKEN_PROBES)

    error_names = [name for name, _, _ in probes.ERROR_PROBES]
    assert len(error_names) == len(set(error_names))
    assert set(error_names) >= probes.QUICK_ERRORS

    cell_ids = [cell_id for cell_id, _ in probes.BEHAVIOUR_CELLS]
    assert len(cell_ids) == len(set(cell_ids))
    assert probes.MIN_TOKENS >= 16


def test_quick_sets_are_strictly_smaller():
    from model_fingerprint import probes

    assert len(probes.QUICK_PROBES) < len(probes.TOKEN_PROBES)
    assert len(probes.QUICK_ERRORS) < len(probes.ERROR_PROBES)


def test_request_count_is_bounded(mock, tmp_path):
    """A --quick run must stay cheap: the budget note in the docs depends on it."""
    import json

    from model_fingerprint.cli import main

    out = tmp_path / "fp.json"
    main(
        [
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
            str(out),
            "--report",
            str(tmp_path / "fp.md"),
        ]
    )
    fingerprint = json.loads(out.read_text())
    assert len(fingerprint["requests"]) < 60

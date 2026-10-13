"""The agent-facing SKILL.md and the installer must stay true to the code.

A skill that documents a flag the CLI does not have, or an installer that
clobbers an unrelated directory, is worse than no skill at all.
"""

import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

from model_fingerprint import __version__
from model_fingerprint.cli import build_parser

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKILL = ROOT / "SKILL.md"
INSTALLER = ROOT / "install.sh"
SKILL_NAME = "model-fingerprint"


def frontmatter(text):
    assert text.startswith("---\n"), "SKILL.md must start with the frontmatter delimiter"
    end = text.index("\n---", 3)
    block = text[4:end]
    parsed = {}
    for line in block.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        key, _, value = line.partition(":")
        parsed[key.strip()] = value.strip()
    return parsed, text[end + 4 :]


@pytest.fixture(scope="module")
def skill():
    text = SKILL.read_text(encoding="utf-8")
    meta, body = frontmatter(text)
    return text, meta, body


# --------------------------------------------------------------- frontmatter
def test_frontmatter_has_the_agent_skills_required_fields(skill):
    _, meta, _ = skill
    assert meta["name"] == SKILL_NAME
    assert meta["description"]


def test_description_is_a_trigger_not_a_paragraph(skill):
    _, meta, _ = skill
    description = meta["description"]
    # Hermes truncates its skill index at 57 chars; other agents cap far higher.
    # Staying short keeps the trigger self-contained everywhere.
    assert len(description) <= 60, "description is %d chars" % len(description)
    assert description.endswith(".")
    assert description.lower().startswith(("use when", "identify"))


def test_frontmatter_version_matches_the_package(skill):
    _, meta, _ = skill
    assert meta["version"] == __version__


def test_name_is_a_valid_skill_directory_name(skill):
    _, meta, _ = skill
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", meta["name"])
    assert len(meta["name"]) <= 64


# ---------------------------------------------------------------------- body
def test_body_has_the_sections_an_agent_needs(skill):
    _, _, body = skill
    for heading in (
        "## When to use",
        "## Setup",
        "## Run the tests",
        "## Fingerprint an endpoint",
        "## Procedure",
        "## Grading",
        "## Pitfalls",
        "## Verification",
    ):
        assert heading in body, "missing section: %s" % heading


def cli_flags():
    parser = build_parser()
    known = set()
    for action in parser._actions:
        known.update(action.option_strings)
    return known


def installer_flags():
    result = subprocess.run(
        ["bash", str(INSTALLER), "--help"], check=False, capture_output=True, text=True
    )
    return set(re.findall(r"--[a-z][a-z0-9-]+", result.stdout))


def test_documented_flags_exist_in_the_cli(skill):
    """Documentation drift is a bug: every flag the skill names must be real."""
    _, _, body = skill
    documented = set(re.findall(r"--[a-z][a-z0-9-]+", body))
    known = cli_flags() | installer_flags()
    assert documented <= known, "documented but not implemented: %s" % sorted(documented - known)


def test_flag_table_only_lists_cli_flags(skill):
    """The fingerprinting flag table must not leak installer flags."""
    _, _, body = skill
    start = body.index("## Fingerprint an endpoint")
    section = body[start : body.index("## Procedure", start)]
    documented = set(re.findall(r"--[a-z][a-z0-9-]+", section))
    assert documented <= cli_flags(), "not a CLI flag: %s" % sorted(documented - cli_flags())
    assert "--api" in documented and "--cell-budget" in documented


def test_skill_documents_the_test_command_and_it_works(skill):
    _, _, body = skill
    assert "python -m pytest -q" in body
    assert "python -m model_fingerprint --help" in body


def test_skill_points_at_the_interpretation_doc(skill):
    _, _, body = skill
    assert "docs/interpretation.md" in body
    assert (ROOT / "docs/interpretation.md").is_file()


def test_skill_warns_about_the_traps_that_produce_false_results(skill):
    _, _, body = skill
    for phrase in (
        "not lineage",
        "Uninformative",
        "not determinism",
        "valid budget",
        "share defaults",
        "Don't hammer",
    ):
        assert phrase in body, "pitfall missing: %s" % phrase


# ----------------------------------------------------------------- installer
def run_installer(*args, home, cwd=None):
    env = dict(os.environ)
    env["HOME"] = str(home)
    env.pop("HERMES_HOME", None)
    env.pop("XDG_CONFIG_HOME", None)
    return subprocess.run(
        ["bash", str(INSTALLER), *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd) if cwd else None,
        timeout=120,
    )


def test_installer_is_executable_and_syntactically_valid():
    assert INSTALLER.is_file()
    assert os.access(INSTALLER, os.X_OK), "install.sh must be executable"
    result = subprocess.run(
        ["bash", "-n", str(INSTALLER)], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_installer_help_lists_the_modes():
    result = subprocess.run(
        ["bash", str(INSTALLER), "--help"], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0
    for flag in ("--target", "--project", "--copy", "--force", "--dry-run", "--list"):
        assert flag in result.stdout
    assert "set -eu" not in result.stdout  # usage must not leak past the comment block


def test_installer_rejects_an_unknown_option(tmp_path):
    result = run_installer("--nope", home=tmp_path)
    assert result.returncode == 1
    assert "unknown option" in result.stdout


def test_copy_install_is_self_contained(tmp_path):
    target = tmp_path / "skills" / SKILL_NAME
    result = run_installer("--target", str(target), "--copy", home=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (target / "SKILL.md").is_file()
    assert (target / "model_fingerprint" / "cli.py").is_file()
    assert (target / "docs" / "interpretation.md").is_file()
    assert not (target / "tests").exists()  # a copy needs the tool, not the suite

    # the copied skill actually runs, with no repo and no install
    run = subprocess.run(
        [sys.executable, "-m", "model_fingerprint", "--version"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(target),
    )
    assert run.returncode == 0, run.stderr
    assert __version__ in run.stdout


def test_link_install_points_at_this_checkout(tmp_path):
    target = tmp_path / "skills" / SKILL_NAME
    result = run_installer("--target", str(target), home=tmp_path)
    assert result.returncode == 0, result.stdout
    assert (target / "SKILL.md").is_file()
    assert pathlib.Path(os.path.realpath(target)) == ROOT


def test_installer_is_idempotent(tmp_path):
    target = tmp_path / "skills" / SKILL_NAME
    first = run_installer("--target", str(target), "--copy", home=tmp_path)
    assert first.returncode == 0
    marker = target / "local-note.txt"
    marker.write_text("keep me", encoding="utf-8")

    second = run_installer("--target", str(target), "--copy", home=tmp_path)
    assert second.returncode == 0
    assert "already installed" in second.stdout
    assert marker.read_text(encoding="utf-8") == "keep me"  # untouched without --force


def test_installer_refuses_to_clobber_an_unrelated_directory(tmp_path):
    target = tmp_path / "skills" / SKILL_NAME
    target.mkdir(parents=True)
    (target / "something-else.txt").write_text("not ours", encoding="utf-8")

    result = run_installer("--target", str(target), home=tmp_path)
    assert result.returncode == 1
    assert "REFUSING" in result.stdout
    assert (target / "something-else.txt").exists()

    forced = run_installer("--target", str(target), "--force", "--copy", home=tmp_path)
    assert forced.returncode == 0, forced.stdout
    assert (target / "SKILL.md").is_file()
    assert not (target / "something-else.txt").exists()


def test_dry_run_writes_nothing(tmp_path):
    target = tmp_path / "skills" / SKILL_NAME
    result = run_installer("--target", str(target), "--copy", "--dry-run", home=tmp_path)
    assert result.returncode == 0
    assert "would:" in result.stdout
    assert not target.exists()


def test_project_install_targets_claude_skills(tmp_path):
    result = run_installer("--project", "--copy", home=tmp_path, cwd=tmp_path)
    assert result.returncode == 0, result.stdout
    assert (tmp_path / ".claude" / "skills" / SKILL_NAME / "SKILL.md").is_file()


def test_autodetect_installs_for_every_agent_home_present(tmp_path):
    (tmp_path / ".hermes").mkdir()
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".codex").mkdir()
    result = run_installer("--copy", home=tmp_path)
    assert result.returncode == 0, result.stdout
    for home_dir in (".hermes", ".claude", ".codex"):
        assert (tmp_path / home_dir / "skills" / SKILL_NAME / "SKILL.md").is_file()
    assert "opencode" not in result.stdout  # no config dir present, so not offered


def test_autodetect_with_no_agents_exits_nonzero(tmp_path):
    result = run_installer(home=tmp_path)
    assert result.returncode == 1
    assert "none" in result.stdout
    assert "--target" in result.stdout


def test_list_changes_nothing(tmp_path):
    (tmp_path / ".claude").mkdir()
    result = run_installer("--list", home=tmp_path)
    assert result.returncode == 0
    assert "[absent]" in result.stdout
    assert not (tmp_path / ".claude" / "skills").exists()


def test_installer_only_touches_the_skill_directory(tmp_path):
    """The repo it links to must survive an install/force cycle untouched."""
    target = tmp_path / "skills" / SKILL_NAME
    run_installer("--target", str(target), home=tmp_path)
    run_installer("--target", str(target), "--force", "--copy", home=tmp_path)
    assert (ROOT / "SKILL.md").is_file()
    assert (ROOT / "model_fingerprint" / "cli.py").is_file()
    assert shutil.which("bash")

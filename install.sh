#!/usr/bin/env bash
# Install the model-fingerprint agent skill into one or more coding agents.
#
#   ./install.sh                 # auto-detect installed agents and install for each
#   ./install.sh --list          # show what would be detected, change nothing
#   ./install.sh --target DIR    # install into an explicit skill directory
#   ./install.sh --project       # install into ./.claude/skills (commit it with the repo)
#   ./install.sh --copy          # copy instead of symlink (self-contained, no clone needed)
#   ./install.sh --force         # replace an existing, unrelated skill directory
#   ./install.sh --dry-run       # print the actions without performing them
#
# Default mode is a symlink to this checkout, so `git pull` keeps the skill current.
# --copy makes the skill directory standalone: SKILL.md plus the package and docs.
set -eu

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SKILL_NAME="model-fingerprint"
MODE="link"
FORCE=0
DRY=0
LIST_ONLY=0
PROJECT=0
TARGETS=""

usage() {
    sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

version() {
    grep -m1 '^version' "$REPO/pyproject.toml" 2>/dev/null | sed 's/.*"\(.*\)".*/\1/' || echo "unknown"
}

log() { printf '%s\n' "$*"; }
act() { if [ "$DRY" -eq 1 ]; then log "  would: $*"; else log "  $*"; fi; }

# Candidate agent skill roots: "<label>:<directory that holds skills>"
candidates() {
    printf '%s\n' \
        "Hermes:${HERMES_HOME:-$HOME/.hermes}/skills" \
        "Claude Code:$HOME/.claude/skills" \
        "Codex:$HOME/.codex/skills" \
        "opencode:${XDG_CONFIG_HOME:-$HOME/.config}/opencode/skill"
}

# Print "label<TAB>skills_root" for each agent whose home directory exists.
detect() {
    candidates | while IFS= read -r entry; do
        label="${entry%%:*}"
        root="${entry#*:}"
        if [ -d "$(dirname "$root")" ]; then
            printf '%s\t%s\n' "$label" "$root"
        fi
    done
}

installed_state() { # $1 = target dir -> absent|linked|present|occupied
    if [ ! -e "$1" ] && [ ! -L "$1" ]; then
        echo "absent"
        return 0
    fi
    resolved="$(cd "$1" 2>/dev/null && pwd -P || true)"
    if [ -n "$resolved" ] && [ "$resolved" = "$REPO" ]; then
        echo "linked"
    elif [ -f "$1/SKILL.md" ]; then
        echo "present"
    else
        echo "occupied"
    fi
}

install_into() { # $1 = target dir; returns 1 only on a refused clobber
    target="$1"
    state="$(installed_state "$target")"
    case "$state" in
        linked)
            log "  already installed (symlink to this checkout): $target"
            return 0 ;;
        present)
            if [ "$FORCE" -ne 1 ]; then
                log "  already installed (v$(version)): $target"
                log "    pass --force to replace it with this checkout"
                return 0
            fi ;;
        occupied)
            if [ "$FORCE" -ne 1 ]; then
                log "  REFUSING to replace a directory that is not a model-fingerprint skill:"
                log "    $target"
                log "    pass --force if that is really what you want"
                return 1
            fi ;;
    esac

    if [ "$DRY" -eq 1 ]; then
        act "remove existing $target"
        if [ "$MODE" = "copy" ]; then
            act "copy skill -> $target"
        else
            act "symlink $REPO -> $target"
        fi
        return 0
    fi

    mkdir -p "$(dirname "$target")"
    rm -rf "$target"
    if [ "$MODE" = "copy" ]; then
        mkdir -p "$target/docs"
        cp "$REPO/SKILL.md" "$REPO/README.md" "$REPO/LICENSE" "$target/"
        cp -R "$REPO/model_fingerprint" "$target/"
        cp "$REPO/docs/interpretation.md" "$target/docs/" 2>/dev/null || true
        act "copied skill (v$(version)) -> $target"
    else
        ln -s "$REPO" "$target"
        act "linked $REPO -> $target"
    fi
    return 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --target)
            [ $# -ge 2 ] || { log "error: --target needs a directory"; exit 1; }
            TARGETS="${TARGETS}${2}
"
            shift 2 ;;
        --project) PROJECT=1; shift ;;
        --copy)    MODE="copy"; shift ;;
        --link)    MODE="link"; shift ;;
        --force)   FORCE=1; shift ;;
        --dry-run) DRY=1; shift ;;
        --list)    LIST_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) log "error: unknown option $1"; usage; exit 1 ;;
    esac
done

log "model-fingerprint skill installer (v$(version), mode: $MODE)"
if [ "$DRY" -eq 1 ]; then
    log "dry run: nothing will be written"
fi

if [ "$PROJECT" -eq 1 ]; then
    TARGETS="${TARGETS}$(pwd -P)/.claude/skills/${SKILL_NAME}
"
fi

# Explicit targets (a file keeps the loop out of a subshell so the exit code survives)
if [ -n "$TARGETS" ]; then
    TMP="$(mktemp)"
    printf '%s' "$TARGETS" | grep -v '^$' > "$TMP" || true
    rc=0
    while IFS= read -r target; do
        log "target: $target"
        install_into "$target" || rc=1
    done < "$TMP"
    rm -f "$TMP"
    log "done. restart the agent (or start a new session) so it picks the skill up."
    exit "$rc"
fi

# Auto-detection
DETECTED="$(detect)"
if [ -z "$DETECTED" ]; then
    log "detected agents: none (looked for ~/.hermes, ~/.claude, ~/.codex, ~/.config/opencode)"
    log "install manually with: $0 --target <dir>"
    exit 1
fi

TMP="$(mktemp)"
printf '%s\n' "$DETECTED" > "$TMP"
rc=0
while IFS="$(printf '\t')" read -r label root; do
    target="$root/$SKILL_NAME"
    if [ "$LIST_ONLY" -eq 1 ]; then
        log "  $label: $target [$(installed_state "$target")]"
    else
        log "  $label:"
        install_into "$target" || rc=1
    fi
done < "$TMP"
rm -f "$TMP"

if [ "$LIST_ONLY" -eq 1 ]; then
    exit 0
fi
log "done. restart the agent (or start a new session) so it picks the skill up."
exit "$rc"

"""Pre-publish leak scan for everything under hf-release/.

Run this BEFORE any upload command in PUBLISH.md. It is the last gate between
a staging directory on someone's laptop and a public Hub repo, and a Hub repo
keeps its git history: a secret pushed once is a secret rotated, not a secret
deleted.

    python hf-release/leak_scan.py            # scan everything staged
    python hf-release/leak_scan.py --path hf-release/space
    python hf-release/leak_scan.py --list-rules

Exit code 0 = clean, 1 = findings. Findings print as `path:line: rule` with the
offending text elided to a short excerpt, so the scanner's own output does not
become the leak.

What it looks for:
  * credentials -- Hub tokens, OpenAI/Anthropic/Google keys, AWS keys, generic
    high-entropy assignments to secret-shaped names, private key headers;
  * author-machine paths -- Windows drive paths, /home/<user>, /Users/<user>,
    and this machine's own account name;
  * unrelated-project and persona names that share this filesystem and must
    never appear in an ML release;
  * workplace/employer references;
  * a private-email heuristic.

Adding a rule is cheap and removing one should be argued for. The persona and
project lists exist because this repository sits on a machine that also hosts
unrelated content work; they are not hypothetical.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Binary and vendored trees are skipped: they are not authored here, and
# scanning a virtualenv produces thousands of false positives that train the
# reader to ignore the output.
SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    "node_modules", ".ruff_cache", "site-packages",
}
SKIP_SUFFIXES = {
    ".safetensors", ".bin", ".pt", ".pth", ".onnx", ".parquet", ".arrow",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp4", ".zip", ".gz", ".pdf",
}

# Third-party corpus rows are a DIFFERENT category from authored files. They
# are verbatim upstream text we redistribute under someone else's license, so
# "my employer" appearing in a Yelp review is not a leak from this machine.
# They are still scanned for credentials and author-machine paths (a key
# committed into a data file is exactly as published as one in a source file);
# content rules are reported as an ADVISORY the owner must read, never
# silently suppressed. See PUBLISH.md step 1.
CORPUS_GLOBS = ("datasets/*/data/*",)

# Rules that apply everywhere, including corpus data. Anything that would be a
# live credential or would identify the author's machine.
HARD_RULE_NAMES = {
    "hf-token", "openai-key", "anthropic-key", "google-api-key",
    "aws-access-key", "github-token", "slack-token", "private-key-block",
    "secret-assignment", "windows-drive-path", "windows-user-path",
    "unix-home-path", "msys-user-path", "account-name", "scratch-cache",
    "local-project-root", "obsidian-vault",
}

Rule = tuple[str, re.Pattern[str]]

RULES: list[Rule] = [
    # ---- credentials -----------------------------------------------------
    ("hf-token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("openai-key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("slack-token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("private-key-block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----")),
    (
        "secret-assignment",
        re.compile(
            r"""(?ix)
            \b(?:api[_-]?key|secret|password|passwd|token|access[_-]?key|
                 client[_-]?secret|auth[_-]?token)\b
            \s*[:=]\s*
            ['"][A-Za-z0-9/+_\-]{16,}['"]
            """
        ),
    ),
    # ---- author-machine paths -------------------------------------------
    ("windows-drive-path", re.compile(r"(?i)\b[C-Z]:[\\/](?:Users|ClaudeProjects|ClaudeCache)\b")),
    ("windows-user-path", re.compile(r"(?i)\b[C-Z]:[\\/]Users[\\/][A-Za-z0-9._-]+")),
    ("unix-home-path", re.compile(r"(?<![\w.])/(?:home|Users)/(?!<)[A-Za-z0-9._-]+/")),
    ("msys-user-path", re.compile(r"(?i)/[c-z]/Users/[A-Za-z0-9._-]+")),
    ("account-name", re.compile(r"(?i)\bcobra\b")),
    ("scratch-cache", re.compile(r"(?i)\bClaudeCache\b")),
    ("local-project-root", re.compile(r"(?i)\bClaudeProjects\b")),
    ("obsidian-vault", re.compile(r"(?i)\bObsidianVault\b")),
    # ---- unrelated projects sharing this machine ------------------------
    (
        "unrelated-project",
        re.compile(
            r"(?i)\b(?:shorts[- ]factory|landon[ _]?lifts|brainrot|gym[- ]music[- ]label|"
            r"loadout[- ]legends|crazy[- ]carl|dividend[- ]desk|going[- ]postal|"
            r"lanari|agentforge|medinsight|job[- ]pipeline|wizard101|"
            r"circle[- ]colosseum|food[- ]court[- ]fight[- ]club|overload|treblo)\b"
        ),
    ),
    ("persona-name", re.compile(r"(?i)\b(?:raven|milo723|sendlark|toast[- ]malone)\b")),
    # ---- workplace ------------------------------------------------------
    (
        "workplace-reference",
        re.compile(
            r"(?i)\b(?:my (?:employer|workplace|company|manager|team lead)|"
            r"at work we|internal[- ]only|confidential|proprietary and confidential|"
            r"do not distribute)\b"
        ),
    ),
    # ---- contact --------------------------------------------------------
    (
        "private-email",
        re.compile(
            r"(?i)\b[A-Za-z0-9._%+-]+@(?!example\.|users\.noreply\.)"
            r"[A-Za-z0-9.-]+\.(?:com|net|org|io|dev|me)\b"
        ),
    ),
]

# Lines that legitimately contain a pattern. Each entry must be justified --
# an allowlist is how a scanner stops finding things.
ALLOW = re.compile(
    r"""(?x)
    ^\s*\#            # a comment in this scanner's own rule table
    | leak_scan\.py   # references to this file by name
    | <local\ checkpoint\ directory  # the smoke proof's redacted placeholder
    | /path/to/checkpoint            # documentation placeholder
    | HOME/\.cache                   # documented cache location, no username
    """
)


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        yield path


def excerpt(line: str, match: re.Match[str]) -> str:
    """Show enough to locate the hit, never enough to reuse it."""
    text = match.group(0)
    shown = text if len(text) <= 12 else text[:8] + "..." + text[-2:]
    return f"{shown!r} in {line.strip()[:70]!r}"


def is_corpus(path: Path, root: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    return any(Path(rel).match(glob) for glob in CORPUS_GLOBS)


def scan_file(path: Path, hard_only: bool) -> list[tuple[int, str, str]]:
    try:
        content = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    rules = [(n, p) for n, p in RULES if n in HARD_RULE_NAMES] if hard_only else RULES
    findings = []
    for lineno, line in enumerate(content.splitlines(), start=1):
        if ALLOW.search(line):
            continue
        for name, pattern in rules:
            match = pattern.search(line)
            if match:
                findings.append((lineno, name, excerpt(line, match)))
    return findings


def main() -> int:
    default_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=default_root)
    parser.add_argument("--list-rules", action="store_true")
    parser.add_argument(
        "--strict-corpus",
        action="store_true",
        help="Treat corpus-content advisories as failures too.",
    )
    args = parser.parse_args()

    if args.list_rules:
        for name, pattern in RULES:
            tier = "HARD " if name in HARD_RULE_NAMES else "authored-only "
            print(f"{tier:15}{name:24} {pattern.pattern.splitlines()[0][:50]}")
        return 0

    root = args.path.resolve()
    if not root.exists():
        print(f"no such path: {root}")
        return 1

    self_path = Path(__file__).resolve()
    failures: list[str] = []
    advisories: list[str] = []
    scanned = 0

    for path in iter_files(root):
        # The rule table necessarily contains every string it searches for.
        if path.resolve() == self_path:
            continue
        scanned += 1
        corpus = is_corpus(path, root)
        rel = path.relative_to(root).as_posix()

        for lineno, name, detail in scan_file(path, hard_only=False):
            line = f"{rel}:{lineno}: [{name}] {detail}"
            if corpus and name not in HARD_RULE_NAMES:
                advisories.append(line)
            else:
                failures.append(line)

    if advisories:
        print("ADVISORY -- third-party corpus content (redistributed, not authored here):")
        for line in advisories:
            print("  " + line)
        print(
            "\n  These are verbatim upstream rows, so they are an UPSTREAM-PROVENANCE\n"
            "  question, not a leak from this machine. Read them, decide, and record the\n"
            "  decision in the dataset card. Do not delete this notice to make it quiet.\n"
        )

    if failures:
        print("FAILURES:")
        for line in failures:
            print("  " + line)

    print(f"scanned {scanned} files under {root.name}/ "
          f"({len(RULES)} rules, {len(HARD_RULE_NAMES)} enforced on corpus data)")

    if failures or (args.strict_corpus and advisories):
        n = len(failures) + (len(advisories) if args.strict_corpus else 0)
        print(f"LEAK SCAN FAILED: {n} finding(s). Do not publish.")
        return 1
    print(f"LEAK SCAN CLEAN ({len(advisories)} corpus advisory/advisories to acknowledge)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Command line entry point."""

import argparse
import json
import os
import sys
import time

from . import __version__, layers
from .client import Client, detect_api, load_env_file
from .report import render

DEFAULT_OUT = "fingerprint.json"
DEFAULT_CELL_BUDGET = 512
CONTEXT_LADDER = (8192, 32768, 131072, 262144, 1048576)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="model-fingerprint",
        description="Probe a black-box LLM endpoint and record layered attribution evidence.",
    )
    parser.add_argument("--base-url", required=True, help="e.g. https://host/v1")
    parser.add_argument("--model", required=True, help="model id exactly as the vendor writes it")
    parser.add_argument("--api-key-env", default=None, help="env var holding the API key")
    parser.add_argument(
        "--env-file", default=None, help="dotenv file to read the key from, e.g. ~/.hermes/.env"
    )
    parser.add_argument(
        "--api-key", default=None, help="literal key (avoid: it lands in shell history)"
    )
    parser.add_argument(
        "--api", default="auto", choices=["auto", "openai", "responses", "anthropic", "gemini"]
    )
    parser.add_argument("--samples", type=int, default=6, help="samples per behaviour cell")
    parser.add_argument("--quick", action="store_true", help="smallest probe set (~40 requests)")
    parser.add_argument("--full", action="store_true", help="largest probe set (~250 requests)")
    parser.add_argument("--cells", default=None, help="comma-separated behaviour cell ids")
    parser.add_argument(
        "--header",
        action="append",
        default=None,
        help="extra request header 'Name: Value' (repeatable)",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["minimal", "low", "medium", "high"],
        help="pin thinking effort (must match the reference run)",
    )
    parser.add_argument(
        "--cell-budget",
        type=int,
        default=DEFAULT_CELL_BUDGET,
        help="output-token budget per behaviour cell (default %s)" % DEFAULT_CELL_BUDGET,
    )
    parser.add_argument(
        "--context-probe",
        action="store_true",
        help="escalating filler requests to find the context ceiling (costly)",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="fingerprint JSON path")
    parser.add_argument("--report", default=None, help="markdown report path")
    parser.add_argument("--quiet", action="store_true", help="do not print the report to stdout")
    parser.add_argument("--version", action="version", version="%(prog)s " + __version__)
    return parser


def parse_headers(raw_headers):
    out = {}
    for header in raw_headers or []:
        if ":" in header:
            key, value = header.split(":", 1)
            out[key.strip()] = value.strip()
    return out


def resolve_key(args):
    if args.api_key:
        return args.api_key
    if not args.api_key_env:
        return ""
    key = os.environ.get(args.api_key_env, "")
    if not key and args.env_file:
        key = load_env_file(args.env_file).get(args.api_key_env, "")
    return key


def run(args, stream=None):
    """Run every layer, returning the fingerprint dict. Never raises on a dead layer."""
    stream = stream if stream is not None else sys.stderr
    key = resolve_key(args)
    if not key:
        print(
            "warning: no API key (--api-key-env NAME, --env-file PATH); auth errors will dominate",
            file=stream,
        )
    api = detect_api(args.base_url, args.model) if args.api == "auto" else args.api
    client = Client(
        args.base_url,
        key,
        args.model,
        api,
        extra_headers=parse_headers(args.header),
        reasoning_effort=args.reasoning_effort,
    )

    samples = 12 if args.full else (3 if args.quick else args.samples)
    cells = set(args.cells.split(",")) if args.cells else None

    fingerprint = {
        "meta": {
            "tool": "model-fingerprint/" + __version__,
            "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "base_url": args.base_url,
            "model": args.model,
            "api": api,
            "key_len": len(key),
            "reasoning_effort": args.reasoning_effort,
            "extra_headers": sorted(parse_headers(args.header)),
            "depth": "full" if args.full else ("quick" if args.quick else "default"),
        }
    }

    steps = [
        ("transport", lambda: layers.transport(client)),
        ("errors", lambda: layers.errors(client, args.quick)),
        ("tokenizer", lambda: layers.tokenizer(client, args.quick)),
        ("capabilities", lambda: layers.capabilities(client)),
        ("limits", lambda: layers.limits(client)),
        ("behaviour", lambda: layers.behaviour(client, samples, cells, args.cell_budget)),
    ]
    if args.context_probe:
        steps.append(("context", lambda: layers.context(client, CONTEXT_LADDER)))

    for name, fn in steps:
        print("  probing %s ..." % name, file=stream, flush=True)
        try:
            fingerprint[name] = fn()
        except Exception as exc:  # noqa: BLE001 - report the dead layer, never hide it
            fingerprint[name] = {"unavailable": "%s: %s" % (type(exc).__name__, exc)}
    fingerprint["requests"] = client.log
    return fingerprint


def main(argv=None):
    args = build_parser().parse_args(argv)
    fingerprint = run(args)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(fingerprint, fh, indent=2, sort_keys=True)
    report_path = args.report or (args.out.rsplit(".", 1)[0] + ".md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(render(fingerprint))
    if not args.quiet:
        print(render(fingerprint))
    print(
        "\nJSON: %s\nReport: %s\nRequests: %d"
        % (args.out, report_path, len(fingerprint["requests"])),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

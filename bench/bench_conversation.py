"""Benchmark Assist conversation latency through the Home Assistant REST API.

Usage:
    HA_TOKEN=... python bench/bench_conversation.py --label before --runs 10
    python bench/bench_conversation.py --label before --merge-log copied-ha.log
    python bench/bench_conversation.py --label after --compare-with before

Writes bench/results/<label>.csv with one row per request. The token is read
from the HA_TOKEN environment variable and is never written to disk. Response
paths are guesses until --merge-log matches conversation IDs to agent timing
lines. The first request per phrase is reported separately from later runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

# (class, phrase, expected path). Light phrases alternate on/off below so the
# lights do not drift from one run to the next.
PHRASES: list[tuple[str, str, str]] = [
    ("builtin", "включи свет на кухне", "builtin"),
    ("builtin", "выключи свет на кухне", "builtin"),
    ("normalizer", "выключи свет в спальне", "normalizer"),
    ("normalizer", "спокойной ночи", "normalizer"),
    ("negation", "включи свет, кроме кухни", "clarification"),
    ("llm_short", "сколько будет два плюс два", "llm"),
    ("llm_tool", "какая погода в Гомеле", "llm_tool"),
    ("llm_border", "погода через три часа", "llm"),
    ("llm_long", "расскажи короткий анекдот", "llm"),
]


def _post(url: str, token: str, body: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


TIMING_RE = re.compile(
    r"gigachain timing: conversation_id=(?P<id>\S+) path=(?P<path>\S+) "
    r"builtin=(?P<builtin>[\d.]+)s normalizer=(?P<normalizer>[\d.]+)s "
    r"apply=(?P<apply>[\d.]+)s llm_data=(?P<llm_data>[\d.]+)s "
    r"llm=(?P<llm>[\d.]+)s total=(?P<total>[\d.]+)s"
)


def _classify(payload: dict) -> str:
    """Best-effort response classification; use --merge-log for exact paths."""
    response = payload.get("response", {})
    if response.get("response_type") == "error":
        return "error"
    if response.get("data", {}).get("success") is not None and response.get("data", {}).get("targets") is not None:
        return "builtin"
    speech = (response.get("speech", {}).get("plain", {}) or {}).get("speech", "")
    if speech in ("Включаю свет.", "Выключаю свет.", "Спокойной ночи."):
        return "normalizer"
    if response.get("data", {}).get("failed"):
        return "clarification"
    return "llm"


def _merge_log(csv_path: Path, log_path: Path) -> None:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    timings = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := TIMING_RE.search(line):
            timings[match["id"]] = match.groupdict()
    for row in rows:
        if timing := timings.get(row.get("conversation_id")):
            row["observed_path"] = timing["path"]
            row["path_source"] = "agent_log"
            for stage in ("builtin", "normalizer", "apply", "llm_data", "llm", "total"):
                row[f"agent_{stage}_seconds"] = timing[stage]
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"matched {sum(bool(timings.get(row.get('conversation_id'))) for row in rows)}/{len(rows)} turns")


def _compare(current_path: Path, other_path: Path) -> None:
    def read(path: Path) -> dict[str, tuple[str, str, str]]:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        result = {}
        for klass in dict.fromkeys(row["class"] for row in rows):
            valid = [row for row in rows if row["class"] == klass and row["status"] == "200"]
            cold = next((float(row["seconds"]) for row in valid if row["run"] == "0"), None)
            warm = sorted(float(row["seconds"]) for row in valid if int(row["run"]) > 0)
            result[klass] = (
                f"{cold:.3f}" if cold is not None else "—",
                f"{statistics.median(warm):.3f}" if warm else "—",
                f"{warm[math.ceil(0.9 * len(warm)) - 1]:.3f}" if warm else "—",
            )
        return result

    before, after = read(other_path), read(current_path)
    print("| Class | Before cold | After cold | Before median | After median | Before p90 | After p90 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for klass in dict.fromkeys((*before, *after)):
        old = before.get(klass, ("—",) * 3)
        new = after.get(klass, ("—",) * 3)
        print(f"| {klass} | {old[0]} | {new[0]} | {old[1]} | {new[1]} | {old[2]} | {new[2]} |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="name of the run, e.g. before or after")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--agent", default="conversation.gigachat")
    parser.add_argument("--base-url", default=os.environ.get("HA_URL", "http://orangepizero2w.lan:8123"))
    parser.add_argument("--merge-log", type=Path, help="merge agent paths and stage times from a copied HA log")
    parser.add_argument("--compare-with", help="print a before/after table against another label")
    args = parser.parse_args()

    out_path = Path(__file__).with_name("results") / f"{args.label}.csv"
    if args.merge_log:
        _merge_log(out_path, args.merge_log)
        return
    if args.compare_with:
        _compare(out_path, out_path.with_name(f"{args.compare_with}.csv"))
        return

    token = os.environ.get("HA_TOKEN")
    if not token:
        raise SystemExit("HA_TOKEN is not set")

    url = f"{args.base_url.rstrip('/')}/api/conversation/process"
    out_dir = Path(__file__).with_name("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.label}.csv"

    rows: list[dict] = []
    for run in range(args.runs):
        for klass, phrase, expected in PHRASES:
            text = phrase
            if "включи свет" in phrase and run % 2:
                text = phrase.replace("включи", "выключи", 1)
            elif "выключи свет" in phrase and run % 2:
                text = phrase.replace("выключи", "включи", 1)
            started = time.monotonic()
            status = 0
            payload: dict = {}
            error = ""
            try:
                status, payload = _post(url, token, {"text": text, "agent_id": args.agent, "language": "ru"})
            except (urllib.error.URLError, TimeoutError) as exc:
                error = str(exc)
            elapsed = time.monotonic() - started
            speech = ""
            if payload:
                speech = (payload.get("response", {}).get("speech", {}).get("plain", {}) or {}).get("speech", "")
            rows.append(
                {
                    "label": args.label,
                    "run": run,
                    "class": klass,
                    "expected_path": expected,
                    "text": text,
                    "conversation_id": payload.get("conversation_id", ""),
                    "status": status,
                    "seconds": round(elapsed, 3),
                    "observed_path": _classify(payload) if payload else "error",
                    "path_source": "response_guess",
                    "speech": speech,
                    "error": error,
                }
            )
            print(f"run={run} {klass:<11} {elapsed:6.2f}s {rows[-1]['observed_path']:<12} {text}")

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {out_path}")
    for klass in dict.fromkeys(row["class"] for row in rows):
        valid = [row for row in rows if row["class"] == klass and row["status"] == 200]
        cold = next((row["seconds"] for row in valid if row["run"] == 0), None)
        warm = sorted(row["seconds"] for row in valid if row["run"] > 0)
        if warm:
            p90 = warm[math.ceil(0.9 * len(warm)) - 1]
            print(f"{klass:<11} cold={cold!s:>5}s median={statistics.median(warm):6.2f}s p90={p90:6.2f}s n={len(warm)}")


if __name__ == "__main__":
    main()

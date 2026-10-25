"""Benchmark Assist conversation latency through the Home Assistant REST API.

Usage:
    HA_TOKEN=... python bench/bench_conversation.py --label before --runs 10

Writes bench/results/<label>.csv with one row per request. The token is read
from the HA_TOKEN environment variable and is never written to disk.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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


def _classify(payload: dict) -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="name of the run, e.g. before or after")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--agent", default="conversation.gigachat")
    parser.add_argument("--base-url", default=os.environ.get("HA_URL", "http://orangepizero2w.lan:8123"))
    args = parser.parse_args()

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
                    "status": status,
                    "seconds": round(elapsed, 3),
                    "observed_path": _classify(payload) if payload else "error",
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
    for klass, _, _ in PHRASES:
        times = sorted(r["seconds"] for r in rows if r["class"] == klass and r["run"] > 0 and not r["error"])
        if not times:
            continue
        p90 = times[min(len(times) - 1, int(round(0.9 * (len(times) - 1))))]
        print(f"{klass:<11} median={statistics.median(times):6.2f}s p90={p90:6.2f}s n={len(times)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fetch OKX perpetual-swap candles into ZetaAlgo's CSV format.

    python3 tools/fetch_okx.py XAU-USDT-SWAP BTC-USDT-SWAP --bar 5m --days 60
    python3 tools/fetch_okx.py --specs XAU-USDT-SWAP   # tick size, contract value

Public market-data endpoints only; no API key, no account access, nothing is
ever sent to OKX but the instrument id.

Timestamps are written in **UTC**, which is also where the VWAP anchors: a
perpetual trades 24/7, so it has no exchange session and the near-universal
convention is to reset VWAP at UTC midnight.  Volume is in contracts; the
contract value (``ctVal``) is what converts that to the underlying, so always
check ``--specs`` before sizing anything.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List

BASE = "https://www.okx.com"
BAR_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
          "30m": 1_800_000, "1H": 3_600_000, "4H": 14_400_000}


def _get(path: str, timeout: int = 45) -> dict:
    request = urllib.request.Request(BASE + path, headers={"User-Agent": "zetaalgo/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if payload.get("code") not in ("0", 0):
        raise RuntimeError(f"OKX error {payload.get('code')}: {payload.get('msg')}")
    return payload


def instrument_specs(inst_type: str = "SWAP") -> Dict[str, dict]:
    data = _get(f"/api/v5/public/instruments?instType={inst_type}")["data"]
    return {row["instId"]: row for row in data}


def fetch_candles(inst_id: str, bar: str = "5m", days: float = 60.0,
                  sleep: float = 0.12) -> List[list]:
    """Page backwards through history-candles until ``days`` are covered.

    OKX returns candles newest-first, 300 per request, and ``after`` asks for
    rows older than a timestamp.  Only *closed* candles are kept: the final
    element of each row is a confirm flag, and acting on an unconfirmed candle
    is the live-trading version of lookahead.
    """
    span = BAR_MS.get(bar)
    if span is None:
        raise SystemExit(f"unsupported bar {bar!r}; choose from {sorted(BAR_MS)}")
    now_ms = int(time.time() * 1000)
    stop_at = now_ms - int(days * 86_400_000)

    rows: Dict[int, list] = {}
    cursor = ""
    while True:
        path = (f"/api/v5/market/history-candles?instId={inst_id}"
                f"&bar={bar}&limit=300{cursor}")
        batch = _get(path)["data"]
        if not batch:
            break
        for row in batch:
            # row: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
            if len(row) > 8 and row[8] != "1":
                continue  # candle still forming
            rows[int(row[0])] = row
        oldest = min(int(r[0]) for r in batch)
        if oldest <= stop_at or len(batch) < 300:
            break
        cursor = f"&after={oldest}"
        time.sleep(sleep)  # stay inside the 20 req / 2s limit
    return [rows[key] for key in sorted(rows) if key >= stop_at]


def write_csv(path: str, rows: List[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for row in rows:
            stamp = datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc)
            writer.writerow([stamp.replace(tzinfo=None).isoformat(sep=" "),
                             row[1], row[2], row[3], row[4], row[5]])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("instruments", nargs="+", help="e.g. XAU-USDT-SWAP")
    parser.add_argument("--bar", default="5m")
    parser.add_argument("--days", type=float, default=60.0)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--specs", action="store_true",
                        help="print contract specs and exit")
    args = parser.parse_args()

    if args.specs:
        specs = instrument_specs()
        print(f"{'instId':18s} {'tickSz':>8s} {'lotSz':>7s} {'minSz':>6s} "
              f"{'ctVal':>8s} {'ctValCcy':>8s} {'maxLever':>8s}")
        for inst in args.instruments:
            row = specs.get(inst)
            if row is None:
                print(f"{inst:18s} NOT FOUND")
                continue
            print(f"{inst:18s} {row['tickSz']:>8s} {row['lotSz']:>7s} "
                  f"{row['minSz']:>6s} {row['ctVal']:>8s} {row['ctValCcy']:>8s} "
                  f"{row['lever']:>8s}")
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    failures = 0
    for inst in args.instruments:
        try:
            rows = fetch_candles(inst, args.bar, args.days)
        except Exception as exc:  # noqa: BLE001
            print(f"{inst:18s} FAILED: {type(exc).__name__}: {exc}")
            failures += 1
            continue
        if not rows:
            print(f"{inst:18s} no candles returned")
            failures += 1
            continue
        name = inst.lower().replace("-", "_")
        path = os.path.join(args.out_dir, f"{name}_{args.bar}.csv")
        write_csv(path, rows)
        first = datetime.fromtimestamp(int(rows[0][0]) / 1000, tz=timezone.utc)
        last = datetime.fromtimestamp(int(rows[-1][0]) / 1000, tz=timezone.utc)
        print(f"{inst:18s} {len(rows):7,} bars  {first:%Y-%m-%d}..{last:%Y-%m-%d}  "
              f"UTC  -> {path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

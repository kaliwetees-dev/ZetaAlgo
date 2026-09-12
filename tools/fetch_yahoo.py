#!/usr/bin/env python3
"""Fetch intraday bars from Yahoo Finance into ZetaAlgo's CSV format.

    python3 tools/fetch_yahoo.py SPY QQQ AAPL --interval 5m --range 60d

Yahoo caps 5-minute history at roughly 60 days, which is a small sample for
evaluating an intraday strategy -- see the README on how little a result over
this window can prove.  Bars are written in exchange-local time with
pre/post-market excluded, so each session's VWAP anchors at the real open.

This uses a public, unauthenticated endpoint that Yahoo may rate-limit
(HTTP 429) or change without notice; it is a convenience for evaluation, not
a production data feed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"


def fetch(symbol: str, interval: str, range_: str, timeout: int = 60) -> Tuple[List[list], str]:
    url = (f"{CHART_URL.format(symbol=symbol)}?interval={interval}"
           f"&range={range_}&includePrePost=false")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)

    error = payload.get("chart", {}).get("error")
    if error:
        raise RuntimeError(f"{symbol}: {error}")
    result = payload["chart"]["result"][0]
    stamps = result.get("timestamp") or []
    quote = result["indicators"]["quote"][0]
    offset = result["meta"].get("gmtoffset", 0)

    rows: List[list] = []
    for i, stamp in enumerate(stamps):
        o, h, l = quote["open"][i], quote["high"][i], quote["low"][i]
        c, v = quote["close"][i], quote["volume"][i]
        if None in (o, h, l, c):
            continue  # Yahoo leaves halted/missing bars as nulls
        local = datetime.fromtimestamp(stamp, tz=timezone.utc) + timedelta(seconds=offset)
        rows.append([local.replace(tzinfo=None).isoformat(sep=" "),
                     f"{o:.4f}", f"{h:.4f}", f"{l:.4f}", f"{c:.4f}", int(v or 0)])
    if not rows:
        raise RuntimeError(f"{symbol}: no usable bars returned")
    return rows, result["meta"].get("exchangeTimezoneName", "?")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("symbols", nargs="+")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--range", dest="range_", default="60d")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--sleep", type=float, default=3.0,
                        help="pause between symbols to avoid rate limiting")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    failures = 0
    for index, symbol in enumerate(args.symbols):
        try:
            rows, tz = fetch(symbol, args.interval, args.range_)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"{symbol:6s} FAILED: {type(exc).__name__}: {exc}")
            failures += 1
            continue
        path = os.path.join(args.out_dir, f"{symbol.lower()}_{args.interval}.csv")
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
            writer.writerows(rows)
        sessions = len({row[0][:10] for row in rows})
        print(f"{symbol:6s} {len(rows):6,} bars  {sessions:3d} sessions  "
              f"{rows[0][0][:10]}..{rows[-1][0][:10]}  tz={tz}  -> {path}")
        if index + 1 < len(args.symbols):
            time.sleep(args.sleep)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

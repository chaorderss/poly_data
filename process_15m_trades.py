#!/usr/bin/env python3
"""
Process btc_updown_15m_all.csv into feature CSV for model training.

Steps:
1. Scan trades to determine time range
2. Fetch 1-second BTC prices from Binance (cached, resumable)
3. Precompute rolling realized volatility (1-15 min windows)
4. Process trades in chunks: aggregate by second, compute features

Output: btc_updown_15m_features.csv
Columns: moneyness, rv, side, tte, price

Usage:
    python lstm_calibration/poly_data/process_15m_trades.py
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from lstm_calibration.fetch_binance import (
    fetch_binance_klines,
    read_last_timestamp_from_csv,
    read_time_range_from_csv_edges,
)

BASE_DIR = os.path.dirname(__file__)
TRADES_PATH = os.path.join(
    BASE_DIR, 'polymarket_data', 'btc_updown_15m_all.csv'
)
BTC_1S_CACHE = os.path.join(BASE_DIR, 'btc_prices_1s.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'btc_updown_15m_features.csv')

CHUNK_SIZE = 5_000_000


# ---------------------------------------------------------------------------
# Step 1: Scan time range (fast: only head/tail)
# ---------------------------------------------------------------------------

def scan_time_range():
    """Fast scan: read only head/tail rows to determine BTC price time range."""
    print("Scanning trades for time range (head/tail only)...")

    # Read timestamp range using existing helper
    min_trade_ts, max_trade_ts = read_time_range_from_csv_edges(
        TRADES_PATH, timestamp_col='timestamp', head_rows=5000, tail_rows=5000
    )
    min_ts = int(min_trade_ts.timestamp())
    max_ts = int(max_trade_ts.timestamp())

    # Also need to check market_slug for start_ts
    # Read head to get earliest start_ts
    head = pd.read_csv(TRADES_PATH, nrows=5000, usecols=['market_slug'])
    head_start = head['market_slug'].str.extract(r'(\d+)$', expand=False).astype(int)
    min_start_ts = int(head_start.min())

    # Read tail to get latest start_ts + window
    with open(TRADES_PATH, 'rb') as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 500_000))  # read last ~500KB
        tail_text = f.read().decode('utf-8', errors='ignore')

    tail_lines = [ln for ln in tail_text.split('\n') if 'btc-updown-15m-' in ln][-1000:]
    tail_slugs = pd.Series(tail_lines).str.extract(r'btc-updown-15m-(\d+)', expand=False)
    tail_slugs = pd.to_numeric(tail_slugs, errors='coerce').dropna()
    max_start_ts = int(tail_slugs.max()) if not tail_slugs.empty else max_ts

    # Final range: account for market windows
    min_ts = min(min_ts, min_start_ts)
    max_ts = max(max_ts, max_start_ts + 15 * 60)

    # Pad: 15 min before (for rv lookback) and 1 min after
    min_ts -= 15 * 60
    max_ts += 60

    print(
        f"BTC price range needed: "
        f"{datetime.fromtimestamp(min_ts, tz=timezone.utc)} -> "
        f"{datetime.fromtimestamp(max_ts, tz=timezone.utc)}"
    )
    return min_ts, max_ts


# ---------------------------------------------------------------------------
# Step 2: Fetch & cache BTC 1s prices (resumable, daily chunks)
# ---------------------------------------------------------------------------

def fetch_and_cache_btc_prices(min_ts, max_ts):
    """Fetch 1-second BTC klines from Binance in daily chunks with resume."""
    first_write = True

    if os.path.exists(BTC_1S_CACHE):
        last_ts = read_last_timestamp_from_csv(BTC_1S_CACHE)
        if last_ts:
            last_unix = int(last_ts.timestamp())
            if last_unix >= max_ts - 60:
                print(f"BTC price cache is up to date: {BTC_1S_CACHE}")
                return
            min_ts = last_unix + 1
            first_write = False
            print(
                f"Resuming fetch from "
                f"{datetime.fromtimestamp(min_ts, tz=timezone.utc)}"
            )
        else:
            os.remove(BTC_1S_CACHE)

    total_seconds = max_ts - min_ts
    total_batches = total_seconds // 1000 + 1
    print(
        f"Fetching BTCUSDT 1s klines "
        f"({total_seconds:,}s, ~{total_batches:,} batches, "
        f"~{total_batches * 0.12 / 60:.0f} min)"
    )

    day_seconds = 86400
    current = min_ts
    total_points = 0

    while current < max_ts:
        chunk_end = min(current + day_seconds, max_ts)
        start_dt = datetime.fromtimestamp(current, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(chunk_end, tz=timezone.utc)

        print(
            f"  {start_dt.strftime('%Y-%m-%d %H:%M')} -> "
            f"{end_dt.strftime('%Y-%m-%d %H:%M')} ...",
            end="",
            flush=True,
        )

        df = fetch_binance_klines(
            'BTCUSDT', start_dt, end_dt, interval='1s', progress=False
        )

        if not df.empty:
            df.to_csv(
                BTC_1S_CACHE,
                mode='w' if first_write else 'a',
                header=first_write,
                index=False,
            )
            first_write = False
            total_points += len(df)
            print(f" {len(df):,} pts (total: {total_points:,})")
        else:
            print(" (no data)")

        current = chunk_end

    print(f"BTC prices complete: {total_points:,} points -> {BTC_1S_CACHE}")


# ---------------------------------------------------------------------------
# Step 3: BTC price & RV lookup
# ---------------------------------------------------------------------------

class BTCPriceLookup:
    """Fast BTC price and realized volatility lookup using numpy arrays."""

    def __init__(self, cache_path):
        print("Loading BTC 1s prices...")
        prices = pd.read_csv(cache_path)
        prices['timestamp'] = pd.to_datetime(prices['timestamp'], utc=True)
        prices['unix_ts'] = (
            prices['timestamp'].astype(np.int64) // 10**9
        ).astype(np.int64)
        prices = (
            prices.sort_values('unix_ts')
            .drop_duplicates(subset='unix_ts')
            .reset_index(drop=True)
        )

        self.ts = prices['unix_ts'].values
        self.price = prices['btc_price'].values.astype(np.float64)

        # Log returns for rv computation
        self.log_ret = np.zeros(len(self.price))
        self.log_ret[1:] = np.log(self.price[1:] / self.price[:-1])

        print(
            f"  {len(self.ts):,} points, "
            f"{datetime.fromtimestamp(int(self.ts[0]), tz=timezone.utc)} -> "
            f"{datetime.fromtimestamp(int(self.ts[-1]), tz=timezone.utc)}"
        )

        self._precompute_rv()

    def _precompute_rv(self):
        """Precompute rolling realized volatility at 1-15 minute windows."""
        print("Precomputing rolling volatility (1-15 min windows)...")
        seconds_per_year = 365.25 * 24 * 3600
        log_ret_series = pd.Series(self.log_ret)
        self.rv = {}

        for w_min in range(1, 16):
            w_sec = w_min * 60
            std = log_ret_series.rolling(
                window=w_sec, min_periods=max(2, w_sec // 4)
            ).std()
            self.rv[w_min] = (std * np.sqrt(seconds_per_year)).values
            print(f"  rv_{w_min}m done")

    def _nearest_idx(self, timestamps):
        """Find nearest price-array index for each timestamp (vectorized)."""
        ts = np.asarray(timestamps, dtype=np.int64)
        idx = np.searchsorted(self.ts, ts, side='left')
        idx = np.clip(idx, 0, len(self.ts) - 1)

        prev = np.clip(idx - 1, 0, len(self.ts) - 1)
        use_prev = np.abs(self.ts[prev] - ts) < np.abs(self.ts[idx] - ts)
        idx[use_prev] = prev[use_prev]
        return idx

    def get_prices(self, timestamps, max_gap=5):
        """Look up BTC prices. Returns NaN if nearest is > max_gap seconds."""
        idx = self._nearest_idx(timestamps)
        ts = np.asarray(timestamps, dtype=np.int64)
        prices = self.price[idx].copy()
        dist = np.abs(self.ts[idx] - ts)
        prices[dist > max_gap] = np.nan
        return prices

    def get_rv(self, timestamps, tte_seconds):
        """Look up precomputed rv. Window = ceil(tte/60) minutes, clamped 1-15."""
        idx = self._nearest_idx(timestamps)
        tte_min = np.clip(
            np.ceil(np.asarray(tte_seconds, dtype=np.float64) / 60).astype(int),
            1,
            15,
        )
        rv = np.full(len(timestamps), np.nan)
        for w in range(1, 16):
            mask = tte_min == w
            if mask.any():
                rv[mask] = self.rv[w][idx[mask]]
        return rv


# ---------------------------------------------------------------------------
# Step 4: Process trades in chunks
# ---------------------------------------------------------------------------

def process_trades(btc: BTCPriceLookup):
    """Read trades in chunks, aggregate by second, compute features, write."""
    print(f"\nProcessing trades: {TRADES_PATH}")
    first_write = True
    total_out = 0
    chunk_num = 0

    for chunk in pd.read_csv(TRADES_PATH, chunksize=CHUNK_SIZE):
        chunk_num += 1

        # Parse start_ts from market_slug, compute expiry & tte
        chunk['start_ts'] = chunk['market_slug'].str.extract(
            r'(\d+)$', expand=False
        ).astype(int)
        chunk['expiry'] = chunk['start_ts'] + 15 * 60
        chunk['tte'] = chunk['expiry'] - chunk['timestamp']
        chunk = chunk[chunk['tte'] > 0].copy()
        if chunk.empty:
            continue

        # side: token1 = Up = 1, token2 = Down = 0
        chunk['side'] = np.where(chunk['token_side'] == 'token1', 1, 0)
        chunk['price'] = chunk['price'].astype(float)

        # Aggregate: average price within same (second, market, side)
        grouped = (
            chunk.groupby(['timestamp', 'market_slug', 'side'])
            .agg(
                price=('price', 'mean'),
                start_ts=('start_ts', 'first'),
                tte=('tte', 'first'),
            )
            .reset_index()
        )

        # BTC price at trade time
        grouped['btc_price'] = btc.get_prices(grouped['timestamp'].values)

        # Strike = BTC price at market start
        grouped['strike'] = btc.get_prices(
            grouped['start_ts'].values, max_gap=10
        )

        grouped = grouped.dropna(subset=['btc_price', 'strike'])
        if grouped.empty:
            continue

        # moneyness
        grouped['moneyness'] = grouped['btc_price'] / grouped['strike']

        # rv (lookback window = tte)
        grouped['rv'] = btc.get_rv(
            grouped['timestamp'].values, grouped['tte'].values
        )
        grouped = grouped.dropna(subset=['rv'])
        if grouped.empty:
            continue

        out = grouped[['moneyness', 'rv', 'side', 'tte', 'price']]
        out.to_csv(
            OUTPUT_PATH,
            mode='w' if first_write else 'a',
            header=first_write,
            index=False,
        )
        first_write = False
        total_out += len(out)

        print(
            f"  Chunk {chunk_num}: {len(chunk):,} trades -> {len(out):,} rows "
            f"(total: {total_out:,})"
        )

    print(f"\nDone. {total_out:,} rows written to {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(TRADES_PATH):
        print(f"Trades file not found: {TRADES_PATH}")
        print("Run collect_15m_data.py first.")
        sys.exit(1)

    print("=" * 60)
    print("Step 1: Scanning time range")
    print("=" * 60)
    min_ts, max_ts = scan_time_range()

    print("\n" + "=" * 60)
    print("Step 2: Fetching BTC 1s prices")
    print("=" * 60)
    fetch_and_cache_btc_prices(min_ts, max_ts)

    print("\n" + "=" * 60)
    print("Step 3: Loading prices & computing volatility")
    print("=" * 60)
    btc = BTCPriceLookup(BTC_1S_CACHE)

    print("\n" + "=" * 60)
    print("Step 4: Processing trades")
    print("=" * 60)
    process_trades(btc)


if __name__ == '__main__':
    main()

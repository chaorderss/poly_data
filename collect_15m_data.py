"""
Collect all btc-up-or-down-15m-* trades from polymarket_data into a single CSV.

Usage:
    python lstm_calibration/collect_15m_data.py
"""

import csv
import os

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'polymarket_data')

MARKETS_CSV = os.path.join(DATA_DIR, 'markets.csv')
TRADES_CSV = os.path.join(DATA_DIR, 'processed', 'trades.csv')
OUTPUT_CSV = os.path.join(DATA_DIR, 'btc_updown_15m_all.csv')


def collect():
    # Step 1: Read markets and filter btc-up-or-down-15m
    print("Reading markets...")
    markets = {}
    with open(MARKETS_CSV, 'r') as f:
        reader = csv.DictReader(f)
        market_fields = reader.fieldnames
        for row in reader:
            if 'btc-up-or-down-15m-' in row.get('market_slug', ''):
                markets[row['id']] = row

    print(f"Found {len(markets)} btc-up-or-down-15m markets")

    # Step 2: Stream through trades.csv and filter
    market_ids = set(markets.keys())
    market_extra_fields = [f for f in market_fields if f != 'id']

    trade_count = 0
    line_count = 0

    print(f"Scanning {TRADES_CSV} ...")

    with open(TRADES_CSV, 'r') as fin:
        reader = csv.DictReader(fin)
        trade_fields = reader.fieldnames
        out_fields = trade_fields + ['market_' + f for f in market_extra_fields]

        with open(OUTPUT_CSV, 'w', newline='') as fout:
            writer = csv.DictWriter(fout, fieldnames=out_fields)
            writer.writeheader()

            for row in reader:
                line_count += 1
                if line_count % 10_000_000 == 0:
                    print(f"  {line_count / 1e6:.0f}M rows scanned, {trade_count} matches")

                mid = row.get('market_id', '')
                if mid in market_ids:
                    market = markets[mid]
                    out_row = dict(row)
                    for f in market_extra_fields:
                        out_row['market_' + f] = market[f]
                    writer.writerow(out_row)
                    trade_count += 1

    print(f"Done. Scanned {line_count} rows, wrote {trade_count} matching trades.")
    print(f"Output: {OUTPUT_CSV}")


if __name__ == '__main__':
    collect()

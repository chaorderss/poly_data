"""
Collect all btc-up-or-down-15m-* trades from orderFilled.csv into a single CSV.

Scans orderFilled.csv directly, matching on token IDs from markets.csv.
This bypasses trades.csv (which may have stale market_id mappings).

Usage:
    python lstm_calibration/poly_data/collect_15m_data.py
"""

import csv
import os

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'polymarket_data')

MARKETS_CSV = os.path.join(DATA_DIR, 'markets.csv')
ORDERFILLED_CSV = os.path.join(DATA_DIR, 'goldsky', 'orderFilled.csv')
OUTPUT_CSV = os.path.join(DATA_DIR, 'btc_updown_15m_all.csv')


def collect():
    # Step 1: Read markets and filter btc-up-or-down-15m
    print("Reading markets...")
    # Map token_id -> {market dict, side_name}
    token_to_market = {}
    market_count = 0

    with open(MARKETS_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            slug = row.get('market_slug', '')
            if 'btc-up-or-down-15m-' not in slug and 'btc-updown-15m-' not in slug:
                continue
            market_count += 1
            if row.get('token1'):
                token_to_market[row['token1']] = {'market': row, 'side': 'token1'}
            if row.get('token2'):
                token_to_market[row['token2']] = {'market': row, 'side': 'token2'}

    print(f"Found {market_count} btc-up-or-down-15m markets ({len(token_to_market)} token IDs)")

    # Step 2: Stream through orderFilled.csv and match on token IDs
    token_ids = set(token_to_market.keys())

    out_fields = [
        'timestamp', 'price', 'usd_amount', 'token_amount',
        'maker', 'taker', 'taker_direction', 'transactionHash',
        'market_id', 'market_slug', 'market_question',
        'answer1', 'answer2', 'token_side'
    ]

    trade_count = 0
    line_count = 0

    print(f"Scanning {ORDERFILLED_CSV} ...")

    with open(ORDERFILLED_CSV, 'r') as fin:
        reader = csv.DictReader(fin)

        with open(OUTPUT_CSV, 'w', newline='') as fout:
            writer = csv.DictWriter(fout, fieldnames=out_fields)
            writer.writeheader()

            for row in reader:
                line_count += 1
                if line_count % 10_000_000 == 0:
                    print(f"  {line_count / 1e6:.0f}M rows scanned, {trade_count:,} matches")

                maker_id = row.get('makerAssetId', '')
                taker_id = row.get('takerAssetId', '')

                # Check which side is the conditional token
                match_info = None
                is_maker_token = False

                if maker_id in token_ids:
                    match_info = token_to_market[maker_id]
                    is_maker_token = True
                elif taker_id in token_ids:
                    match_info = token_to_market[taker_id]
                    is_maker_token = False
                else:
                    continue

                market = match_info['market']
                token_side = match_info['side']

                try:
                    maker_amount = float(row.get('makerAmountFilled', 0)) / 1e6
                    taker_amount = float(row.get('takerAmountFilled', 0)) / 1e6
                except (ValueError, TypeError):
                    continue

                if maker_amount <= 0 or taker_amount <= 0:
                    continue

                if is_maker_token:
                    # Maker has token, taker has USDC -> taker is BUYing
                    usd_amount = taker_amount
                    token_amount = maker_amount
                    taker_direction = 'BUY'
                else:
                    # Taker has token, maker has USDC -> taker is SELLing
                    usd_amount = maker_amount
                    token_amount = taker_amount
                    taker_direction = 'SELL'

                price = usd_amount / token_amount if token_amount > 0 else 0

                out_row = {
                    'timestamp': row.get('timestamp', ''),
                    'price': f"{price:.6f}",
                    'usd_amount': f"{usd_amount:.6f}",
                    'token_amount': f"{token_amount:.6f}",
                    'maker': row.get('maker', ''),
                    'taker': row.get('taker', ''),
                    'taker_direction': taker_direction,
                    'transactionHash': row.get('transactionHash', ''),
                    'market_id': market.get('id', ''),
                    'market_slug': market.get('market_slug', ''),
                    'market_question': market.get('question', ''),
                    'answer1': market.get('answer1', ''),
                    'answer2': market.get('answer2', ''),
                    'token_side': token_side,
                }

                writer.writerow(out_row)
                trade_count += 1

    print(f"Done. Scanned {line_count:,} rows, wrote {trade_count:,} matching trades.")
    print(f"Output: {OUTPUT_CSV}")


if __name__ == '__main__':
    collect()

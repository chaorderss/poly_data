import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
from poly_utils.utils import get_markets, update_missing_tokens

import subprocess
import json

import pandas as pd

CURSOR_FILE = 'processed/cursor_state.json'

def save_cursor(timestamp_str, transaction_hash, maker, taker, orderfilled_line_count):
    """Save cursor state for efficient resume."""
    state = {
        'timestamp': timestamp_str,
        'transactionHash': transaction_hash,
        'maker': maker,
        'taker': taker,
        'orderfilled_line_count': orderfilled_line_count
    }
    os.makedirs('processed', exist_ok=True)
    with open(CURSOR_FILE, 'w') as f:
        json.dump(state, f)

def get_cursor():
    """Get cursor state. Returns (last_processed dict, orderfilled_line_count)."""
    if os.path.exists(CURSOR_FILE):
        try:
            with open(CURSOR_FILE, 'r') as f:
                state = json.load(f)
            last_processed = {
                'timestamp': pd.to_datetime(state['timestamp']),
                'transactionHash': state['transactionHash'],
                'maker': state['maker'],
                'taker': state['taker'],
            }
            line_count = state.get('orderfilled_line_count', None)
            print(f"✓ Loaded cursor from {CURSOR_FILE}")
            print(f"  Last processed: {last_processed['timestamp']}")
            print(f"  OrderFilled line count: {line_count}")
            return last_processed, line_count
        except Exception as e:
            print(f"⚠ Error reading cursor file: {e}")
    return None, None

def get_orderfilled_line_count():
    """Count lines in orderFilled.csv (excluding header)."""
    try:
        result = subprocess.run(['wc', '-l', 'goldsky/orderFilled.csv'],
                              capture_output=True, text=True, check=True)
        return int(result.stdout.split()[0]) - 1  # Subtract header
    except Exception as e:
        print(f"⚠ Error counting orderFilled lines: {e}")
        return None

def get_processed_df(df):
    markets_df = get_markets()
    markets_df = markets_df.rename({'id': 'market_id'})

    # 1) Make markets long: (market_id, side, asset_id) where side ∈ {"token1", "token2"}
    markets_long = (
        markets_df
        .select(["market_id", "token1", "token2"])
        .melt(id_vars="market_id", value_vars=["token1", "token2"],
            variable_name="side", value_name="asset_id")
    )

    # 2) Identify the non-USDC asset for each trade (the one that isn't 0)
    df = df.with_columns(
        pl.when(pl.col("makerAssetId") != "0")
        .then(pl.col("makerAssetId"))
        .otherwise(pl.col("takerAssetId"))
        .alias("nonusdc_asset_id")
    )

    # 3) Join once on that non-USDC asset to recover the market + side ("token1" or "token2")
    df = df.join(
        markets_long,
        left_on="nonusdc_asset_id",
        right_on="asset_id",
        how="left",
    )

    # 4) label columns and keep market_id
    df = df.with_columns([
        pl.when(pl.col("makerAssetId") == "0").then(pl.lit("USDC")).otherwise(pl.col("side")).alias("makerAsset"),
        pl.when(pl.col("takerAssetId") == "0").then(pl.lit("USDC")).otherwise(pl.col("side")).alias("takerAsset"),
        pl.col("market_id"),
    ])

    df = df[['timestamp', 'market_id', 'maker', 'makerAsset', 'makerAmountFilled', 'taker', 'takerAsset', 'takerAmountFilled', 'transactionHash']]

    df = df.with_columns([
        (pl.col("makerAmountFilled") / 10**6).alias("makerAmountFilled"),
        (pl.col("takerAmountFilled") / 10**6).alias("takerAmountFilled"),
    ])

    df = df.with_columns(
        pl.when(pl.col("takerAsset") == "USDC")
        .then(pl.lit("BUY"))
        .otherwise(pl.lit("SELL"))
        .alias("taker_direction")
    )

    df = df.with_columns([
        pl.when(pl.col("takerAsset") == "USDC")
        .then(pl.lit("BUY"))
        .otherwise(pl.lit("SELL"))
        .alias("taker_direction"),

        # reverse of taker_direction
        pl.when(pl.col("takerAsset") == "USDC")
        .then(pl.lit("SELL"))
        .otherwise(pl.lit("BUY"))
        .alias("maker_direction"),
    ])

    df = df.with_columns([
        pl.when(pl.col("makerAsset") != "USDC")
        .then(pl.col("makerAsset"))
        .otherwise(pl.col("takerAsset"))
        .alias("nonusdc_side"),

        pl.when(pl.col("takerAsset") == "USDC")
        .then(pl.col("takerAmountFilled"))
        .otherwise(pl.col("makerAmountFilled"))
        .alias("usd_amount"),
        pl.when(pl.col("takerAsset") != "USDC")
        .then(pl.col("takerAmountFilled"))
        .otherwise(pl.col("makerAmountFilled"))
        .alias("token_amount"),
        pl.when(pl.col("takerAsset") == "USDC")
        .then(pl.col("takerAmountFilled") / pl.col("makerAmountFilled"))
        .otherwise(pl.col("makerAmountFilled") / pl.col("takerAmountFilled"))
        .cast(pl.Float64)
        .alias("price")
    ])


    df = df[['timestamp', 'market_id', 'maker', 'taker', 'nonusdc_side', 'maker_direction', 'taker_direction', 'price', 'usd_amount', 'token_amount', 'transactionHash']]
    return df



def process_live():
    processed_file = 'processed/trades.csv'

    print("=" * 60)
    print("🔄 Processing Live Trades")
    print("=" * 60)

    # Try to load cursor first (more efficient)
    last_processed, prev_line_count = get_cursor()

    # Fallback to reading processed file if no cursor
    if last_processed is None and os.path.exists(processed_file):
        print(f"✓ Found existing processed file: {processed_file}")
        result = subprocess.run(['tail', '-n', '1', processed_file], capture_output=True, text=True)
        last_line = result.stdout.strip()
        splitted = last_line.split(',')

        last_processed = {
            'timestamp': pd.to_datetime(splitted[0]),
            'transactionHash': splitted[-1],
            'maker': splitted[2],
            'taker': splitted[3],
        }

        print(f"📍 Resuming from processed file: {last_processed['timestamp']}")
        print(f"   Last hash: {last_processed['transactionHash'][:16]}...")
    elif last_processed is None:
        print("⚠ No cursor or processed file found - processing from beginning")

    # Get current orderFilled line count
    current_line_count = get_orderfilled_line_count()

    schema_overrides = {
        "takerAssetId": pl.Utf8,
        "makerAssetId": pl.Utf8,
    }

    # Decide whether to read full file or only new lines
    if prev_line_count is not None and current_line_count is not None:
        new_lines = current_line_count - prev_line_count
        if new_lines <= 0:
            print(f"✓ No new data in orderFilled.csv")
            return

        print(f"📂 Reading only new data from goldsky/orderFilled.csv")
        print(f"   Previous: {prev_line_count:,} lines, Current: {current_line_count:,} lines")
        print(f"   Reading last {new_lines + 100:,} lines (with buffer)...")

        # Read only the tail of the file (new lines + small buffer for overlap)
        buffer_lines = new_lines + 100  # Add buffer to ensure we catch the resume point
        try:
            result = subprocess.run(['tail', '-n', str(buffer_lines), 'goldsky/orderFilled.csv'],
                                  capture_output=True, text=True, check=True)

            # Write to temp file and read with polars
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
                # Get header from original file
                header_result = subprocess.run(['head', '-n', '1', 'goldsky/orderFilled.csv'],
                                             capture_output=True, text=True, check=True)
                tmp.write(header_result.stdout)
                tmp.write(result.stdout)
                tmp_path = tmp.name

            df = pl.read_csv(tmp_path, schema_overrides=schema_overrides)
            os.unlink(tmp_path)

        except Exception as e:
            print(f"⚠ Error reading tail: {e}, falling back to full file read")
            df = pl.scan_csv("goldsky/orderFilled.csv", schema_overrides=schema_overrides).collect(streaming=True)
    else:
        print(f"📂 Reading full goldsky/orderFilled.csv (first run or no cursor)")
        df = pl.scan_csv("goldsky/orderFilled.csv", schema_overrides=schema_overrides).collect(streaming=True)

    df = df.with_columns(
        pl.from_epoch(pl.col('timestamp'), time_unit='s').alias('timestamp')
    )

    print(f"✓ Loaded {len(df):,} rows")

    if last_processed is not None:
        df = df.with_row_index()

        same_timestamp = df.filter(pl.col('timestamp') == last_processed['timestamp'])
        same_timestamp = same_timestamp.filter(
            (pl.col("transactionHash") == last_processed['transactionHash']) &
            (pl.col("maker") == last_processed['maker']) &
            (pl.col("taker") == last_processed['taker'])
        )

        if len(same_timestamp) > 0:
            df_process = df.filter(pl.col('index') > same_timestamp.row(0)[0])
            df_process = df_process.drop('index')
        else:
            print("⚠ Could not find exact resume point, processing all loaded rows")
            df_process = df.drop('index')
    else:
        df_process = df

    print(f"⚙️  Processing {len(df_process):,} new rows...")

    if len(df_process) == 0:
        print("✓ No new rows to process")
        return

    new_df = get_processed_df(df_process)

    if not os.path.isdir('processed'):
        os.makedirs('processed')

    op_file = 'processed/trades.csv'

    if not os.path.isfile(op_file):
        new_df.write_csv(op_file)
        print(f"✓ Created new file: processed/trades.csv")
    else:
        print(f"✓ Appending {len(new_df):,} rows to processed/trades.csv")
        with open(op_file, mode="a") as f:
            new_df.write_csv(f, include_header=False)

    # Save cursor for next run
    if len(new_df) > 0:
        last_row = new_df.row(-1, named=True)
        save_cursor(
            timestamp_str=str(last_row['timestamp']),
            transaction_hash=last_row['transactionHash'],
            maker=last_row['maker'],
            taker=last_row['taker'],
            orderfilled_line_count=current_line_count
        )
        print(f"✓ Saved cursor state")

    print("=" * 60)
    print("✅ Processing complete!")
    print("=" * 60)

if __name__ == "__main__":
    process_live()
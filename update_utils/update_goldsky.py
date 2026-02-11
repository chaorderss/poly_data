import os
import json
import pandas as pd
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport
from flatten_json import flatten
from datetime import datetime, timezone
import subprocess
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from update_utils.update_markets import update_markets

# Global runtime timestamp - set once when program starts
RUNTIME_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# Columns to save
COLUMNS_TO_SAVE = ['timestamp', 'maker', 'makerAssetId', 'makerAmountFilled', 'taker', 'takerAssetId', 'takerAmountFilled', 'transactionHash']

if not os.path.isdir('goldsky'):
    os.mkdir('goldsky')

CURSOR_FILE = 'goldsky/cursor_state.json'

def save_cursor(timestamp, last_id, sticky_timestamp=None):
    """Save cursor state to file for efficient resume."""
    state = {
        'last_timestamp': timestamp,
        'last_id': last_id,
        'sticky_timestamp': sticky_timestamp
    }
    with open(CURSOR_FILE, 'w') as f:
        json.dump(state, f)

def get_latest_cursor():
    """Get the latest cursor state for efficient resume.
    Returns (timestamp, last_id, sticky_timestamp) tuple."""
    # First try to load from cursor state file (most efficient)
    if os.path.isfile(CURSOR_FILE):
        try:
            with open(CURSOR_FILE, 'r') as f:
                state = json.load(f)
            timestamp = state.get('last_timestamp', 0)
            last_id = state.get('last_id')
            sticky_timestamp = state.get('sticky_timestamp')

            # Validate cursor state: if sticky_timestamp is set, last_id must also be set
            if sticky_timestamp is not None and last_id is None:
                print(f"Warning: Invalid cursor state (sticky_timestamp={sticky_timestamp} but last_id=None), clearing sticky state")
                sticky_timestamp = None

            if timestamp > 0:
                readable_time = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                print(f'Resuming from cursor state: timestamp {timestamp} ({readable_time}), id: {last_id}, sticky: {sticky_timestamp}')
                return timestamp, last_id, sticky_timestamp
        except Exception as e:
            print(f"Error reading cursor file: {e}")

    # Fallback: read from CSV file
    cache_file = 'goldsky/orderFilled.csv'

    if not os.path.isfile(cache_file):
        print("No existing file found, starting from beginning of time (timestamp 0)")
        return 0, None, None

    try:
        # Use tail to get the last line efficiently
        result = subprocess.run(['tail', '-n', '1', cache_file], capture_output=True, text=True, check=True)
        last_line = result.stdout.strip()
        if last_line:
            # Get header to find column indices
            header_result = subprocess.run(['head', '-n', '1', cache_file], capture_output=True, text=True, check=True)
            headers = header_result.stdout.strip().split(',')

            if 'timestamp' in headers:
                timestamp_index = headers.index('timestamp')
                values = last_line.split(',')
                if len(values) > timestamp_index:
                    last_timestamp = int(values[timestamp_index])
                    readable_time = datetime.fromtimestamp(last_timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                    print(f'Resuming from CSV (no cursor file): timestamp {last_timestamp} ({readable_time})')
                    # Go back 1 second to ensure no data loss (may create some duplicates)
                    return last_timestamp - 1, None, None
    except Exception as e:
        print(f"Error reading latest file with tail: {e}")
        # Fallback to pandas
        try:
            df = pd.read_csv(cache_file)
            if len(df) > 0 and 'timestamp' in df.columns:
                last_timestamp = df.iloc[-1]['timestamp']
                readable_time = datetime.fromtimestamp(int(last_timestamp), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                print(f'Resuming from CSV (no cursor file): timestamp {last_timestamp} ({readable_time})')
                return int(last_timestamp) - 1, None, None
        except Exception as e2:
            print(f"Error reading with pandas: {e2}")

    # Fallback to beginning of time
    print("Falling back to beginning of time (timestamp 0)")
    return 0, None, None


def _scrape_range(query_url, start_ts, end_ts, worker_id, at_once=1000):
    """Scrape a specific time range (start_ts, end_ts]. Returns list of dicts."""
    transport = RequestsHTTPTransport(url=query_url, verify=True, retries=3)
    client = Client(transport=transport)

    all_rows = []
    last_timestamp = start_ts
    last_id = None
    sticky_timestamp = None
    batch_count = 0

    while True:
        if sticky_timestamp is not None:
            where_clause = f'timestamp: "{sticky_timestamp}", id_gt: "{last_id}"'
        else:
            where_clause = f'timestamp_gt: "{last_timestamp}", timestamp_lte: "{end_ts}"'

        q_string = '''query MyQuery {
                        orderFilledEvents(orderBy: timestamp, orderDirection: asc
                                             first: ''' + str(at_once) + '''
                                             where: {''' + where_clause + '''}) {
                            fee
                            id
                            maker
                            makerAmountFilled
                            makerAssetId
                            orderHash
                            taker
                            takerAmountFilled
                            takerAssetId
                            timestamp
                            transactionHash
                        }
                    }
                '''

        try:
            query = gql(q_string)
            res = client.execute(query)
        except Exception as e:
            print(f"  [Worker {worker_id}] Query error: {e}, retrying...")
            time.sleep(3)
            transport = RequestsHTTPTransport(url=query_url, verify=True, retries=3)
            client = Client(transport=transport)
            continue

        events = res.get('orderFilledEvents', [])
        if not events:
            if sticky_timestamp is not None:
                last_timestamp = sticky_timestamp
                sticky_timestamp = None
                last_id = None
                continue
            break

        rows = [flatten(x) for x in events]
        all_rows.extend(rows)
        batch_count += 1

        # Sort to determine cursor advancement
        rows_sorted = sorted(rows, key=lambda r: (int(r['timestamp']), r['id']))
        batch_last_ts = int(rows_sorted[-1]['timestamp'])
        batch_last_id = rows_sorted[-1]['id']

        just_cleared_sticky = False
        if len(events) >= at_once:
            sticky_timestamp = batch_last_ts
            last_id = batch_last_id
        else:
            if sticky_timestamp is not None:
                last_timestamp = sticky_timestamp
                sticky_timestamp = None
                last_id = None
                just_cleared_sticky = True
            else:
                last_timestamp = batch_last_ts

        if batch_count % 20 == 0:
            readable = datetime.fromtimestamp(batch_last_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  [Worker {worker_id}] batch {batch_count}, rows {len(all_rows):,}, at {readable}")

        if len(events) < at_once and sticky_timestamp is None and not just_cleared_sticky:
            break

    readable_start = datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime('%m-%d %H:%M')
    readable_end = datetime.fromtimestamp(end_ts, tz=timezone.utc).strftime('%m-%d %H:%M')
    print(f"  [Worker {worker_id}] Done: {readable_start} -> {readable_end}, {len(all_rows):,} rows in {batch_count} batches")
    return all_rows


def scrape(at_once=1000, num_workers=16):
    QUERY_URL = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/0.0.1/gn"
    print(f"Query URL: {QUERY_URL}")
    print(f"Runtime timestamp: {RUNTIME_TIMESTAMP}")

    last_timestamp, last_id, sticky_timestamp = get_latest_cursor()

    # Handle any remaining sticky state first (sequentially)
    if sticky_timestamp is not None:
        print(f"Clearing sticky state at timestamp {sticky_timestamp}...")
        transport = RequestsHTTPTransport(url=QUERY_URL, verify=True, retries=3)
        client = Client(transport=transport)
        output_file = 'goldsky/orderFilled.csv'

        while sticky_timestamp is not None:
            where_clause = f'timestamp: "{sticky_timestamp}", id_gt: "{last_id}"'
            q_string = '''query MyQuery {
                            orderFilledEvents(orderBy: timestamp, orderDirection: asc
                                                 first: ''' + str(at_once) + '''
                                                 where: {''' + where_clause + '''}) {
                                fee id maker makerAmountFilled makerAssetId orderHash
                                taker takerAmountFilled takerAssetId timestamp transactionHash
                            }
                        }
                    '''
            try:
                res = client.execute(gql(q_string))
            except Exception as e:
                print(f"Sticky query error: {e}, retrying...")
                time.sleep(3)
                transport = RequestsHTTPTransport(url=QUERY_URL, verify=True, retries=3)
                client = Client(transport=transport)
                continue

            events = res.get('orderFilledEvents', [])
            if not events:
                last_timestamp = sticky_timestamp
                sticky_timestamp = None
                last_id = None
                break

            df = pd.DataFrame([flatten(x) for x in events]).reset_index(drop=True)
            df = df.sort_values(['timestamp', 'id'], ascending=True).reset_index(drop=True)
            batch_last_ts = int(df.iloc[-1]['timestamp'])
            batch_last_id = df.iloc[-1]['id']

            if len(events) >= at_once:
                sticky_timestamp = batch_last_ts
                last_id = batch_last_id
            else:
                last_timestamp = sticky_timestamp
                sticky_timestamp = None
                last_id = None

            df = df.drop_duplicates(subset=['id'])
            df_to_save = df[COLUMNS_TO_SAVE].copy()
            if os.path.isfile(output_file):
                df_to_save.to_csv(output_file, index=None, mode='a', header=None)
            else:
                df_to_save.to_csv(output_file, index=None)

        save_cursor(last_timestamp, last_id, sticky_timestamp)
        print(f"Sticky state cleared. Cursor at {last_timestamp}")

    # Now do parallel fetching in day-sized batches
    now_ts = int(datetime.now(timezone.utc).timestamp())
    total_range = now_ts - last_timestamp

    if total_range <= 0:
        print("Already up to date.")
        if os.path.isfile(CURSOR_FILE):
            os.remove(CURSOR_FILE)
        return

    DAY_SECONDS = 86400
    output_file = 'goldsky/orderFilled.csv'
    total_records = 0
    final_ts = None
    batch_num = 0

    start_readable = datetime.fromtimestamp(last_timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    end_readable = datetime.fromtimestamp(now_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    total_days = total_range / DAY_SECONDS
    print(f"\nScraping: {start_readable} -> {end_readable} ({total_days:.1f} days)")
    print(f"Strategy: {num_workers} workers x 1 day each, batch every {num_workers} days")

    batch_start = last_timestamp
    while batch_start < now_ts:
        batch_num += 1
        batch_end = min(batch_start + num_workers * DAY_SECONDS, now_ts)

        # Split this batch into daily ranges for workers
        ranges = []
        t = batch_start
        while t < batch_end:
            chunk_end = min(t + DAY_SECONDS, batch_end)
            ranges.append((t, chunk_end))
            t = chunk_end

        actual_workers = len(ranges)
        batch_start_readable = datetime.fromtimestamp(batch_start, tz=timezone.utc).strftime('%Y-%m-%d')
        batch_end_readable = datetime.fromtimestamp(batch_end, tz=timezone.utc).strftime('%Y-%m-%d')
        batch_days = (batch_end - batch_start) / DAY_SECONDS
        print(f"\n--- Batch {batch_num}: {batch_start_readable} -> {batch_end_readable} ({batch_days:.1f} days, {actual_workers} workers) ---")

        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = {
                executor.submit(_scrape_range, QUERY_URL, r[0], r[1], i, at_once): i
                for i, r in enumerate(ranges)
            }

            worker_results = {}
            for future in as_completed(futures):
                worker_id = futures[future]
                try:
                    rows = future.result()
                    worker_results[worker_id] = rows
                except Exception as e:
                    print(f"  [Worker {worker_id}] FAILED: {e}")
                    worker_results[worker_id] = []

        # Merge results in order and write to file
        file_exists = os.path.isfile(output_file)
        batch_records = 0

        for i in range(actual_workers):
            rows = worker_results.get(i, [])
            if not rows:
                continue
            df = pd.DataFrame(rows).reset_index(drop=True)
            df = df.sort_values(['timestamp', 'id'], ascending=True).reset_index(drop=True)
            df = df.drop_duplicates(subset=['id'])

            batch_records += len(df)
            total_records += len(df)
            df_to_save = df[COLUMNS_TO_SAVE].copy()

            if file_exists:
                df_to_save.to_csv(output_file, index=None, mode='a', header=False)
            else:
                df_to_save.to_csv(output_file, index=None)
                file_exists = True

            final_ts = int(df.iloc[-1]['timestamp'])
            del df, rows, df_to_save

        worker_results.clear()

        # Save cursor after each batch
        if final_ts is not None:
            save_cursor(final_ts, None, None)
            readable = datetime.fromtimestamp(final_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            print(f"  Batch {batch_num} done: {batch_records:,} records, cursor -> {readable}")

        batch_start = batch_end

    if final_ts is not None:
        readable = datetime.fromtimestamp(final_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        print(f"\nFinal cursor: {final_ts} ({readable})")
    else:
        print("No new records fetched.")

    # Clear cursor file on successful completion (caught up)
    if total_records == 0 and os.path.isfile(CURSOR_FILE):
        os.remove(CURSOR_FILE)

    print(f"Finished scraping orderFilledEvents")
    print(f"Total new records: {total_records:,}")
    print(f"Output file: {output_file}")

def update_goldsky():
    """Run scraping for orderFilledEvents"""
    print(f"\n{'='*50}")
    print(f"Starting to scrape orderFilledEvents")
    print(f"Runtime: {RUNTIME_TIMESTAMP}")
    print(f"{'='*50}")
    try:
        scrape()
        print(f"Successfully completed orderFilledEvents")
    except Exception as e:
        print(f"Error scraping orderFilledEvents: {str(e)}")

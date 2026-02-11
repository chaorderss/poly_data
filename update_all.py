from datetime import datetime, timedelta, timezone
import os
import time

from update_utils.update_markets import update_markets
from update_utils.update_goldsky import update_goldsky, save_cursor, get_latest_cursor
from update_utils.process_live import process_live

TWO_MONTHS_DAYS = 60

if __name__ == "__main__":
    # two_months_ago = datetime.now(timezone.utc) - timedelta(days=TWO_MONTHS_DAYS)
    # two_months_ago_ts = int(two_months_ago.timestamp())

    # os.makedirs('goldsky', exist_ok=True)

    # Ensure goldsky cursor starts no earlier than 2 months ago
    # current_ts, _, _ = get_latest_cursor()
    # if current_ts < two_months_ago_ts:
    #     print(f"Limiting goldsky fetch to last {TWO_MONTHS_DAYS} days "
    #           f"(from {two_months_ago.strftime('%Y-%m-%d %H:%M:%S UTC')})")
    #     save_cursor(two_months_ago_ts, None, None)

    # print("Updating markets")
    # update_markets()

    # Loop until we catch up to present
    iteration = 0
    while True:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"Iteration {iteration} - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"{'='*60}")

        # Get cursor state before update
        ts_before, _, _ = get_latest_cursor()
        ts_before_readable = datetime.fromtimestamp(ts_before, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        print(f"Current cursor: {ts_before} ({ts_before_readable})")

        print("\nUpdating goldsky...")
        update_goldsky()

        # Get cursor state after update
        ts_after, _, _ = get_latest_cursor()
        ts_after_readable = datetime.fromtimestamp(ts_after, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        print(f"New cursor: {ts_after} ({ts_after_readable})")

        print("\nProcessing live...")
        process_live()

        # Check if we're caught up (no new data fetched)
        if ts_after == ts_before:
            now_ts = int(datetime.now(timezone.utc).timestamp())
            time_lag = now_ts - ts_after
            print(f"\nNo new data fetched.")
            print(f"Current time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"Data lag: {time_lag} seconds ({time_lag/60:.1f} minutes)")

            if time_lag < 600:  # Less than 10 minutes behind
                print("✅ Caught up to present! Exiting.")
                break
            else:
                print(f"⚠️  Still {time_lag/60:.1f} minutes behind, but no data returned.")
                print("Waiting 10 seconds before retrying...")
                time.sleep(10)
        else:
            time_advanced = ts_after - ts_before
            print(f"Advanced by {time_advanced} seconds ({time_advanced/60:.1f} minutes)")
            print("Continuing to next iteration...")
            time.sleep(2)  # Brief pause between iterations

    print(f"\n{'='*60}")
    print("Update complete!")
    print(f"{'='*60}")
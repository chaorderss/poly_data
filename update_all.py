from datetime import datetime, timedelta, timezone
import os

from update_utils.update_markets import update_markets
from update_utils.update_goldsky import update_goldsky, save_cursor, get_latest_cursor
from update_utils.process_live import process_live

TWO_MONTHS_DAYS = 60

if __name__ == "__main__":
    two_months_ago = datetime.now(timezone.utc) - timedelta(days=TWO_MONTHS_DAYS)
    two_months_ago_ts = int(two_months_ago.timestamp())

    os.makedirs('goldsky', exist_ok=True)

    # Ensure goldsky cursor starts no earlier than 2 months ago
    current_ts, _, _ = get_latest_cursor()
    if current_ts < two_months_ago_ts:
        print(f"Limiting goldsky fetch to last {TWO_MONTHS_DAYS} days "
              f"(from {two_months_ago.strftime('%Y-%m-%d %H:%M:%S UTC')})")
        save_cursor(two_months_ago_ts, None, None)

    print("Updating markets")
    update_markets()
    print("Updating goldsky")
    update_goldsky()
    print("Processing live")
    process_live()
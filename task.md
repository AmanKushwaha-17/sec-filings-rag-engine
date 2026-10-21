# Task Checklist: Make README Mermaid Diagrams Black and White Only

- [X] Plan black-and-white monochrome styling for Mermaid diagrams in `README.md`
- [X] Create `implementation_plan.md` artifact with proposed diffs and request user feedback
- [X] Await explicit user approval (Proceed button)
- [X] Update `README.md` to remove colored classes and apply black-and-white formatting
- [X] Verify formatting and markdown rendering




```python
"""Fetch 1-minute EUR/USD data for the past 1 year from Dukascopy.

Uses dukascopy-python to download historical 1-minute BID and ASK bars,
saving them in monthly Parquet files (matching repository backtest format)
or as a single consolidated CSV/Parquet file.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import sys

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dukascopy_fetcher")


def get_dukascopy():
    """Import dukascopy_python with helpful installation prompt if missing."""
    try:
        import dukascopy_python
        from dukascopy_python import instruments
        return dukascopy_python, instruments
    except ImportError:
        logger.error(
            "\nThe 'dukascopy-python' package is not installed.\n"
            "Please run:\n"
            "    pip install dukascopy-python\n"
            "or (with uv):\n"
            "    uv add dukascopy-python\n"
        )
        sys.exit(1)


def parse_date(date_str: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Unable to parse date string '{date_str}'. Expected format: YYYY-MM-DD.")


def get_monthly_windows(start_dt: datetime, end_dt: datetime) -> list[tuple[datetime, datetime, str]]:
    """Split [start_dt, end_dt] into monthly slices for reliable chunked downloading."""
    windows = []
    current_start = start_dt
  
    while current_start < end_dt:
        # Determine start of next month
        if current_start.month == 12:
            next_month = current_start.replace(year=current_start.year + 1, month=1, day=1, hour=0, minute=0, second=0)
        else:
            next_month = current_start.replace(month=current_start.month + 1, day=1, hour=0, minute=0, second=0)
      
        current_end = min(next_month - timedelta(seconds=1), end_dt)
        month_label = current_start.strftime("%Y-%m")
        windows.append((current_start, current_end, month_label))
        current_start = next_month

    return windows


def clean_bars_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format DataFrame to canonical schema: timestamp, open, high, low, close, volume."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    out = df.copy()
    if not isinstance(out.index, pd.RangeIndex):
        out = out.reset_index()

    out.columns = [str(c).strip().lower() for c in out.columns]

    time_cols = ("timestamp", "time", "datetime", "date")
    tcol = next((c for c in time_cols if c in out.columns), None)
    if tcol is None:
        raise ValueError(f"Timestamp column missing in returned data: {list(out.columns)}")

    out["timestamp"] = pd.to_datetime(out[tcol], utc=True)
    if tcol != "timestamp":
        out.drop(columns=[tcol], inplace=True)

    for col in ("open", "high", "low", "close"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
        else:
            raise ValueError(f"Missing price column '{col}' in data.")

    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(-1.0).astype("float64")
    else:
        out["volume"] = -1.0

    out = out[["timestamp", "open", "high", "low", "close", "volume"]]
    out.sort_values("timestamp", inplace=True)
    out.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    out.dropna(subset=["open", "high", "low", "close"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def main() -> None:
    dp, instruments = get_dukascopy()

    parser = argparse.ArgumentParser(
        description="Fetch 1-minute historical data for EUR/USD from Dukascopy."
    )
    parser.add_argument("--pair", default="EURUSD", help="Currency pair (default: EURUSD)")
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD). If omitted, defaults to 1 year ago.")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD). If omitted, defaults to now.")
    parser.add_argument("--timeframe", default="1m", choices=["1m", "5m", "15m", "1h", "1d"], help="Candle timeframe (default: 1m)")
    parser.add_argument("--side", default="both", choices=["both", "bid", "ask"], help="Side to fetch (default: both)")
    parser.add_argument("--layout", default="monthly", choices=["monthly", "single"], help="Output layout: 'monthly' or 'single'")
    parser.add_argument("--format", default="parquet", choices=["parquet", "csv"], help="Format for single layout: parquet or csv")
    parser.add_argument("--out", default="data/raw", help="Output directory or filepath (default: data/raw)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files instead of resuming/skipping")

    args = parser.parse_args()

    # Determine 1-year date range if not specified
    now_utc = datetime.now(timezone.utc)
    end_dt = parse_date(args.end) if args.end else now_utc
    start_dt = parse_date(args.start) if args.start else (end_dt - timedelta(days=365))

    if start_dt >= end_dt:
        parser.error(f"Start date ({start_dt}) must be earlier than end date ({end_dt}).")

    # Timeframe mapping
    tf_map = {
        "1m": dp.INTERVAL_MIN_1,
        "5m": dp.INTERVAL_MIN_5,
        "15m": dp.INTERVAL_MIN_15,
        "1h": dp.INTERVAL_HOUR_1,
        "1d": dp.INTERVAL_DAY_1,
    }
    interval = tf_map[args.timeframe]

    # Resolve EURUSD instrument
    instrument = instruments.INSTRUMENT_FX_MAJORS_EUR_USD

    sides = ["bid", "ask"] if args.side == "both" else [args.side]
    monthly_slices = get_monthly_windows(start_dt, end_dt)
    out_root = Path(args.out)

    logger.info("Fetching %s %s bars from %s to %s (%d monthly chunks)",
                args.pair, args.timeframe, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"), len(monthly_slices))

    for side in sides:
        dp_side = dp.OFFER_SIDE_BID if side == "bid" else dp.OFFER_SIDE_ASK
        side_frames = []

        logger.info("\n=== Processing side: %s ===", side.upper())
        target_dir = out_root / args.pair.upper() / side
        target_dir.mkdir(parents=True, exist_ok=True)

        for idx, (m_start, m_end, m_label) in enumerate(monthly_slices, 1):
            month_file = target_dir / f"{m_label}.parquet"

            if args.layout == "monthly" and month_file.exists() and not args.force:
                logger.info("[%d/%d] Skipping %s (%s) - file already exists", idx, len(monthly_slices), m_label, side.upper())
                continue

            logger.info("[%d/%d] Fetching %s %s [%s -> %s]...", idx, len(monthly_slices), m_label, side.upper(),
                        m_start.strftime("%Y-%m-%d"), m_end.strftime("%Y-%m-%d"))

            try:
                raw_df = dp.fetch(
                    instrument=instrument,
                    interval=interval,
                    offer_side=dp_side,
                    start=m_start,
                    end=m_end,
                )
            except Exception as e:
                logger.error("Error fetching %s for %s: %s", m_label, side, e)
                continue

            if raw_df is None or raw_df.empty:
                logger.warning("No data for %s %s (weekend/holiday or unavailable).", m_label, side)
                continue

            clean_df = clean_bars_df(raw_df)
            logger.info("  -> Got %d clean bars for %s.", len(clean_df), m_label)

            if args.layout == "monthly":
                clean_df.to_parquet(month_file, index=False)
                logger.info("  -> Saved to %s", month_file)
            else:
                side_frames.append(clean_df)

        if args.layout == "single" and side_frames:
            combined = pd.concat(side_frames, ignore_index=True)
            combined.sort_values("timestamp", inplace=True)
            combined.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
            combined.reset_index(drop=True, inplace=True)

            out_root.mkdir(parents=True, exist_ok=True)
            single_path = out_root / f"{args.pair.upper()}_{side}_{args.timeframe}_1y.{args.format}"
            if args.format == "csv":
                combined.to_csv(single_path, index=False)
            else:
                combined.to_parquet(single_path, index=False)
            logger.info("Saved consolidated file: %s (%d total bars)", single_path, len(combined))

    logger.info("\nAll requests completed successfully.")


if __name__ == "__main__":
    main()
```

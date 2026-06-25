#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
match_drives.py
===============
Matches .BIN (STM32 IMU) recordings with .CSV (OBD/GPS) recordings
by comparing drive durations.

Since BIN and CSV files may have started at slightly different times
(OBD often connects a few seconds before/after the STM32 starts),
matching is done by closest duration rather than absolute timestamps.

Output: numbered subfolders (1, 2, 3, ...) each containing one matched
        .bin + .csv pair, ready for the AutoDNA pipeline.

Usage:
    python match_drives.py --bin-dir /path/to/bin_files
                           --csv-dir /path/to/csv_files
                           --out-dir /path/to/output
                           --start-index 1
"""

import argparse
import shutil
import pandas as pd
import numpy as np
from pathlib import Path


# ── try to import your existing BIN parser ────────────────────────────────────
import sys
from pathlib import Path

# add project root (PROJEKT_REPO) to path so AutoDNA package is findable
# script is at: PROJEKT_REPO/AutoDNA/data/drive_data/miha-hyundai/match/match_drives.py
# so parents[5] = PROJEKT_REPO
_SCRIPT_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[4]  # go up: match -> miha-hyundai -> drive_data -> data -> AutoDNA -> PROJEKT_REPO

sys.path.insert(0, str(_PROJECT_ROOT))                                          # finds AutoDNA package
sys.path.insert(0, str(_PROJECT_ROOT / 'AutoDNA' / 'stm32' / 'bin_parser'))    # finds stm_utils.py

from stm_utils import read_packets_from_file


# ── duration extraction ────────────────────────────────────────────────────────

def get_bin_info(bin_path: Path) -> dict | None:
    """
    Extract start time and duration from a BIN file via packet timestamps.
    BIN timestamps are in milliseconds (STM32 elapsed time).
    """
    try:
        packets = read_packets_from_file(str(bin_path))
        if not packets:
            print(f"  [BIN] {bin_path.name}: no packets parsed")
            return None

        timestamps = [p.ts for p in packets]
        first_ts = min(timestamps)
        last_ts  = max(timestamps)
        duration_s = (last_ts - first_ts) / 1000.0   # ms -> seconds

        return {
            'path':       bin_path,
            'duration_s': duration_s,
            'first_ts':   first_ts,
            'last_ts':    last_ts,
            'n_packets':  len(packets),
        }
    except Exception as e:
        print(f"  [BIN] Error reading {bin_path.name}: {e}")
        return None


def get_csv_info(csv_path: Path) -> dict | None:
    """
    Extract start time and duration from a CSV file via the seconds column.
    Handles both comma and semicolon separators.
    """
    try:
        # try semicolon first (OBD default), fall back to comma
        try:
            df = pd.read_csv(csv_path, sep=';', quotechar='"')
            if df.shape[1] < 3:
                raise ValueError("too few columns with ';' separator")
        except Exception:
            df = pd.read_csv(csv_path, sep=',', quotechar='"')

        df.columns = df.columns.str.lower().str.strip()

        if 'seconds' not in df.columns:
            print(f"  [CSV] {csv_path.name}: no 'seconds' column")
            return None

        seconds = pd.to_numeric(df['seconds'], errors='coerce').dropna()
        if len(seconds) < 2:
            print(f"  [CSV] {csv_path.name}: too few valid rows ({len(seconds)})")
            return None

        first_ts = float(seconds.iloc[0])
        last_ts  = float(seconds.iloc[-1])
        duration = last_ts - first_ts

        # if duration is suspiciously large it might be in milliseconds
        if duration > 100_000:
            duration /= 1000.0

        if duration <= 0:
            print(f"  [CSV] {csv_path.name}: non-positive duration ({duration:.1f}s)")
            return None

        return {
            'path':       csv_path,
            'duration_s': duration,
            'first_ts':   first_ts,
            'last_ts':    last_ts,
            'n_rows':     len(df),
        }
    except Exception as e:
        print(f"  [CSV] Error reading {csv_path.name}: {e}")
        return None


# ── matching ───────────────────────────────────────────────────────────────────

def match_by_duration(bin_infos: list[dict],
                      csv_infos: list[dict],
                      max_diff_s: float = 120.0) -> list[tuple]:
    """
    Greedy nearest-duration matching.
    Sorts both lists by duration, then matches each BIN to its closest CSV.

    :param bin_infos:  List of BIN info dicts (must have 'duration_s').
    :param csv_infos:  List of CSV info dicts (must have 'duration_s').
    :param max_diff_s: Maximum allowed duration difference in seconds.
                       Pairs further apart than this are rejected.
    :returns: List of (bin_info, csv_info, diff_s) tuples, one per match.
    """
    # sort both by duration
    bins_sorted = sorted(bin_infos, key=lambda x: x['duration_s'])
    csvs_sorted = sorted(csv_infos, key=lambda x: x['duration_s'])

    # build cost matrix: rows=BIN files, cols=CSV files
    n_bins = len(bins_sorted)
    n_csvs = len(csvs_sorted)
    cost = np.zeros((n_bins, n_csvs), dtype=float)
    for i, b in enumerate(bins_sorted):
        for j, c in enumerate(csvs_sorted):
            cost[i, j] = abs(b['duration_s'] - c['duration_s'])

    # greedy matching: repeatedly pick the smallest cost pair
    matched    = []
    used_bins  = set()
    used_csvs  = set()

    # flatten all pairs sorted by cost
    pairs = sorted(
        [(cost[i, j], i, j) for i in range(n_bins) for j in range(n_csvs)],
        key=lambda x: x[0]
    )

    for diff, i, j in pairs:
        if i in used_bins or j in used_csvs:
            continue
        if diff > max_diff_s:
            break   # remaining pairs are all worse, sorted list so we can stop
        matched.append((bins_sorted[i], csvs_sorted[j], diff))
        used_bins.add(i)
        used_csvs.add(j)

    return matched


# ── output ─────────────────────────────────────────────────────────────────────

def create_drive_folders(matches: list[tuple],
                         output_dir: Path,
                         start_index: int = 1):
    """
    Create numbered subfolders and copy matched file pairs into them.

    :param matches:     List of (bin_info, csv_info, diff_s) from match_by_duration.
    :param output_dir:  Root output directory (created if missing).
    :param start_index: First folder number (default 1).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for offset, (bin_info, csv_info, diff_s) in enumerate(matches):
        folder_name = str(start_index + offset)
        folder = output_dir / folder_name
        folder.mkdir(exist_ok=True)

        shutil.copy2(bin_info['path'], folder / bin_info['path'].name)
        shutil.copy2(csv_info['path'], folder / csv_info['path'].name)

        print(f"  [{folder_name}]  {bin_info['path'].name} ({bin_info['duration_s']:.0f}s)"
              f"  ↔  {csv_info['path'].name} ({csv_info['duration_s']:.0f}s)"
              f"  diff={diff_s:.1f}s")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Match STM32 .BIN files with OBD .CSV files by drive duration.'
    )
    parser.add_argument('--bin-dir',      required=True, help='Folder containing .BIN files')
    parser.add_argument('--csv-dir',      required=True, help='Folder containing .CSV files')
    parser.add_argument('--out-dir',      required=True, help='Output folder for numbered subfolders')
    parser.add_argument('--start-index',  type=int, default=1,  help='First subfolder number (default: 1)')
    parser.add_argument('--max-diff',     type=float, default=120.0,
                        help='Max duration difference in seconds to accept a match (default: 120)')
    args = parser.parse_args()

    bin_dir = Path(args.bin_dir)
    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir)

    bin_files = sorted(bin_dir.glob('*.bin')) + sorted(bin_dir.glob('*.BIN'))
    csv_files = sorted(csv_dir.glob('*.csv')) + sorted(csv_dir.glob('*.CSV'))

    if not bin_files:
        print(f"No .BIN files found in {bin_dir}")
        sys.exit(1)
    if not csv_files:
        print(f"No .CSV files found in {csv_dir}")
        sys.exit(1)

    print(f"Found {len(bin_files)} BIN files and {len(csv_files)} CSV files.\n")

    # extract durations
    print("Reading BIN files...")
    bin_infos = [info for f in bin_files if (info := get_bin_info(f)) is not None]

    print("\nReading CSV files...")
    csv_infos = [info for f in csv_files if (info := get_csv_info(f)) is not None]

    # print duration overview so user can spot obvious mismatches
    print("\n--- BIN durations ---")
    for b in sorted(bin_infos, key=lambda x: x['duration_s']):
        print(f"  {b['path'].name:30s}  {b['duration_s']:7.1f}s  ({b['duration_s']/60:.1f} min)")

    print("\n--- CSV durations ---")
    for c in sorted(csv_infos, key=lambda x: x['duration_s']):
        print(f"  {c['path'].name:30s}  {c['duration_s']:7.1f}s  ({c['duration_s']/60:.1f} min)")

    # match
    print(f"\nMatching (max allowed difference: {args.max_diff:.0f}s)...")
    matches = match_by_duration(bin_infos, csv_infos, max_diff_s=args.max_diff)

    print(f"\nMatched {len(matches)} / {len(bin_infos)} BIN files:\n")
    create_drive_folders(matches, out_dir, start_index=args.start_index)

    # report unmatched files
    matched_bins = {m[0]['path'] for m in matches}
    matched_csvs = {m[1]['path'] for m in matches}

    unmatched_bins = [b for b in bin_infos if b['path'] not in matched_bins]
    unmatched_csvs = [c for c in csv_infos if c['path'] not in matched_csvs]

    if unmatched_bins:
        print(f"\nUnmatched BIN files ({len(unmatched_bins)}):")
        for b in unmatched_bins:
            print(f"  {b['path'].name}  ({b['duration_s']:.1f}s)")

    if unmatched_csvs:
        print(f"\nUnmatched CSV files ({len(unmatched_csvs)}):")
        for c in unmatched_csvs:
            print(f"  {c['path'].name}  ({c['duration_s']:.1f}s)")

    print(f"\nDone. Folders written to: {out_dir}")


if __name__ == '__main__':
    # hardcoded paths — edit these
    bin_dir     = Path(r'C:\Users\mihal\OneDrive - Univerza v Mariboru\FAKS\4 SEMESTER\PROJEKT\PROJEKT_REPO\AutoDNA\data\drive_data\miha-hyundai\match')
    csv_dir     = Path(r'C:\Users\mihal\OneDrive - Univerza v Mariboru\FAKS\4 SEMESTER\PROJEKT\PROJEKT_REPO\AutoDNA\data\drive_data\miha-hyundai\match')
    out_dir     = Path(r'C:\Users\mihal\OneDrive - Univerza v Mariboru\FAKS\4 SEMESTER\PROJEKT\PROJEKT_REPO\AutoDNA\data\drive_data\miha-hyundai\match\output')
    start_index = 1
    max_diff    = 120.0

    # skip argparse, call functions directly
    bin_files = sorted(bin_dir.glob('*.bin')) + sorted(bin_dir.glob('*.BIN'))
    csv_files = sorted(csv_dir.glob('*.csv')) + sorted(csv_dir.glob('*.CSV'))

    print(f"Found {len(bin_files)} BIN files and {len(csv_files)} CSV files.\n")

    print("Reading BIN files...")
    bin_infos = [info for f in bin_files if (info := get_bin_info(f)) is not None]

    print("\nReading CSV files...")
    csv_infos = [info for f in csv_files if (info := get_csv_info(f)) is not None]

    print("\n--- BIN durations ---")
    for b in sorted(bin_infos, key=lambda x: x['duration_s']):
        print(f"  {b['path'].name:40s}  {b['duration_s']:7.1f}s  ({b['duration_s']/60:.1f} min)")

    print("\n--- CSV durations ---")
    for c in sorted(csv_infos, key=lambda x: x['duration_s']):
        print(f"  {c['path'].name:40s}  {c['duration_s']:7.1f}s  ({c['duration_s']/60:.1f} min)")

    print(f"\nMatching (max allowed difference: {max_diff:.0f}s)...")
    matches = match_by_duration(bin_infos, csv_infos, max_diff_s=max_diff)

    print(f"\nMatched {len(matches)} pairs:\n")
    create_drive_folders(matches, out_dir, start_index=start_index)

    matched_bins = {m[0]['path'] for m in matches}
    matched_csvs = {m[1]['path'] for m in matches}

    unmatched_bins = [b for b in bin_infos if b['path'] not in matched_bins]
    unmatched_csvs = [c for c in csv_infos if c['path'] not in matched_csvs]

    if unmatched_bins:
        print(f"\nUnmatched BIN files ({len(unmatched_bins)}):")
        for b in unmatched_bins:
            print(f"  {b['path'].name}  ({b['duration_s']:.1f}s)")

    if unmatched_csvs:
        print(f"\nUnmatched CSV files ({len(unmatched_csvs)}):")
        for c in unmatched_csvs:
            print(f"  {c['path'].name}  ({c['duration_s']:.1f}s)")

    print(f"\nDone. Output: {out_dir}")
    input("\nPress Enter to close...")
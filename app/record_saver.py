# -*- coding: utf-8 -*-
"""Early per-segment fuel-record tracker, not used by the live app.

Superseded by :mod:`app.segment_records`, which buckets segments by
type/magnitude instead of a single flat key and also tracks
record/close/worse classification for the map. Kept around for
reference; nothing in ``app/`` imports this module anymore.

Created on Fri Jun  5 16:41:07 2026

@author: mihal
"""

import json
from AutoDNA.app.get_path import get_data_dir

RECORDS_FILE = get_data_dir() / "autodna_records.json"


def load_records() -> dict:
    """Load the flat fuel-record store from disk, returning ``{}`` if missing."""
    if RECORDS_FILE.exists():
        return json.loads(RECORDS_FILE.read_text())
    return {}


def save_records(records: dict):
    """Persist the record store to disk as JSON."""
    RECORDS_FILE.write_text(json.dumps(records, indent=2))


def update_record(segment_key: str, fuel_used: float):
    """Update if this run beat the record for this segment."""
    records = load_records()
    
    if segment_key not in records or fuel_used < records[segment_key]:
        records[segment_key] = fuel_used
        save_records(records)
        return 0  # new record
    
    else:
        diff = fuel_used - records[segment_key]
        return diff      #unnecessary fuel used
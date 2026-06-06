# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 16:41:07 2026

@author: mihal
"""

import json
from pathlib import Path

RECORDS_FILE = Path(__file__).resolve().parent.parent / "autodna_records.json"


def load_records() -> dict:
    if RECORDS_FILE.exists():
        return json.loads(RECORDS_FILE.read_text())
    return {}


def save_records(records: dict):
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
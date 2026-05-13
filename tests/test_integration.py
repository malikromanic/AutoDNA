# Integration tests using real .BIN/.npz/.json files from Desktop\Voznje.
# These tests are skipped automatically if the data files are not present (e.g. in CI).

import json

import numpy as np
import pytest

from stm_utils import read_packets_from_file


# ---------------------------------------------------------------------------
# .BIN parsing (real data)
# ---------------------------------------------------------------------------

def test_log001_returns_packets(log001_bin):
    packets = read_packets_from_file(str(log001_bin))
    assert len(packets) > 0


def test_log001_has_all_three_sensors(log001_bin):
    packets = read_packets_from_file(str(log001_bin))
    sensors = {p.sensor for p in packets}
    assert "gyro" in sensors
    assert "accel" in sensors
    assert "mag" in sensors


def test_log001_timestamps_are_positive(log001_bin):
    packets = read_packets_from_file(str(log001_bin))
    assert all(p.ts >= 0 for p in packets)


def test_log001_timestamps_monotonically_increasing(log001_bin):
    packets = read_packets_from_file(str(log001_bin))
    timestamps = [p.ts for p in packets]
    assert timestamps == sorted(timestamps)


def test_log001_sample_data_shape(log001_bin):
    packets = read_packets_from_file(str(log001_bin))
    for p in packets:
        assert p.data.ndim == 2
        assert p.data.shape[1] == 3
        assert p.data.dtype == np.int16


# ---------------------------------------------------------------------------
# .BIN vs .npz consistency
# ---------------------------------------------------------------------------

def test_log001_parsed_matches_npz_sensors(log001_bin, log001_npz):
    packets = read_packets_from_file(str(log001_bin))
    reference = np.load(str(log001_npz))

    parsed_sensors = {p.sensor for p in packets}
    for sensor in reference.files:
        assert sensor in parsed_sensors, f"Sensor '{sensor}' from .npz not found in parsed packets"


def test_log001_parsed_matches_npz_sample_counts(log001_bin, log001_npz):
    packets = read_packets_from_file(str(log001_bin))
    reference = np.load(str(log001_npz))

    for sensor in reference.files:
        expected = len(reference[sensor])
        got = sum(p.sample_count for p in packets if p.sensor == sensor)
        assert got == expected, f"{sensor}: parsed {got} samples, .npz has {expected}"


# ---------------------------------------------------------------------------
# Label JSON schema validation
# ---------------------------------------------------------------------------

VALID_TURN_DIRS = {"left", "right", None}
VALID_HILL_DIRS = {"up", "down", None}
VALID_INTENSITIES = {"blag", "srednji", "intenzivna"}


def _validate_label(label: dict, index: int) -> None:
    assert "t_start" in label, f"Label {index}: missing t_start"
    assert "t_end" in label, f"Label {index}: missing t_end"
    assert label["t_start"] < label["t_end"], f"Label {index}: t_start >= t_end"
    assert "straight" in label, f"Label {index}: missing straight"
    assert isinstance(label["straight"], bool), f"Label {index}: straight must be bool"

    if label.get("turn") is not None:
        turn = label["turn"]
        assert turn.get("dir") in VALID_TURN_DIRS, f"Label {index}: invalid turn dir"
        assert isinstance(turn.get("angleDeg"), (int, float)), f"Label {index}: turn angleDeg must be numeric"
        assert turn.get("intensity") in VALID_INTENSITIES, f"Label {index}: invalid turn intensity"

    if label.get("hill") is not None:
        hill = label["hill"]
        assert hill.get("dir") in VALID_HILL_DIRS, f"Label {index}: invalid hill dir"
        assert isinstance(hill.get("angleDeg"), (int, float)), f"Label {index}: hill angleDeg must be numeric"
        assert hill.get("intensity") in VALID_INTENSITIES, f"Label {index}: invalid hill intensity"


def test_log001_labels_schema(log001_labels):
    with open(log001_labels, encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("schema") == "autodna-v2-multilayer"
    assert "labels" in data
    assert isinstance(data["labels"], list)
    assert len(data["labels"]) > 0

    for i, label in enumerate(data["labels"]):
        _validate_label(label, i)


def test_log001_labels_reference_existing_npz(log001_labels, log001_npz):
    with open(log001_labels, encoding="utf-8") as f:
        data = json.load(f)

    reference = np.load(str(log001_npz))
    if "gyro" not in reference.files:
        pytest.skip("No gyro data in npz to compare against")

    gyro = reference["gyro"]
    max_ts_sec = gyro[:, 0].max() / 1000.0

    for i, label in enumerate(data["labels"]):
        assert label["t_end"] <= max_ts_sec + 1.0, (
            f"Label {i}: t_end={label['t_end']:.2f}s exceeds recording length {max_ts_sec:.2f}s"
        )

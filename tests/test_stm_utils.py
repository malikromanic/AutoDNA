import struct

import numpy as np
import pytest

from stm_utils import (
    crc16_compute,
    crc16_update,
    load_from_npz,
    parse_packet,
    save_to_npz,
    unstuff_bytes,
)
from packet import Packet


# ---------------------------------------------------------------------------
# Helpers: build synthetic valid packets
# ---------------------------------------------------------------------------

def _stuff_bytes(data: bytes) -> bytes:
    """Inverse of unstuff_bytes: escape 0xFE and 0xFF bytes."""
    result = []
    for byte in data:
        if byte in (0xFE, 0xFF):
            result.append(0xFE)
            result.append(byte ^ 0xFE)
        else:
            result.append(byte)
    return bytes(result)


def _build_packet_bytes(timestamp: int, chunks: list) -> bytes:
    """
    Build a complete valid packet (0xFF 0xFF + stuffed payload) for parse_packet.

    chunks: list of (chunk_id, [(x, y, z), ...])
    """
    ts_bytes = struct.pack("<I", timestamp)
    unknown_header = b"\x00\x00"

    chunks_bytes = b""
    for chunk_id, samples in chunks:
        chunk_data = b"".join(struct.pack("<hhh", x, y, z) for x, y, z in samples)
        chunk_size = len(chunk_data)
        chunks_bytes += (
            bytes([chunk_id])
            + struct.pack("<H", chunk_size - 1)
            + b"\x00"
            + chunk_data
        )

    payload_no_crc = ts_bytes + unknown_header + chunks_bytes
    crc = crc16_compute(payload_no_crc)
    payload = payload_no_crc + struct.pack("<H", crc)

    return b"\xFF\xFF" + _stuff_bytes(payload)


def _build_bin_stream(packets_data: list) -> bytes:
    """
    Build a byte stream matching the STM32 .BIN file format.

    packets_data: list of (timestamp, chunks)
    Each packet in the file has the format: 0xFF 0xFF [counter] [stuffed payload]
    """
    stream = b""
    for counter, (timestamp, chunks) in enumerate(packets_data):
        ts_bytes = struct.pack("<I", timestamp)
        unknown_header = b"\x00\x00"

        chunks_bytes = b""
        for chunk_id, samples in chunks:
            chunk_data = b"".join(struct.pack("<hhh", x, y, z) for x, y, z in samples)
            chunk_size = len(chunk_data)
            chunks_bytes += (
                bytes([chunk_id])
                + struct.pack("<H", chunk_size - 1)
                + b"\x00"
                + chunk_data
            )

        payload_no_crc = ts_bytes + unknown_header + chunks_bytes
        crc = crc16_compute(payload_no_crc)
        payload = payload_no_crc + struct.pack("<H", crc)

        stream += b"\xFF\xFF" + bytes([counter & 0xFF]) + _stuff_bytes(payload)

    return stream


# ---------------------------------------------------------------------------
# crc16
# ---------------------------------------------------------------------------

def test_crc16_empty_is_initial_value():
    assert crc16_compute(b"") == 0xFFFF


def test_crc16_consistent():
    data = b"\x01\x02\x03\x04\x05"
    assert crc16_compute(data) == crc16_compute(data)


def test_crc16_different_inputs_different_results():
    assert crc16_compute(b"\x01") != crc16_compute(b"\x02")


def test_crc16_update_matches_compute():
    data = b"\xAB\xCD\xEF"
    crc = 0xFFFF
    for byte in data:
        crc = crc16_update(crc, byte)
    assert crc == crc16_compute(data)


def test_crc16_byte_order_matters():
    assert crc16_compute(b"\x01\x02") != crc16_compute(b"\x02\x01")


# ---------------------------------------------------------------------------
# unstuff_bytes
# ---------------------------------------------------------------------------

def test_unstuff_no_stuffed_bytes():
    data = b"\x01\x02\x03\x04"
    assert unstuff_bytes(data) == data


def test_unstuff_escape_produces_fe():
    # 0xFE 0x00 → 0x00 ^ 0xFE = 0xFE
    assert unstuff_bytes(b"\xFE\x00") == b"\xFE"


def test_unstuff_escape_produces_ff():
    # 0xFE 0x01 → 0x01 ^ 0xFE = 0xFF
    assert unstuff_bytes(b"\xFE\x01") == b"\xFF"


def test_unstuff_mixed_bytes():
    result = unstuff_bytes(b"\x42\xFE\x01\x43")
    assert result == b"\x42\xFF\x43"


def test_unstuff_roundtrip_with_stuff():
    original = bytes(range(256))
    assert unstuff_bytes(_stuff_bytes(original)) == original


def test_unstuff_incomplete_escape_raises():
    with pytest.raises(ValueError):
        unstuff_bytes(b"\xFE")


def test_unstuff_empty():
    assert unstuff_bytes(b"") == b""


# ---------------------------------------------------------------------------
# parse_packet
# ---------------------------------------------------------------------------

def test_parse_packet_invalid_header_returns_none():
    assert parse_packet(b"\x00\x00" + b"\x00" * 20) is None


def test_parse_packet_wrong_first_byte_returns_none():
    data = _build_packet_bytes(1000, [(0x01, [(1, 2, 3)])])
    corrupted = b"\xAA" + data[1:]
    assert parse_packet(corrupted) is None


def test_parse_packet_bad_crc_returns_none():
    data = _build_packet_bytes(1000, [(0x01, [(100, 200, 300)])])
    corrupted = data[:-1] + bytes([data[-1] ^ 0xFF])
    assert parse_packet(corrupted) is None


def test_parse_packet_single_gyro_chunk():
    samples = [(100, 200, 300), (-100, -200, -300)]
    data = _build_packet_bytes(5000, [(0x01, samples)])

    packets = parse_packet(data)

    assert packets is not None
    assert len(packets) == 1
    assert packets[0].sensor == "gyro"
    assert packets[0].ts == 5000.0
    assert packets[0].sample_count == 2
    np.testing.assert_array_equal(
        packets[0].data, np.array(samples, dtype=np.int16)
    )


def test_parse_packet_single_accel_chunk():
    samples = [(1000, -2000, 3000)]
    data = _build_packet_bytes(9999, [(0x02, samples)])

    packets = parse_packet(data)

    assert packets is not None
    assert packets[0].sensor == "accel"
    assert packets[0].ts == 9999.0


def test_parse_packet_multiple_chunks_same_timestamp():
    gyro = [(10, 20, 30)]
    accel = [(40, 50, 60), (70, 80, 90)]
    data = _build_packet_bytes(1000, [(0x01, gyro), (0x02, accel)])

    packets = parse_packet(data)

    assert packets is not None
    assert len(packets) == 2
    assert packets[0].sensor == "gyro"
    assert packets[1].sensor == "accel"
    assert packets[1].sample_count == 2


def test_parse_packet_timestamp_max():
    data = _build_packet_bytes(0xFFFFFFFF, [(0x03, [(1, 2, 3)])])
    packets = parse_packet(data)
    assert packets[0].ts == float(0xFFFFFFFF)


def test_parse_packet_extreme_int16_values():
    samples = [(32767, -32768, 0)]
    data = _build_packet_bytes(1, [(0x01, samples)])
    packets = parse_packet(data)
    np.testing.assert_array_equal(
        packets[0].data, np.array(samples, dtype=np.int16)
    )


# ---------------------------------------------------------------------------
# read_packets_from_file (synthetic .BIN stream)
# ---------------------------------------------------------------------------

def test_read_packets_from_file_synthetic(tmp_path):
    from stm_utils import read_packets_from_file

    stream = _build_bin_stream([
        (1000, [(0x01, [(10, 20, 30)])]),
        (2000, [(0x02, [(40, 50, 60)])]),
        (3000, [(0x03, [(70, 80, 90)])]),
    ])

    bin_file = tmp_path / "test.BIN"
    bin_file.write_bytes(stream)

    packets = read_packets_from_file(str(bin_file))

    assert len(packets) == 3
    sensors = {p.sensor for p in packets}
    assert sensors == {"gyro", "accel", "mag"}


def test_read_packets_from_file_empty(tmp_path):
    from stm_utils import read_packets_from_file

    bin_file = tmp_path / "empty.BIN"
    bin_file.write_bytes(b"")

    packets = read_packets_from_file(str(bin_file))
    assert packets == []


def test_read_packets_from_file_timestamps_preserved(tmp_path):
    from stm_utils import read_packets_from_file

    stream = _build_bin_stream([
        (100, [(0x01, [(1, 2, 3)])]),
        (200, [(0x01, [(4, 5, 6)])]),
        (300, [(0x01, [(7, 8, 9)])]),
    ])

    bin_file = tmp_path / "ts.BIN"
    bin_file.write_bytes(stream)

    packets = read_packets_from_file(str(bin_file))
    timestamps = [p.ts for p in packets]

    assert 100.0 in timestamps
    assert 200.0 in timestamps
    assert 300.0 in timestamps


# ---------------------------------------------------------------------------
# save_to_npz / load_from_npz round-trip
# ---------------------------------------------------------------------------

def test_save_load_roundtrip_sensors(tmp_path):
    original = [
        Packet(id=0x01, ts=1000.0, data=np.array([[10, 20, 30]], dtype=np.int16)),
        Packet(id=0x02, ts=1000.0, data=np.array([[40, 50, 60]], dtype=np.int16)),
        Packet(id=0x03, ts=1000.0, data=np.array([[70, 80, 90]], dtype=np.int16)),
    ]
    npz_path = str(tmp_path / "out.npz")

    save_to_npz(original, npz_path)
    loaded = load_from_npz(npz_path)

    sensors_original = {p.sensor for p in original}
    sensors_loaded = {p.sensor for p in loaded}
    assert sensors_original == sensors_loaded


def test_save_load_roundtrip_sample_counts(tmp_path):
    original = [
        Packet(id=0x01, ts=1000.0, data=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)),
        Packet(id=0x01, ts=2000.0, data=np.array([[7, 8, 9]], dtype=np.int16)),
    ]
    npz_path = str(tmp_path / "out.npz")

    save_to_npz(original, npz_path)
    loaded = load_from_npz(npz_path)

    total_original = sum(p.sample_count for p in original)
    total_loaded = sum(p.sample_count for p in loaded)
    assert total_original == total_loaded


def test_save_load_roundtrip_data_values(tmp_path):
    samples = np.array([[100, 200, 300], [-100, -200, -300]], dtype=np.int16)
    original = [Packet(id=0x01, ts=500.0, data=samples)]
    npz_path = str(tmp_path / "out.npz")

    save_to_npz(original, npz_path)
    loaded = load_from_npz(npz_path)

    gyro_packets = [p for p in loaded if p.sensor == "gyro"]
    loaded_data = np.vstack([p.data for p in gyro_packets])

    for row in samples:
        assert any(np.array_equal(row, loaded_row) for loaded_row in loaded_data)


def test_save_npz_creates_file(tmp_path):
    p = Packet(id=0x01, ts=0.0, data=np.zeros((1, 3), dtype=np.int16))
    npz_path = str(tmp_path / "out")

    save_to_npz([p], npz_path)

    assert (tmp_path / "out.npz").exists()

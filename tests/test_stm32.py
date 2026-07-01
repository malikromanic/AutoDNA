# ============================================================================
# STM32 binary parser tests (packet + stm_utils) — core behaviors only.
# ============================================================================
import struct

import numpy as np
import pytest

from packet import CHUNK_NAMES, Packet
from stm_utils import (
    crc16_compute,
    crc16_update,
    load_from_npz,
    parse_packet,
    read_packets_from_file,
    save_to_npz,
    unstuff_bytes,
)


# ── Packet ───────────────────────────────────────────────────────────────--
def test_chunk_names_sensor_mapping():
    assert CHUNK_NAMES[0x01] == "gyro"
    assert CHUNK_NAMES[0x02] == "accel"
    assert CHUNK_NAMES[0x03] == "mag"


def test_packet_unknown_sensor_and_sample_count():
    assert Packet(id=0x99, ts=0.0, data=np.zeros((0, 3), dtype=np.int16)).sensor == "unknown_0x99"
    assert Packet(id=0x01, ts=0.0, data=np.zeros((7, 3), dtype=np.int16)).sample_count == 7


# ── Helpers: build synthetic valid packets ──────────────────────────────────
def _stuff_bytes(data: bytes) -> bytes:
    result = []
    for byte in data:
        if byte in (0xFE, 0xFF):
            result += [0xFE, byte ^ 0xFE]
        else:
            result.append(byte)
    return bytes(result)


def _one_packet_payload(timestamp, chunks):
    chunks_bytes = b""
    for chunk_id, samples in chunks:
        chunk_data = b"".join(struct.pack("<hhh", x, y, z) for x, y, z in samples)
        chunks_bytes += bytes([chunk_id]) + struct.pack("<H", len(chunk_data) - 1) + b"\x00" + chunk_data
    payload = struct.pack("<I", timestamp) + b"\x00\x00" + chunks_bytes
    return payload + struct.pack("<H", crc16_compute(payload))


def _build_packet_bytes(timestamp, chunks):
    return b"\xFF\xFF" + _stuff_bytes(_one_packet_payload(timestamp, chunks))


def _build_bin_stream(packets_data):
    stream = b""
    for counter, (timestamp, chunks) in enumerate(packets_data):
        stream += b"\xFF\xFF" + bytes([counter & 0xFF]) + _stuff_bytes(_one_packet_payload(timestamp, chunks))
    return stream


# ── CRC16 + byte unstuffing ─────────────────────────────────────────────--
def test_crc16_update_matches_compute():
    crc = 0xFFFF
    for byte in b"\xAB\xCD\xEF":
        crc = crc16_update(crc, byte)
    assert crc == crc16_compute(b"\xAB\xCD\xEF")
    assert crc16_compute(b"\x01") != crc16_compute(b"\x02")


def test_unstuff_roundtrip_and_incomplete_escape():
    original = bytes(range(256))
    assert unstuff_bytes(_stuff_bytes(original)) == original
    with pytest.raises(ValueError):
        unstuff_bytes(b"\xFE")


# ── parse_packet ─────────────────────────────────────────────────────────--
def test_parse_packet_bad_crc_returns_none():
    data = _build_packet_bytes(1000, [(0x01, [(100, 200, 300)])])
    assert parse_packet(data[:-1] + bytes([data[-1] ^ 0xFF])) is None


def test_parse_packet_single_gyro_chunk():
    samples = [(100, 200, 300), (-100, -200, -300)]
    packets = parse_packet(_build_packet_bytes(5000, [(0x01, samples)]))
    assert packets[0].sensor == "gyro"
    assert packets[0].ts == 5000.0
    assert packets[0].sample_count == 2
    np.testing.assert_array_equal(packets[0].data, np.array(samples, dtype=np.int16))


def test_parse_packet_multiple_chunks_one_timestamp():
    packets = parse_packet(_build_packet_bytes(
        1000, [(0x01, [(10, 20, 30)]), (0x02, [(40, 50, 60), (70, 80, 90)])]))
    assert len(packets) == 2
    assert packets[0].sensor == "gyro" and packets[1].sensor == "accel"
    assert packets[1].sample_count == 2


# ── read_packets_from_file + npz round-trip ─────────────────────────────────
def test_read_packets_from_file(tmp_path):
    stream = _build_bin_stream([
        (1000, [(0x01, [(10, 20, 30)])]),
        (2000, [(0x02, [(40, 50, 60)])]),
        (3000, [(0x03, [(70, 80, 90)])]),
    ])
    bin_file = tmp_path / "test.BIN"
    bin_file.write_bytes(stream)
    assert {p.sensor for p in read_packets_from_file(str(bin_file))} == {"gyro", "accel", "mag"}

    empty = tmp_path / "empty.BIN"
    empty.write_bytes(b"")
    assert read_packets_from_file(str(empty)) == []


def test_save_load_npz_roundtrip(tmp_path):
    samples = np.array([[100, 200, 300], [-100, -200, -300]], dtype=np.int16)
    npz_path = str(tmp_path / "out.npz")
    save_to_npz([Packet(id=0x01, ts=500.0, data=samples)], npz_path)
    loaded = load_from_npz(npz_path)
    assert {p.sensor for p in loaded} == {"gyro"}
    loaded_data = np.vstack([p.data for p in loaded if p.sensor == "gyro"])
    for row in samples:
        assert any(np.array_equal(row, lr) for lr in loaded_data)
